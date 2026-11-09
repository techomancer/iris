//! The tables the X server loads and the scan-out reads for windows that are
//! not the textport's: the X colour map, and the display-ID table that says,
//! for each window's ID, how its pixel words become colours.
//!
//! - 0x12e (`x_color_map`): a start address, a count and that many entries,
//!   each 10:10:10 with red in bits 29:20 (software expectation from the X
//!   server's install routine). Entry `i` lands at `start + i`.
//! - 0x132 (`x_did_map`): a display ID and a word, the ID-table entry
//!   `(cmap << 7) & 0xffff80 | ((planes - 1) & 0xf) << 3 | lum << 2 | grey << 1`
//!   (established layout; the same word the X server stores through the XMAP's
//!   indirect registers 0x118 (index) and 0x11c (data), which `indirect` takes
//!   too). The model reads `cmap` as the first entry of the window's colour map
//!   and `planes` as the width of its index.
//! - 0x1e6 (`display_mode`): zero, a display ID and the X-built mode byte. The
//!   mode byte is a second word, separate from the entry (established from the
//!   X server's case split): bit 0 back buffer shown,
//!   bit 1 stereo, bit 2 RGB below 24 bits, bit 3 colour index or luminance,
//!   bit 4 direct colour; for overlays bits 3:1 hold a code (2, or 3 when double
//!   buffered and starting in the back buffer, plus 4 for an underlay), bit 0 is
//!   set for non-RGB overlays and bit 4 for opaque ones. A 30-bit RGB window
//!   sends no mode bits. What the hardware does with the bits is not read
//!   anywhere; the model uses them to choose the window kind.
//!
//! # Window kinds
//!
//! The kind of a display ID comes from the two words together (`decode`).
//! Main-plane IDs are 0 to 62; 63 is the popup (a 2-plane colour index); 64 to
//! 95 are overlays. Decoded, because the recordings use them: an 8-plane and a
//! 12-plane colour index (the entry's index-plane field 7 and 11) through the X
//! colour map; RGB, from mode bit 2 or 4, and 30-bit RGB (field 13, no map
//! offset, no mode bits), as direct colour; the popup and colour-index overlays
//! of 2, 4 or 8 planes as `Layer`s. Counted and shown black or transparent, with
//! the reason (`Why`): 16-plane luminance, 15-bit RGB overlays and any other
//! width. The pipe treats a colour index as a left-justified 12-bit component (an 8-bit
//! index is its value shifted left 4, `defs::colour_index`), and the draw-pixels path puts
//! an index of 8 bits or fewer at bits 15:8 of a 16-bit lane, which is the top byte the model
//! reads (established for the component, the lane inferred). Which lane holds the index and its
//! offset in the stored bank word are unknown. The ID in the pixel is the painted value: the
//! main ID at bits 7:2 and an overlay or popup group at bits 12:8 (`defs::PAINT_MAIN_SHIFT`); the
//! model keeps one 13-bit ID plane that holds both fields.
//!
//! A window's pixel words are an index shifted left by 8 for the X server's
//! 16-bit main-plane index (a pixel write shifts by the setup's 8), and the
//! colour map index is the word's top byte; for a 12-bit window the index is
//! bits 15:4. Direct colour is `0x00RRGGBB`, as the fills write it.
//!
//! # Baked when loaded
//!
//! What the scan-out reads is a `Baked` snapshot the loads keep up to date: the
//! colour map through the gamma RAM at full table size, direct colour's bytes
//! through it, each ID's window kind and the glyph cursor's colours and rows. A
//! load changes only what it names (a colour map load, the entries it covers; a
//! gamma load, every entry), and a frame takes the snapshot by `Arc`, so a frame
//! on a board with nothing loaded, and one between loads, copies and computes
//! nothing.

use crate::dev::kona::defs::*;
use super::dac_cursor::{Dac, DacCursor, DacDraw};
use std::sync::Arc;

/// Entries in the X colour map.
pub const ENTRIES: usize = 0x8000;
/// Display IDs the table holds: main planes 0 to 62, the popup at 63 and the
/// overlays at 64 to 95. The ID plane's main field is six bits.
pub const IDS: usize = DISPLAY_IDS;

/// What a window's pixels are, resolved from its table entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    /// No table entry for the ID: the textport's own colour map by the word's
    /// low byte (the PROM's console).
    Textport,
    /// An 8-bit index in the word's top byte, through the map from `base`.
    Index8 { base: u32 },
    /// A 12-bit index in the word's bits 15:4, through the map from `base`.
    Index12 { base: u32 },
    /// The word's low 24 bits are red, green and blue, in the colour word of
    /// buffer `bank` (0 front, 1 back): which buffer a double-buffered window
    /// shows follows its swaps.
    Direct { bank: u8 },
    /// The popup or an overlay: a colour index of `planes` bits through the map
    /// from `base`, in a plane of its own, laid over the layer below. Index 0 lets
    /// that layer through unless the window is `opaque` (inferred). `bank` is the
    /// overlay buffer its swaps last showed.
    Layer { base: u32, planes: u8, opaque: bool, bank: u8 },
    /// A table word of zero: an ID the server cleared. Shown black.
    Cleared,
    /// A kind the model does not decode, counted by reason. Shown black (a
    /// main-plane window) or not shown (a layer).
    Unsupported(Why),
}

/// Why a display ID is not decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Why {
    /// A 16-plane luminance or grey window.
    Luminance,
    /// A 15-bit RGB overlay, whose transparency rule is unread.
    Rgb15Overlay,
    /// An index width the model has no kind for, with its planes.
    Planes(u8),
    /// A colour map the table cannot hold from the entry's offset.
    Map,
}

/// The glyph cursor as the scan-out draws it: where its 64 x 64 box starts, its
/// colours and its rows (four words each: source bits, then mask bits).
#[derive(Clone, Debug)]
struct Cursor {
    x0: i64,
    y0: i64,
    third: u32,
    fg: u32,
    bg: u32,
    rows: [u32; 256],
}

/// What a frame reads: the tables, through the gamma RAM, ready to index.
#[derive(Clone, Debug)]
pub struct Baked {
    /// The colour map's entries as opaque `0xAABBGGRR` pixels, `ENTRIES` of them.
    cmap: Vec<u32>,
    /// Direct colour's bytes through the gamma RAM, by channel (red, green, blue).
    lut: [[u8; 256]; 3],
    windows: [Window; IDS],
    default: bool,
    /// Some popup or overlay window is decoded, so a frame has layers to compose.
    layers: bool,
    cursor: Option<Cursor>,
    /// The DAC cursor, drawn after everything else and past the gamma RAM.
    dac_cursor: Option<DacDraw>,
    /// Channel 0 is blanked: the frame is black.
    blank: bool,
}

#[derive(Clone, Debug)]
pub struct Tables {
    /// The colour map's entries as loaded, 10:10:10.
    raw: Vec<u32>,
    /// The gamma RAM: a load address is the entry in bits 11:0, the colour in
    /// 13:12 (red 0, green 1, blue 2), the page in 17:14 and a flag in 18 (what
    /// the flag does is unknown; both halves are held). Zero until loaded.
    gamma: Vec<u16>,
    gamma_loaded: bool,
    /// The XMAP's index register (0x118): which ID or channel the data registers
    /// write, and the set-up word of channel 0, the one frame the model shows. Its
    /// bits 3:0 pick the gamma page and bit 5 means no gamma; bit 4 (external
    /// gamma) is not modelled.
    xmap_index: u32,
    channel0: u32,
    /// A display-ID table word, once one was loaded for the ID.
    did: [Option<u32>; IDS],
    /// The mode word of each ID, 0 until `display_mode` sends one, and the
    /// buffer each ID's swaps last showed.
    mode: [u32; IDS],
    bank: [u8; IDS],
    /// The cursor: its glyph memory, which glyph, whether it shows, and where.
    cursor_ram: Vec<u32>,
    cursor_mode: u32,
    cursor_at: u32,
    cursor_pos: u32,
    /// The DAC's cursor and its registers, and whether the frame is blanked.
    dac: Dac,
    blank: bool,
    baked: Arc<Baked>,
    /// Colour-map and gamma entries past the tables, and cursor words past the
    /// glyph memory, dropped.
    pub dropped: u64,
}

impl Default for Tables {
    fn default() -> Self {
        let mut t = Tables {
            raw: vec![0; ENTRIES],
            gamma: vec![0; GAMMA_ENTRIES],
            gamma_loaded: false,
            xmap_index: 0,
            channel0: 0,
            did: [None; IDS],
            mode: [0; IDS],
            bank: [0; IDS],
            cursor_ram: vec![0; CURSOR_WORDS],
            cursor_mode: 0,
            cursor_at: 0,
            cursor_pos: 0,
            dac: Dac::default(),
            blank: false,
            baked: Arc::new(Baked { cmap: vec![BLACK; ENTRIES], lut: [[0; 256]; 3], windows: [Window::Textport; IDS], default: true, layers: false, cursor: None, dac_cursor: None, blank: false }),
            dropped: 0,
        };
        t.rebake_all();
        t
    }
}

const BLACK: u32 = 0xff00_0000;
/// Words of glyph memory: 64 slots of 260.
pub const CURSOR_SLOT: usize = 260;
const CURSOR_WORDS: usize = 64 * CURSOR_SLOT;
/// Gamma entries held: the 19-bit load address space.
const GAMMA_ENTRIES: usize = 1 << 19;

/// A 12-bit gamma output to the DAC's 8-bit inputs: its top eight bits. Established on the link from
/// the colour map's output chips to the video-output chips, which carry only bits 11:4 of each colour
/// and have no rounding or dither control (read from the board's wiring); inferred end to end, since a chip between
/// them and the DAC is not read to add none.
fn narrow(c: u32) -> u32 {
    c >> 4
}

impl Tables {
    pub fn reset(&mut self) {
        *self = Tables::default();
    }

    /// The tables as the guest loaded them, as the stream module reads and writes them.
    /// What a frame reads (`Baked`) is made again from them on a load; the glyph memory
    /// is the pixel store's chunks'.
    pub fn visit<S: crate::dev::kona::stream::Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { raw: _, gamma: _, gamma_loaded: _, xmap_index: _, channel0: _, did: _, mode: _, bank: _, cursor_ram: _, cursor_mode: _, cursor_at: _, cursor_pos: _, dac: _, blank: _, baked: _, dropped: _ } = self;
        s.vec_u32(&mut self.raw);
        s.bool(&mut self.gamma_loaded);
        if self.gamma_loaded {
            let n = s.len(self.gamma.len());
            if s.loading() {
                self.gamma.clear();
                self.gamma.resize(n, 0);
            }
            s.u16s(&mut self.gamma);
        } else if s.loading() {
            self.gamma.iter_mut().for_each(|g| *g = 0);
        }
        s.u32(&mut self.xmap_index);
        s.u32(&mut self.channel0);
        for e in self.did.iter_mut() {
            s.opt(e, |s, v| s.u32(v));
        }
        s.u32s(&mut self.mode);
        for b in self.bank.iter_mut() {
            s.u8(b);
        }
        s.u32(&mut self.cursor_mode);
        s.u32(&mut self.cursor_at);
        s.u32(&mut self.cursor_pos);
        self.dac.visit(s);
        // `blank` is not saved: the back end derives it from the channels it restores.
        if s.loading() {
            if self.raw.len() != ENTRIES || self.gamma.len() != GAMMA_ENTRIES {
                s.fail("the saved display tables are the wrong size");
                self.raw.resize(ENTRIES, 0);
                self.gamma.resize(GAMMA_ENTRIES, 0);
            }
            self.rebake_all();
        }
    }

    /// The tables a frame reads. Cheap: a reference count.
    pub fn snapshot(&self) -> Arc<Baked> {
        self.baked.clone()
    }

    /// One gamma output: entry `v12` of colour `ch`'s table in the page channel 0
    /// selects (its set-up word's bits 3:0; the page of a channel is chosen per
    /// channel, established), or `v12` itself with no gamma loaded or the channel's
    /// no-gamma bit (5) set. Only bank 0 of the load address is read.
    fn gamma_out(&self, ch: usize, v12: u32) -> u32 {
        if self.gamma_loaded && ChannelSetup::NO_GAMMA.get(self.channel0) == 0 {
            let address = GammaAddress::PAGE.put(GammaAddress::COLOUR.put(v12, ch as u32), ChannelSetup::GAMMA_PAGE.get(self.channel0));
            narrow(self.gamma[address as usize] as u32)
        } else {
            narrow(v12)
        }
    }

    /// A 10:10:10 word as the scan-out shows it: each component widened to 12
    /// bits by replicating its top two bits (`c << 2 | c >> 8`, inferred: the only rule the
    /// board's gamma addressing is known to fit), through the gamma RAM, narrowed to 8 by dropping
    /// the low four. Without a gamma load the RAM is the identity. That the output goes through the
    /// RAM is inferred. The glyph cursor's colours go through it too (inferred); the DAC cursor
    /// does not (`rebake_dac`).
    fn through_gamma(&self, e: u32) -> u32 {
        let w10 = |c: u32| c << 2 | c >> 8;
        BLACK | self.gamma_out(2, w10(e & 0x3ff)) << 16 | self.gamma_out(1, w10((e >> 10) & 0x3ff)) << 8 | self.gamma_out(0, w10((e >> 20) & 0x3ff))
    }

    /// Redo the colour map entries `from..to` of the snapshot.
    fn rebake_colours(&mut self, from: usize, to: usize) {
        let mut cmap = std::mem::take(&mut Arc::make_mut(&mut self.baked).cmap);
        for i in from..to.min(ENTRIES) {
            cmap[i] = self.through_gamma(self.raw[i]);
        }
        Arc::make_mut(&mut self.baked).cmap = cmap;
    }

    /// Redo what the gamma RAM shapes: every colour map entry, direct colour's
    /// bytes and the cursor's colours.
    fn rebake_gamma(&mut self) {
        self.rebake_colours(0, ENTRIES);
        let mut lut = [[0u8; 256]; 3];
        for c in 0..256u32 {
            for ch in 0..3 {
                lut[ch][c as usize] = self.gamma_out(ch, c << 4 | c >> 4) as u8;
            }
        }
        Arc::make_mut(&mut self.baked).lut = lut;
        self.rebake_cursor();
    }

    /// The cursor's glyph memory, for a snapshot.
    pub fn cursor_words(&self) -> &[u32] {
        &self.cursor_ram
    }

    /// Restore the glyph memory from a snapshot (the same length), and redo the cursor.
    pub fn load_cursor_words(&mut self, words: &[u32]) {
        if words.len() == self.cursor_ram.len() {
            self.cursor_ram.copy_from_slice(words);
        }
        self.rebake_cursor();
    }

    /// Redo the glyph cursor: whether it shows, where, and its colours and rows.
    /// The position's fields less the 32 the driver adds give the top-left pixel
    /// of the 64 x 64 box (x from the left, y from the top). The crosshair
    /// (mode 1) is not drawn.
    fn rebake_cursor(&mut self) {
        let at = self.cursor_at as usize;
        let cursor = self.cursor_ram.get(at..at + CURSOR_SLOT).filter(|_| self.cursor_mode == 2).map(|g| Cursor {
            x0: CursorPosition::X.get(self.cursor_pos) as i64 - CursorPosition::BIAS,
            y0: CursorPosition::Y.get(self.cursor_pos) as i64 - CursorPosition::BIAS,
            third: self.through_gamma(g[256]),
            fg: self.through_gamma(g[257]),
            bg: self.through_gamma(g[258]),
            rows: g[..256].try_into().expect("256 words"),
        });
        let b = Arc::make_mut(&mut self.baked);
        b.cursor = cursor;
        self.refresh_default();
    }

    /// Redo what the DAC adds to a frame: its cursor, and the blank.
    fn rebake_dac(&mut self) {
        // The DAC is after gamma: 10-bit components, narrowed to the eight the screen shows.
        let narrow10 = |c: [u16; 3]| BLACK | (c[2] as u32 >> 2) << 16 | (c[1] as u32 >> 2) << 8 | c[0] as u32 >> 2;
        let draw = self.dac.draw(narrow10);
        let b = Arc::make_mut(&mut self.baked);
        b.dac_cursor = draw;
        b.blank = self.blank;
        self.refresh_default();
    }

    /// Blank the frame, or show it: channel 0 blanked, or every channel temporarily.
    pub fn set_blank(&mut self, blank: bool) {
        if self.blank != blank {
            self.blank = blank;
            self.rebake_dac();
        }
    }

    /// A DAC cursor load.
    pub fn load_dac_cursor(&mut self, c: DacCursor) {
        self.dac.load(c);
        self.rebake_dac();
    }

    /// The DAC's state, for the monitor.
    pub fn dac(&self) -> &Dac {
        &self.dac
    }

    fn refresh_default(&mut self) {
        let default = self.baked.cursor.is_none() && self.baked.dac_cursor.is_none() && !self.baked.blank && !self.gamma_loaded && self.channel0 == 0 && self.did.iter().all(Option::is_none) && self.mode.iter().all(|&m| m == 0);
        Arc::make_mut(&mut self.baked).default = default;
    }

    fn rebake_all(&mut self) {
        for id in 0..IDS {
            self.set_window(id);
        }
        self.rebake_gamma();
        self.rebake_dac();
    }

    /// 0x12e: `words` are the entries from `start`.
    pub fn load_colours(&mut self, start: u32, words: &[u32]) {
        for (i, &w) in words.iter().enumerate() {
            match self.raw.get_mut(start as usize + i) {
                Some(e) => *e = w,
                None => self.dropped += 1,
            }
        }
        self.rebake_colours(start as usize, start as usize + words.len());
    }

    /// 0x12f: `words` are gamma entries from load address `start` (entry in bits
    /// 11:0, colour 13:12, page 17:14, flag 18). An address past the 19 bits is
    /// dropped and counted. Which page and bank the display reads is the channel's
    /// choice (`indirect`), page 0 until one is sent.
    pub fn load_gamma(&mut self, start: u32, words: &[u32]) {
        self.gamma_loaded = true;
        for (i, &w) in words.iter().enumerate() {
            let at = start as usize + i;
            match self.gamma.get_mut(at) {
                Some(e) => *e = w as u16 & 0xfff,
                None => self.dropped += 1,
            }
        }
        self.rebake_gamma();
        self.refresh_default();
    }

    /// 0x130: a glyph's 260 words at glyph-memory address `at`.
    pub fn load_cursor(&mut self, at: u32, words: &[u32]) {
        for (i, &w) in words.iter().enumerate() {
            match self.cursor_ram.get_mut(at as usize + i) {
                Some(e) => *e = w,
                None => self.dropped += 1,
            }
        }
        self.rebake_cursor();
    }

    /// A store to the display's indirect register, the XMAP's: 0x101 (glyph
    /// address), 0x102 (position) and 0x103 (cursor mode: 0 off, 1 crosshair, 2
    /// glyph) are the cursor's; 0x118 selects an ID or channel, and 0x11c writes
    /// that ID's table entry and 0x11d that channel's set-up word. Others are not
    /// modelled and are ignored.
    pub fn indirect(&mut self, selector: u32, value: u32) {
        match selector {
            XMAP_CURSOR_IMAGE_BASE => self.cursor_at = value,
            XMAP_CURSOR_POSITION => self.cursor_pos = value,
            XMAP_CURSOR_MODE => self.cursor_mode = value,
            XMAP_INDEX => self.xmap_index = value,
            XMAP_ID_TABLE_DATA => return self.load_did(self.xmap_index, value),
            XMAP_CHANNEL_SETUP_DATA => {
                // Only channel 0, the one frame the model shows, is kept.
                if self.xmap_index == 0 {
                    self.channel0 = value;
                    self.rebake_gamma();
                    self.refresh_default();
                }
                return;
            }
            _ if self.dac.write(selector, value) => return self.rebake_dac(),
            _ => return,
        }
        self.rebake_cursor();
    }

    /// 0x132: the table word of display ID `id`.
    pub fn load_did(&mut self, id: u32, word: u32) {
        if let Some(e) = self.did.get_mut(id as usize) {
            *e = Some(word);
            self.set_window(id as usize);
            self.refresh_default();
        }
    }

    /// The buffer display ID `id`'s swaps last showed (0 before any).
    pub fn bank(&self, id: u32) -> u8 {
        self.bank.get(id as usize).copied().unwrap_or(0)
    }

    /// A swap (0x1e6 with a window id): the buffer display ID `id` shows now.
    pub fn load_bank(&mut self, id: u32, bank: u8) {
        if let Some(e) = self.bank.get_mut(id as usize) {
            *e = bank;
            self.set_window(id as usize);
        }
    }

    /// 0x1e6: the mode word of display ID `id`.
    pub fn load_mode(&mut self, id: u32, mode: u32) {
        if let Some(e) = self.mode.get_mut(id as usize) {
            *e = mode;
            self.set_window(id as usize);
            self.refresh_default();
        }
    }

    /// What display ID `id`'s pixels are, from its table entry and mode byte
    /// together (see the module doc).
    fn decode(&self, id: usize) -> Window {
        let (mode, entry) = (self.mode[id], self.did[id]);
        let bank = self.bank[id];
        if id >= POPUP_ID {
            return Self::decode_layer(id, mode, entry, bank);
        }
        // RGB below 24 bits (mode bit 2) or direct colour (bit 4) says what the
        // words are, entry or none.
        if ModeByte::RGB.get(mode) != 0 || ModeByte::DIRECT_OR_OPAQUE.get(mode) != 0 {
            return Window::Direct { bank };
        }
        let Some(w) = entry else { return Window::Textport };
        if w == 0 {
            return Window::Cleared;
        }
        let (base, field) = (DisplayIdEntry::COLOUR_MAP_BASE.get(w), DisplayIdEntry::INDEX_PLANES.get(w));
        let fits = |n: usize| base as usize + n <= ENTRIES;
        match field {
            // A 30-bit RGB window sends no mode bits, and its entry has no map
            // offset, no luminance or grey bit and an index-plane field of 13 (the
            // 30 planes less one, kept to four bits). The recorded GL windows show
            // exactly that, 0x68, beside the 24-bit RGB ones that say so in mode bit
            // 0x4 (above). A 24-plane entry also has no offset and a field of 7, as
            // an 8-plane index window's does, so only 13 is read as colour here;
            // whether the hardware tells them apart by anything but the mode byte
            // is not known.
            13 if base == 0 && DisplayIdEntry::LUMINANCE.get(w) == 0 && DisplayIdEntry::GREY.get(w) == 0 && ModeByte::COLOUR_INDEX.get(mode) == 0 => Window::Direct { bank },
            7 if fits(256) => Window::Index8 { base },
            11 if fits(4096) => Window::Index12 { base },
            7 | 11 => Window::Unsupported(Why::Map),
            15 => Window::Unsupported(Why::Luminance),
            f => Window::Unsupported(Why::Planes(f as u8 + 1)),
        }
    }

    /// The popup (ID 63) or an overlay (64 to 95). A colour-index layer of 2, 4 or
    /// 8 planes is decoded; an overlay whose mode byte has bit 0 clear is RGB (15
    /// bits), whose transparency rule is unread, and is counted. An opaque overlay
    /// (bit 4) replaces what is under it whatever its index.
    fn decode_layer(id: usize, mode: u32, entry: Option<u32>, bank: u8) -> Window {
        let Some(w) = entry else { return Window::Textport };
        if w == 0 {
            return Window::Cleared;
        }
        if id >= OVERLAY_ID_FIRST && ModeByte::OVERLAY_NON_RGB.get(mode) == 0 {
            return Window::Unsupported(Why::Rgb15Overlay);
        }
        let (base, planes) = (DisplayIdEntry::COLOUR_MAP_BASE.get(w), DisplayIdEntry::INDEX_PLANES.get(w) as u8 + 1);
        match planes {
            2 | 4 | 8 if base as usize + (1 << planes) <= ENTRIES => Window::Layer { base, planes, opaque: id >= OVERLAY_ID_FIRST && ModeByte::DIRECT_OR_OPAQUE.get(mode) != 0, bank },
            2 | 4 | 8 => Window::Unsupported(Why::Map),
            n => Window::Unsupported(Why::Planes(n)),
        }
    }

    /// Decode display ID `id` into the snapshot and note whether any layer is in use.
    fn set_window(&mut self, id: usize) {
        let w = self.decode(id);
        let b = Arc::make_mut(&mut self.baked);
        b.windows[id] = w;
        b.layers = b.windows[POPUP_ID..].iter().any(|w| matches!(w, Window::Layer { .. }));
    }

    /// A line for the monitor: what is loaded, and the kinds not decoded by reason.
    pub fn describe(&self) -> String {
        let ids = self.did.iter().filter(|e| e.is_some()).count();
        let modes = self.mode.iter().filter(|&&m| m != 0).count();
        let cleared = self.baked.windows.iter().filter(|&&w| w == Window::Cleared).count();
        let mut why: std::collections::BTreeMap<Why, usize> = Default::default();
        for w in &self.baked.windows {
            if let Window::Unsupported(y) = w {
                *why.entry(*y).or_default() += 1;
            }
        }
        let unsupported: usize = why.values().sum();
        let reasons = if why.is_empty() { String::new() } else { format!(": {}", why.iter().map(|(y, n)| format!("{n} {y:?}")).collect::<Vec<_>>().join(", ")) };
        format!(
            "{ids} display-ID entries ({cleared} cleared, {unsupported} not decoded{reasons}), {modes} mode words, gamma {}, cursor mode {} at {:#x}, {} entries dropped",
            if self.gamma_loaded { "loaded" } else { "none" },
            self.cursor_mode,
            self.cursor_at,
            self.dropped
        )
    }
}

impl Baked {
    /// Whether nothing but the textport's own colour map is in use.
    /// The frame is blanked: black, whatever the store holds.
    pub fn is_blank(&self) -> bool {
        self.blank
    }

    /// A DAC cursor is drawn.
    pub fn has_dac_cursor(&self) -> bool {
        self.dac_cursor.is_some()
    }

    /// Draw the DAC cursor over the composed frame, as `draw_cursor` does the XMAP's.
    pub fn draw_dac_cursor(&self, out: &mut [u32], stride: usize, w: usize, h: usize) {
        if let Some(d) = &self.dac_cursor {
            d.draw(out, stride, w, h);
        }
    }

    pub fn is_default(&self) -> bool {
        self.default
    }

    pub fn window(&self, id: usize) -> Window {
        self.windows[id]
    }

    /// Whether a popup or an overlay is decoded, so a frame has layers to compose.
    pub fn has_layers(&self) -> bool {
        self.layers
    }

    /// The colour of index `v` of a layer whose map starts at `base`; `Layer` was
    /// only made with its whole map inside the table.
    pub fn layer(&self, base: u32, v: u32) -> u32 {
        self.cmap[base as usize + v as usize]
    }

    /// Entry `i` of the colour map as an opaque `0xAABBGGRR` pixel; black past
    /// the table. The frame reads whole windows of it (`index8`, `index12`).
    #[cfg(test)]
    pub fn colour(&self, i: usize) -> u32 {
        self.cmap.get(i).copied().unwrap_or(BLACK)
    }

    /// The 256 entries from `base` of an 8-bit window; `base + 256` is inside the
    /// table, as `Window::Index8` was only made so.
    pub fn index8(&self, base: u32) -> &[u32; 256] {
        self.cmap[base as usize..base as usize + 256].try_into().expect("the window was checked against the table")
    }

    /// The 4096 entries from `base` of a 12-bit window.
    pub fn index12(&self, base: u32) -> &[u32; 4096] {
        self.cmap[base as usize..base as usize + 4096].try_into().expect("the window was checked against the table")
    }

    /// A direct-colour word `0x00RRGGBB` as an opaque `0xAABBGGRR` pixel, through
    /// the gamma RAM.
    pub fn direct(&self, word: u32) -> u32 {
        BLACK | (self.lut[2][(word & 0xff) as usize] as u32) << 16 | (self.lut[1][((word >> 8) & 0xff) as usize] as u32) << 8 | self.lut[0][((word >> 16) & 0xff) as usize] as u32
    }

    /// The glyph cursor as the host draws a pointer: its 64 x 64 box's top-left and
    /// which pixels are opaque (any of the pixel's source and mask bits, as
    /// `draw_cursor` shows it). None with the cursor off or the crosshair, whose shape
    /// is unknown, so the host keeps its own pointer.
    pub fn hw_cursor(&self) -> Option<crate::gfx_display::HwCursor> {
        let Some(c) = self.cursor.as_ref() else {
            let d = self.dac_cursor.as_ref()?;
            let mask = (0..super::dac_cursor::SIZE).map(|y| (0..super::dac_cursor::SIZE).fold(0u64, |m, x| m | ((d.cursor.pixel(x, y) >= 2) as u64) << x)).collect();
            return Some(crate::gfx_display::HwCursor { x: d.x0, y: d.y0, size: 64, mask });
        };
        let mask = (0..64usize)
            .map(|r| {
                let row = &c.rows[4 * r..4 * r + 4];
                (0..64usize).fold(0u64, |m, col| {
                    let bit = |hi: u32, lo: u32| if col < 32 { hi >> (31 - col) & 1 } else { lo >> (63 - col) & 1 };
                    m | ((bit(row[0], row[1]) | bit(row[2], row[3])) as u64) << col
                })
            })
            .collect();
        Some(crate::gfx_display::HwCursor { x: c.x0 as i32, y: c.y0 as i32, size: 64, mask })
    }

    /// Draw the glyph cursor into `out`, a frame `stride` pixels a row, top row
    /// first, `w` x `h` of it the screen. A glyph is 64 rows of four words, the
    /// first two the row's source bits and the next two its mask bits, most
    /// significant bit leftmost; words 256 to 258 are the third colour, the
    /// foreground and the background, 10:10:10. A pixel with only its mask bit
    /// takes the background, only its source bit the foreground, both the third
    /// colour, neither leaves the screen. Which bit pair means which colour is
    /// inferred from one recorded arrow (white outline from the mask, red fill
    /// from the source), as is the bit order. Colours go through the gamma RAM
    /// as the colour map's do.
    ///
    /// Known gaps: the plan reads the glyph as a two-bit value picking colour word
    /// `CURSOR_COLOUR_WORDS + (3 - v)` with no transparent value; this rule, with
    /// (0, 0) leaving the screen, is what one recorded arrow showed, and the two
    /// disagree about transparency. The crosshair (mode 1) is not drawn. The DAC
    /// cursor is drawn by `draw_dac_cursor`.
    pub fn draw_cursor(&self, out: &mut [u32], stride: usize, w: usize, h: usize) {
        let Some(c) = &self.cursor else { return };
        for r in 0..64usize {
            let y = c.y0 + r as i64;
            if y < 0 || y >= h as i64 {
                continue;
            }
            let row = &c.rows[4 * r..4 * r + 4];
            for col in 0..64usize {
                let x = c.x0 + col as i64;
                if x < 0 || x >= w as i64 {
                    continue;
                }
                let bit = |hi: u32, lo: u32| if col < 32 { hi >> (31 - col) & 1 } else { lo >> (63 - col) & 1 };
                let px = match (bit(row[0], row[1]), bit(row[2], row[3])) {
                    (0, 0) => continue,
                    (1, 0) => c.fg,
                    (0, 1) => c.bg,
                    _ => c.third,
                };
                out[y as usize * stride + x as usize] = px;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_load_at_their_address_and_narrow_to_eight_bits() {
        let mut t = Tables::default();
        t.load_colours(0x2d00, &[0x3ff << 20, 0x3ff << 10, 0x3ff, 0x1ff << 20 | 0x1ff << 10 | 0x1ff]);
        assert_eq!(t.snapshot().colour(0x2d00), 0xff00_00ff, "red is the low byte of the pixel");
        assert_eq!(t.snapshot().colour(0x2d01), 0xff00_ff00);
        assert_eq!(t.snapshot().colour(0x2d02), 0xffff_0000);
        assert_eq!(t.snapshot().colour(0x2d03), 0xff7f_7f7f, "511 of 1023 widens to 2047 of 4095, 127 of 255");
        t.load_colours(ENTRIES as u32 - 1, &[1, 2]);
        assert_eq!(t.dropped, 1, "an entry past the table is dropped and counted");
        assert_eq!(t.snapshot().colour(ENTRIES + 5), 0xff00_0000);
    }

    /// The table words the X server sent for the greeter: 8-bit windows at
    /// 0x2c00, 0x2d00 and so on, 12-bit ones at 0x3800.., the popup a two-bit
    /// window the model does not read.
    #[test]
    fn a_display_id_word_names_a_colour_map_and_an_index_width() {
        let mut t = Tables::default();
        assert_eq!(t.snapshot().window(1), Window::Textport);
        assert!(t.snapshot().is_default());
        t.load_did(1, 0x16803c);
        t.load_did(6, 0x1c005c);
        t.load_did(0x3f, 0x140808);
        t.load_did(9, 0x7e);
        t.load_mode(9, 8);
        assert_eq!(t.snapshot().window(1), Window::Index8 { base: 0x2d00 });
        assert_eq!(t.snapshot().window(6), Window::Index12 { base: 0x3800 });
        assert_eq!(t.snapshot().window(0x3f), Window::Layer { base: 0x2810, planes: 2, opaque: false, bank: 0 }, "the popup: a 2-plane map");
        assert_eq!(t.snapshot().window(9), Window::Unsupported(Why::Luminance), "a 16-plane luminance window is counted, not read");
        // A window with no map offset that is not colour index holds colour: 0x38 with mode 4, 0x68 with mode 0.
        t.load_did(20, 0x38);
        t.load_mode(20, 4);
        t.load_did(21, 0x68);
        assert_eq!([t.snapshot().window(20), t.snapshot().window(21)], [Window::Direct { bank: 0 }; 2]);
        t.load_bank(21, 1);
        assert_eq!(t.snapshot().window(21), Window::Direct { bank: 1 }, "a swap names the buffer shown");
        assert!(!t.snapshot().is_default());
        t.load_did(2, 0);
        assert_eq!(t.snapshot().window(2), Window::Cleared, "a word of zero is an ID the server cleared");
        assert!(t.describe().contains("7 display-ID entries (1 cleared, 1 not decoded: 1 Luminance)"), "{}", t.describe());
    }

    /// A mode word that says RGB makes the window direct colour, whatever the
    /// table says; a colour-index mode with no table entry stays the default.
    #[test]
    fn a_mode_word_that_says_rgb_makes_the_window_direct() {
        let mut t = Tables::default();
        t.load_mode(10, 4);
        t.load_mode(11, 0x14);
        t.load_mode(2, 8);
        t.load_did(11, 0);
        assert_eq!([t.snapshot().window(10), t.snapshot().window(11), t.snapshot().window(2)], [Window::Direct { bank: 0 }, Window::Direct { bank: 0 }, Window::Textport]);
        assert_eq!(t.snapshot().direct(0x00_11_22_33), 0xff33_2211, "no gamma: the identity");
        t.load_did(99, 1);
        t.load_mode(99, 4);
        t.reset();
        assert!(t.snapshot().is_default());
        assert_eq!(t.describe(), Tables::default().describe());
    }
}

#[cfg(test)]
mod gamma_tests {
    use super::*;

    /// The gamma RAM maps a widened component to 12 bits and the table narrows
    /// that to 8; a load past the first page is dropped and counted.
    #[test]
    fn the_gamma_ram_shapes_the_colour_map_and_direct_colour() {
        let mut t = Tables::default();
        // Red halves, green is the identity, blue is zero.
        let red: Vec<u32> = (0..4096).map(|i| i / 2).collect();
        let green: Vec<u32> = (0..4096).collect();
        t.load_gamma(0, &red);
        t.load_gamma(0x1000, &green);
        t.load_gamma(0x2000, &[0; 4096]);
        assert!(!t.snapshot().is_default());
        t.load_colours(0, &[0x3ff << 20 | 0x3ff << 10 | 0x3ff]);
        assert_eq!(t.snapshot().colour(0), 0xff00_ff7f, "red 4095 -> 2047 -> 127 of 255; green full; blue none");
        assert_eq!(t.snapshot().direct(0x00ff_ffff), 0xff00_ff7f);
        t.load_gamma((1 << 19) - 1, &[1, 2]);
        assert_eq!(t.dropped, 1, "the last entry fits, the next is past the address space");
    }
}

#[cfg(test)]
mod cursor_tests {
    use super::*;

    const RED: u32 = 0x3ff << 20;
    const WHITE: u32 = 0x3fff_ffff;
    const GREEN: u32 = 0x3ff << 10;

    fn glyph() -> Vec<u32> {
        let mut g = vec![0u32; CURSOR_SLOT];
        // Row 0: pixel 0 source only, pixel 1 mask only. Row 1: pixel 0 both.
        g[0] = 0x8000_0000;
        g[2] = 0x4000_0000;
        g[4] = 0x8000_0000;
        g[6] = 0x8000_0000;
        // A pixel in the right half of a row: the second word of each pair.
        g[8 + 1] = 1;
        g[256] = GREEN;
        g[257] = RED;
        g[258] = WHITE;
        g
    }

    /// A glyph cursor lands with its 64 x 64 box's corner at the position less
    /// the 32 the driver adds; each bit pair picks a colour or leaves the pixel.
    #[test]
    fn a_glyph_cursor_is_drawn_where_its_registers_say() {
        let mut t = Tables::default();
        t.load_cursor(260, &glyph());
        t.indirect(0x101, 260);
        t.indirect(0x102, 34 | 33 << 13);
        t.indirect(0x103, 2);
        let (w, h) = (80usize, 80usize);
        let mut out = vec![7u32; w * h];
        t.snapshot().draw_cursor(&mut out, w, w, h);
        let at = |x: usize, y: usize| out[y * w + x];
        assert_eq!(at(2, 1), 0xff00_00ff, "source only: the foreground");
        assert_eq!(at(3, 1), 0xffff_ffff, "mask only: the background");
        assert_eq!(at(4, 1), 7, "neither: the screen");
        assert_eq!(at(2, 2), 0xff00_ff00, "both: the third colour");
        assert_eq!(at(2 + 63, 1 + 2), 0xff00_00ff, "the right half is the second word of each pair");
        assert_eq!(at(1, 1), 7);
        // A cursor off, or the crosshair, draws nothing; a glyph off the frame is clipped.
        t.indirect(0x103, 1);
        let mut out = vec![7u32; w * h];
        t.snapshot().draw_cursor(&mut out, w, w, h);
        assert!(out.iter().all(|&p| p == 7));
        t.indirect(0x103, 2);
        t.indirect(0x102, 0);
        t.snapshot().draw_cursor(&mut out, w, w, h);
        assert_eq!(at_of(&out, w, 0, 0), 7, "its corner is 32 left of the frame");
        assert!(t.dropped == 0);
        t.load_cursor(CURSOR_WORDS as u32 - 1, &[1, 2]);
        assert_eq!(t.dropped, 1);
    }

    fn at_of(out: &[u32], w: usize, x: usize, y: usize) -> u32 {
        out[y * w + x]
    }

    /// The host's pointer follows the registers: nothing with the cursor off or the
    /// crosshair, the glyph's opaque pixels and top-left otherwise.
    #[test]
    fn the_host_cursor_is_read_from_the_registers() {
        let mut t = Tables::default();
        assert_eq!(t.snapshot().hw_cursor(), None, "out of reset the cursor is off");
        t.load_cursor(260, &glyph());
        t.indirect(0x101, 260);
        t.indirect(0x102, 34 | 33 << 13);
        assert_eq!(t.snapshot().hw_cursor(), None, "loaded but not enabled");
        t.indirect(0x103, 1);
        assert_eq!(t.snapshot().hw_cursor(), None, "the crosshair keeps the host's pointer");
        t.indirect(0x103, 2);
        let c = t.snapshot().hw_cursor().expect("enabled");
        assert_eq!((c.x, c.y, c.size, c.mask.len()), (2, 1, 64, 64));
        assert_eq!(c.mask[0], 0b11, "row 0: pixel 0 source only and pixel 1 mask only are both opaque");
        assert_eq!(c.mask[1], 1, "row 1: pixel 0 both");
        assert_eq!(c.mask[2], 1 << 63, "row 2: the last pixel, in the second word of the pair");
        assert_eq!(c.mask[3], 0);
        t.indirect(0x102, 0);
        let c = t.snapshot().hw_cursor().expect("enabled");
        assert_eq!((c.x, c.y), (-32, -32), "the box starts 32 left of and above the frame");
    }

    /// A shown cursor takes the frame off the default composition, so it is
    /// drawn even with no table loaded.
    #[test]
    fn a_shown_cursor_is_not_the_default() {
        let mut t = Tables::default();
        assert!(t.snapshot().is_default());
        t.indirect(0x103, 2);
        assert!(!t.snapshot().is_default());
        t.indirect(0x103, 0);
        assert!(t.snapshot().is_default());
    }
}

#[cfg(test)]
mod kind_tests {
    use super::*;

    /// The kind comes from the entry and the mode byte together: the same
    /// 8-plane entry is an index window with a colour-index mode, and an RGB one
    /// when the mode byte says RGB or direct colour.
    #[test]
    fn the_mode_byte_and_the_entry_together_choose_the_kind() {
        let mut t = Tables::default();
        t.load_did(1, 0x16803c);
        assert_eq!(t.snapshot().window(1), Window::Index8 { base: 0x2d00 });
        t.load_mode(1, 8);
        assert_eq!(t.snapshot().window(1), Window::Index8 { base: 0x2d00 }, "colour index says what the entry says");
        t.load_mode(1, 4);
        assert_eq!(t.snapshot().window(1), Window::Direct { bank: 0 }, "RGB below 24 bits");
        t.load_mode(1, 0x10);
        assert_eq!(t.snapshot().window(1), Window::Direct { bank: 0 }, "direct colour");
        // A 24-plane entry (field 7, no map offset) with no mode bits is an 8-plane index window.
        t.load_did(2, 0x38);
        assert_eq!(t.snapshot().window(2), Window::Index8 { base: 0 });
        // A 30-plane entry with no mode bits is RGB; with the colour-index bit it is not decoded.
        t.load_did(3, 0x68);
        assert_eq!(t.snapshot().window(3), Window::Direct { bank: 0 });
        t.load_mode(3, 8);
        assert_eq!(t.snapshot().window(3), Window::Unsupported(Why::Planes(14)));
        // 16-plane luminance, and a map that does not fit.
        t.load_did(4, 0x7e);
        assert_eq!(t.snapshot().window(4), Window::Unsupported(Why::Luminance));
        t.load_did(5, ((ENTRIES as u32 - 8) << 7) | 7 << 3);
        assert_eq!(t.snapshot().window(5), Window::Unsupported(Why::Map));
        assert!(t.describe().contains("3 not decoded: 1 Luminance, 1 Planes(14), 1 Map"), "{}", t.describe());
    }

    /// The popup is ID 63 and overlays 64 to 95; an overlay is decoded from its
    /// entry when its mode byte says non-RGB, counted when RGB, and opaque with
    /// mode bit 4. The XMAP's indirect registers write the same table.
    #[test]
    fn popup_and_overlay_ids_decode_as_layers() {
        let mut t = Tables::default();
        t.load_did(64, 0x1400 << 7 | 7 << 3);
        t.load_mode(64, 0x05);
        assert_eq!(t.snapshot().window(64), Window::Layer { base: 0x1400, planes: 8, opaque: false, bank: 0 });
        t.load_mode(64, 0x15);
        assert_eq!(t.snapshot().window(64), Window::Layer { base: 0x1400, planes: 8, opaque: true, bank: 0 });
        t.load_mode(64, 0x04);
        assert_eq!(t.snapshot().window(64), Window::Unsupported(Why::Rgb15Overlay), "an RGB overlay's transparency rule is unread");
        t.load_mode(64, 0x05);
        t.load_did(65, 0x1500 << 7 | 3 << 3);
        t.load_mode(65, 1);
        assert_eq!(t.snapshot().window(65), Window::Layer { base: 0x1500, planes: 4, opaque: false, bank: 0 });
        t.load_bank(65, 1);
        assert_eq!(t.snapshot().window(65), Window::Layer { base: 0x1500, planes: 4, opaque: false, bank: 1 });
        assert!(t.snapshot().has_layers());
        t.load_did(200, 1);
        t.indirect(0x118, 66);
        t.indirect(0x11c, 0x1600 << 7 | 1 << 3);
        t.load_mode(66, 1);
        assert_eq!(t.snapshot().window(66), Window::Layer { base: 0x1600, planes: 2, opaque: false, bank: 0 }, "the XMAP's index and data registers write the entry");
        t.load_did(67, 0x16 << 7 | 4 << 3);
        t.load_mode(67, 1);
        assert_eq!(t.snapshot().window(67), Window::Unsupported(Why::Planes(5)));
    }
}

#[cfg(test)]
mod page_tests {
    use super::*;

    /// Gamma pages are chosen per channel by the set-up word's bits 3:0, loaded
    /// at the address's bits 17:14; bit 5 of the word means no gamma.
    #[test]
    fn a_channel_picks_its_gamma_page_and_can_have_none() {
        let mut t = Tables::default();
        let ident: Vec<u32> = (0..4096).collect();
        let half: Vec<u32> = (0..4096).map(|i| i / 2).collect();
        for ch in 0..3u32 {
            t.load_gamma(ch << 12, &ident);
            t.load_gamma(1 << 14 | ch << 12, &half);
        }
        let white = 0x3ff << 20 | 0x3ff << 10 | 0x3ff;
        t.load_colours(0, &[white]);
        assert_eq!(t.snapshot().colour(0), 0xffff_ffff, "page 0 is the identity");
        t.indirect(0x118, 0);
        t.indirect(0x11d, 1);
        assert_eq!(t.snapshot().colour(0), 0xff7f_7f7f, "page 1 halves");
        assert_eq!(t.snapshot().direct(0x00ff_ffff), 0xff7f_7f7f, "direct colour too");
        t.indirect(0x11d, 0x21);
        assert_eq!(t.snapshot().colour(0), 0xffff_ffff, "bit 5: no gamma, whatever the page");
        // Another channel's set-up word is not the displayed channel's.
        t.indirect(0x118, 1);
        t.indirect(0x11d, 1);
        assert_eq!(t.snapshot().colour(0), 0xffff_ffff);
        t.load_gamma(1 << 19, &[1]);
        assert_eq!(t.dropped, 1, "an address past the 19 bits is dropped");
    }
}
