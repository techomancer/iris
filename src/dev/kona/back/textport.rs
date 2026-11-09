//! The textport: the PROM's text console, drawn before any window exists.
//!
//! Commands, after the textport's sync word: 2 sets the draw colour, 4 the
//! raster position, 5 fills a box, 6 plots a point and a bitmap blit draws a
//! glyph or icon at the raster position, and 3 loads the textport's colour map
//! (an index and three 10-bit components). 1, 7 and 8 are framed and counted,
//! with no drawing role known. 9 and 0xc are answered by the front end.
//!
//! Grades. The command meanings are software expectations; a textport pixel is
//! a colour index (inferred from command 2 taking an index, 0 to 255). The
//! pixels go to the colour word of a single-buffered level-0 window, word 1 of
//! the record: no buffer configuration precedes the console, so this default is
//! the model's. A box lights columns and rows by the fill rule. A bitmap's
//! header is width, height, x origin, y origin, x advance and y advance, the
//! last four signed 16-bit; pixel (column c, row r) lands at (raster x - x
//! origin + c, raster y - y origin + r), row 0 lowest, and the raster position
//! then advances by the advances. That placement is approximated: the model's
//! reading, matched to the glyph routines that send it, not confirmed.

use crate::dev::kona::back::display::ColourMap;
use crate::dev::kona::defs::*;
use crate::dev::kona::back::raster::Surface;
use parking_lot::Mutex;
use std::sync::Arc;

/// The word of a pixel's record the textport draws into: the main colour word of
/// the planes' layout, which the display composes and the GL path draws.
pub const COLOUR_WORD: usize = crate::dev::kona::back::store::layout::MAIN_COLOUR_WORD as usize;

#[derive(Debug, Default)]
pub struct Textport {
    colour: u32,
    raster: (i32, i32),
    /// Shared with the display, which copies it once a frame.
    colour_map: Arc<Mutex<ColourMap>>,
    /// Commands that drew, and those accepted without effect.
    pub drawn: u64,
    pub unmodelled: u64,
}

impl Textport {
    /// A textport drawing with `map` as its colour map.
    pub fn with_map(map: Arc<Mutex<ColourMap>>) -> Textport {
        Textport { colour_map: map, ..Textport::default() }
    }

    pub fn colour(&self) -> u32 {
        self.colour
    }

    pub fn raster(&self) -> (i32, i32) {
        self.raster
    }

    #[cfg(test)]
    /// The colour map a textport pixel (an index) is resolved through at
    /// scan-out.
    pub fn colour_map(&self) -> ColourMap {
        self.colour_map.lock().clone()
    }

    /// The colour, the raster position and the colour map, as the stream module reads and
    /// writes them.
    pub fn visit<S: crate::dev::kona::stream::Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { colour: _, raster: _, colour_map: _, drawn: _, unmodelled: _ } = self;
        s.u32(&mut self.colour);
        s.i32(&mut self.raster.0);
        s.i32(&mut self.raster.1);
        self.colour_map.lock().visit(s);
    }

    /// Power-on state.
    pub fn reset(&mut self) {
        let map = self.colour_map.clone();
        map.lock().reset();
        *self = Textport::with_map(map);
    }

    /// A textport command. False if `selector` is not one.
    /// `surface` makes the surface a paint draws on, only when one does.
    pub fn command<'a>(&mut self, surface: impl FnOnce() -> Surface<'a>, selector: u32, args: &[u32]) -> bool {
        let a = |i: usize| args.get(i).copied().unwrap_or(0);
        match selector {
            TEXTPORT_SET_COLOUR => self.colour = a(0) & 0xff,
            TEXTPORT_SET_RASTER_POSITION => self.raster = (a(0) as i32, a(1) as i32),
            TEXTPORT_FILL_BOX => {
                surface().fill_box(a(0) as i32 as i64, a(1) as i32 as i64, a(2) as i32 as i64, a(3) as i32 as i64, self.colour);
                self.drawn += 1;
            }
            TEXTPORT_PLOT_POINT => {
                surface().plot(a(0) as i32 as i64, a(1) as i32 as i64, self.colour);
                self.drawn += 1;
            }
            TEXTPORT_LOAD_COLOUR_MAP => self.colour_map.lock().load(args),
            TEXTPORT_UNMODELLED_1 | TEXTPORT_UNMODELLED_7 | TEXTPORT_UNMODELLED_8 => self.unmodelled += 1,
            _ => return false,
        }
        true
    }

    /// A bitmap blit: six header words, then the bits.
    pub fn bitmap(&mut self, surface: &Surface<'_>, header: &[u32], payload: &[u32]) {
        let [w, h, xo, yo, xa, ya] = header else { return };
        let (xo, yo, xa, ya) = (*xo as i16 as i64, *yo as i16 as i64, *xa as i16 as i64, *ya as i16 as i64);
        let (rx, ry) = (self.raster.0 as i64, self.raster.1 as i64);
        surface.bitmap(rx - xo, ry - yo, *w, *h, payload, self.colour);
        // The position advances with wrapping 32-bit arithmetic: its range is
        // unknown.
        self.raster = ((rx + xa) as i32, (ry + ya) as i32);
        self.drawn += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::kona::back::store::PixelStore;


    /// What the tests draw on: the store's colour word, with nothing queued.
    fn sf<'a>(s: &'a PixelStore) -> impl FnOnce() -> Surface<'a> {
        move || Surface::new(s, COLOUR_WORD)
    }

    fn tp() -> (Textport, PixelStore) {
        (Textport::default(), PixelStore::new(32, 32))
    }

    #[test]
    fn colour_then_a_box_draws_the_index() {
        let (mut t, s) = tp();
        assert!(t.command(sf(&s), 2, &[0x1ff]));
        assert_eq!(t.colour(), 0xff, "an index is a byte");
        assert!(t.command(sf(&s), 5, &[1, 1, 4, 3]));
        assert_eq!(s.read(2, 2, COLOUR_WORD), 0xff);
        assert_eq!(s.read(1, 2, COLOUR_WORD), 0);
        assert_eq!(s.read(4, 3, COLOUR_WORD), 0xff);
        assert_eq!(s.allocated(), [COLOUR_WORD]);
    }

    /// A colour-map load reaches the map the display resolves through.
    #[test]
    fn a_colour_map_load_sets_an_entry() {
        let (mut t, s) = tp();
        assert!(t.command(sf(&s), 3, &[7, 1020, 512, 0]));
        assert_eq!(t.colour_map().get(7), 0xff00_80ff);
        assert_eq!(t.unmodelled, 0);
    }

    #[test]
    fn a_point_and_the_raster_position() {
        let (mut t, s) = tp();
        t.command(sf(&s), 2, &[9]);
        t.command(sf(&s), 6, &[7, 8]);
        t.command(sf(&s), 4, &[10, 20]);
        assert_eq!(s.read(7, 8, COLOUR_WORD), 9);
        assert_eq!(t.raster(), (10, 20));
        assert_eq!(t.drawn, 1, "the position draws nothing");
    }

    /// A box with negative corners (sent as 32-bit words) is clipped.
    #[test]
    fn corners_are_signed() {
        let (mut t, s) = tp();
        t.command(sf(&s), 2, &[1]);
        t.command(sf(&s), 5, &[-5i32 as u32, -5i32 as u32, 2, 2]);
        assert_eq!(s.read(0, 0, COLOUR_WORD), 1);
        assert_eq!(s.read(2, 2, COLOUR_WORD), 1);
        assert_eq!(s.read(3, 3, COLOUR_WORD), 0);
    }

    /// A glyph lands at the raster position minus its origin, and the position
    /// advances by the advance.
    #[test]
    fn a_bitmap_lands_relative_to_the_raster_position_and_advances_it() {
        let (mut t, s) = tp();
        t.command(sf(&s), 2, &[5]);
        t.command(sf(&s), 4, &[10, 10]);
        // 2 x 2, origin (-1, 0): column 0 at x = 11; advance (3, 0).
        t.bitmap(&Surface::new(&s, COLOUR_WORD), &[2, 2, -1i32 as u32, 0, 3, 0], &[0b10_01u32 << 28]);
        assert_eq!(s.read(11, 10, COLOUR_WORD), 5);
        assert_eq!(s.read(12, 10, COLOUR_WORD), 0);
        assert_eq!(s.read(12, 11, COLOUR_WORD), 5);
        assert_eq!(s.read(11, 11, COLOUR_WORD), 0);
        assert_eq!(t.raster(), (13, 10));
        // A header of the wrong length is ignored.
        let before = (t.colour(), t.raster(), t.drawn, t.unmodelled);
        t.bitmap(&Surface::new(&s, COLOUR_WORD), &[1, 1], &[0]);
        assert_eq!((t.colour(), t.raster(), t.drawn, t.unmodelled), before);
    }

    #[test]
    fn commands_with_no_drawing_role_are_counted_and_others_refused() {
        let (mut t, s) = tp();
        for sel in [1, 7, 8] {
            assert!(t.command(sf(&s), sel, &[0, 0, 0, 0]));
        }
        assert_eq!(t.unmodelled, 3);
        assert!(!t.command(sf(&s), 9, &[]));
        assert!(!t.command(sf(&s), 0xc, &[]));
        assert!(s.allocated().is_empty());
        t.command(sf(&s), 3, &[1, 400, 0, 0]);
        t.reset();
        assert_eq!((t.colour(), t.raster(), t.drawn, t.unmodelled), (0, (0, 0), 0, 0));
        assert_eq!(t.colour_map(), ColourMap::default(), "a reset clears the shared map too");
    }
}
