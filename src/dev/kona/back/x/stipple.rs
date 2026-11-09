//! The X server's stipple cache.
//!
//! Stippled fills are masked by a pattern the X server stores once and then
//! names:
//!
//! - 0x13d (`x_store_stipple`): an entry, the pattern's width and height, then
//!   `height` rows, one word each, most significant bit leftmost (the rows come
//!   behind the command, as a data run the back end asks for);
//! - 0x13e (`x_rotate_stipple`): an entry, then the pattern origin x, y;
//! - 0x13c (`x_load_stipple`): an entry whose pattern masks the box, rectangle,
//!   span and point fills that follow, or all ones to end the masking.
//!
//! Grades. The command meanings and the row order are software expectations
//! from the senders; the bit order (most significant bit leftmost) is inferred
//! from a traced I-beam. The cache's size, 32 patterns of up to 32 by 32, is
//! the model's choice: a store outside it is refused, its rows consumed, and
//! the entry left holding no pattern. A store announcing more than 4096 rows
//! is this model's own bound: it is refused and its rows are not asked for. A store replaces the entry's pattern and
//! keeps its origin; whether the pipe does either is unknown. The pattern is
//! measured from the entry's origin plus the drawable's position, both
//! inferred, and bits past the width are ignored. A clear bit leaves its pixel
//! unchanged unless the context is opaque (inferred; no stippled fill in the
//! recorded streams shows which a real pipe does). With no stipple loaded, or
//! one holding no pattern, every pixel is painted.

use crate::dev::kona::defs::*;

pub const X_LOAD_STIPPLE: u32 = LOAD_WINDOW_SYSTEM_STIPPLE;
pub const X_STORE_STIPPLE: u32 = STORE_WINDOW_SYSTEM_STIPPLE;
pub const X_ROTATE_STIPPLE: u32 = ROTATE_WINDOW_SYSTEM_STIPPLE;

/// Entries in the cache, and the largest pattern side.
pub const ENTRIES: usize = 32;
pub const MAX_SIDE: u32 = 32;
const NONE: u32 = u32::MAX;
/// The most rows a store may announce before it is refused.
const ROWS_MAX: u32 = 4096;

#[derive(Clone, Debug)]
pub struct Stipple {
    width: [u32; ENTRIES],
    height: [u32; ENTRIES],
    origin: [(i32, i32); ENTRIES],
    rows: [[u32; MAX_SIDE as usize]; ENTRIES],
    loaded: u32,
    /// A store waiting for its rows: entry, width, height.
    pending: Option<(u32, u32, u32)>,
    /// Stores and loads refused (outside the cache, or an entry with no
    /// pattern).
    pub refused: u64,
}

impl Default for Stipple {
    fn default() -> Self {
        Stipple { width: [0; ENTRIES], height: [0; ENTRIES], origin: [(0, 0); ENTRIES], rows: [[0; 32]; ENTRIES], loaded: NONE, pending: None, refused: 0 }
    }
}

fn fits(entry: u32, width: u32, height: u32) -> bool {
    (entry as usize) < ENTRIES && (1..=MAX_SIDE).contains(&width) && (1..=MAX_SIDE).contains(&height)
}

impl Stipple {
    /// The cache, as the stream module reads and writes it (the counter is not saved).
    pub fn visit<S: crate::dev::kona::stream::Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { width: _, height: _, origin: _, rows: _, loaded: _, pending: _, refused: _ } = self;
        s.u32s(&mut self.width);
        s.u32s(&mut self.height);
        for o in self.origin.iter_mut() {
            s.i32(&mut o.0);
            s.i32(&mut o.1);
        }
        for r in self.rows.iter_mut() {
            s.u32s(r);
        }
        s.u32(&mut self.loaded);
        s.opt(&mut self.pending, |s, p| {
            s.u32(&mut p.0);
            s.u32(&mut p.1);
            s.u32(&mut p.2);
        });
    }

    /// A store's header: the rows to expect behind it. The entry loses its
    /// pattern until they arrive.
    pub fn begin_store(&mut self, args: &[u32]) -> u32 {
        let [entry, width, height] = *args else { return 0 };
        if (entry as usize) < ENTRIES {
            self.width[entry as usize] = 0;
            self.height[entry as usize] = 0;
        }
        // The rows are asked for as one data run: a count past this model's
        // own bound is refused whole rather than swallow the stream. A store
        // refused for its size still has its rows consumed, up to that bound.
        if !fits(entry, width, height) || height > ROWS_MAX {
            self.refused += 1;
        }
        if height > ROWS_MAX {
            self.pending = None;
            return 0;
        }
        self.pending = (height != 0).then_some((entry, width, height));
        if height == 0 {
            // No rows: the store is complete and holds nothing.
            return 0;
        }
        height
    }

    /// The rows behind a store.
    pub fn rows(&mut self, words: &[u32]) {
        let Some((entry, width, height)) = self.pending.take() else { return };
        if fits(entry, width, height) && words.len() == height as usize {
            let e = entry as usize;
            self.rows[e][..words.len()].copy_from_slice(words);
            self.width[e] = width;
            self.height[e] = height;
        }
    }

    /// Rows still owed to a store.
    pub fn pending_rows(&self) -> u32 {
        self.pending.map_or(0, |(_, _, h)| h)
    }

    pub fn rotate(&mut self, args: &[u32]) {
        let [entry, x, y] = *args else { return };
        if (entry as usize) < ENTRIES {
            self.origin[entry as usize] = (x as i32, y as i32);
        } else {
            self.refused += 1;
        }
    }

    pub fn load(&mut self, args: &[u32]) {
        let [entry] = *args else { return };
        self.loaded = if entry == NONE {
            NONE
        } else if (entry as usize) >= ENTRIES {
            self.refused += 1;
            NONE
        } else {
            if self.width[entry as usize] == 0 {
                self.refused += 1;
            }
            entry
        };
    }

    /// Whether a stipple with a pattern is loaded.
    pub fn is_loaded(&self) -> bool {
        (self.loaded as usize) < ENTRIES && self.width[self.loaded as usize] != 0
    }

    /// The loaded pattern: its rows, width, height and origin, if one with a
    /// pattern is loaded.
    pub fn loaded_pattern(&self) -> Option<([u32; 32], i64, i64, (i64, i64))> {
        if !self.is_loaded() {
            return None;
        }
        let e = self.loaded as usize;
        Some((self.rows[e], self.width[e] as i64, self.height[e] as i64, (self.origin[e].0 as i64, self.origin[e].1 as i64)))
    }

    #[cfg(test)]
    /// Whether the loaded stipple lets a fill paint (`x`, `y`), the drawable
    /// at (`ox`, `oy`).
    pub fn bit(&self, x: i64, y: i64, ox: i64, oy: i64) -> bool {
        if !self.is_loaded() {
            return true;
        }
        let e = self.loaded as usize;
        let (w, h) = (self.width[e] as i64, self.height[e] as i64);
        let (px, py) = (ox + self.origin[e].0 as i64, oy + self.origin[e].1 as i64);
        let row = self.rows[e][((y - py).rem_euclid(h)) as usize];
        row & (0x8000_0000u32 >> (x - px).rem_euclid(w)) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored(s: &mut Stipple, entry: u32, w: u32, rows: &[u32]) {
        assert_eq!(s.begin_store(&[entry, w, rows.len() as u32]), rows.len() as u32);
        s.rows(rows);
    }

    /// A 3 x 2 pattern, most significant bit leftmost, tiles from the origin.
    #[test]
    fn a_stored_pattern_tiles_from_its_origin() {
        let mut s = Stipple::default();
        stored(&mut s, 5, 3, &[0b101 << 29, 0b010 << 29]);
        assert!(!s.is_loaded());
        assert!(s.bit(0, 0, 0, 0), "nothing loaded: every pixel paints");
        s.load(&[5]);
        assert!(s.is_loaded());
        let row = |y| (0..6).map(|x| s.bit(x, y, 0, 0)).collect::<Vec<_>>();
        assert_eq!(row(0), [true, false, true, true, false, true]);
        assert_eq!(row(1), [false, true, false, false, true, false]);
        assert_eq!(row(2), row(0), "rows repeat by the height");
        // Negative coordinates wrap the same way.
        assert_eq!(s.bit(-1, 0, 0, 0), s.bit(2, 0, 0, 0));
        // The entry's origin and the drawable's position both shift it.
        s.rotate(&[5, 1, 1]);
        // Pattern bit (0, 0) is set and (1, 0) is clear.
        assert!(s.bit(1, 1, 0, 0) && !s.bit(2, 1, 0, 0));
        assert!(s.bit(4, 3, 3, 2) && !s.bit(5, 3, 3, 2));
    }

    #[test]
    fn load_none_or_a_bad_entry_ends_the_masking() {
        let mut s = Stipple::default();
        stored(&mut s, 0, 1, &[0]);
        s.load(&[0]);
        assert!(!s.bit(0, 0, 0, 0));
        s.load(&[u32::MAX]);
        assert!(s.bit(0, 0, 0, 0));
        s.load(&[0]);
        s.load(&[ENTRIES as u32]);
        assert!(s.bit(0, 0, 0, 0));
        assert_eq!(s.refused, 1);
        // An entry with no pattern loads, refused, and does not mask.
        s.load(&[7]);
        assert!(!s.is_loaded());
        assert_eq!(s.refused, 2);
    }

    /// A store outside the cache is refused, its rows are consumed (the
    /// reply asks for them) and the entry holds nothing; a store keeps the
    /// entry's origin and replaces its pattern.
    #[test]
    fn a_store_outside_the_cache_is_refused_and_a_store_replaces() {
        let mut s = Stipple::default();
        assert_eq!(s.begin_store(&[40, 4, 2]), 2, "the rows still follow");
        s.rows(&[1, 2]);
        assert_eq!(s.refused, 1);
        assert_eq!(s.begin_store(&[3, 33, 1]), 1);
        s.rows(&[1]);
        assert_eq!(s.refused, 2);
        s.load(&[3]);
        assert!(!s.is_loaded());
        stored(&mut s, 3, 4, &[0xf000_0000]);
        s.rotate(&[3, 2, 0]);
        stored(&mut s, 3, 4, &[0x8000_0000]);
        s.load(&[3]);
        assert!(s.bit(2, 0, 0, 0), "the origin survived the second store");
        assert!(!s.bit(3, 0, 0, 0));
        // A store with no rows is complete at once and leaves the entry empty.
        assert_eq!(s.begin_store(&[3, 4, 0]), 0);
        assert_eq!(s.pending_rows(), 0);
        s.load(&[3]);
        assert!(!s.is_loaded());
    }

    /// An absurd row count is refused without asking for the rows.
    #[test]
    fn a_store_announcing_too_many_rows_is_refused() {
        let mut s = Stipple::default();
        assert_eq!(s.begin_store(&[0, 8, u32::MAX]), 0);
        assert_eq!((s.pending_rows(), s.refused), (0, 1));
    }

    #[test]
    fn rotate_outside_the_cache_is_refused() {
        let mut s = Stipple::default();
        s.rotate(&[ENTRIES as u32, 1, 1]);
        assert_eq!(s.refused, 1);
    }
}
