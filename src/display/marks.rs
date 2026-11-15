//! Where drawing landed since the last frame, for a board whose buffers
//! share memory words: the back buffer of a double-buffered window is a
//! field of the same pixel the front buffer is in, so "memory was written"
//! does not say whether the picture changed.
//!
//! A raster thread marks each thing it draws with the store it wrote (a
//! buffer, however the board names one), the planes of the pixel it could
//! have changed, and a box around it. Once a frame the display takes the
//! marks and tests each against what the screen shows inside its box; when
//! none shows, the last frame stands and nothing is composed.
//!
//! A mark follows its data: the thread stores, then marks. A take that
//! misses the mark leaves it for the next frame, and one that has it is
//! followed by a compose that reads the data.
//!
//! Marking is cheap when it repeats: each thread remembers what the table
//! already holds and returns without touching shared memory but for one
//! load, until the display takes the table.

use parking_lot::Mutex;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// A box of pixels, corners included, in the store's own coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Rect {
    pub fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Rect {
        Rect { x0, y0, x1, y1 }
    }

    pub fn is_empty(&self) -> bool {
        self.x0 > self.x1 || self.y0 > self.y1
    }

    pub fn contains(&self, o: &Rect) -> bool {
        self.x0 <= o.x0 && self.y0 <= o.y0 && self.x1 >= o.x1 && self.y1 >= o.y1
    }

    pub fn union(&self, o: &Rect) -> Rect {
        Rect { x0: self.x0.min(o.x0), y0: self.y0.min(o.y0), x1: self.x1.max(o.x1), y1: self.y1.max(o.y1) }
    }

    pub fn intersect(&self, o: &Rect) -> Rect {
        Rect { x0: self.x0.max(o.x0), y0: self.y0.max(o.y0), x1: self.x1.min(o.x1), y1: self.y1.min(o.y1) }
    }
}

/// Drawing into `planes` of the pixels of `store` inside `rect`. What a
/// store and a plane are is the board's: the display that reads the marks
/// is the same board's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mark {
    pub store: u64,
    pub planes: u64,
    pub rect: Rect,
}

/// Marks kept apart: one per store and plane set, its boxes joined. Past
/// this the table says "anything".
const SLOTS: usize = 8;

#[derive(Clone, Copy)]
struct Table {
    marks: [Mark; SLOTS],
    len: usize,
    /// Something was drawn that no mark describes.
    all: bool,
}

impl Table {
    const EMPTY: Table = Table { marks: [Mark { store: 0, planes: 0, rect: Rect { x0: 0, y0: 0, x1: 0, y1: 0 } }; SLOTS], len: 0, all: false };

    fn covers(&self, m: &Mark) -> bool {
        self.all || self.marks[..self.len].iter().any(|t| t.store == m.store && t.planes == m.planes && t.rect.contains(&m.rect))
    }

    fn add(&mut self, m: &Mark) {
        if self.all {
            return;
        }
        if let Some(t) = self.marks[..self.len].iter_mut().find(|t| t.store == m.store && t.planes == m.planes) {
            t.rect = t.rect.union(&m.rect);
        } else if self.len < SLOTS {
            self.marks[self.len] = *m;
            self.len += 1;
        } else {
            self.all = true;
        }
    }
}

/// What a thread knows the table holds, and the take it knew it at.
struct Seen {
    epoch: u64,
    table: Table,
}

thread_local! {
    static SEEN: RefCell<Seen> = const { RefCell::new(Seen { epoch: 0, table: Table::EMPTY }) };
}

/// Each table's epochs are its own: a thread's memory of one table never
/// passes for another's.
static EPOCHS: AtomicU64 = AtomicU64::new(1 << 32);

/// One board's marks.
pub struct Marks {
    /// Moves at every take.
    epoch: AtomicU64,
    table: Mutex<Table>,
    /// `Table::all`, for a marker with no box to give: no lock.
    all: AtomicBool,
}

impl Default for Marks {
    fn default() -> Self {
        Self::new()
    }
}

impl Marks {
    pub fn new() -> Marks {
        Marks {
            epoch: AtomicU64::new(EPOCHS.fetch_add(1 << 32, Ordering::Relaxed)),
            table: Mutex::new(Table::EMPTY),
            all: AtomicBool::new(false),
        }
    }

    /// Drawing landed in `planes` of `store` inside `rect`. Call after the
    /// stores it describes. From any thread.
    #[inline]
    pub fn mark(&self, store: u64, planes: u64, rect: Rect) {
        if planes == 0 || rect.is_empty() {
            return;
        }
        let m = Mark { store, planes, rect };
        SEEN.with_borrow_mut(|seen| {
            // A take since this thread last looked empties what it knew.
            if seen.epoch == self.epoch.load(Ordering::Acquire) && seen.table.covers(&m) {
                return;
            }
            let mut table = self.table.lock();
            table.add(&m);
            // The epoch moves only under the lock: this is the table's.
            seen.epoch = self.epoch.load(Ordering::Acquire);
            seen.table = *table;
        });
    }

    /// Something was drawn that no box or plane set describes: the picture
    /// may have changed anywhere.
    #[inline]
    pub fn mark_all(&self) {
        // The load keeps the line shared between the threads that draw.
        if !self.all.load(Ordering::Relaxed) {
            self.all.store(true, Ordering::Release);
        }
    }

    /// The marks since the last take, into `out` (emptied first). True when
    /// they do not describe everything drawn: the picture may have changed
    /// anywhere.
    pub fn take(&self, out: &mut Vec<Mark>) -> bool {
        out.clear();
        let mut all = self.all.swap(false, Ordering::AcqRel);
        let mut table = self.table.lock();
        all |= table.all;
        out.extend_from_slice(&table.marks[..table.len]);
        *table = Table::EMPTY;
        self.epoch.fetch_add(1, Ordering::AcqRel);
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_of_one_store_and_plane_set_join_their_boxes() {
        let m = Marks::new();
        let mut out = Vec::new();
        m.mark(1, 0xFFF, Rect::new(10, 10, 20, 20));
        m.mark(1, 0xFFF, Rect::new(15, 5, 30, 12));
        m.mark(1, 0xFFF000, Rect::new(0, 0, 1, 1));
        m.mark(2, 0xFFF, Rect::new(7, 7, 7, 7));
        assert!(!m.take(&mut out));
        assert_eq!(out, [
            Mark { store: 1, planes: 0xFFF, rect: Rect::new(10, 5, 30, 20) },
            Mark { store: 1, planes: 0xFFF000, rect: Rect::new(0, 0, 1, 1) },
            Mark { store: 2, planes: 0xFFF, rect: Rect::new(7, 7, 7, 7) },
        ]);
        assert!(!m.take(&mut out));
        assert!(out.is_empty(), "a take empties the table");
    }

    #[test]
    fn a_mark_repeated_after_a_take_is_kept() {
        // The thread remembers the table held it; the take must undo that.
        let m = Marks::new();
        let mut out = Vec::new();
        let r = Rect::new(0, 0, 9, 9);
        m.mark(1, 1, r);
        m.mark(1, 1, r);
        m.take(&mut out);
        assert_eq!(out.len(), 1);
        m.mark(1, 1, r);
        m.take(&mut out);
        assert_eq!(out, [Mark { store: 1, planes: 1, rect: r }]);
    }

    #[test]
    fn one_table_s_memory_is_not_another_s() {
        let (a, b) = (Marks::new(), Marks::new());
        let mut out = Vec::new();
        let r = Rect::new(0, 0, 9, 9);
        a.mark(1, 1, r);
        b.mark(1, 1, r);
        b.take(&mut out);
        assert_eq!(out.len(), 1);
        a.take(&mut out);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn empty_marks_are_dropped_and_too_many_say_anything() {
        let m = Marks::new();
        let mut out = Vec::new();
        m.mark(1, 0, Rect::new(0, 0, 9, 9));
        m.mark(1, 1, Rect::new(5, 0, 4, 9));
        assert!(!m.take(&mut out));
        assert!(out.is_empty());
        for store in 0..=SLOTS as u64 {
            m.mark(store, 1, Rect::new(0, 0, 0, 0));
        }
        assert!(m.take(&mut out));
        m.mark_all();
        assert!(m.take(&mut out));
        assert!(!m.take(&mut out));
    }

    #[test]
    fn marks_from_other_threads_survive_takes() {
        // Every mark made lands in some take: none is lost to a thread's
        // memory of a table the display has since emptied.
        let m = std::sync::Arc::new(Marks::new());
        let done = std::sync::Arc::new(AtomicBool::new(false));
        let workers: Vec<_> = (0..4u64)
            .map(|t| {
                let (m, done) = (m.clone(), done.clone());
                std::thread::spawn(move || {
                    let mut n = 0u64;
                    while !done.load(Ordering::Relaxed) {
                        m.mark(t, 1, Rect::new(0, 0, 9, 9));
                        n += 1;
                    }
                    // The last mark of each thread must be seen below.
                    m.mark(t, 2, Rect::new(0, 0, 9, 9));
                    n
                })
            })
            .collect();
        let mut out = Vec::new();
        let mut seen = [false; 4];
        // Take until every thread has been seen marking, and then some: a
        // thread may be slow to start.
        let mut takes = 0;
        while takes < 2000 || !seen.iter().all(|&s| s) {
            m.take(&mut out);
            for k in &out {
                seen[k.store as usize] = true;
            }
            takes += 1;
            if takes >= 2000 {
                std::thread::yield_now();
            }
        }
        done.store(true, Ordering::Relaxed);
        for w in workers {
            assert!(w.join().unwrap() > 0);
        }
        m.take(&mut out);
        for t in 0..4u64 {
            assert!(out.contains(&Mark { store: t, planes: 2, rect: Rect::new(0, 0, 9, 9) }), "thread {t}'s last mark");
            assert!(seen[t as usize] || out.iter().any(|k| k.store == t));
        }
    }
}
