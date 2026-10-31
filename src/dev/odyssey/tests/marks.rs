//! The ordering rules between the threads that write the planes and the
//! display that copies them, checked on small models over every
//! interleaving of their steps.
//!
//! The steps run in sequentially consistent order, so these tests pin the
//! program order the rules need (store, then mark; take, then read), not
//! the Release and Acquire the marks carry on a weaker machine. Each rule
//! has a negative control: the same schedule with the order reversed must
//! leave a stale row, or the model proves nothing.

use super::*;
use std::cell::{Cell, RefCell};

const ROWS: usize = 4;

/// Everything a model run shares: the real marks, plane cells, a lookup
/// value, and what the display made of them.
struct Model {
    marks: RowMarks,
    cells: [Cell<u32>; ROWS],
    lut: Cell<u32>,
    lut_seen: Cell<u32>,
    taken: RefCell<Vec<u32>>,
    /// What the display holds per row: the cell and the lookup it saw.
    shown: [Cell<u32>; ROWS],
}

impl Model {
    fn new() -> Model {
        let marks = RowMarks::default();
        marks.set_height(ROWS as u32);
        Model { marks, cells: Default::default(), lut: Cell::new(0), lut_seen: Cell::new(0), taken: Default::default(), shown: Default::default() }
    }

    fn take(&self) {
        self.taken.borrow_mut().clear();
        self.marks.take_dirty(ROWS, &mut self.taken.borrow_mut());
    }

    fn see_lut(&self) {
        self.lut_seen.set(self.lut.get());
    }

    fn copy(&self) {
        for &y in self.taken.borrow().iter() {
            self.shown[y as usize].set(self.cells[y as usize].get() + 100 * self.lut_seen.get());
        }
    }

    /// A quiet frame after the writers stopped: take, look, copy.
    fn settle(&self) {
        self.take();
        self.see_lut();
        self.copy();
    }

    fn stale_rows(&self) -> Vec<usize> {
        (0..ROWS).filter(|&y| self.shown[y].get() != self.cells[y].get() + 100 * self.lut.get()).collect()
    }
}

type Step = fn(&Model);

/// Run every interleaving of the writer's steps with the display's, each
/// from a fresh model, then a quiet frame. The schedules that end with a
/// stale row.
fn failing_schedules(writer: &[Step], display: &[Step]) -> usize {
    let (n, m) = (writer.len(), display.len());
    let mut failing = 0;

    for pattern in 0u32..1 << (n + m) {
        if pattern.count_ones() as usize != n {
            continue;
        }
        let model = Model::new();
        let (mut w, mut d) = (0, 0);

        for step in 0..n + m {
            if pattern >> step & 1 != 0 {
                writer[w](&model);
                w += 1;
            } else {
                display[d](&model);
                d += 1;
            }
        }
        model.settle();
        if !model.stale_rows().is_empty() {
            failing += 1;
        }
    }
    failing
}

fn store_a(m: &Model) {
    m.cells[1].set(7);
}
fn mark_a(m: &Model) {
    m.marks.invalidate(DirtySite::Pixel, 1, 1);
}
fn store_b(m: &Model) {
    m.cells[2].set(9);
}
fn mark_b(m: &Model) {
    m.marks.invalidate(DirtySite::Pixel, 2, 1);
}

/// Two frames of the display, each taking the marks and then copying the
/// rows it took.
const FRAMES: [Step; 4] = [Model::take, Model::copy, Model::take, Model::copy];

/// A writer that stores its pixels and then marks them leaves no stale row
/// whenever the display's frames fall; the reverse order does.
#[test]
fn a_row_marked_after_its_pixels_is_never_left_stale() {
    assert_eq!(failing_schedules(&[store_a, mark_a, store_b, mark_b], &FRAMES), 0);
    assert!(failing_schedules(&[mark_a, store_a, mark_b, store_b], &FRAMES) > 0, "marking first must be able to lose a row");
}

fn set_lut(m: &Model) {
    m.lut.set(1);
}
fn repaint_all(m: &Model) {
    m.marks.repaint(DirtySite::Lut, 0, ROWS as i32);
}

/// The display takes the marks and only then reads what they cover (the
/// lookup RAM, the display modes and the cursor): the other order resolves
/// with the old table and drops the mark.
#[test]
fn the_lookup_is_read_after_the_marks_are_taken() {
    let ordered: [Step; 6] = [Model::take, Model::see_lut, Model::copy, Model::take, Model::see_lut, Model::copy];
    let reversed: [Step; 6] = [Model::see_lut, Model::take, Model::copy, Model::see_lut, Model::take, Model::copy];

    assert_eq!(failing_schedules(&[set_lut, repaint_all], &ordered), 0);
    assert!(failing_schedules(&[set_lut, repaint_all], &reversed) > 0, "reading the lookup first must be able to lose the repaint");
}

/// The display's own clear, measure, set of the lines a writer did not
/// name: a line marked after the clear is measured at the next swap, where
/// a load-modify-store of the whole word would lose it.
#[test]
fn an_unnamed_line_marked_during_a_measure_is_kept_for_the_next() {
    let marks = RowMarks::default();
    marks.set_height(8);
    marks.invalidate(DirtySite::Pixel, 2, 1);

    let taken = marks.take_unnamed();
    assert_eq!(taken[0], 1 << 2);
    // A write lands while the display measures.
    marks.invalidate(DirtySite::Pixel, 3, 1);
    assert!(marks.unnamed_pending(), "the later line stays marked");
    assert_eq!(marks.take_unnamed()[0], 1 << 3);
}

/// The swap model: a writer stores into the buffer a window does not show
/// and marks the line; a swap shows that buffer and repaints the lines the
/// buffers differ in, found among the lines marked unnamed. A line that
/// the display took as dirty before the swap, but that was not yet named
/// unnamed when the swap measured, would stay stale for good.
struct Swap {
    marks: RowMarks,
    buffer: [Cell<u32>; 2],
    displayed: Cell<usize>,
    shown: Cell<u32>,
    taken: RefCell<Vec<u32>>,
}

impl Swap {
    fn run(writer: &[fn(&Swap)], display: &[fn(&Swap)]) -> usize {
        let (n, m) = (writer.len(), display.len());
        let mut failing = 0;

        for pattern in 0u32..1 << (n + m) {
            if pattern.count_ones() as usize != n {
                continue;
            }
            let model = Swap { marks: RowMarks::default(), buffer: Default::default(), displayed: Cell::new(0), shown: Cell::new(0), taken: Default::default() };
            model.marks.set_height(1);
            let (mut w, mut d) = (0, 0);

            for step in 0..n + m {
                if pattern >> step & 1 != 0 {
                    writer[w](&model);
                    w += 1;
                } else {
                    display[d](&model);
                    d += 1;
                }
            }
            model.frame_take();
            model.frame_copy();
            if model.shown.get() != model.buffer[model.displayed.get()].get() {
                failing += 1;
            }
        }
        failing
    }

    fn frame_take(&self) {
        self.taken.borrow_mut().clear();
        self.marks.take_dirty(1, &mut self.taken.borrow_mut());
    }
    fn frame_copy(&self) {
        if !self.taken.borrow().is_empty() {
            self.shown.set(self.buffer[self.displayed.get()].get());
        }
    }
    /// The retrace latches the swap and the display repaints the lines
    /// where the buffers differ among those not named.
    fn flip(&self) {
        self.displayed.set(1);
        let unnamed = self.marks.take_unnamed();
        if unnamed[0] & 1 != 0 && self.buffer[0].get() != self.buffer[1].get() {
            self.marks.repaint(DirtySite::Flip, 0, 1);
        }
    }
    fn store(&self) {
        self.buffer[1].set(5);
    }
    fn name(&self) {
        self.marks.name_unnamed(0, 1);
    }
    fn mark(&self) {
        self.marks.repaint(DirtySite::Pixel, 0, 1);
    }
}

/// A line is named unnamed before it is marked dirty: in that order a swap
/// never misses a line whose buffers differ, and in the other it can.
#[test]
fn a_line_is_named_unnamed_before_it_is_marked_dirty() {
    let display: [fn(&Swap); 5] = [Swap::frame_take, Swap::frame_copy, Swap::flip, Swap::frame_take, Swap::frame_copy];
    assert_eq!(Swap::run(&[Swap::store, Swap::name, Swap::mark], &display), 0);
    assert!(Swap::run(&[Swap::store, Swap::mark, Swap::name], &display) > 0, "marking dirty first must be able to lose the swap's repaint");
}

/// Marks past the frame are dropped, and a new height forgets the old marks.
#[test]
fn marks_are_bounded_by_the_frame() {
    let marks = RowMarks::default();
    let mut rows = Vec::new();

    marks.repaint(DirtySite::Other, 0, 10);
    marks.take_dirty(1024, &mut rows);
    assert!(rows.is_empty(), "no frame, no marks");

    marks.set_height(100);
    marks.repaint(DirtySite::Other, 90, 50);
    marks.take_dirty(100, &mut rows);
    assert_eq!(rows, (90..100).collect::<Vec<_>>());

    marks.repaint(DirtySite::Other, 0, 3);
    marks.set_height(50);
    rows.clear();
    marks.take_dirty(50, &mut rows);
    assert!(rows.is_empty());
}
