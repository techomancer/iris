//! BUZZ's raster unit on a thread of its own, and the queue that feeds it.
//!
//! BUZZ is the board's one graphics chip. Its front end parses the command
//! FIFO, writes registers, transforms, lights and sets up primitives; behind
//! it the raster unit rasterises, textures, shades and writes the framebuffer.
//! US 6,417,858 puts an output FIFO between the transform engine and the
//! rasteriser (inferred to apply to BUZZ). The model splits there: the
//! board's thread (worker.rs) is the front end, and this file's thread,
//! `odyssey-raster`, is the raster unit. The back end, PB&J, stays with the
//! display thread.
//!
//! What crosses the queue is the model's own encoding, because the
//! hardware's is unknown: each entry is one job, a closure over everything
//! the raster work needs, captured when the front end sets it up. A GL
//! primitive's job carries its raster rules, draw state, vertices, texture
//! chain and the texture tables it reads; a fence carries its address and
//! token. The state a job captures is the state at its place in the
//! command stream, so raster state reaches the raster unit in order. The
//! guest sees nothing of the split.
//!
//! The planes belong to the raster unit. They stay where the board keeps
//! them, and a job reaches them through raw spans (`PlaneSpans`), as the
//! IMPACT model's raster engines reach their pixel memory through pointers.
//! Two other threads need them:
//!
//! - The front end's own plane work (integer and image drawing, pixel DMA,
//!   the object store, reset) first waits for the raster unit to finish
//!   every job queued (`Board::raster_wait`), so the two never touch the
//!   planes at once. A DMA's texels for an object that already has its
//!   rows go to the raster unit as a job instead (`Board::object_stage`).
//! - The display thread resolves the frame while holding the raster unit's
//!   context lock (`RasterQueue::planes`), which the raster thread holds for
//!   each slice of jobs, so a frame waits out a slice in flight and never
//!   reads a half-drawn primitive.
//!
//! The rows a job wrote are the display's business too. The raster thread
//! keeps them in its context while it draws and publishes them to a cell
//! the display takes from (`RasterCtx::publish`): before it runs a job
//! that tells the guest something (`RasterQueue::submit_completion`), when
//! the queue drains, and when a slice ends. A guest that waits for a
//! fence or a query before it arms a swap therefore has all of its rows
//! published before the swap can latch.
//!
//! The front end waits when the queued work, estimated in pixels per job,
//! passes `RASTER_WORK_BUDGET`, so the drawing stays close behind the commands.
//!
//! The ring takes the shared `GFifo`. With no raster thread running (unit
//! tests, a stopped machine) a job runs inside the submit.

use super::*;

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::SeqCst};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex, MutexGuard};

use crate::gfifo::GFifo;

/// The raster thread's name.
pub const ODYSSEY_RASTER_THREAD: &str = "odyssey-raster";

/// Jobs the queue holds.
pub const ODSY_RASTER_QUEUE_MAX: usize = 65536;

/// Work, in pixels, the front end may have queued for the raster unit
/// before it waits for the raster unit to catch up, as the board's front
/// end waits on a full output FIFO between transform and rasteriser
/// (US 6,417,858; the depth on BUZZ is unknown). It keeps the drawing
/// close behind the commands, so the display does not show a frame the
/// processor sent long ago. Measured in work, not jobs: Smoke & Mirrors
/// queues thousands of tiny triangles, which a job-count bound stalled
/// the processor behind. The model rasterises 5 to 30 Mpixel/s a
/// textured or blended fill (`raster_bench`, release build), so 2^17 pixels
/// is a few milliseconds to a few tens of milliseconds of raster time, and
/// well past any one frame's small primitives.
pub const RASTER_WORK_BUDGET: u64 = 1 << 17;

/// What a job costs before its pixels: setup, dispatch and the state it
/// carries, in pixels of fill. A queue of tiny primitives still adds up.
pub const RASTER_JOB_OVERHEAD: u32 = 64;

/// Clears and block fills run about this many times faster per pixel than
/// a shaded fill (`raster_bench`: 200 against 5 to 30 Mpixel/s), so their
/// pixels count for this fraction.
pub const RASTER_CLEAR_DIVISOR: u32 = 8;

/// The cost a ring entry's word can hold.
const COST_MAX: u64 = (1 << 30) - 1;

/// The longest the raster thread holds its context lock before letting a
/// waiting display frame in, checked between jobs.
pub const RASTER_SLICE: Duration = Duration::from_millis(1);

/// What a GL draw hands back to its context in place of the line-stipple
/// phase while the raster unit keeps the phase: the next draw that carries
/// it continues from the raster unit's own count. The context resets its
/// phase to 0 at glBegin and glEnd, which replaces this marker.
pub const LINE_PHASE_ON_RASTER: u32 = u32::MAX;

/// Words of a row set: one bit a plane row.
pub const ROW_WORDS: usize = (SGI_ODYSSEY_HEIGHT as usize).div_ceil(64);

/// The rows a draw wrote, a bit each, by where it wrote: colour buffer 0,
/// colour buffer 1 and the overlay byte.
pub type TouchedRows = [[Cell<u64>; ROW_WORDS]; ODSY_BUFFERS as usize + 1];

/// Slot of the overlay byte in `TouchedRows`.
pub const TOUCHED_OVERLAY: usize = ODSY_BUFFERS as usize;

/// Note that a draw wrote row y of `slot`.
#[inline(always)]
pub fn touch_row(rows: &TouchedRows, slot: usize, y: i32) {
    if (y as u32) < SGI_ODYSSEY_HEIGHT {
        let w = &rows[slot][y as usize >> 6];
        w.set(w.get() | 1 << (y & 63));
    }
}

/// The colour rows the raster unit has written since the display last
/// looked, for the display to resolve those and no others. A row's pixels
/// show a colour buffer by the window their id plane names and that
/// window's displayed-buffer bit, so a row written to a buffer is kept by
/// the windows the draw's id test lets it reach. The overlay shows
/// whatever the windows do.
pub struct DrawnRows {
    by_id: Box<[[[u64; ROW_WORDS]; ODSY_DISPMODE_COUNT as usize]; ODSY_BUFFERS as usize]>,
    overlay: [u64; ROW_WORDS],
    /// Some row is held.
    any: bool,
}

impl Default for DrawnRows {
    fn default() -> Self {
        DrawnRows { by_id: Box::new([[[0; ROW_WORDS]; ODSY_DISPMODE_COUNT as usize]; ODSY_BUFFERS as usize]), overlay: [0; ROW_WORDS], any: false }
    }
}

impl DrawnRows {
    /// Keep what a draw wrote, which passed the id test `id_compare` (the
    /// board's 0x840 word) at each pixel it stored.
    pub fn add(&mut self, touched: &TouchedRows, id_compare: u32) {
        self.any = true;
        let value = id_compare as u8;
        let mask = (id_compare >> ODSY_ID_COMPARE_MASK_SHIFT) as u8;

        for (slot, rows) in touched.iter().enumerate() {
            if rows.iter().all(|w| w.get() == 0) {
                continue;
            }
            if slot == TOUCHED_OVERLAY {
                for (o, w) in self.overlay.iter_mut().zip(rows) {
                    *o |= w.get();
                }
                continue;
            }
            for id in 0..ODSY_DISPMODE_COUNT {
                // The main drawable is the id byte's bits 7:3.
                if id_compare & ODSY_ID_COMPARE_ENABLE != 0 && (((id << ODSY_DID_MAIN_SHIFT) as u8 ^ value) & mask) >> ODSY_DID_MAIN_SHIFT != 0 {
                    continue;
                }
                for (o, w) in self.by_id[slot][id as usize].iter_mut().zip(rows) {
                    *o |= w.get();
                }
            }
        }
    }

    /// Move every row `from` holds into this set, leaving `from` empty.
    pub fn absorb(&mut self, from: &mut DrawnRows) {
        if !std::mem::take(&mut from.any) {
            return;
        }
        self.any = true;
        for (o, f) in self.overlay.iter_mut().zip(from.overlay.iter_mut()) {
            *o |= std::mem::take(f);
        }
        for (ids, from_ids) in self.by_id.iter_mut().zip(from.by_id.iter_mut()) {
            for (rows, from_rows) in ids.iter_mut().zip(from_ids.iter_mut()) {
                for (o, f) in rows.iter_mut().zip(from_rows.iter_mut()) {
                    *o |= std::mem::take(f);
                }
            }
        }
        from.overlay = [0; ROW_WORDS];
    }

    /// Hand the display the rows the draws so far changed, and forget
    /// them. A row shows a change when a buffer a window shows was written
    /// there: the scanout repaints it. A row written to either buffer
    /// joins the rows where the window's buffers may differ (`differ`),
    /// which a swap of that window repaints.
    pub fn take(&mut self, displayed: u32, differ: &mut Differ, marks: &RowMarks) {
        if !std::mem::take(&mut self.any) {
            return;
        }
        let mut shown = self.overlay;

        for (b, ids) in self.by_id.iter_mut().enumerate() {
            for (id, rows) in ids.iter_mut().enumerate() {
                if (displayed >> id) & 1 == b as u32 {
                    for (s, w) in shown.iter_mut().zip(rows.iter()) {
                        *s |= w;
                    }
                }
                for (d, w) in differ[id].iter_mut().zip(rows.iter_mut()) {
                    *d |= std::mem::take(w);
                }
            }
        }
        self.overlay = [0; ROW_WORDS];
        for (i, mut w) in shown.into_iter().enumerate() {
            while w != 0 {
                marks.repaint(DirtySite::Raster, (i as u32 * 64 + w.trailing_zeros()) as i32, 1);
                w &= w - 1;
            }
        }
    }
}

/// The cell the raster unit publishes its drawn rows to and the display takes
/// them from.
pub type PublishedRows = Arc<Mutex<DrawnRows>>;

/// The raster unit's own state.
#[derive(Default)]
pub struct RasterCtx {
    /// The standalone line stipple's phase after the last draw that
    /// carried one.
    pub line_phase: u32,
    /// The colour rows drawn since the last publish. A job that writes the
    /// colour planes records its rows here, or the display never resolves
    /// them.
    pub drawn: DrawnRows,
    /// Where `publish` puts them.
    published: PublishedRows,
}

impl RasterCtx {
    fn new(published: PublishedRows) -> Self {
        RasterCtx { line_phase: 0, drawn: DrawnRows::default(), published }
    }

    /// Hand the rows drawn so far to the display. Called before anything the
    /// guest can see that follows the drawing, when the queue drains and
    /// when a slice ends; the cell's lock is held for a copy of at most the
    /// rows' bitsets and never across a draw.
    pub fn publish(&mut self) {
        if self.drawn.any {
            self.published.lock().absorb(&mut self.drawn);
        }
    }
}

/// One unit of raster work.
pub type RasterJob = Box<dyn FnOnce(&mut RasterCtx) + Send>;

/// The ring entry's address word: the value is a boxed `RasterJob`.
const ENTRY_JOB: u32 = 1;
/// The value is a boxed `GlDrawJob`, which goes back to the front end
/// when drawn.
const ENTRY_DRAW: u32 = 2;
/// The entry's address word is its kind in the low bits and its cost
/// (`RasterQueue::push`) above them.
const ENTRY_KIND_MASK: u32 = 3;
const ENTRY_COST_SHIFT: u32 = 2;

/// The cost of a rectangle of pixels clipped to the screen, a clear's or a
/// block fill's counted at `RASTER_CLEAR_DIVISOR`, plus the job's overhead.
pub fn rect_cost(x0: i32, y0: i32, x1: i32, y1: i32, fast: bool) -> u32 {
    let w = (x1.min(SGI_ODYSSEY_WIDTH as i32) - x0.max(0)).max(0) as u64;
    let h = (y1.min(SGI_ODYSSEY_HEIGHT as i32) - y0.max(0)).max(0) as u64;
    let px = w * h / if fast { RASTER_CLEAR_DIVISOR as u64 } else { 1 };

    RASTER_JOB_OVERHEAD.saturating_add(px.min(COST_MAX) as u32)
}

/// The cost of a primitive from its vertices' window positions, None for a
/// vertex with no position (behind the eye, where clipping will cut the
/// primitive and what is left may be any size, so the whole screen counts).
/// Points cost their square, lines their length times their width, a
/// triangle its area and any other polygon its bounding box, each within
/// the screen. An estimate, from the corners alone.
pub fn primitive_cost(mode: u32, points: impl Iterator<Item = Option<(f64, f64)>>, point_size: f32, line_width: f32) -> u32 {
    let (w, h) = (SGI_ODYSSEY_WIDTH as f64, SGI_ODYSSEY_HEIGHT as f64);
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    let mut first = [(0.0f64, 0.0f64); 3];
    let mut n = 0;

    for p in points {
        let Some((x, y)) = p else { return RASTER_JOB_OVERHEAD.saturating_add((w * h) as u32) };
        if n < 3 {
            first[n] = (x, y);
        }
        n += 1;
        lo = [lo[0].min(x), lo[1].min(y)];
        hi = [hi[0].max(x), hi[1].max(y)];
    }
    if n == 0 {
        return RASTER_JOB_OVERHEAD;
    }
    // A NaN position is no extent; the clamp then yields zero.
    let (bw, bh) = ((hi[0].min(w) - lo[0].max(0.0)).max(0.0), (hi[1].min(h) - lo[1].max(0.0)).max(0.0));
    let boxed = bw * bh;
    let px = match mode {
        SGI_GL_POINTS => (point_size.max(1.0) as f64).powi(2) * n as f64,
        SGI_GL_LINES | SGI_GL_LINE_LOOP | SGI_GL_LINE_STRIP => (bw.max(bh) + 1.0) * (line_width.max(1.0) as f64),
        SGI_GL_TRIANGLES if n == 3 => {
            let (a, b, c) = (first[0], first[1], first[2]);
            (((b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1)).abs() / 2.0).min(boxed)
        }
        _ => boxed,
    };
    RASTER_JOB_OVERHEAD.saturating_add(px.min(COST_MAX as f64) as u32)
}

/// Finished GL draw jobs kept for the front end to fill again. A draw
/// that finds none allocates; one finished while this many wait is freed.
const SPARE_DRAWS: usize = 1024;

/// The planes as the display holds them: no raster job runs meanwhile.
pub struct PlanesHeld<'a> {
    _ctx: MutexGuard<'a, RasterCtx>,
    published: &'a Mutex<DrawnRows>,
}

impl PlanesHeld<'_> {
    /// The colour rows the raster unit published since the last call, to
    /// the scanout (DrawnRows::take()).
    pub fn take_rows(&mut self, displayed: u32, differ: &mut Differ, marks: &RowMarks) {
        self.published.lock().take(displayed, differ, marks);
    }
}

/// The queue between the front end and the raster unit. `HEADS` is the
/// GFifo's consumer heads: one, because the board has one raster unit. A
/// second would need the jobs to split the rows between the heads, as the
/// IMPACT model's two engines interleave scanlines.
pub struct RasterQueue<const HEADS: usize = 1> {
    ring: Box<GFifo<ODSY_RASTER_QUEUE_MAX, HEADS>>,
    /// Finished draw jobs, from the raster thread (or a draw run inside
    /// its submit) back to the front end, which takes them under the board
    /// lock.
    spare: Box<GFifo<SPARE_DRAWS>>,
    /// Held by the raster thread for a slice of jobs, by a job run inside
    /// the submit, and by the display while it reads the planes.
    ctx: Mutex<RasterCtx>,
    /// The rows the raster unit has published and the display has not yet
    /// taken (RasterCtx::publish()).
    published: PublishedRows,
    /// Jobs submitted and not yet finished.
    pending: AtomicU32,
    /// The estimated work of those jobs, in pixels (`RASTER_WORK_BUDGET`).
    queued: AtomicU64,
    /// The work the front end waits under (`RASTER_WORK_BUDGET`; tests
    /// set their own).
    budget: AtomicU64,
    /// The front end waits for the raster unit to bring `queued` down; the
    /// raster unit wakes it at half the budget, and a stop wakes it.
    room_waiting: AtomicBool,
    closing: AtomicBool,
    room: Mutex<()>,
    room_kick: Condvar,
    /// Jobs pushed onto the ring, and taken off it by the raster thread,
    /// for its wake-up. Both run on across a stop and a start.
    pushed: AtomicU64,
    taken: AtomicU64,
    /// The raster thread runs, so jobs queue.
    running: AtomicBool,
    exit: AtomicBool,
    sleeping: AtomicBool,
    wake: Mutex<()>,
    kick: Condvar,
    idle: Mutex<()>,
    idle_kick: Condvar,
    thread: Mutex<Option<JoinHandle<()>>>,
    /// Tests: each slice first holds the context lock this long, as a long
    /// draw would.
    #[cfg(test)]
    pub test_slice_sleep_ms: AtomicU32,
    /// Tests: each slice sleeps this long between its last job and its
    /// publish, so a test can look at the cell in between.
    #[cfg(test)]
    pub test_slice_end_ms: AtomicU32,
    /// Tests: the longest a slice runs, in milliseconds, in place of
    /// `RASTER_SLICE`; 0 keeps it.
    #[cfg(test)]
    pub test_slice_limit_ms: AtomicU32,
}

// SAFETY: the jobs in the ring are Send; the ring itself is shared as GFifo
// documents (one producer at a time, the board lock; one consumer, the
// raster thread).
unsafe impl<const HEADS: usize> Send for RasterQueue<HEADS> {}
unsafe impl<const HEADS: usize> Sync for RasterQueue<HEADS> {}

impl<const HEADS: usize> Default for RasterQueue<HEADS> {
    fn default() -> Self {
        const { assert!(HEADS == 1, "the board has one raster unit") };
        let published = PublishedRows::default();
        RasterQueue {
            // SAFETY: a zeroed GFifo is an empty one (GFifo::new); built in
            // place because the ring is too big for the stack.
            ring: unsafe { Box::new_zeroed().assume_init() },
            // SAFETY: as the ring.
            spare: unsafe { Box::new_zeroed().assume_init() },
            ctx: Mutex::new(RasterCtx::new(published.clone())),
            published,
            pending: AtomicU32::new(0),
            queued: AtomicU64::new(0),
            budget: AtomicU64::new(RASTER_WORK_BUDGET),
            room_waiting: AtomicBool::new(false),
            closing: AtomicBool::new(false),
            room: Mutex::new(()),
            room_kick: Condvar::new(),
            pushed: AtomicU64::new(0),
            taken: AtomicU64::new(0),
            running: AtomicBool::new(false),
            exit: AtomicBool::new(false),
            sleeping: AtomicBool::new(false),
            wake: Mutex::new(()),
            kick: Condvar::new(),
            idle: Mutex::new(()),
            idle_kick: Condvar::new(),
            thread: Mutex::new(None),
            #[cfg(test)]
            test_slice_sleep_ms: AtomicU32::new(0),
            #[cfg(test)]
            test_slice_end_ms: AtomicU32::new(0),
            #[cfg(test)]
            test_slice_limit_ms: AtomicU32::new(0),
        }
    }
}

impl<const HEADS: usize> Drop for RasterQueue<HEADS> {
    fn drop(&mut self) {
        // Jobs never run still own their captures.
        while let Some((kind, value)) = self.ring.peek() {
            self.ring.consume();
            // SAFETY: every entry holds a job from submit() or
            // submit_draw(), taken once.
            unsafe { drop_entry(kind & ENTRY_KIND_MASK, value) };
        }
        while let Some((_, value)) = self.spare.peek() {
            self.spare.consume();
            // SAFETY: give_back() put each there, taken once.
            drop(unsafe { Box::from_raw(value as *mut GlDrawJob) });
        }
    }
}

/// Free a ring entry's job without running it.
///
/// # Safety
/// The entry came from submit() or submit_draw() and is taken once.
unsafe fn drop_entry(kind: u32, value: u64) {
    unsafe {
        if kind == ENTRY_DRAW {
            drop(Box::from_raw(value as *mut GlDrawJob));
        } else {
            drop(Box::from_raw(value as *mut RasterJob));
        }
    }
}

impl RasterQueue {
    /// Whether jobs queue for the raster thread.
    pub fn threaded(&self) -> bool {
        self.running.load(SeqCst)
    }

    /// Jobs submitted and not yet finished.
    pub fn pending(&self) -> u32 {
        self.pending.load(SeqCst)
    }

    /// The estimated work of the jobs submitted and not yet finished.
    pub fn queued(&self) -> u64 {
        self.queued.load(SeqCst)
    }

    /// Hand a job to the raster unit, behind every job before it, at the
    /// cost of a job with no pixels of its own (`submit_work`).
    pub fn submit(&self, job: impl FnOnce(&mut RasterCtx) + Send + 'static) {
        self.submit_work(RASTER_JOB_OVERHEAD, job);
    }

    /// Hand a job to the raster unit, behind every job before it, with an
    /// estimate of its work in pixels (`rect_cost`, `primitive_cost`). With
    /// no raster thread it runs now. A job that writes the colour planes
    /// records the rows it wrote in the context (RasterCtx::drawn). The
    /// caller is the front end, under the board lock.
    pub fn submit_work(&self, cost: u32, job: impl FnOnce(&mut RasterCtx) + Send + 'static) {
        if !self.threaded() {
            let mut ctx = self.ctx.lock();
            job(&mut ctx);
            ctx.publish();
            return;
        }
        let entry = Box::into_raw(Box::new(Box::new(job) as RasterJob)) as u64;
        self.push(ENTRY_JOB, entry, cost);
    }

    /// Hand the raster unit a job whose effect the guest can see, a fence's
    /// token or a query's answer, behind every job before it. The rows those
    /// jobs drew are published before it runs: a guest that sees the
    /// completion then arms a swap, and the swap's repaint must already
    /// know which rows differ.
    pub fn submit_completion(&self, job: impl FnOnce(&mut RasterCtx) + Send + 'static) {
        self.submit_work(RASTER_JOB_OVERHEAD, move |ctx| {
            ctx.publish();
            job(ctx);
        });
    }

    /// Hand a GL draw to the raster unit, behind every job before it. The
    /// raster thread gives the job back when drawn (`take_draw`). The
    /// caller is the front end, under the board lock, with the raster
    /// thread running.
    pub fn submit_draw(&self, job: Box<GlDrawJob>, cost: u32) {
        debug_assert!(self.threaded());
        self.push(ENTRY_DRAW, Box::into_raw(job) as u64, cost);
    }

    /// Tests: bound the queued work to `pixels`.
    #[cfg(test)]
    pub fn set_budget(&self, pixels: u64) {
        self.budget.store(pixels, SeqCst);
    }

    /// Wait while the raster unit holds more work than the budget, down to
    /// half of it. The caller is the front end, under the board lock; the
    /// raster thread needs neither that nor anything the front end holds. A
    /// stop, or no raster thread, ends the wait.
    fn wait_room(&self) {
        let mut g = self.room.lock();
        // Raised before the level is looked at, so a raster unit that
        // drains past half after the look sees it and wakes this.
        self.room_waiting.store(true, SeqCst);
        while self.threaded() && !self.closing.load(SeqCst) && self.queued() > self.budget.load(SeqCst) / 2 {
            self.room_kick.wait(&mut g);
        }
        self.room_waiting.store(false, SeqCst);
    }

    fn push(&self, kind: u32, entry: u64, cost: u32) {
        let cost = (cost as u64).min(COST_MAX);
        let kind = kind | (cost as u32) << ENTRY_COST_SHIFT;

        if self.queued() >= self.budget.load(SeqCst) {
            self.wait_room();
        }
        self.queued.fetch_add(cost, SeqCst);
        self.pending.fetch_add(1, SeqCst);
        // The ring is deep; it fills only when the raster unit is far behind,
        // and then the front end waits for it, as the board's would.
        while !self.ring.try_push(kind, entry) {
            std::thread::yield_now();
        }
        self.pushed.fetch_add(1, SeqCst);
        if self.sleeping.load(SeqCst) {
            let _g = self.wake.lock();
            self.kick.notify_one();
        }
    }

    /// A finished draw job to fill again, if one is back. The caller is the
    /// front end, under the board lock: the only taker.
    pub fn take_draw(&self) -> Option<Box<GlDrawJob>> {
        let (_, value) = self.spare.peek()?;
        self.spare.consume();
        // SAFETY: give_back() put it there, and it is taken once.
        Some(unsafe { Box::from_raw(value as *mut GlDrawJob) })
    }

    /// Keep a finished draw job for a later draw, or free it when enough
    /// wait already.
    pub fn give_back(&self, job: Box<GlDrawJob>) {
        let entry = Box::into_raw(job) as u64;
        while !self.spare.try_push(0, entry) {
            if self.spare.len() >= SPARE_DRAWS - 1 {
                // SAFETY: the entry was not pushed, so it is still ours.
                drop(unsafe { Box::from_raw(entry as *mut GlDrawJob) });
                return;
            }
            // The other giver holds the push lock for a moment.
            std::hint::spin_loop();
        }
    }

    /// Run a job now, on the caller's thread, for a caller that needs its
    /// answer: with no raster thread, a GL draw's result and stipple phase.
    pub fn run_now<R>(&self, job: impl FnOnce(&mut RasterCtx) -> R) -> R {
        self.wait_idle();
        let mut ctx = self.ctx.lock();
        let result = job(&mut ctx);
        ctx.publish();
        result
    }

    /// Wait until every job submitted has finished.
    pub fn wait_idle(&self) {
        if self.pending() == 0 {
            return;
        }
        crate::pf_scope!(crate::perfetto::events::ODSY_RASTER_WAIT, self.pending());
        let mut g = self.idle.lock();
        while self.pending() != 0 {
            self.idle_kick.wait(&mut g);
        }
    }

    /// The planes for the display: waits out a slice in flight, and holds
    /// off the next until dropped.
    pub fn planes(&self) -> PlanesHeld<'_> {
        let ctx = self.ctx.lock();
        PlanesHeld { _ctx: ctx, published: &self.published }
    }

    /// The planes for the display if no slice is running now.
    pub fn try_planes(&self) -> Option<PlanesHeld<'_>> {
        let ctx = self.ctx.try_lock()?;
        Some(PlanesHeld { _ctx: ctx, published: &self.published })
    }

    /// Wait out the slice in flight, holding nothing.
    pub fn wait_slice(&self) {
        drop(self.ctx.lock());
    }

    /// Tests: finished draw jobs waiting to be filled again.
    #[cfg(test)]
    pub fn spare_draws(&self) -> usize {
        self.spare.len()
    }

    /// The rows the raster unit published and the display has not taken:
    /// those a window shows go to `marks`, and all join `differ`. Takes the
    /// cell's lock and nothing else, so it does not wait for a slice in
    /// flight.
    pub fn take_published_rows(&self, displayed: u32, differ: &mut Differ, marks: &RowMarks) {
        self.published.lock().take(displayed, differ, marks);
    }

    /// Tests: a slice holds the raster unit's context.
    #[cfg(test)]
    pub fn in_slice(&self) -> bool {
        self.ctx.is_locked()
    }

    /// Start the raster thread.
    pub fn start(self: &std::sync::Arc<Self>) {
        let mut thread = self.thread.lock();
        if thread.is_some() {
            return;
        }
        self.exit.store(false, SeqCst);
        self.closing.store(false, SeqCst);
        self.running.store(true, SeqCst);
        let queue = self.clone();
        *thread = Some(
            std::thread::Builder::new()
                .name(ODYSSEY_RASTER_THREAD.into())
                .spawn(move || queue.run())
                .expect("odyssey raster thread"),
        );
    }

    /// Finish every job queued, then stop the raster thread. Later jobs
    /// run inside their submit.
    pub fn stop(&self) {
        let thread = self.thread.lock().take();
        if let Some(t) = thread {
            // A front end waiting for room goes on and queues behind the
            // rest, which the wait below finishes.
            self.closing.store(true, SeqCst);
            {
                let _g = self.room.lock();
                self.room_kick.notify_all();
            }
            self.wait_idle();
            self.running.store(false, SeqCst);
            self.exit.store(true, SeqCst);
            {
                let _g = self.wake.lock();
                self.kick.notify_one();
            }
            let _ = t.join();
        }
        self.running.store(false, SeqCst);
    }

    /// The raster thread: sleep until a job is pushed, then run the ring in
    /// slices under the context lock.
    fn run(&self) {
        loop {
            {
                let mut g = self.wake.lock();
                self.sleeping.store(true, SeqCst);
                while self.pushed.load(SeqCst) == self.taken.load(SeqCst) && !self.exit.load(SeqCst) {
                    self.kick.wait(&mut g);
                }
                self.sleeping.store(false, SeqCst);
            }
            if self.pushed.load(SeqCst) == self.taken.load(SeqCst) && self.exit.load(SeqCst) {
                return;
            }
            let mut ctx = self.ctx.lock();
            #[cfg(test)]
            {
                let ms = self.test_slice_sleep_ms.swap(0, SeqCst);
                if ms != 0 {
                    std::thread::sleep(Duration::from_millis(ms as u64));
                }
            }
            #[cfg(test)]
            let slice = match self.test_slice_limit_ms.load(SeqCst) {
                0 => RASTER_SLICE,
                ms => Duration::from_millis(ms as u64),
            };
            #[cfg(not(test))]
            let slice = RASTER_SLICE;
            let deadline = Instant::now() + slice;
            crate::pf_begin!(crate::perfetto::events::ODSY_RASTER_SLICE, self.pending());
            while let Some((kind, value)) = self.ring.peek() {
                self.ring.consume();
                self.taken.fetch_add(1, SeqCst);
                let cost = (kind >> ENTRY_COST_SHIFT) as u64;
                let kind = kind & ENTRY_KIND_MASK;
                if kind == ENTRY_DRAW {
                    // SAFETY: submit_draw() put it there, and it is taken once.
                    let job = unsafe { Box::from_raw(value as *mut GlDrawJob) };
                    job.run(&mut ctx);
                    self.give_back(job);
                } else {
                    // SAFETY: submit() put it there, and it is taken once.
                    let job = unsafe { Box::from_raw(value as *mut RasterJob) };
                    job(&mut ctx);
                }
                let left = self.queued.fetch_sub(cost, SeqCst) - cost;
                if self.pending.load(SeqCst) == 1 {
                    // The last job: whoever sees the queue empty then looks
                    // at the planes, so the rows are out before the count
                    // says so.
                    ctx.publish();
                }
                if self.pending.fetch_sub(1, SeqCst) == 1 {
                    let _g = self.idle.lock();
                    self.idle_kick.notify_all();
                }
                if left <= self.budget.load(SeqCst) / 2 && self.room_waiting.load(SeqCst) {
                    let _g = self.room.lock();
                    self.room_kick.notify_all();
                }
                if Instant::now() >= deadline {
                    break;
                }
            }
            #[cfg(test)]
            {
                let ms = self.test_slice_end_ms.load(SeqCst);
                if ms != 0 {
                    std::thread::sleep(Duration::from_millis(ms as u64));
                }
            }
            // A slice's rows are out by its end, bounding how long the
            // display can go without them.
            ctx.publish();
            crate::pf_end!(crate::perfetto::events::ODSY_RASTER_SLICE);
            // A display frame waiting for the planes gets them next.
            MutexGuard::unlock_fair(ctx);
        }
    }
}

/// A byte span of one of the board's planes, for a raster job.
#[derive(Clone, Copy)]
pub struct Span {
    ptr: *mut u8,
    len: usize,
}

impl Span {
    pub fn of(bytes: &mut [u8]) -> Self {
        Span { ptr: bytes.as_mut_ptr(), len: bytes.len() }
    }

    /// The bytes, for the raster unit.
    ///
    /// # Safety
    /// The span's storage outlives the use, and nothing else writes it
    /// meanwhile: the front end waits for the raster unit before it
    /// touches the planes, and never moves or frees them while a job is
    /// queued (`Board::raster_wait`).
    pub unsafe fn bytes<'a>(self) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// The board's planes as a raster job reaches them: each colour buffer,
/// depth and stencil plane from its base, the overlay and id bytes, and the
/// object store the texels come from.
#[derive(Clone, Copy)]
pub struct PlaneSpans {
    pub colour: [Span; ODSY_BUFFERS as usize],
    pub depth: Span,
    pub stencil: Span,
    pub overlay: Span,
    pub did: Span,
    pub obj: Span,
}

// SAFETY: the spans point into the board's planes, which outlive every job
// (Board::raster_wait), and only the raster unit uses them while one runs.
unsafe impl Send for PlaneSpans {}

impl Board {
    /// Wait for the raster unit to finish its queued jobs. The front end
    /// calls it before it touches the planes or the object store itself,
    /// and before it writes a completion word that must follow the drawing.
    pub fn raster_wait(&self) {
        self.raster.wait_idle();
    }

    /// The planes as a raster job reaches them.
    pub fn plane_spans(&mut self) -> PlaneSpans {
        fn rows(m: &mut BoardMemory, plane: u32) -> Span {
            let p = m.plane(plane).expect("the board's planes are allocated");
            let end = (p.base + p.len).min(m.size) as usize;
            Span::of(&mut m.bytes[p.base as usize..end])
        }
        PlaneSpans {
            colour: [rows(&mut self.mem, self.colour_plane), rows(&mut self.buffer1_mem, self.buffer1_plane)],
            depth: rows(&mut self.depth_mem, self.depth_plane),
            stencil: rows(&mut self.stencil_mem, self.stencil_plane),
            overlay: Span::of(&mut self.overlay),
            did: Span::of(&mut self.did),
            obj: Span::of(&mut self.obj_mem.bytes),
        }
    }
}
