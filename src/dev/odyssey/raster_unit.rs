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
//! Board memory belongs to the raster unit while it draws. A job reaches
//! it through raw pointers (`VramPtr`, view::Rows), as the IMPACT model's
//! raster engines reach their pixel memory. Two other threads touch it:
//!
//! - The front end's own memory work (integer and image drawing, pixel DMA,
//!   copies, reset) first waits for the raster unit to finish every job
//!   queued (`Board::raster_wait`). That orders the command stream, in
//!   which the board's pipeline runs a memory access behind the drawing
//!   before it.
//! - The display thread composes each frame from memory with no lock and
//!   no wait, as the hardware's back end reads memory while the pipeline
//!   writes it (reader.rs).
//!
//! The front end waits when the queued work, estimated in pixels per job,
//! passes `RASTER_WORK_BUDGET`, so the drawing stays close behind the commands.
//!
//! The ring takes the shared `GFifo`. With no raster thread running (unit
//! tests, a stopped machine) a job runs inside the submit.

use super::board::{Board, ODSY_HEIGHT, ODSY_WIDTH};
use super::gl::GlDrawJob;
use super::assembly::{LINES, LINE_LOOP, LINE_STRIP, POINTS, TRIANGLES};

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::SeqCst};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::gfifo::GFifo;

/// The raster thread's name.
pub(super) const ODYSSEY_RASTER_THREAD: &str = "odyssey-raster";

/// Jobs the queue holds.
pub(super) const ODSY_RASTER_QUEUE_MAX: usize = 65536;

/// How often a raster thread with an empty queue looks whether the
/// pipeline it holds work in has run dry.
const HELD_POLL: Duration = Duration::from_millis(1);

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
pub(super) const RASTER_WORK_BUDGET: u64 = 1 << 17;

/// What a job costs before its pixels: setup, dispatch and the state it
/// carries, in pixels of fill. A queue of tiny primitives still adds up.
pub(super) const RASTER_JOB_OVERHEAD: u32 = 64;

/// Clears and block fills run about this many times faster per pixel than
/// a shaded fill (`raster_bench`: 200 against 5 to 30 Mpixel/s), so their
/// pixels count for this fraction.
pub(super) const RASTER_CLEAR_DIVISOR: u32 = 8;

/// The cost a ring entry's word can hold.
const COST_MAX: u64 = (1 << 29) - 1;

/// The longest the raster thread runs jobs before it looks at its queue
/// again, checked between jobs.
pub(super) const RASTER_SLICE: Duration = Duration::from_millis(1);

/// What a GL draw hands back to its context in place of the line-stipple
/// phase while the raster unit keeps the phase: the next draw that carries
/// it continues from the raster unit's own count. The context resets its
/// phase to 0 at glBegin and glEnd, which replaces this marker.
pub(super) const LINE_PHASE_ON_RASTER: u32 = u32::MAX;

/// The raster unit's own state.
#[derive(Default)]
pub(super) struct RasterCtx {
    /// The standalone line stipple's phase after the last draw that
    /// carried one.
    pub line_phase: u32,
    /// The quad ring to the shader and back-end threads, while the raster
    /// thread runs (pipe/ring.rs).
    pub ring: Option<Arc<super::pipe::ring::QuadRing>>,
}

impl RasterCtx {
    fn new() -> Self {
        RasterCtx { line_phase: 0, ring: None }
    }

    /// Wait for the back-end thread to store everything queued to it, so
    /// board memory is this thread's (and the front end's) again.
    pub fn drain_back(&mut self) {
        if let Some(r) = &self.ring {
            r.drain();
        }
    }

    /// Whether the shader and back end have stored every quad in the ring.
    pub fn pipe_idle(&self) -> bool {
        self.ring.as_ref().is_none_or(|r| r.idle())
    }

    /// The draw jobs the stages have finished with.
    pub fn reclaim_shade(&mut self) -> Vec<Box<GlDrawJob>> {
        self.ring.as_ref().map(|r| r.reclaim()).unwrap_or_default()
    }
}

/// One unit of raster work.
pub(super) type RasterJob = Box<dyn FnOnce(&mut RasterCtx) + Send>;

/// The ring entry's address word: the value is a boxed `RasterJob`.
const ENTRY_JOB: u32 = 1;
/// The value is a boxed `GlDrawJob`, which goes back to the front end
/// when drawn.
const ENTRY_DRAW: u32 = 2;
/// A boxed `RasterJob` that only puts work into the quad ring (a clear's
/// spans): ordered by the ring, it waits for no stage.
const ENTRY_RING: u32 = 4;
/// The entry's address word is its kind in the low bits and its cost
/// (`RasterQueue::push`) above them.
const ENTRY_KIND_MASK: u32 = 7;
const ENTRY_COST_SHIFT: u32 = 3;

/// The cost of a rectangle of pixels clipped to the screen, a clear's or a
/// block fill's counted at `RASTER_CLEAR_DIVISOR`, plus the job's overhead.
pub(super) fn rect_cost(x0: i32, y0: i32, x1: i32, y1: i32, fast: bool) -> u32 {
    let w = (x1.min(ODSY_WIDTH as i32) - x0.max(0)).max(0) as u64;
    let h = (y1.min(ODSY_HEIGHT as i32) - y0.max(0)).max(0) as u64;
    let px = w * h / if fast { RASTER_CLEAR_DIVISOR as u64 } else { 1 };

    RASTER_JOB_OVERHEAD.saturating_add(px.min(COST_MAX) as u32)
}

/// The cost of a primitive from its vertices' window positions, None for a
/// vertex with no position (behind the eye, where clipping will cut the
/// primitive and what is left may be any size, so the whole screen counts).
/// Points cost their square, lines their length times their width, a
/// triangle its area and any other polygon its bounding box, each within
/// the screen. An estimate, from the corners alone.
pub(super) fn primitive_cost(mode: u32, points: impl Iterator<Item = Option<(f64, f64)>>, point_size: f32, line_width: f32) -> u32 {
    let (w, h) = (ODSY_WIDTH as f64, ODSY_HEIGHT as f64);
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
        POINTS => (point_size.max(1.0) as f64).powi(2) * n as f64,
        LINES | LINE_LOOP | LINE_STRIP => (bw.max(bh) + 1.0) * (line_width.max(1.0) as f64),
        TRIANGLES if n == 3 => {
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

/// The queue between the front end and the raster unit. `HEADS` is the
/// GFifo's consumer heads: one, because the board has one raster unit. A
/// second would need the jobs to split the rows between the heads, as the
/// IMPACT model's two engines interleave scanlines.
pub(super) struct RasterQueue<const HEADS: usize = 1> {
    ring: Box<GFifo<ODSY_RASTER_QUEUE_MAX, HEADS>>,
    /// Finished draw jobs, from the raster thread (or a draw run inside
    /// its submit) back to the front end, which takes them under the board
    /// lock.
    spare: Box<GFifo<SPARE_DRAWS>>,
    /// The raster unit's own state (the stipple phase and the quad ring,
    /// not board memory). The raster thread owns it while it runs jobs; the
    /// front end uses it only for a job it runs itself, with no raster thread
    /// (`RasterQueue::ctx`).
    ctx: UnsafeCell<RasterCtx>,
    /// Jobs submitted and not yet finished, and one more while `held`.
    pending: AtomicU32,
    /// The shader and back end hold quads of finished draws: the raster
    /// thread keeps one count in `pending` for them rather than drain the
    /// pipeline whenever its queue runs dry, so the stages keep overlapping
    /// when the front end submits draws one at a time. Whoever waits for
    /// the raster unit asks for the drain (`flush`).
    held: AtomicBool,
    /// A waiter wants the held work stored.
    flush: AtomicBool,
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
    /// The fragment pipeline's workers and tile size (`[graphics] threads`,
    /// `tile`), for the pool the raster thread starts.
    pool: (AtomicU32, AtomicU32),
    /// The raster thread runs, so jobs queue.
    running: AtomicBool,
    exit: AtomicBool,
    sleeping: AtomicBool,
    wake: Mutex<()>,
    kick: Condvar,
    idle: Mutex<()>,
    idle_kick: Condvar,
    thread: Mutex<Option<JoinHandle<()>>>,
    /// Tests: the raster thread is in a slice.
    #[cfg(test)]
    test_in_slice: AtomicBool,
    /// Tests: each slice first runs this long, as a long draw would.
    #[cfg(test)]
    pub test_slice_sleep_ms: AtomicU32,
    /// Tests: each slice sleeps this long after its last job, so a test can
    /// look at the cell in between.
    #[cfg(test)]
    pub test_slice_end_ms: AtomicU32,
    /// Tests: the longest a slice runs, in milliseconds, in place of
    /// `RASTER_SLICE`; 0 keeps it.
    #[cfg(test)]
    pub test_slice_limit_ms: AtomicU32,
}

// SAFETY: the jobs in the ring are Send; the ring itself is shared as GFifo
// documents (one producer at a time, the front end; one consumer, the raster
// thread). `ctx` is the raster thread's while it runs; the front end touches
// it only with no raster thread (`!threaded()`), as the other boards' CPU
// threads touch an engine's state only after its threads have stopped. A guest that stores while
// it waits for a drain gets whatever ordering it asked for, not a lock.
unsafe impl<const HEADS: usize> Send for RasterQueue<HEADS> {}
unsafe impl<const HEADS: usize> Sync for RasterQueue<HEADS> {}

impl<const HEADS: usize> Default for RasterQueue<HEADS> {
    fn default() -> Self {
        const { assert!(HEADS == 1, "the board has one raster unit") };
        RasterQueue {
            // SAFETY: a zeroed GFifo is an empty one (GFifo::new); built in
            // place because the ring is too big for the stack.
            ring: unsafe { Box::new_zeroed().assume_init() },
            // SAFETY: as the ring.
            spare: unsafe { Box::new_zeroed().assume_init() },
            ctx: UnsafeCell::new(RasterCtx::new()),
            pending: AtomicU32::new(0),
            held: AtomicBool::new(false),
            flush: AtomicBool::new(false),
            queued: AtomicU64::new(0),
            budget: AtomicU64::new(RASTER_WORK_BUDGET),
            room_waiting: AtomicBool::new(false),
            closing: AtomicBool::new(false),
            room: Mutex::new(()),
            room_kick: Condvar::new(),
            pushed: AtomicU64::new(0),
            taken: AtomicU64::new(0),
            pool: (AtomicU32::new(2), AtomicU32::new(32)),
            running: AtomicBool::new(false),
            exit: AtomicBool::new(false),
            sleeping: AtomicBool::new(false),
            wake: Mutex::new(()),
            kick: Condvar::new(),
            idle: Mutex::new(()),
            idle_kick: Condvar::new(),
            thread: Mutex::new(None),
            #[cfg(test)]
            test_in_slice: AtomicBool::new(false),
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

#[cfg(test)]
    /// Hand a job to the raster unit, behind every job before it, at the
    /// cost of a job with no pixels of its own (`submit_work`).
    pub fn submit(&self, job: impl FnOnce(&mut RasterCtx) + Send + 'static) {
        self.submit_work(RASTER_JOB_OVERHEAD, job);
    }

    /// Hand a job to the raster unit, behind every job before it, with an
    /// estimate of its work in pixels (`rect_cost`, `primitive_cost`). With
    /// no raster thread it runs now. The caller is the front end.
    pub fn submit_work(&self, cost: u32, job: impl FnOnce(&mut RasterCtx) + Send + 'static) {
        if !self.threaded() {
            // SAFETY: no raster thread runs, so the front end owns the
            // context (see the Sync impl).
            let ctx = unsafe { &mut *self.ctx.get() };
            job(ctx);
            return;
        }
        let entry = Box::into_raw(Box::new(Box::new(job) as RasterJob)) as u64;
        self.push(ENTRY_JOB, entry, cost);
    }

    /// Hand the raster unit a job that only writes into the quad ring
    /// (RasterCtx::ring), behind every job before it; with no ring it
    /// runs as `submit_work` does, and must store by itself.
    pub fn submit_ring(&self, cost: u32, job: impl FnOnce(&mut RasterCtx) + Send + 'static) {
        if !self.threaded() {
            self.submit_work(cost, job);
            return;
        }
        let entry = Box::into_raw(Box::new(Box::new(job) as RasterJob)) as u64;
        self.push(ENTRY_RING, entry, cost);
    }

    /// Hand the raster unit a job whose effect the guest can see, a fence's
    /// token or a query's answer, behind every job before it.
    pub fn submit_completion(&self, job: impl FnOnce(&mut RasterCtx) + Send + 'static) {
        self.submit_work(RASTER_JOB_OVERHEAD, move |ctx| {
            job(ctx);
        });
    }

    /// Hand a GL draw to the raster unit, behind every job before it. The
    /// raster thread gives the job back when drawn (`take_draw`). The
    /// caller is the front end, with the raster
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
    /// half of it. The caller is the front end; the
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
    /// front end: the only taker.
    pub fn take_draw(&self) -> Option<Box<GlDrawJob>> {
        let (_, value) = self.spare.peek()?;
        self.spare.consume();
        // SAFETY: give_back() put it there, and it is taken once.
        Some(unsafe { Box::from_raw(value as *mut GlDrawJob) })
    }

    /// Keep a finished draw job for a later draw, or free it when enough
    /// wait already.
    pub fn give_back(&self, mut job: Box<GlDrawJob>) {
        job.release();
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
        // SAFETY: the queue is idle and this is the front end, which queues
        // the jobs, so nothing else runs one (see the Sync impl).
        let ctx = unsafe { &mut *self.ctx.get() };
        let result = job(ctx);
        result
    }

    /// Ask the raster thread to store the work the pipeline holds, so the
    /// count of pending jobs reaches zero.
    pub fn request_flush(&self) {
        if self.held.load(SeqCst) && !self.flush.swap(true, SeqCst) {
            let _g = self.wake.lock();
            self.kick.notify_one();
        }
    }

    /// The raster thread: store what the pipeline holds and give up the
    /// count kept for it.
    fn release_held(&self, ctx: &mut RasterCtx) {
        ctx.drain_back();
        for job in ctx.reclaim_shade() {
            self.give_back(job);
        }
        self.flush.store(false, SeqCst);
        if self.held.swap(false, SeqCst) && self.pending.fetch_sub(1, SeqCst) == 1 {
            let _g = self.idle.lock();
            self.idle_kick.notify_all();
        }
    }

    /// Wait until every job submitted has finished.
    pub fn wait_idle(&self) {
        if self.pending() == 0 {
            return;
        }
        crate::pf_scope!(crate::perfetto::events::ODSY_RASTER_WAIT, self.pending());
        self.request_flush();
        let mut g = self.idle.lock();
        while self.pending() != 0 {
            self.idle_kick.wait(&mut g);
        }
    }

    /// Tests: finished draw jobs waiting to be filled again.
    #[cfg(test)]
    pub fn spare_draws(&self) -> usize {
        self.spare.len()
    }

    /// Tests: the front end is waiting for room in the queue.
    #[cfg(test)]
    pub fn waiting_for_room(&self) -> bool {
        self.room_waiting.load(SeqCst)
    }

    /// Tests: the raster thread is running a slice of jobs.
    #[cfg(test)]
    pub fn in_slice(&self) -> bool {
        self.test_in_slice.load(SeqCst)
    }

    /// Start the raster thread and the back end's (pipe/thread.rs);
    /// IRIS_ODSY_BACKEND=inline keeps the back end on the raster thread.
    pub fn start(self: &std::sync::Arc<Self>) {
        self.start_threads(std::env::var("IRIS_ODSY_BACKEND").as_deref() != Ok("inline"));
    }

    /// The fragment pipeline's workers and tile size for the next start.
    pub fn set_pool(&self, threads: usize, tile: u32) {
        self.pool.0.store(threads.max(1) as u32, SeqCst);
        self.pool.1.store(tile, SeqCst);
    }

    /// Start the raster thread, and with `backend` the back end's.
    pub fn start_threads(self: &std::sync::Arc<Self>, backend: bool) {
        let mut thread = self.thread.lock();
        if thread.is_some() {
            return;
        }
        self.exit.store(false, SeqCst);
        self.closing.store(false, SeqCst);
        // SAFETY: no raster thread runs yet, so the context is this thread's.
        let ctx = unsafe { &mut *self.ctx.get() };
        if ctx.ring.is_none() && backend {
            // The shader's thread shades in the ring, so it comes with the
            // back end's.
            let shader = std::env::var("IRIS_ODSY_SHADER").as_deref() != Ok("inline");

            ctx.ring = Some(super::pipe::ring::QuadRing::start(shader, self.pool.0.load(SeqCst) as usize, self.pool.1.load(SeqCst)));
        }
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
            // SAFETY: the raster thread has stopped, so the context is ours.
            let ctx = unsafe { &mut *self.ctx.get() };
            if let Some(r) = ctx.ring.take() {
                r.stop();
            }
        }
        self.running.store(false, SeqCst);
    }

    /// The raster thread: sleep until a job is pushed, then run the ring in
    /// slices.
    fn run(&self) {
        loop {
            // SAFETY: this is the raster thread, which owns the context
            // while it runs (see the Sync impl).
            let ctx = unsafe { &mut *self.ctx.get() };
            {
                let mut g = self.wake.lock();
                self.sleeping.store(true, SeqCst);
                while self.pushed.load(SeqCst) == self.taken.load(SeqCst) && !self.exit.load(SeqCst) {
                    if !self.held.load(SeqCst) {
                        self.kick.wait(&mut g);
                    } else if self.flush.load(SeqCst) || ctx.pipe_idle() {
                        break;
                    } else {
                        // The pipeline gives up its count once it runs dry
                        // even when no one asks, for the readers of the
                        // count that do not.
                        self.kick.wait_for(&mut g, HELD_POLL);
                    }
                }
                self.sleeping.store(false, SeqCst);
            }
            if self.pushed.load(SeqCst) == self.taken.load(SeqCst) {
                if self.held.load(SeqCst) {
                    self.release_held(ctx);
                    continue;
                }
                if self.exit.load(SeqCst) {
                    return;
                }
                continue;
            }
            #[cfg(test)]
            self.test_in_slice.store(true, SeqCst);
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
                    if let Some(job) = job.run_owned(ctx) {
                        self.give_back(job);
                    }
                    for job in ctx.reclaim_shade() {
                        self.give_back(job);
                    }
                } else if kind == ENTRY_RING {
                    // SAFETY: submit_ring() put it there, and it is taken once.
                    let job = unsafe { Box::from_raw(value as *mut RasterJob) };
                    job(ctx);
                } else {
                    // SAFETY: submit() put it there, and it is taken once.
                    let job = unsafe { Box::from_raw(value as *mut RasterJob) };
                    // Any other job touches board memory itself.
                    ctx.drain_back();
                    job(ctx);
                    if self.held.load(SeqCst) {
                        self.release_held(ctx);
                    }
                }
                let left = self.queued.fetch_sub(cost, SeqCst) - cost;
                // This job's count, unless the pipeline keeps it.
                let mut done = 1;
                if self.pending.load(SeqCst) == 1 + self.held.load(SeqCst) as u32 {
                    // The last job: whoever sees the count reach zero then
                    // looks at board memory, so the stores are
                    // out before it does. A draw's quads may stay in the
                    // pipeline under the count kept for them, until a
                    // waiter asks.
                    if kind != ENTRY_JOB && !self.flush.load(SeqCst) && !ctx.pipe_idle() {
                        if !self.held.swap(true, SeqCst) {
                            done = 0;
                        }
                    } else {
                        ctx.drain_back();
                        for job in ctx.reclaim_shade() {
                            self.give_back(job);
                        }
                        self.flush.store(false, SeqCst);
                        if self.held.swap(false, SeqCst) {
                            done += 1;
                        }
                    }
                }
                if done != 0 && self.pending.fetch_sub(done, SeqCst) == done {
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
            crate::pf_end!(crate::perfetto::events::ODSY_RASTER_SLICE);
            #[cfg(test)]
            self.test_in_slice.store(false, SeqCst);
        }
    }
}

impl Board {
    /// Wait for the raster unit to finish its queued jobs. The front end
    /// calls it before it touches board memory itself,
    /// and before it writes a completion word that must follow the drawing.
    pub fn raster_wait(&self) {
        self.raster.wait_idle();
    }
}
