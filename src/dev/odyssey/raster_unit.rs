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
//! The ring takes the shared `GFifo`. With no raster thread running (unit
//! tests, a stopped machine) a job runs inside the submit.

use super::*;

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::SeqCst};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex, MutexGuard};

use crate::gfifo::GFifo;

/// The raster thread's name.
pub const ODYSSEY_RASTER_THREAD: &str = "odyssey-raster";

/// Jobs the queue holds.
pub const ODSY_RASTER_QUEUE_MAX: usize = 65536;

/// The longest the raster thread holds its context lock before letting a
/// waiting display frame in, checked between jobs.
pub const RASTER_SLICE: Duration = Duration::from_millis(1);

/// What a GL draw hands back to its context in place of the line-stipple
/// phase while the raster unit keeps the phase: the next draw that carries
/// it continues from the raster unit's own count. The context resets its
/// phase to 0 at glBegin and glEnd, which replaces this marker.
pub const LINE_PHASE_ON_RASTER: u32 = u32::MAX;

/// The raster unit's own state.
#[derive(Default)]
pub struct RasterCtx {
    /// The standalone line stipple's phase after the last draw that
    /// carried one.
    pub line_phase: u32,
}

/// One unit of raster work.
pub type RasterJob = Box<dyn FnOnce(&mut RasterCtx) + Send>;

/// The ring entry's address word: the value is a boxed `RasterJob`.
const ENTRY_JOB: u32 = 1;

thread_local! {
    /// This thread holds the raster unit's context lock and reads the
    /// planes under it (the display thread's resolve).
    static PLANES_HELD: Cell<bool> = const { Cell::new(false) };
}

/// The planes as the display holds them: no raster job runs meanwhile.
pub struct PlanesHeld<'a> {
    _ctx: MutexGuard<'a, RasterCtx>,
}

impl Drop for PlanesHeld<'_> {
    fn drop(&mut self) {
        PLANES_HELD.with(|h| h.set(false));
    }
}

/// The queue between the front end and the raster unit. `HEADS` is the
/// GFifo's consumer heads: one, because the board has one raster unit. A
/// second would need the jobs to split the rows between the heads, as the
/// IMPACT model's two engines interleave scanlines.
pub struct RasterQueue<const HEADS: usize = 1> {
    ring: Box<GFifo<ODSY_RASTER_QUEUE_MAX, HEADS>>,
    /// Held by the raster thread for a slice of jobs, by a job run inside
    /// the submit, and by the display while it reads the planes.
    ctx: Mutex<RasterCtx>,
    /// Jobs submitted and not yet finished.
    pending: AtomicU32,
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
    /// The raster unit drew since the display last looked.
    drawn: AtomicBool,
    thread: Mutex<Option<JoinHandle<()>>>,
    /// Tests: each slice first holds the context lock this long, as a long
    /// draw would.
    #[cfg(test)]
    pub test_slice_sleep_ms: AtomicU32,
}

// SAFETY: the jobs in the ring are Send; the ring itself is shared as GFifo
// documents (one producer at a time, the board lock; one consumer, the
// raster thread).
unsafe impl<const HEADS: usize> Send for RasterQueue<HEADS> {}
unsafe impl<const HEADS: usize> Sync for RasterQueue<HEADS> {}

impl<const HEADS: usize> Default for RasterQueue<HEADS> {
    fn default() -> Self {
        const { assert!(HEADS == 1, "the board has one raster unit") };
        RasterQueue {
            // SAFETY: a zeroed GFifo is an empty one (GFifo::new); built in
            // place because the ring is too big for the stack.
            ring: unsafe { Box::new_zeroed().assume_init() },
            ctx: Mutex::new(RasterCtx::default()),
            pending: AtomicU32::new(0),
            pushed: AtomicU64::new(0),
            taken: AtomicU64::new(0),
            running: AtomicBool::new(false),
            exit: AtomicBool::new(false),
            sleeping: AtomicBool::new(false),
            wake: Mutex::new(()),
            kick: Condvar::new(),
            idle: Mutex::new(()),
            idle_kick: Condvar::new(),
            drawn: AtomicBool::new(false),
            thread: Mutex::new(None),
            #[cfg(test)]
            test_slice_sleep_ms: AtomicU32::new(0),
        }
    }
}

impl<const HEADS: usize> Drop for RasterQueue<HEADS> {
    fn drop(&mut self) {
        // Jobs never run still own their captures.
        while let Some((_, value)) = self.ring.peek() {
            self.ring.consume();
            // SAFETY: every entry holds a job from submit(), taken once.
            drop(unsafe { Box::from_raw(value as *mut RasterJob) });
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

    /// Hand a job to the raster unit, behind every job before it. With no
    /// raster thread it runs now. The caller is the front end, under the
    /// board lock.
    pub fn submit(&self, job: impl FnOnce(&mut RasterCtx) + Send + 'static) {
        if !self.threaded() {
            let mut ctx = self.ctx.lock();
            job(&mut ctx);
            self.drawn.store(true, SeqCst);
            return;
        }
        let entry = Box::into_raw(Box::new(Box::new(job) as RasterJob)) as u64;
        self.pending.fetch_add(1, SeqCst);
        // The ring is deep; it fills only when the raster unit is far behind,
        // and then the front end waits for it, as the board's would.
        while !self.ring.try_push(ENTRY_JOB, entry) {
            std::thread::yield_now();
        }
        self.pushed.fetch_add(1, SeqCst);
        if self.sleeping.load(SeqCst) {
            let _g = self.wake.lock();
            self.kick.notify_one();
        }
    }

    /// Run a job now, on the caller's thread, for a caller that needs its
    /// answer: with no raster thread, a GL draw's result and stipple phase.
    pub fn run_now<R>(&self, job: impl FnOnce(&mut RasterCtx) -> R) -> R {
        self.wait_idle();
        let mut ctx = self.ctx.lock();
        let r = job(&mut ctx);
        self.drawn.store(true, SeqCst);
        r
    }

    /// Wait until every job submitted has finished. The display thread,
    /// which holds the planes against the raster thread already, returns
    /// at once.
    pub fn wait_idle(&self) {
        if self.pending() == 0 || PLANES_HELD.with(Cell::get) {
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
        PLANES_HELD.with(|h| h.set(true));
        PlanesHeld { _ctx: ctx }
    }

    /// Tests: a slice holds the raster unit's context.
    #[cfg(test)]
    pub fn in_slice(&self) -> bool {
        self.ctx.is_locked()
    }

    /// Whether the raster unit drew since the last call.
    pub fn take_drawn(&self) -> bool {
        self.drawn.swap(false, SeqCst)
    }

    /// Start the raster thread.
    pub fn start(self: &std::sync::Arc<Self>) {
        let mut thread = self.thread.lock();
        if thread.is_some() {
            return;
        }
        self.exit.store(false, SeqCst);
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
            let deadline = Instant::now() + RASTER_SLICE;
            crate::pf_begin!(crate::perfetto::events::ODSY_RASTER_SLICE, self.pending());
            while let Some((_, value)) = self.ring.peek() {
                self.ring.consume();
                self.taken.fetch_add(1, SeqCst);
                // SAFETY: every entry holds a job from submit(), taken once.
                let job = unsafe { Box::from_raw(value as *mut RasterJob) };
                job(&mut ctx);
                self.drawn.store(true, SeqCst);
                if self.pending.fetch_sub(1, SeqCst) == 1 {
                    let _g = self.idle.lock();
                    self.idle_kick.notify_all();
                }
                if Instant::now() >= deadline {
                    break;
                }
            }
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
