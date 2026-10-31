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
}

impl Default for DrawnRows {
    fn default() -> Self {
        DrawnRows { by_id: Box::new([[[0; ROW_WORDS]; ODSY_DISPMODE_COUNT as usize]; ODSY_BUFFERS as usize]), overlay: [0; ROW_WORDS] }
    }
}

impl DrawnRows {
    /// Keep what a draw wrote, which passed the id test `id_compare` (the
    /// board's 0x840 word) at each pixel it stored.
    pub fn add(&mut self, touched: &TouchedRows, id_compare: u32) {
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

    /// Hand over the rows that show a change under the displayed-buffer
    /// bits `displayed`, and forget every row. A row written to a buffer no
    /// window shows is forgotten too: the window shows it only after a flip
    /// of its displayed buffer, and that repaints the whole frame.
    pub fn take(&mut self, displayed: u32, mut row: impl FnMut(u32)) {
        let mut shown = self.overlay;

        for (b, ids) in self.by_id.iter().enumerate() {
            for (id, rows) in ids.iter().enumerate() {
                if (displayed >> id) & 1 == b as u32 {
                    for (s, w) in shown.iter_mut().zip(rows) {
                        *s |= w;
                    }
                }
            }
        }
        for (i, mut w) in shown.into_iter().enumerate() {
            while w != 0 {
                row(i as u32 * 64 + w.trailing_zeros());
                w &= w - 1;
            }
        }
        for ids in self.by_id.iter_mut() {
            for rows in ids.iter_mut() {
                *rows = [0; ROW_WORDS];
            }
        }
        self.overlay = [0; ROW_WORDS];
    }
}

/// The raster unit's own state.
#[derive(Default)]
pub struct RasterCtx {
    /// The standalone line stipple's phase after the last draw that
    /// carried one.
    pub line_phase: u32,
    /// The colour rows drawn since the display last looked. A job that
    /// writes the colour planes records its rows here, or the display never
    /// resolves them.
    pub drawn: DrawnRows,
}

/// One unit of raster work.
pub type RasterJob = Box<dyn FnOnce(&mut RasterCtx) + Send>;

/// The ring entry's address word: the value is a boxed `RasterJob`.
const ENTRY_JOB: u32 = 1;
/// The value is a boxed `GlDrawJob`, which goes back to the front end
/// when drawn.
const ENTRY_DRAW: u32 = 2;

/// Finished GL draw jobs kept for the front end to fill again. A draw
/// that finds none allocates; one finished while this many wait is freed.
const SPARE_DRAWS: usize = 1024;

thread_local! {
    /// This thread holds the raster unit's context lock and reads the
    /// planes under it (the display thread's resolve).
    static PLANES_HELD: Cell<bool> = const { Cell::new(false) };
}

/// The planes as the display holds them: no raster job runs meanwhile.
pub struct PlanesHeld<'a> {
    _ctx: MutexGuard<'a, RasterCtx>,
}

impl PlanesHeld<'_> {
    /// The colour rows the raster unit wrote since the last call that show
    /// a change under the displayed-buffer bits (DrawnRows::take()).
    pub fn take_rows(&mut self, displayed: u32, row: impl FnMut(u32)) {
        self._ctx.drawn.take(displayed, row);
    }
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
    /// Finished draw jobs, from the raster thread (or a draw run inside
    /// its submit) back to the front end, which takes them under the board
    /// lock.
    spare: Box<GFifo<SPARE_DRAWS>>,
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
            // SAFETY: as the ring.
            spare: unsafe { Box::new_zeroed().assume_init() },
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
            thread: Mutex::new(None),
            #[cfg(test)]
            test_slice_sleep_ms: AtomicU32::new(0),
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
            unsafe { drop_entry(kind, value) };
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

    /// Hand a job to the raster unit, behind every job before it. With no
    /// raster thread it runs now. A job that writes the colour planes
    /// records the rows it wrote in the context (RasterCtx::drawn). The
    /// caller is the front end, under the board lock.
    pub fn submit(&self, job: impl FnOnce(&mut RasterCtx) + Send + 'static) {
        if !self.threaded() {
            let mut ctx = self.ctx.lock();
            job(&mut ctx);
            return;
        }
        let entry = Box::into_raw(Box::new(Box::new(job) as RasterJob)) as u64;
        self.push(ENTRY_JOB, entry);
    }

    /// Hand a GL draw to the raster unit, behind every job before it. The
    /// raster thread gives the job back when drawn (`take_draw`). The
    /// caller is the front end, under the board lock, with the raster
    /// thread running.
    pub fn submit_draw(&self, job: Box<GlDrawJob>) {
        debug_assert!(self.threaded());
        self.push(ENTRY_DRAW, Box::into_raw(job) as u64);
    }

    fn push(&self, kind: u32, entry: u64) {
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
        job(&mut ctx)
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

    /// Tests: finished draw jobs waiting to be filled again.
    #[cfg(test)]
    pub fn spare_draws(&self) -> usize {
        self.spare.len()
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
            while let Some((kind, value)) = self.ring.peek() {
                self.ring.consume();
                self.taken.fetch_add(1, SeqCst);
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
