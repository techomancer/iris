//! The fragment pipeline's screen tiles and their worker pool
//! (`clanker/odyssey/screen-tiles.md`).
//!
//! The raster thread sets primitives up and writes each as one command in
//! the next slot of a ring: a triangle (its record and bounding box), a
//! quad it shaded itself (lines, points, smoothed polygons), or a clear's
//! rectangle. It then appends the slot's index to the queue of every screen
//! tile (`TILE` pixels square, in plane coordinates) the command touches.
//! A tile with commands waiting goes on the ready queue; a worker takes it
//! and runs its commands in order: walks the triangle within the tile and
//! shades and stores each quad, stores the shaded quad, or clears its part
//! of the rectangle. One worker owns a tile at a time, so every pixel sees
//! its fragments in command order with no other ordering, and a large
//! primitive is drawn by as many workers as it covers tiles.
//!
//! A slot counts the tiles still to run it; the raster thread retires
//! slots whose count is zero in order (`tail`), behind which it writes
//! slots again, and a draw keeps its job and context until the tail is past
//! its last slot (`hold`, `reclaim`). The display composes each frame from
//! board memory as it stands, so nobody tracks the rows the stores wrote.

use super::backend::{DestMem, BackendState, ClearDraw};
use super::raster::{triangle_quads, RasterState};
use super::super::gl::{DrawFrame, GlDrawJob};
use super::setup::Tri;
use parking_lot::{Condvar, Mutex};
use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::{Acquire, Relaxed, Release, SeqCst}};
use std::sync::Arc;
use std::thread::JoinHandle;

/// The pool threads' name, numbered.
pub(crate) const ODYSSEY_POOL_THREAD: &str = "odyssey-pool";

/// Slots in the ring.
const RING: usize = 1 << 14;
/// How long a thread spins before it yields or sleeps.
const SPINS: u32 = 2000;
/// How long an idle worker spins for a tile before it sleeps.
const IDLE_SPINS: u32 = 200;

/// A slot's command.
/// A triangle: its record in ptr, its bounding box in plane pixels in x, y
/// and zq[0], zq[1] (the far corner).
pub(crate) const KIND_TRI: u8 = 0;
/// A quad shaded by the raster thread: colours in data, back-end state in ptr.
pub(crate) const KIND_SHADED: u8 = 1;
/// A clear's rectangle: x, y to zq[0], zq[1] in plane pixels, the clear's
/// record (`backend::ClearDraw`) in ptr.
pub(crate) const KIND_RECT: u8 = 3;

/// A draw's back-end state, as the ring's slots name it.
#[repr(C, align(64))]
pub(crate) struct BackendDraw {
    pub state: BackendState,
    pub planes: DestMem,
}

/// Up to four shaded fragments of one quad, as the raster thread shades
/// them itself: pixel i is plane pixel (x + (i & 1), y + (i >> 1)) where
/// `cov` bit i is set, with its depth word (polygon offset applied) and
/// 12-bit colour components.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(crate) struct BackQuad {
    pub draw: *const BackendDraw,
    pub x: i32,
    pub y: i32,
    pub zq: [u32; 4],
    pub rgba: [[u16; 4]; 4],
    pub cov: u8,
}

impl Default for BackQuad {
    fn default() -> Self {
        BackQuad { draw: std::ptr::null(), x: 0, y: 0, zq: [0; 4], rgba: [[0; 4]; 4], cov: 0 }
    }
}

impl BackQuad {
}

/// One slot: a quad on its way through the stages, one cache line.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(crate) struct RingQuad {
    /// Plane pixel of the quad's first pixel.
    pub x: i16,
    pub y: i16,
    pub cov: u8,
    pub kind: u8,
    _pad: u16,
    /// The depth words, polygon offset applied.
    pub zq: [u32; 4],
    /// Before shading, the four pixels' barycentrics l1 then l2 as f32
    /// bits; after, each pixel's colour as four 12-bit components in u16s
    /// (`colour`, `set_colour`).
    pub data: [u32; 8],
    /// The triangle's record (KIND_SHADE, which leads to the draw's context
    /// and back-end state), the draw's back-end state (KIND_SHADED) or the
    /// clear's record (KIND_SPAN).
    pub ptr: *const (),
}

const _: () = assert!(std::mem::size_of::<RingQuad>() == 64);

impl RingQuad {
    /// Pixel i's 12-bit colour components.
    #[inline(always)]
    pub fn colour(&self, i: usize) -> [u32; 4] {
        let (a, b) = (self.data[2 * i], self.data[2 * i + 1]);

        [a & 0xffff, a >> 16, b & 0xffff, b >> 16]
    }

    #[inline(always)]
    pub fn set_colour(&mut self, i: usize, c: [u16; 4]) {
        self.data[2 * i] = c[0] as u32 | (c[1] as u32) << 16;
        self.data[2 * i + 1] = c[2] as u32 | (c[3] as u32) << 16;
    }
}

impl Default for RingQuad {
    fn default() -> Self {
        RingQuad {
            x: 0,
            y: 0,
            cov: 0,
            kind: KIND_SHADED,
            _pad: 0,
            zq: [0; 4],
            data: [0; 8],
            ptr: std::ptr::null(),
        }
    }
}

/// The pool's sleep.
#[derive(Default)]
struct Waiter {
    sleepers: AtomicU32,
    m: Mutex<()>,
    cv: Condvar,
}

impl Waiter {
    /// Sleep unless `more` (read after the count is up) or exit.
    fn sleep(&self, more: impl Fn() -> bool, exit: &AtomicBool) {
        let mut g = self.m.lock();

        self.sleepers.fetch_add(1, SeqCst);
        if !more() && !exit.load(SeqCst) {
            self.cv.wait(&mut g);
        }
        self.sleepers.fetch_sub(1, SeqCst);
    }

    /// Wake a sleeper, if any; the caller has just published work with
    /// SeqCst.
    fn wake(&self, all: bool) {
        if self.sleepers.load(SeqCst) != 0 {
            let _g = self.m.lock();
            if all {
                self.cv.notify_all();
            } else {
                self.cv.notify_one();
            }
        }
    }
}

/// A draw kept until the tail passes `end`: its job and context.
type Held = (u64, Box<GlDrawJob>, Box<DrawFrame>);

/// The plane the tile grid covers, per axis; coordinates wrap onto it
/// (pixels that far apart then share a tile, which only orders them).
const GRID_SPAN: i32 = 4096;
/// Slot indices the tile queues hold in all.
const QUEUE_BUDGET: usize = 1 << 22;

/// One tile's queue of slot indices: appended by the raster thread, run by
/// the worker that owns the tile.
#[repr(C, align(64))]
struct Tile {
    head: AtomicU64,
    tail: AtomicU64,
    /// On the ready queue or owned by a worker.
    scheduled: AtomicBool,
}

pub(crate) struct QuadRing {
    ring: Box<[UnsafeCell<RingQuad>]>,
    /// Each slot's tiles still to run it.
    refs: Box<[AtomicU32]>,
    /// Slots written, and slots retired.
    staged: crossbeam_utils::CachePadded<AtomicU64>,
    tail: crossbeam_utils::CachePadded<AtomicU64>,
    tiles: Box<[Tile]>,
    queues: Box<[UnsafeCell<u32>]>,
    /// Tiles a grid row, log2 of the tile size, entries a tile queue.
    grid: usize,
    tile_log2: u32,
    queue: u64,
    ready: crossbeam_queue::ArrayQueue<u32>,
    /// Workers shade filled triangles; without it the raster thread shades
    /// every quad itself.
    shader: bool,
    pool: Waiter,
    exit: AtomicBool,
    held: Mutex<VecDeque<Held>>,
    /// Clears' records, kept as the draws are.
    held_clears: Mutex<VecDeque<(u64, Box<ClearDraw>)>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

// SAFETY: a slot is written by the raster thread before its index goes on
// any tile's queue and read-only after, until every tile has run it and the
// raster thread retires it. A tile queue's entry is written by the raster
// thread before it moves the tile's head past it and read by the tile's one
// owner after.
unsafe impl Sync for QuadRing {}
unsafe impl Send for QuadRing {}

impl QuadRing {
    /// Start the pool of `workers` threads on tiles `tile` pixels square (a
    /// power of two, 8 to 256); with `shader` the workers shade filled
    /// triangles.
    pub fn start(shader: bool, workers: usize, tile: u32) -> Arc<QuadRing> {
        let tile = (tile.clamp(8, 256).next_power_of_two()) as i32;
        let grid = (GRID_SPAN / tile) as usize;
        let ntiles = grid * grid;
        let queue = (QUEUE_BUDGET / ntiles).clamp(16, 4096).next_power_of_two() as u64;
        let r = Arc::new(QuadRing {
            ring: (0..RING).map(|_| UnsafeCell::new(RingQuad::default())).collect(),
            refs: (0..RING).map(|_| AtomicU32::new(0)).collect(),
            staged: Default::default(),
            tail: Default::default(),
            tiles: (0..ntiles).map(|_| Tile { head: AtomicU64::new(0), tail: AtomicU64::new(0), scheduled: AtomicBool::new(false) }).collect(),
            queues: (0..ntiles * queue as usize).map(|_| UnsafeCell::new(0)).collect(),
            grid,
            tile_log2: tile.trailing_zeros(),
            queue,
            ready: crossbeam_queue::ArrayQueue::new(ntiles),
            shader,
            pool: Waiter::default(),
            exit: AtomicBool::new(false),
            held: Mutex::new(VecDeque::new()),
            held_clears: Mutex::new(VecDeque::new()),
            threads: Mutex::new(Vec::new()),
        });
        let mut threads = r.threads.lock();
        for i in 0..workers.max(1) {
            let c = r.clone();
            let name = format!("{ODYSSEY_POOL_THREAD}-{i}");

            threads.push(std::thread::Builder::new().name(name).spawn(move || c.run_worker()).expect("odyssey pool thread"));
        }
        drop(threads);
        r
    }

    /// Whether workers shade filled triangles.
    pub fn shader(&self) -> bool {
        self.shader
    }

    // The raster thread's side.

    /// Retire the slots every tile has run, in order.
    fn retire(&self) {
        let mut t = self.tail.load(Relaxed);
        let staged = self.staged.load(Relaxed);

        while t < staged && self.refs[t as usize & (RING - 1)].load(Acquire) == 0 {
            t += 1;
        }
        self.tail.store(t, Release);
    }

    /// Wait while `busy`, running ready tiles meanwhile.
    #[cold]
    fn wait(&self, busy: impl Fn(&Self) -> bool) {
        let mut spins = 0u32;
        loop {
            self.retire();
            if !busy(self) {
                return;
            }
            if let Some(t) = self.ready.pop() {
                self.run_tile(t);
                continue;
            }
            spins += 1;
            if spins < SPINS {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// The next slot to write, once retired.
    #[inline]
    fn slot(&self) -> &mut RingQuad {
        let staged = self.staged.load(Relaxed);

        if staged - self.tail.load(Relaxed) >= RING as u64 {
            self.wait(|r| staged - r.tail.load(Relaxed) >= RING as u64);
        }
        // SAFETY: the slot is retired and on no tile's queue: only the
        // raster thread reaches it.
        unsafe { &mut *self.ring[staged as usize & (RING - 1)].get() }
    }

    /// The tile holding plane pixel (x, y).
    #[inline(always)]
    fn tile_of(&self, x: i32, y: i32) -> usize {
        let m = self.grid - 1;
        ((y >> self.tile_log2) as usize & m) * self.grid + ((x >> self.tile_log2) as usize & m)
    }

    /// Commit the slot just written to the tiles under plane rectangle
    /// (x0, y0)-(x1, y1), inclusive.
    fn commit(&self, x0: i32, y0: i32, x1: i32, y1: i32) {
        let seq = self.staged.load(Relaxed);
        let idx = seq as usize & (RING - 1);
        let (tx0, ty0, tx1, ty1) = (x0 >> self.tile_log2, y0 >> self.tile_log2, x1 >> self.tile_log2, y1 >> self.tile_log2);
        let n = if x0 > x1 || y0 > y1 {
            0
        } else {
            // A rectangle wider than the grid wraps onto every column once.
            ((tx1 - tx0 + 1).min(self.grid as i32) * (ty1 - ty0 + 1).min(self.grid as i32)) as u32
        };

        self.refs[idx].store(n, Relaxed);
        self.staged.store(seq + 1, Release);
        if n == 0 {
            return;
        }
        for ty in ty0..=ty1.min(ty0 + self.grid as i32 - 1) {
            for tx in tx0..=tx1.min(tx0 + self.grid as i32 - 1) {
                self.enqueue(self.tile_of(tx << self.tile_log2, ty << self.tile_log2), idx as u32);
            }
        }
    }

    /// Append slot idx to tile t's queue, and put the tile on the ready
    /// queue unless it is there or owned.
    #[inline]
    fn enqueue(&self, t: usize, idx: u32) {
        let tile = &self.tiles[t];
        let head = tile.head.load(Relaxed);

        if head - tile.tail.load(Acquire) >= self.queue {
            self.wait(|r| head - r.tiles[t].tail.load(Acquire) >= r.queue);
        }
        // SAFETY: the entry is past the tile's tail: only the raster thread
        // reaches it until the head moves past it.
        unsafe { *self.queues[t * self.queue as usize + (head & (self.queue - 1)) as usize].get() = idx };
        tile.head.store(head + 1, SeqCst);
        if !tile.scheduled.swap(true, SeqCst) {
            self.ready.push(t as u32).expect("a tile on the ready queue once");
            self.pool.wake(false);
        }
    }

    /// Write a triangle for the tiles to walk: its record as its draw keeps
    /// it, and its bounding box in plane pixels.
    #[inline]
    pub fn push_tri(&self, tri: *const Tri, x0: i32, y0: i32, x1: i32, y1: i32) {
        let s = self.slot();

        *s = RingQuad { x: x0 as i16, y: y0 as i16, kind: KIND_TRI, ptr: tri.cast(), ..RingQuad::default() };
        s.zq[0] = x1 as u32;
        s.zq[1] = y1 as u32;
        self.commit(x0, y0, x1, y1);
    }

    /// Write a quad the raster thread shaded itself.
    #[inline]
    pub fn push_shaded(&self, q: &BackQuad) {
        let s = self.slot();

        s.x = q.x as i16;
        s.y = q.y as i16;
        s.cov = q.cov;
        s.kind = KIND_SHADED;
        s.zq = q.zq;
        for i in 0..4 {
            s.set_colour(i, q.rgba[i]);
        }
        s.ptr = q.draw.cast();
        // The pixels it covers, which may straddle a tile edge.
        let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for i in 0..4 {
            if q.cov & (1 << i) != 0 {
                let (x, y) = (q.x + (i & 1) as i32, q.y + (i >> 1) as i32);
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            }
        }
        self.commit(x0, y0, x1, y1);
    }

    /// Write a clear's rectangle, (x0, y0)-(x1, y1) inclusive in plane
    /// pixels.
    #[inline]
    pub fn push_rect(&self, x0: i32, y0: i32, x1: i32, y1: i32, clear: *const ClearDraw) {
        let s = self.slot();

        *s = RingQuad { x: x0 as i16, y: y0 as i16, kind: KIND_RECT, ptr: clear.cast(), ..RingQuad::default() };
        s.zq[0] = x1 as u32;
        s.zq[1] = y1 as u32;
        self.commit(x0, y0, x1, y1);
    }

    /// Keep a clear's record until the tail is past every slot written so
    /// far.
    pub fn hold_clear(&self, clear: Box<ClearDraw>) {
        self.held_clears.lock().push_back((self.staged.load(Relaxed), clear));
    }

    /// Keep a draw's job and context until the tail is past every slot
    /// written so far.
    pub fn hold(&self, job: Box<GlDrawJob>, frame: Box<DrawFrame>) {
        self.held.lock().push_back((self.staged.load(Relaxed), job, frame));
    }

    /// The jobs of the held draws the tail is past, to give back; their
    /// contexts are freed.
    pub fn reclaim(&self) -> Vec<Box<GlDrawJob>> {
        self.retire();
        let tail = self.tail.load(Acquire);
        let mut h = self.held.lock();
        let mut out = Vec::new();

        while h.front().is_some_and(|(end, _, _)| *end <= tail) {
            let (_, job, _) = h.pop_front().expect("a held draw");

            out.push(job);
        }
        drop(h);
        let mut c = self.held_clears.lock();
        while c.front().is_some_and(|(end, _)| *end <= tail) {
            c.pop_front();
        }
        out
    }

    /// Whether every slot written has run on all its tiles.
    pub fn idle(&self) -> bool {
        self.retire();
        self.tail.load(Acquire) == self.staged.load(Relaxed)
    }

    /// Wait until every slot written has run, so board memory is the raster
    /// thread's again; it runs tiles meanwhile.
    pub fn drain(&self) {
        let target = self.staged.load(Relaxed);
        self.wait(|r| r.tail.load(Acquire) != target);
    }

    /// Stop the pool once everything has run; the held draws go.
    pub fn stop(&self) {
        self.drain();
        self.exit.store(true, SeqCst);
        self.pool.wake(true);
        for t in self.threads.lock().drain(..) {
            let _ = t.join();
        }
        self.held.lock().clear();
        self.held_clears.lock().clear();
    }

    // The pool.

    fn run_worker(&self) {
        let mut spins = 0u32;

        loop {
            if let Some(t) = self.ready.pop() {
                spins = 0;
                self.run_tile(t);
                continue;
            }
            if self.exit.load(SeqCst) {
                return;
            }
            spins += 1;
            if spins < IDLE_SPINS {
                std::hint::spin_loop();
            } else {
                self.pool.sleep(|| !self.ready.is_empty(), &self.exit);
            }
        }
    }

    /// Own tile t: run its commands in order until its queue is empty.
    fn run_tile(&self, t: u32) {
        let tile = &self.tiles[t as usize];
        let size = 1i32 << self.tile_log2;
        // The tile's plane rectangle (its first copy on the grid).
        let (px, py) = ((t as usize % self.grid) as i32 * size, (t as usize / self.grid) as i32 * size);

        loop {
            let head = tile.head.load(Acquire);
            let mut c = tile.tail.load(Relaxed);

            while c < head {
                // SAFETY: the raster thread wrote the entry before moving
                // the head past it.
                let idx = unsafe { *self.queues[t as usize * self.queue as usize + (c & (self.queue - 1)) as usize].get() };
                self.run(idx as usize, px, py, size);
                self.refs[idx as usize].fetch_sub(1, Release);
                c += 1;
                tile.tail.store(c, Release);
            }
            tile.scheduled.store(false, SeqCst);
            if tile.head.load(SeqCst) == c || tile.scheduled.swap(true, SeqCst) {
                return;
            }
        }
    }

    /// Run slot idx's command within the tile at plane (px, py), `size`
    /// pixels square. A rectangle's tiles wrap onto the grid, so the
    /// command's own pixels decide which copy of the tile it means.
    fn run(&self, idx: usize, px: i32, py: i32, size: i32) {
        // SAFETY: on this tile's queue, so written and not yet retired.
        let s = unsafe { &*self.ring[idx].get() };
        match s.kind {
            KIND_SHADED => {
                // SAFETY: the draw outlives its slots (QuadRing::hold).
                let d = unsafe { &*s.ptr.cast::<BackendDraw>() };
                // The quad's pixels in this tile.
                let mut q = BackQuad { draw: d, x: s.x as i32, y: s.y as i32, zq: s.zq, cov: 0, ..BackQuad::default() };
                for i in 0..4usize {
                    let (x, y) = (q.x + (i & 1) as i32, q.y + (i >> 1) as i32);
                    if s.cov & (1 << i) != 0 && (x & !(size - 1) & (GRID_SPAN - 1)) == px && (y & !(size - 1) & (GRID_SPAN - 1)) == py {
                        q.cov |= 1 << i;
                        q.rgba[i] = s.colour(i).map(|c| c as u16);
                    }
                }
                (d.state.store)(&d.state, &d.planes, &q);
            }
            KIND_RECT => {
                // SAFETY: the clear outlives its slots (QuadRing::hold_clear).
                let c = unsafe { &*s.ptr.cast::<ClearDraw>() };
                let (x0, y0, x1, y1) = (s.x as i32, s.y as i32, s.zq[0] as i32, s.zq[1] as i32);
                for (tx, ty) in tile_copies(x0, y0, x1, y1, px, py, size) {
                    for y in y0.max(ty)..=y1.min(ty + size - 1) {
                        c.span(y, x0.max(tx), x1.min(tx + size - 1));
                    }
                }
            }
            _ => {
                // SAFETY: the draw, which keeps the triangle's record,
                // outlives its slots (QuadRing::hold).
                let t = unsafe { &*s.ptr.cast::<Tri>() };
                // SAFETY: as above. A fresh copy of the draw's context each
                // time: a cached one could be a freed draw's whose address
                // a later draw took.
                let d = unsafe { &*t.live }.shader_draw();
                let p = d.pipe;
                let rst = RasterState { offset_steps: p.fill_steps, offset_fill: true, ..p.raster };
                let Some((bx0, by0, bx1, by1)) = d.bounds else {
                    return;
                };
                let (x0, y0, x1, y1) = (s.x as i32, s.y as i32, s.zq[0] as i32, s.zq[1] as i32);
                for (tx, ty) in tile_copies(x0, y0, x1, y1, px, py, size) {
                    // The tile in window pixels, within the draw's bounds.
                    let (wx0, wy0) = (tx.wrapping_sub(rst.ox).max(bx0), ty.wrapping_sub(rst.oy).max(by0));
                    let (wx1, wy1) = ((tx + size - 1).wrapping_sub(rst.ox).min(bx1), (ty + size - 1).wrapping_sub(rst.oy).min(by1));
                    triangle_quads(&rst, &t.tri, Some((wx0, wy0, wx1, wy1)), |q| {
                        let mut out = BackQuad { x: rst.ox.wrapping_add(q.x), y: rst.oy.wrapping_add(q.y), ..BackQuad::default() };
                        d.shade_quad(t, q, &mut out);
                        (p.backend.store)(&p.backend, &p.planes, &out);
                    });
                }
            }
        }
    }
}

/// The copies of the grid tile at (px, py), `size` pixels square, that
/// rectangle (x0, y0)-(x1, y1) overlaps, by their plane corners: one,
/// unless the rectangle is wider or taller than the grid.
fn tile_copies(x0: i32, y0: i32, x1: i32, y1: i32, px: i32, py: i32, size: i32) -> impl Iterator<Item = (i32, i32)> {
    // The first copy whose tile reaches lo, then one a grid span on.
    let axis = move |lo: i32, hi: i32, p: i32| {
        let first = p + GRID_SPAN * (lo - size + 1 - p).div_euclid(GRID_SPAN) + if (lo - size + 1 - p).rem_euclid(GRID_SPAN) != 0 { GRID_SPAN } else { 0 };
        std::iter::successors(Some(first), |&c| Some(c + GRID_SPAN)).take_while(move |&c| c <= hi)
    };
    axis(y0, y1, py).flat_map(move |ty| axis(x0, x1, px).map(move |tx| (tx, ty)))
}
