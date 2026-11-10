//! KONA's tiles: the screen cut into square tiles, one owner each, and the
//! scheduler that gives each tile's owner the raster jobs that touch it.
//!
//! The producer (`kona-hip`, through the raster stage) writes each job into the
//! next slot of a ring and appends the slot's index to the queue of every tile
//! the job's bounding box touches; the slot counts
//! those tiles. A tile with work goes on a ready queue. A worker takes a tile
//! from it and owns the tile until its queue is empty, running each command
//! clipped to the tile's rectangle, in the order the jobs were written.
//! The producer retires slots whose count has reached zero, in order, and
//! writes them again only then.
//!
//! The rules (they are the screen-tile rules VPro's fragment pipeline follows;
//! nothing here is VPro's code, and VPro is untouched):
//!
//! - One owner per tile is the only ordering. A command must go to every tile
//!   it can touch, so its bounding box must contain every pixel it writes: a
//!   box too small lets a pixel be written out of order. The scheduler cannot
//!   check this; the pixel module's tests do.
//! - The pixel module clips to the tile it is given and keeps the window's own
//!   grid (the caller's quad grid, in the raster stage), so a tiled run is
//!   bit-exact with an inline one: `threads == 0` runs each job
//!   immediately over the whole screen, which is the inline run to compare with.
//! - A slot is not written again while any tile queue can name it. A tile's
//!   queue holds a slot's index until a worker takes it, and the slot's count
//!   drops only after the worker has finished running it, so a count of zero
//!   means no queue and no worker has the slot.
//! - The slot is the job ring: it holds a `RasterJob` for good, plain data the
//!   producer fills in place (`submit_with`), so a draw allocates nothing. A job
//!   names the draw state it runs under by an index into the draw-state table
//!   (`state`); an entry is not written again while any job in flight names it.
//! - A work budget bounds what is in flight: the producer waits, retiring as it
//!   goes, while the commands' estimated costs (pixels) exceed it, and resumes
//!   at half.
//! - A panic in a worker poisons the scheduler: the slot still retires, so a
//!   drain finishes, and every later `submit_with` and `drain` says so.
//!
//! Tiles are a plain grid from the screen's origin. The wrapping 4096-pixel
//! grid VPro's multi-window planes use is not needed: the managed area is at
//! most 4096 wide.

use super::raster::pixel;
use super::raster::setup::TriJob;
use super::raster::{ClearReq, DrawState, Rect};
use super::super::handle::DisplayHandle;
use super::store::PixelStore;
use crossbeam_queue::ArrayQueue;
use parking_lot::{Condvar, Mutex};
use std::cell::UnsafeCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::Arc;
use std::thread::JoinHandle;

/// A clear, under draw state `state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClearJob {
    pub req: ClearReq,
    pub state: u16,
}

/// What a slot of the ring holds: plain data with no pointers, naming its draw
/// state by index. `RasterJob::default` is the empty one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RasterJob {
    Tri(TriJob),
    Clear(ClearJob),
}

impl Default for RasterJob {
    fn default() -> Self {
        RasterJob::Clear(ClearJob::default())
    }
}

impl RasterJob {
    /// Where the command can write.
    pub fn rect(&self) -> Rect {
        match self {
            RasterJob::Tri(t) => t.rect,
            RasterJob::Clear(c) => c.req.rect,
        }
    }

    /// The draw-state table entry it runs under.
    pub fn state(&self) -> u16 {
        match self {
            RasterJob::Tri(t) => t.state,
            RasterJob::Clear(c) => c.state,
        }
    }

    /// An estimate of the work in pixels, for the tiles' budget.
    pub fn cost(&self) -> u64 {
        let r = self.rect();
        (r.x1 - r.x0).max(0) as u64 * (r.y1 - r.y0).max(0) as u64
    }
}

/// A tile size or ring size that is not a power of two (or a ring under two slots).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Invalid {
    Tile(u32),
    Ring(usize),
}

/// A panic in a worker poisoned the tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poisoned;

/// What the tiles counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileStats {
    pub submitted: u64,
    pub retired: u64,
    /// Jobs run by a tile owner, summed over tiles.
    pub tile_runs: u64,
    /// The most cost in flight at once, and times the producer waited for
    /// the ring or the budget.
    pub peak_cost: u64,
    pub waits: u64,
    /// Times something that touches the store asked for every job to finish,
    /// and those that found jobs still queued, which is what a drain costs.
    pub drains: u64,
    pub busy_drains: u64,
}

struct Slot {
    job: UnsafeCell<RasterJob>,
    /// Tiles that have yet to finish running the job.
    pending: AtomicU32,
}

// SAFETY: the producer alone writes `job`, and only while `pending` is zero and
// no tile queue holds the slot's index (the slot has retired, or is new);
// workers only read it, and only between taking the index from a queue and
// decrementing `pending`. `pending` is the hand-over: Release by the worker
// after its last read, Acquire by the producer before it writes again.
unsafe impl Sync for Slot {}

/// One entry of the draw-state table: written by the producer while no job in
/// flight names it, read by the workers while one does.
struct StateCell(UnsafeCell<DrawState>);

// SAFETY: as for `Slot`: the producer writes an entry only when its count of
// jobs in flight, which it keeps itself and decrements after the slot's
// Acquire load of `pending`, is zero; workers read it only while a job that
// names it is between a tile queue and its `pending` decrement.
unsafe impl Sync for StateCell {}

/// Entries in the draw-state table.
const STATES: usize = 64;

struct Tile {
    /// The slots to run, in order: a ring the producer appends to at `tail`
    /// and the owner takes from at `head`. It has a place for every slot, and a
    /// slot is named at most once, so it never overflows.
    queue: Box<[AtomicU32]>,
    head: AtomicU32,
    tail: AtomicU32,
    /// On the ready queue, or owned by a worker.
    scheduled: AtomicBool,
    rect: Rect,
}

impl Tile {
    /// Producer: append a slot's index.
    fn push(&self, slot: u32) {
        let t = self.tail.load(Relaxed);
        self.queue[t as usize & (self.queue.len() - 1)].store(slot, Relaxed);
        self.tail.store(t.wrapping_add(1), Release);
    }

    /// Owner: take the next slot's index.
    fn pop(&self) -> Option<u32> {
        let h = self.head.load(Relaxed);
        if h == self.tail.load(Acquire) {
            return None;
        }
        let slot = self.queue[h as usize & (self.queue.len() - 1)].load(Relaxed);
        self.head.store(h.wrapping_add(1), Relaxed);
        Some(slot)
    }

    fn is_empty(&self) -> bool {
        self.head.load(Relaxed) == self.tail.load(Acquire)
    }
}

struct Shared {
    store: Arc<PixelStore>,
    slots: Vec<Slot>,
    states: Vec<StateCell>,
    tiles: Vec<Tile>,
    ready: ArrayQueue<u32>,
    stop: AtomicBool,
    poisoned: AtomicBool,
    sleepers: AtomicUsize,
    gate: Mutex<()>,
    wake: Condvar,
    tile_runs: AtomicU64,
}

impl Shared {
    /// Wake a sleeping worker, if one sleeps.
    fn notify(&self) {
        if self.sleepers.load(SeqCst) > 0 {
            let _g = self.gate.lock();
            self.wake.notify_one();
        }
    }

    fn run_tile(&self, t: &Tile) {
        loop {
            while let Some(s) = t.pop() {
                let slot = &self.slots[s as usize];
                // SAFETY: the slot's index came off a tile queue, so its count is
                // above zero until this tile decrements it below.
                let job = unsafe { &*slot.job.get() };
                // SAFETY: the entry the job names stays as it is until the job retires.
                let ds = unsafe { &*self.states[job.state() as usize].0.get() };
                if catch_unwind(AssertUnwindSafe(|| pixel::run(&self.store, job, ds, t.rect))).is_err() {
                    self.poisoned.store(true, SeqCst);
                }
                self.tile_runs.fetch_add(1, Relaxed);
                slot.pending.fetch_sub(1, Release);
            }
            t.scheduled.store(false, SeqCst);
            // A job pushed after the last pop and before the store above saw
            // the tile scheduled and left it to us.
            if t.is_empty() || t.scheduled.swap(true, SeqCst) {
                return;
            }
        }
    }

    fn worker(&self) {
        let mut idle = 0u32;
        loop {
            if let Some(t) = self.ready.pop() {
                idle = 0;
                self.run_tile(&self.tiles[t as usize]);
                continue;
            }
            if self.stop.load(SeqCst) {
                return;
            }
            idle += 1;
            if idle < 200 {
                std::hint::spin_loop();
                continue;
            }
            self.sleepers.fetch_add(1, SeqCst);
            {
                let mut g = self.gate.lock();
                if self.ready.is_empty() && !self.stop.load(SeqCst) {
                    self.wake.wait(&mut g);
                }
            }
            self.sleepers.fetch_sub(1, SeqCst);
            idle = 0;
        }
    }
}

pub struct Tiles {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    screen: Rect,
    tile: i32,
    cols: i32,
    mask: u64,
    /// Producer-only: slots written, slots retired, each slot's cost and
    /// draw-state entry, the cost in flight, the budget and whether the
    /// producer is waiting for half of it.
    head: u64,
    tail: u64,
    costs: Vec<u64>,
    slot_state: Vec<u16>,
    /// Jobs in flight naming each draw-state entry, and one more for the entry
    /// the producer last set (`held`), which the next `state` call can reuse.
    refs: Vec<u32>,
    held: Option<u16>,
    inflight: u64,
    budget: u64,
    throttled: bool,
    /// With no workers, the job to run at once.
    scratch: RasterJob,
    stats: TileStats,
    /// Told the producer is working while it waits, so the driver's watchdog
    /// does not take a long drain for a dead pipe.
    handle: Option<DisplayHandle>,
}

impl Tiles {
    /// Tiles for a `store`'s screen cut into `tile`-pixel squares (a power of
    /// two), `threads` workers (0: inline), `ring` slots (a power of two) and a
    /// budget in pixels of cost.
    pub fn new(store: Arc<PixelStore>, tile: u32, threads: usize, ring: usize, budget: u64, handle: Option<DisplayHandle>) -> Result<Tiles, Invalid> {
        if !tile.is_power_of_two() {
            return Err(Invalid::Tile(tile));
        }
        if !ring.is_power_of_two() || ring < 2 {
            return Err(Invalid::Ring(ring));
        }
        let (w, h) = (store.width() as i32, store.height() as i32);
        let t = tile as i32;
        let (cols, rows) = ((w + t - 1) / t, (h + t - 1) / t);
        let tiles: Vec<Tile> = (0..rows * cols)
            .map(|i| {
                let (x0, y0) = ((i % cols) * t, (i / cols) * t);
                Tile {
                    queue: (0..ring).map(|_| AtomicU32::new(0)).collect(),
                    head: AtomicU32::new(0),
                    tail: AtomicU32::new(0),
                    scheduled: AtomicBool::new(false),
                    rect: Rect { x0, y0, x1: (x0 + t).min(w), y1: (y0 + t).min(h) },
                }
            })
            .collect();
        let ready = ArrayQueue::new(tiles.len().max(1));
        let slots = (0..ring).map(|_| Slot { job: UnsafeCell::new(RasterJob::default()), pending: AtomicU32::new(0) }).collect();
        let states = (0..STATES).map(|_| StateCell(UnsafeCell::new(DrawState::default()))).collect();
        let shared = Arc::new(Shared { store, slots, states, tiles, ready, stop: AtomicBool::new(false), poisoned: AtomicBool::new(false), sleepers: AtomicUsize::new(0), gate: Mutex::new(()), wake: Condvar::new(), tile_runs: AtomicU64::new(0) });
        let workers = (0..threads)
            .map(|n| {
                let s = shared.clone();
                std::thread::Builder::new().name(format!("kona-tile-{n}")).spawn(move || s.worker()).expect("a thread")
            })
            .collect();
        Ok(Tiles { shared, workers, screen: Rect { x0: 0, y0: 0, x1: w, y1: h }, tile: t, cols, mask: ring as u64 - 1, head: 0, tail: 0, costs: vec![0; ring], slot_state: vec![0; ring], refs: vec![0; STATES], held: None, inflight: 0, budget, throttled: false, scratch: RasterJob::default(), stats: TileStats::default(), handle })
    }

    pub fn stats(&self) -> TileStats {
        TileStats { retired: self.tail, tile_runs: self.shared.tile_runs.load(Relaxed), ..self.stats }
    }

    /// Workers asleep on the wake condition, for the tests of that path.
    #[cfg(test)]
    fn sleeping(&self) -> usize {
        self.shared.sleepers.load(SeqCst)
    }

    pub fn is_poisoned(&self) -> bool {
        self.shared.poisoned.load(SeqCst)
    }

    /// The producer is waiting on the workers: the driver's watchdog is told.
    fn working(&self) {
        if let Some(h) = &self.handle {
            h.activity();
        }
    }

    /// Retire the slots whose tiles are all done, in order.
    fn retire(&mut self) {
        while self.tail < self.head {
            let i = (self.tail & self.mask) as usize;
            if self.shared.slots[i].pending.load(Acquire) != 0 {
                break;
            }
            self.inflight -= self.costs[i];
            self.refs[self.slot_state[i] as usize] -= 1;
            self.tail += 1;
        }
    }

    /// Enter `ds` in the draw-state table and return its index, for the jobs
    /// that follow to name. The entry last returned is reused when it is the
    /// same state. Waits, retiring as it goes, while no entry is free.
    pub fn state(&mut self, ds: DrawState) -> u16 {
        if let Some(h) = self.held {
            // SAFETY: the producer set the entry and nobody writes it but the producer.
            if unsafe { *self.shared.states[h as usize].0.get() } == ds {
                return h;
            }
            self.refs[h as usize] -= 1;
            self.held = None;
        }
        let i = loop {
            self.retire();
            match self.refs.iter().position(|&r| r == 0) {
                Some(i) => break i,
                None => {
                    self.stats.waits += 1;
                    self.working();
                    std::thread::yield_now();
                }
            }
        };
        // SAFETY: no job in flight names the entry, so no worker reads it.
        unsafe { *self.shared.states[i].0.get() = ds };
        self.refs[i] = 1;
        self.held = Some(i as u16);
        i as u16
    }

    /// Hand `job` to every tile its rectangle touches. Waits, retiring as it
    /// goes, while the ring is full or the budget is spent. The draw state it
    /// names must have come from `state` and not been replaced since.
    pub fn submit(&mut self, job: RasterJob) -> Result<(), Poisoned> {
        if self.is_poisoned() {
            return Err(Poisoned);
        }
        self.stats.submitted += 1;
        if self.workers.is_empty() {
            self.scratch = job;
            let r = job.rect().intersect(&self.screen);
            if !r.is_empty() {
                // SAFETY: the entry is the producer's and nothing else reads it.
                let ds = unsafe { &*self.shared.states[job.state() as usize].0.get() };
                pixel::run(&self.shared.store, &self.scratch, ds, self.screen);
            }
            self.tail = self.stats.submitted;
            return Ok(());
        }
        self.retire();
        let mut waited = false;
        while self.head - self.tail > self.mask || (self.inflight > 0 && self.over_budget()) {
            waited = true;
            self.working();
            std::thread::yield_now();
            self.retire();
        }
        self.stats.waits += waited as u64;
        let i = (self.head & self.mask) as usize;
        let slot = &self.shared.slots[i];
        // SAFETY: the slot has retired (or never ran): its count is zero, so no
        // tile queue names it and no worker reads it.
        unsafe { *slot.job.get() = job };
        self.refs[job.state() as usize] += 1;
        self.slot_state[i] = job.state();
        let (bounds, cost) = (job.rect().intersect(&self.screen), job.cost());
        self.head += 1;
        self.costs[i] = cost;
        self.inflight += cost;
        self.stats.peak_cost = self.stats.peak_cost.max(self.inflight);
        if bounds.is_empty() {
            slot.pending.store(0, Relaxed);
            self.retire();
            return Ok(());
        }
        let (tx0, tx1) = (bounds.x0 / self.tile, (bounds.x1 - 1) / self.tile);
        let (ty0, ty1) = (bounds.y0 / self.tile, (bounds.y1 - 1) / self.tile);
        slot.pending.store(((tx1 - tx0 + 1) * (ty1 - ty0 + 1)) as u32, Relaxed);
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                let id = (ty * self.cols + tx) as u32;
                let t = &self.shared.tiles[id as usize];
                t.push(i as u32);
                if !t.scheduled.swap(true, SeqCst) {
                    // The ready queue holds each tile at most once and has room for
                    // every tile, so this cannot fail.
                    assert!(self.shared.ready.push(id).is_ok(), "the ready queue is full");
                    self.shared.notify();
                }
            }
        }
        Ok(())
    }

    /// Over the budget; once over, stay waiting until half of it is free.
    fn over_budget(&mut self) -> bool {
        if self.inflight > self.budget {
            self.throttled = true;
        } else if self.inflight <= self.budget / 2 {
            self.throttled = false;
        }
        self.throttled
    }

    /// Jobs submitted and not yet retired.
    pub fn queued(&mut self) -> u64 {
        self.retire();
        // Inline, nothing is ever queued: the tail follows the submissions and the head does not.
        self.head.saturating_sub(self.tail)
    }

    /// Wait until every job submitted has been run and retired.
    pub fn drain(&mut self) -> Result<(), Poisoned> {
        self.stats.drains += 1;
        self.stats.busy_drains += (self.queued() > 0) as u64;
        while self.tail < self.head {
            self.working();
            self.retire();
            std::thread::yield_now();
        }
        if self.is_poisoned() {
            Err(Poisoned)
        } else {
            Ok(())
        }
    }
}

impl Drop for Tiles {
    fn drop(&mut self) {
        let _ = self.drain();
        self.shared.stop.store(true, SeqCst);
        {
            let _g = self.shared.gate.lock();
            self.shared.wake.notify_all();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 100;
    const H: u32 = 70;

    /// The draw state of a draw into word 1 over the whole screen.
    fn draw_state() -> DrawState {
        DrawState { clip: Rect { x0: 0, y0: 0, x1: W as i32, y1: H as i32 }, colour: [1, 0], colours: 1, ..DrawState::default() }
    }

    /// Tiles with the draw state of the above entered.
    fn tiles(store: Arc<PixelStore>, tile: u32, threads: usize, ring: usize, budget: u64) -> (Tiles, u16) {
        let mut t = Tiles::new(store, tile, threads, ring, budget, None).unwrap();
        let s = t.state(draw_state());
        (t, s)
    }

    /// A clear of `r` to `colour`, as a job under draw state `state`.
    fn clear(r: Rect, colour: u32, state: u16) -> RasterJob {
        RasterJob::Clear(ClearJob { req: ClearReq { rect: r, colour: Some(colour), ..Default::default() }, state })
    }

    fn store() -> Arc<PixelStore> {
        let s = Arc::new(PixelStore::new(W, H));
        s.ensure(1);
        s
    }

    /// The inline hash of `scene(400)`: a regression net that moves only if the
    /// scene or the clear does.
    const INLINE_400: u64 = 0xe30a_20fc_d4c9_4467;

    /// An order-sensitive scene of overlapping clears.
    fn scene(n: usize) -> Vec<(Rect, u32)> {
        let mut seed = 0x1234_5678u64;
        let mut next = move |m: i32| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) % m as u64) as i32
        };
        (0..n)
            .map(|k| {
                let (x0, y0) = (next(110) - 10, next(80) - 10);
                (Rect { x0, y0, x1: x0 + next(60) + 1, y1: y0 + next(40) + 1 }, 0xff00_0000 | k as u32)
            })
            .collect()
    }

    fn run(scene: &[(Rect, u32)], threads: usize, tile: u32, ring: usize, budget: u64) -> (u64, TileStats) {
        let store = store();
        let (mut t, s) = tiles(store.clone(), tile, threads, ring, budget);
        for (r, c) in scene {
            t.submit(clear(*r, *c, s)).unwrap();
        }
        t.drain().unwrap();
        let stats = t.stats();
        drop(t);
        (store.checksum(), stats)
    }

    /// A tiled run is bit-exact with the inline one at every worker count, tile
    /// size and ring size, and the scene is order sensitive, so a lost ordering shows.
    #[test]
    fn a_tiled_run_is_bit_exact_with_the_inline_one() {
        let s = scene(400);
        let (inline, st) = run(&s, 0, 32, 64, 1 << 20);
        assert_eq!(st.submitted, 400);
        assert_eq!(inline, INLINE_400, "the pinned inline hash");
        for threads in [1, 2, 4] {
            for tile in [8, 16, 32, 64] {
                for ring in [4, 64] {
                    let (h, st) = run(&s, threads, tile, ring, 1 << 14);
                    assert_eq!(h, inline, "{threads} workers, tile {tile}, ring {ring}");
                    assert_eq!(st.retired, 400);
                }
            }
        }
        let mut r = s.clone();
        r.reverse();
        assert_ne!(run(&r, 0, 32, 64, 1 << 20).0, inline, "the scene depends on its order");
    }

    /// A job reaches every tile its rectangle touches and no pixel outside it
    /// (cut at the screen's edge) is written.
    #[test]
    fn a_job_reaches_every_tile_it_touches() {
        let boxes = [Rect { x0: 0, y0: 0, x1: 1, y1: 1 }, Rect { x0: 15, y0: 15, x1: 17, y1: 17 }, Rect { x0: 10, y0: 60, x1: 100, y1: 70 }, Rect { x0: -50, y0: -50, x1: 500, y1: 500 }, Rect { x0: 32, y0: 0, x1: 48, y1: 16 }, Rect { x0: 200, y0: 0, x1: 300, y1: 10 }];
        for b in boxes {
            let store = store();
            let (mut t, s) = tiles(store.clone(), 16, 3, 8, 1 << 20);
            t.submit(clear(b, 0xff11_2233, s)).unwrap();
            t.drain().unwrap();
            for y in 0..H as i32 {
                for x in 0..W as i32 {
                    let inside = x >= b.x0 && x < b.x1 && y >= b.y0 && y < b.y1;
                    assert_eq!(store.read(x as u32, y as u32, 1) != 0, inside, "{b:?} at ({x}, {y})");
                }
            }
        }
    }

    /// Jobs on one tile run in submission order and a slot is not written again under
    /// a worker: a two-slot ring and a thousand overlapping jobs leave the last one's colour.
    #[test]
    fn tiles_run_in_order_and_a_slot_is_not_recycled_under_a_worker() {
        let store = store();
        let (mut t, s) = tiles(store.clone(), 16, 2, 2, 1 << 20);
        let all = Rect { x0: 0, y0: 0, x1: W as i32, y1: H as i32 };
        for k in 0..1000u32 {
            t.submit(clear(all, 0xff00_0000 | k, s)).unwrap();
        }
        t.drain().unwrap();
        assert_eq!(t.stats().retired, 1000);
        assert!((0..H).all(|y| (0..W).all(|x| store.read(x, y, 1) == 0xff00_0000 | 999)));
    }

    /// The cost in flight stays near the budget: the producer waits for work to retire.
    #[test]
    fn the_budget_bounds_what_is_in_flight() {
        let store = store();
        let (mut t, s) = tiles(store, 16, 2, 64, 3000);
        let all = Rect { x0: 0, y0: 0, x1: W as i32, y1: H as i32 };
        for k in 0..200u32 {
            t.submit(clear(all, k, s)).unwrap();
        }
        t.drain().unwrap();
        let st = t.stats();
        assert!(st.peak_cost <= 3000 + (W * H) as u64, "{st:?}");
        assert!(st.waits > 0, "{st:?}");
    }

    /// A job wholly off the screen retires at once, with no tile named.
    #[test]
    fn a_job_off_the_screen_retires_at_once() {
        let store = store();
        let (mut t, s) = tiles(store, 16, 2, 4, 1 << 20);
        t.submit(clear(Rect { x0: 500, y0: 500, x1: 600, y1: 600 }, 1, s)).unwrap();
        assert_eq!((t.stats().tile_runs, t.queued()), (0, 0));
    }

    /// A panic in a worker poisons the tiles: the slot still retires, a drain ends and
    /// later submissions say so. (A job that names a colour word set-up never allocated
    /// panics in the pixel module.)
    #[test]
    fn a_panicking_job_poisons_the_tiles() {
        let store = Arc::new(PixelStore::new(W, H));
        let (mut t, s) = tiles(store, 16, 2, 4, 1 << 20);
        t.submit(clear(Rect { x0: 0, y0: 0, x1: 10, y1: 10 }, 1, s)).unwrap();
        assert_eq!(t.drain(), Err(Poisoned));
        assert!(t.is_poisoned());
        assert_eq!(t.submit(RasterJob::default()), Err(Poisoned));
    }

    /// Dropping runs what is queued and joins the workers.
    #[test]
    fn dropping_runs_what_is_queued_and_joins() {
        let store = store();
        let (mut t, s) = tiles(store.clone(), 16, 2, 16, 1 << 20);
        let all = Rect { x0: 0, y0: 0, x1: W as i32, y1: H as i32 };
        for k in 0..10u32 {
            t.submit(clear(all, 0xff00_0000 | k, s)).unwrap();
        }
        drop(t);
        assert_eq!(store.read(5, 5, 1), 0xff00_0009);
    }

    /// A pool that has gone to sleep is woken by a job.
    #[test]
    fn a_sleeping_pool_is_woken_by_a_job() {
        let store = store();
        let (mut t, s) = tiles(store.clone(), 16, 2, 4, 1 << 20);
        let end = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while t.sleeping() < 2 && std::time::Instant::now() < end {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(t.sleeping(), 2, "both workers went to sleep");
        t.submit(clear(Rect { x0: 0, y0: 0, x1: 20, y1: 20 }, 0xff12_3456, s)).unwrap();
        t.drain().unwrap();
        assert_eq!(store.read(3, 3, 1), 0xff12_3456);
    }

    /// A retired slot lets go of the draw state its job named, and the same
    /// state entered again is the same entry.
    #[test]
    fn a_retired_slot_lets_go_of_its_draw_state() {
        let store = store();
        for threads in [0, 2] {
            let (mut t, s) = tiles(store.clone(), 16, threads, 4, 1 << 20);
            assert_eq!(t.state(draw_state()), s, "{threads} workers: unchanged state, same entry");
            t.submit(clear(Rect { x0: 0, y0: 0, x1: 20, y1: 20 }, 1, s)).unwrap();
            let other = t.state(DrawState { colours: 0, ..draw_state() });
            t.drain().unwrap();
            // Only the entry set last is held: the job's hold went when it retired.
            assert_eq!((t.refs.iter().sum::<u32>(), t.refs[other as usize]), (1, 1), "{threads} workers");
        }
    }

    /// A draw-state table far smaller than the states used makes the producer
    /// wait for entries, and the result is the inline one: a state is not
    /// overwritten while a job in flight names it.
    #[test]
    fn more_states_than_entries_wait_for_free_ones() {
        let run = |threads: usize| {
            let store = store();
            (2..=3).for_each(|w| store.ensure(w));
            let mut t = Tiles::new(store.clone(), 8, threads, 256, 1 << 20, None).unwrap();
            for k in 0..(4 * STATES as i32) {
                // Each state names one of three colour words, and no two in a row are alike.
                let s = t.state(DrawState { colour: [1 + (k % 3) as u32, k as u32 + 100], ..draw_state() });
                t.submit(clear(Rect { x0: 0, y0: 0, x1: W as i32, y1: H as i32 }, 0xff00_0000 | k as u32, s)).unwrap();
            }
            t.drain().unwrap();
            drop(t);
            store.checksum()
        };
        assert_eq!(run(0), run(3));
    }

    /// A producer that waits on the workers tells the driver's watchdog the pipe
    /// is working: for a full ring, a spent budget, a full state table and a drain.
    #[test]
    fn a_producer_that_waits_feeds_the_watchdog() {
        use std::sync::atomic::Ordering::Relaxed;
        let f = crate::dev::kona::tests::Fixture::new();
        let host = f.kona.shared.host.clone();
        let handle = DisplayHandle::new(&host);
        let all = Rect { x0: 0, y0: 0, x1: W as i32, y1: H as i32 };
        // A two-slot ring: the third job has to wait for a worker.
        let mut t = Tiles::new(store(), 16, 1, 2, 1 << 20, Some(handle.clone())).unwrap();
        let s = t.state(draw_state());
        host.activity.store(false, Relaxed);
        for k in 0..64u32 {
            t.submit(clear(all, k, s)).unwrap();
        }
        t.drain().unwrap();
        assert!(t.stats().waits > 0 && host.activity.load(Relaxed), "{:?}", t.stats());
        // A budget below one job's cost makes the next submission wait.
        host.activity.store(false, Relaxed);
        let mut t = Tiles::new(store(), 16, 1, 64, 10, Some(handle)).unwrap();
        let s = t.state(draw_state());
        for k in 0..8u32 {
            t.submit(clear(all, k, s)).unwrap();
        }
        t.drain().unwrap();
        assert!(host.activity.load(Relaxed));
    }

    #[test]
    fn a_bad_tile_or_ring_size_is_refused() {
        let store = store();
        assert_eq!(Tiles::new(store.clone(), 24, 1, 4, 1, None).err(), Some(Invalid::Tile(24)));
        assert_eq!(Tiles::new(store.clone(), 16, 1, 3, 1, None).err(), Some(Invalid::Ring(3)));
        assert_eq!(Tiles::new(store, 16, 1, 1, 1, None).err(), Some(Invalid::Ring(1)));
    }
}
