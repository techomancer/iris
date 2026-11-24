//! The compile pool's ring of code arenas: which one the workers compile
//! into, which still hold code that runs, which are empty, and how the
//! oldest is taken away without stopping a CPU.
//!
//! One arena is **active**. When it fills, a worker switches the pool to a
//! **spare** one ([`ArenaRing::switch`]); the arena it left stays **live**,
//! and its code keeps running. A switch that uses the last spare also starts
//! retiring the oldest live arena. The worker that switched then:
//!
//! 1. refuses further publishes from that arena and sweeps its code off every
//!    page;
//! 2. asks every CPU to pass a batch boundary, and says so
//!    ([`ArenaRing::kicked`]);
//!
//! and any worker, later, once every CPU has answered and every worker has
//! followed the switch, empties the arena in place and makes it a spare
//! ([`ArenaRing::try_reclaim`]).
//!
//! No CPU parks and no worker parks. If the active arena fills while there is
//! no spare, the workers stop compiling until the reclaim: `switch` says
//! [`Switch::NoSpare`].
//!
//! The ring only keeps the books. What an arena is, how a sweep, a kick and a
//! recycle are done, are the caller's, so the order of those steps can be
//! modelled with tokens for arenas (`loom_tests`):
//!
//! `RUSTFLAGS="--cfg loom" cargo test --release --features jitv2 --lib
//! arena_ring::loom_tests`.

#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use std::collections::VecDeque;

use crate::sync::Mutex;

struct State {
    /// Index of the arena the workers compile into.
    active: usize,
    /// Arenas left behind by a switch whose code still runs, oldest first.
    live: VecDeque<usize>,
    /// Empty arenas, ready to become active.
    spare: Vec<usize>,
    /// The arena being taken away, if any.
    retiring: Option<usize>,
    /// Its code is off every page and every CPU has been asked to pass a
    /// batch boundary. Until then a CPU's "answer" means nothing.
    kicked: bool,
    /// How many switches there have been. A worker that has seen this many
    /// compiles into `active`.
    switches: u64,
    /// The switch that began the current retirement: every worker must have
    /// followed it before the arena is emptied.
    retire_switch: u64,
}

/// What [`ArenaRing::switch`] did.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Switch<A> {
    /// The pool moved to a spare arena. With `Some`, the caller must retire
    /// that arena: stop publishes from it, sweep its code off the pages, ask
    /// the CPUs to pass a batch boundary, then call [`ArenaRing::kicked`].
    Switched { retire: Option<A> },
    /// Another worker already switched: follow it.
    Behind,
    /// No arena is empty yet. Compile nothing until a reclaim.
    NoSpare,
}

/// A snapshot of the ring for the monitor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingStatus {
    pub active: usize,
    pub live: Vec<usize>,
    pub spare: Vec<usize>,
    pub retiring: Option<usize>,
    pub switches: u64,
}

pub(crate) struct ArenaRing<A> {
    arenas: Vec<A>,
    state: Mutex<State>,
    /// `State::switches`, for a worker's check at the top of its loop.
    switches: AtomicU64,
    /// `State::retiring.is_some()`, likewise.
    retiring: AtomicBool,
    /// The switch each worker has followed.
    workers: Vec<AtomicU64>,
}

impl<A: Clone> ArenaRing<A> {
    /// A ring of `arenas`, the first one active and the rest spare, for
    /// `workers` workers.
    pub(crate) fn new(arenas: Vec<A>, workers: usize) -> Self {
        assert!(arenas.len() >= 2, "a ring needs an arena to switch to");
        let spare = (1..arenas.len()).rev().collect();
        Self {
            arenas,
            state: Mutex::new(State {
                active: 0,
                live: VecDeque::new(),
                spare,
                retiring: None,
                kicked: false,
                switches: 0,
                retire_switch: 0,
            }),
            switches: AtomicU64::new(0),
            retiring: AtomicBool::new(false),
            workers: (0..workers).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    pub(crate) fn arenas(&self) -> &[A] {
        &self.arenas
    }

    /// The arena to compile into, and the switch count it goes with.
    pub(crate) fn active(&self) -> (A, u64) {
        let state = self.state.lock();
        (self.arenas[state.active].clone(), state.switches)
    }

    /// How many switches there have been. A worker compares it with the count
    /// it last followed.
    #[inline]
    pub(crate) fn switches(&self) -> u64 {
        self.switches.load(Ordering::Acquire)
    }

    /// An arena is waiting to be reclaimed.
    #[inline]
    pub(crate) fn retire_pending(&self) -> bool {
        self.retiring.load(Ordering::Acquire)
    }

    /// Worker `worker` now compiles into the arena of switch `switch`, and
    /// holds nothing of an older one.
    pub(crate) fn reached(&self, worker: usize, switch: u64) {
        if let Some(w) = self.workers.get(worker) {
            w.store(switch, Ordering::Release);
        }
    }

    /// The active arena is full, as seen by a worker that follows switch
    /// `from`: move the pool to a spare one.
    pub(crate) fn switch(&self, from: u64) -> Switch<A> {
        let mut state = self.state.lock();
        if state.switches != from {
            return Switch::Behind;
        }
        let Some(next) = state.spare.pop() else {
            return Switch::NoSpare;
        };
        let left = std::mem::replace(&mut state.active, next);
        state.live.push_back(left);
        state.switches += 1;
        self.switches.store(state.switches, Ordering::Release);
        let mut retire = None;
        if state.spare.is_empty() && state.retiring.is_none() {
            let oldest = state.live.pop_front().expect("the arena just left is live");
            state.retiring = Some(oldest);
            state.kicked = false;
            state.retire_switch = state.switches;
            self.retiring.store(true, Ordering::Release);
            retire = Some(self.arenas[oldest].clone());
        }
        Switch::Switched { retire }
    }

    /// The retiring arena's code is off every page and every CPU has been
    /// asked to pass a batch boundary.
    pub(crate) fn kicked(&self) {
        self.state.lock().kicked = true;
    }

    /// Empty the retiring arena and make it a spare, if it can go: its sweep
    /// and kick are done, `cpus_out` says every CPU has answered, and every
    /// worker has followed the switch that began the retirement. `recycle`
    /// empties it. Returns whether it went.
    ///
    /// Workers call this at the top of their loop while
    /// [`Self::retire_pending`]. The ring is locked for the whole of
    /// `recycle`, so one worker does it and a switch waits for it.
    pub(crate) fn try_reclaim(&self, cpus_out: impl FnOnce() -> bool, recycle: impl FnOnce(&A)) -> bool {
        let mut state = self.state.lock();
        let Some(idx) = state.retiring else {
            return false;
        };
        if !state.kicked {
            return false;
        }
        let need = state.retire_switch;
        if self.workers.iter().any(|w| w.load(Ordering::Acquire) < need) {
            return false;
        }
        if !cpus_out() {
            return false;
        }
        recycle(&self.arenas[idx]);
        state.retiring = None;
        state.kicked = false;
        state.spare.push(idx);
        self.retiring.store(false, Ordering::Release);
        true
    }

    /// Start over with every arena empty: the first active, the rest spare.
    /// `recycle` empties each. For a flush of the whole page pool, with no
    /// worker running and every CPU outside compiled code.
    pub(crate) fn reset(&self, mut recycle: impl FnMut(&A)) {
        let mut state = self.state.lock();
        for arena in &self.arenas {
            recycle(arena);
        }
        state.active = 0;
        state.live.clear();
        state.spare = (1..self.arenas.len()).rev().collect();
        state.retiring = None;
        state.kicked = false;
        state.switches += 1;
        state.retire_switch = 0;
        self.switches.store(state.switches, Ordering::Release);
        self.retiring.store(false, Ordering::Release);
        for w in &self.workers {
            w.store(state.switches, Ordering::Release);
        }
    }

    /// Every worker starts on the active arena: for a pool start.
    pub(crate) fn workers_start(&self) {
        let state = self.state.lock();
        for w in &self.workers {
            w.store(state.switches, Ordering::Release);
        }
    }

    pub(crate) fn status(&self) -> RingStatus {
        let state = self.state.lock();
        RingStatus {
            active: state.active,
            live: state.live.iter().copied().collect(),
            spare: state.spare.clone(),
            retiring: state.retiring,
            switches: state.switches,
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    fn ring(n: usize) -> ArenaRing<usize> {
        ArenaRing::new((0..n).collect(), 1)
    }

    /// Follow the ring's switches as the one worker.
    fn follow(r: &ArenaRing<usize>) -> u64 {
        let (_, s) = r.active();
        r.reached(0, s);
        s
    }

    #[test]
    fn the_first_switch_of_three_retires_nothing() {
        let r = ring(3);
        assert_eq!(r.active(), (0, 0));
        assert_eq!(r.switch(0), Switch::Switched { retire: None });
        assert_eq!(r.active(), (1, 1));
        assert_eq!(r.status().live, vec![0], "the arena left behind still runs");
        assert!(!r.retire_pending());
    }

    #[test]
    fn using_the_last_spare_retires_the_oldest_and_keeps_the_previous() {
        let r = ring(3);
        r.switch(0);
        assert_eq!(r.switch(1), Switch::Switched { retire: Some(0) });
        let status = r.status();
        assert_eq!((status.active, status.live, status.retiring), (2, vec![1], Some(0)));
        assert!(r.retire_pending());
    }

    #[test]
    fn a_ring_of_two_retires_the_arena_it_just_left() {
        let r = ring(2);
        assert_eq!(r.switch(0), Switch::Switched { retire: Some(0) });
        assert_eq!(r.status().live, Vec::<usize>::new());
    }

    #[test]
    fn a_worker_that_is_behind_follows_instead_of_switching_again() {
        let r = ring(3);
        r.switch(0);
        assert_eq!(r.switch(0), Switch::Behind);
        assert_eq!(r.active(), (1, 1));
    }

    #[test]
    fn nothing_compiles_while_no_arena_is_empty() {
        let r = ring(3);
        r.switch(0);
        r.switch(1);
        assert_eq!(r.switch(2), Switch::NoSpare);
        assert_eq!(r.active(), (2, 2), "a refused switch moves nothing");
    }

    #[test]
    fn an_arena_is_reclaimed_only_after_the_kick_the_workers_and_the_cpus() {
        let r = ring(3);
        r.switch(0);
        r.switch(1);
        let mut recycled = Vec::new();
        // Not kicked: the CPUs have not even been asked.
        assert!(!r.try_reclaim(|| true, |a| recycled.push(*a)));
        r.kicked();
        // The worker still holds a `Codegen` on an older arena.
        assert!(!r.try_reclaim(|| true, |a| recycled.push(*a)));
        follow(&r);
        // A CPU has not passed a batch boundary.
        assert!(!r.try_reclaim(|| false, |a| recycled.push(*a)));
        assert!(recycled.is_empty());
        assert!(r.try_reclaim(|| true, |a| recycled.push(*a)));
        assert_eq!(recycled, vec![0]);
        let status = r.status();
        assert_eq!((status.spare, status.retiring), (vec![0], None));
        assert!(!r.retire_pending());
        // And the ring goes round: 2 fills, 0 takes over, 1 retires.
        assert_eq!(r.switch(2), Switch::Switched { retire: Some(1) });
        assert_eq!(r.active(), (0, 3));
    }

    #[test]
    fn a_reset_empties_every_arena_and_moves_every_worker() {
        let r = ring(3);
        r.switch(0);
        r.switch(1);
        let mut recycled = Vec::new();
        r.reset(|a| recycled.push(*a));
        assert_eq!(recycled, vec![0, 1, 2]);
        let status = r.status();
        assert_eq!((status.active, status.live, status.spare, status.retiring), (0, vec![], vec![2, 1], None));
        assert_eq!(r.switches(), 3, "a reset is a switch: workers rebuild");
        assert_eq!(r.switch(3), Switch::Switched { retire: None });
    }
}

/// A worker that fills arenas, a second worker, and a CPU that runs whatever
/// the one page points at, in every interleaving loom finds. The CPU must
/// never run code from an arena after it was emptied.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use loom::sync::atomic::AtomicUsize;
    use loom::sync::Arc;
    use loom::thread;

    const NO_CODE: usize = usize::MAX;

    struct Pool {
        ring: ArenaRing<usize>,
        /// Per arena: how many times it has been emptied. Code is "arena,
        /// filling"; running code whose filling is gone is the bug.
        fillings: [AtomicUsize; 3],
        /// Per arena: publishes are refused.
        retired: [AtomicBool; 3],
        /// The one page: its code, as `arena * 16 + filling`, under its
        /// publish lock.
        page: Mutex<usize>,
        /// The CPU has been asked to pass a batch boundary.
        request: AtomicBool,
        /// The CPU thread has ended, which counts as an answer.
        cpu_done: AtomicBool,
    }

    impl Pool {
        fn publish(&self, arena: usize) {
            let mut page = self.page.lock();
            if !self.retired[arena].load(Ordering::SeqCst) {
                *page = arena * 16 + self.fillings[arena].load(Ordering::SeqCst);
            }
        }

        fn retire(&self, arena: usize) {
            self.retired[arena].store(true, Ordering::SeqCst);
            {
                let mut page = self.page.lock();
                if *page != NO_CODE && *page / 16 == arena {
                    *page = NO_CODE;
                }
            }
            self.request.store(true, Ordering::SeqCst);
            self.ring.kicked();
        }

        fn reclaim(&self) {
            self.ring.try_reclaim(
                || !self.request.load(Ordering::SeqCst) || self.cpu_done.load(Ordering::SeqCst),
                |&arena| {
                    self.fillings[arena].fetch_add(1, Ordering::SeqCst);
                    self.retired[arena].store(false, Ordering::SeqCst);
                },
            );
        }

        /// One worker: follow the ring, publish into the arena it is on, and
        /// switch `fills` times.
        fn worker(&self, id: usize, fills: usize) {
            let (mut arena, mut at) = self.ring.active();
            self.ring.reached(id, at);
            for _ in 0..fills {
                self.publish(arena);
                if let Switch::Switched { retire: Some(old) } = self.ring.switch(at) {
                    self.retire(old);
                }
                (arena, at) = self.ring.active();
                self.ring.reached(id, at);
                if self.ring.retire_pending() {
                    self.reclaim();
                }
            }
            self.publish(arena);
        }

        /// The CPU: load the page's code, run it, and answer a request at
        /// the boundary between runs.
        fn cpu(&self, runs: usize) {
            for _ in 0..runs {
                let code = *self.page.lock();
                if code != NO_CODE {
                    let (arena, filling) = (code / 16, code % 16);
                    assert_eq!(
                        self.fillings[arena].load(Ordering::SeqCst), filling,
                        "the CPU ran code from an arena that was emptied under it"
                    );
                }
                self.request.store(false, Ordering::SeqCst);
            }
            self.cpu_done.store(true, Ordering::SeqCst);
        }
    }

    fn model(arenas: usize, fills: usize) {
        let mut builder = loom::model::Builder::new();
        builder.preemption_bound = Some(3);
        builder.check(move || {
            let p = Arc::new(Pool {
                ring: ArenaRing::new((0..arenas).collect(), 2),
                fillings: [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)],
                retired: [AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false)],
                page: Mutex::new(NO_CODE),
                request: AtomicBool::new(false),
                cpu_done: AtomicBool::new(false),
            });
            let a = { let p = p.clone(); thread::spawn(move || p.worker(0, fills)) };
            let b = { let p = p.clone(); thread::spawn(move || p.worker(1, 1)) };
            let c = { let p = p.clone(); thread::spawn(move || p.cpu(2)) };
            a.join().unwrap();
            b.join().unwrap();
            c.join().unwrap();
            // Whatever the page holds at the end is code that still exists.
            let code = *p.page.lock();
            if code != NO_CODE {
                assert_eq!(p.fillings[code / 16].load(Ordering::SeqCst), code % 16);
            }
        });
    }

    #[test]
    fn two_arenas_two_workers_and_a_cpu() {
        model(2, 2);
    }

    #[test]
    fn three_arenas_two_workers_and_a_cpu() {
        model(3, 3);
    }
}
