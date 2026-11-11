//! The compile pool's quiesce barrier: how one worker stops the others for a
//! flush or a forced seal, and how `CompileQueue::stop` gets every worker out.
//!
//! The first worker to see a trigger wins [`PoolBarrier::try_lead`]. Every
//! other worker that sees the cycle in progress parks
//! ([`PoolBarrier::park`]) until the leader ends it with
//! [`PoolBarrier::release`]. The leader waits for the others to park
//! ([`PoolBarrier::wait_for_followers`]) before it touches anything they
//! share. One barrier serves both kinds of cycle, a full arena flush and a
//! forced-seal sweep. They used to have a barrier each, and deadlocked: one
//! worker led a seal and waited for the other to park, while the other led a
//! flush and waited for the first.
//!
//! The pool's run flag lives here too, so that clearing it
//! ([`PoolBarrier::stop`]) always wakes whoever waits on the barrier. Every
//! change to a waiter's condition is made under the barrier's mutex and
//! followed by a notify, so no wait needs a timeout to make progress.

#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, Ordering};
#[cfg(loom)]
use loom::sync::{Condvar, Mutex, MutexGuard};
#[cfg(not(loom))]
use parking_lot::{Condvar, Mutex, MutexGuard};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, Ordering};

struct State<A> {
    /// Workers parked in the current cycle. Reset by the leader's release.
    parked: usize,
    /// Bumped once per cycle, by the leader's release. A parked worker waits
    /// for it to move, which tells a real release from a spurious wakeup.
    generation: u64,
    /// Whether the last cycle replaced the arena: parked workers then rebuild
    /// their `Codegen` on `arena`. A seal-only cycle leaves both alone.
    was_flush: bool,
    /// The arena every worker builds on: set at start and by each flush.
    arena: Option<A>,
}

/// `A` is what a flush hands every worker to rebuild on: the shared arena
/// and its state in production, a token in the models.
pub(crate) struct PoolBarrier<A> {
    state: Mutex<State<A>>,
    cv: Condvar,
    running: AtomicBool,
    /// A cycle is in progress: set by the leader's election, cleared by its
    /// release.
    quiesce: AtomicBool,
}

#[cfg(not(loom))]
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock()
}
#[cfg(loom)]
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap()
}
#[cfg(not(loom))]
fn wait<'a, T>(cv: &Condvar, mut g: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    cv.wait(&mut g);
    g
}
#[cfg(loom)]
fn wait<'a, T>(cv: &Condvar, g: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    cv.wait(g).unwrap()
}

impl<A: Clone> PoolBarrier<A> {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(State { parked: 0, generation: 0, was_flush: false, arena: None }),
            cv: Condvar::new(),
            running: AtomicBool::new(false),
            quiesce: AtomicBool::new(false),
        }
    }

    /// Mark the pool running, on the arena its workers are about to build on.
    pub(crate) fn start(&self, arena: A) {
        lock(&self.state).arena = Some(arena);
        self.running.store(true, Ordering::SeqCst);
    }

    #[inline]
    pub(crate) fn running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Clear the run flag and wake every waiter, so a leader waiting for a
    /// worker that has already left, and a worker parked behind a leader
    /// that is gone, both see it. Taking the mutex orders the store before
    /// any waiter's next look at the flag: a waiter between its check and
    /// its wait holds the mutex.
    pub(crate) fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        let _state = lock(&self.state);
        self.cv.notify_all();
    }

    /// A worker leads a cycle.
    #[inline]
    pub(crate) fn in_progress(&self) -> bool {
        self.quiesce.load(Ordering::Acquire)
    }

    /// Win the election for a new cycle. The winner must end it with
    /// [`Self::release`] whatever happens, or the others park forever.
    pub(crate) fn try_lead(&self) -> bool {
        self.quiesce.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok()
    }

    /// The leader: wait until `followers` workers have parked. `false` if the
    /// pool stops first: a worker that saw the stop at the top of its loop
    /// leaves rather than parks, so the count may never arrive.
    pub(crate) fn wait_for_followers(&self, followers: usize) -> bool {
        let mut state = lock(&self.state);
        while state.parked < followers {
            if !self.running() {
                return false;
            }
            state = wait(&self.cv, state);
        }
        true
    }

    /// The leader: end the cycle and wake the parked workers. `Some` for a
    /// flush, with the arena they must rebuild on; `None` for a seal-only or
    /// abandoned cycle, after which they resume in place. The flag clears
    /// under the mutex: a worker that saw the cycle in progress but reaches
    /// [`Self::park`] after this sees it over, rather than parking for a
    /// generation nobody will bump.
    pub(crate) fn release(&self, arena: Option<A>) {
        let mut state = lock(&self.state);
        state.was_flush = arena.is_some();
        if let Some(a) = arena {
            state.arena = Some(a);
        }
        state.parked = 0;
        state.generation += 1;
        self.quiesce.store(false, Ordering::Release);
        self.cv.notify_all();
    }

    /// A worker that saw a cycle in progress: park until the leader releases
    /// it. Returns the arena to rebuild on if the cycle was a flush, `None`
    /// if there is nothing to rebuild: a seal-only cycle, a cycle already
    /// over when this worker got here, or the pool stopping (the worker then
    /// leaves at the top of its loop).
    pub(crate) fn park(&self) -> Option<A> {
        let mut state = lock(&self.state);
        if !self.in_progress() || !self.running() {
            return None;
        }
        let generation = state.generation;
        state.parked += 1;
        self.cv.notify_all();
        while state.generation == generation {
            if !self.running() {
                // Leave the count as it was for the leader, which may still
                // be counting.
                state.parked -= 1;
                return None;
            }
            state = wait(&self.cv, state);
        }
        if state.was_flush {
            Some(state.arena.clone().expect("a flush publishes an arena"))
        } else {
            None
        }
    }

    /// The current arena, for monitor commands. `None` before the first start.
    pub(crate) fn arena(&self) -> Option<A> {
        lock(&self.state).arena.clone()
    }

    #[cfg(all(test, not(loom)))]
    pub(crate) fn parked(&self) -> usize {
        lock(&self.state).parked
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    /// Run `f` on a thread and fail, rather than hang, if it does not return.
    fn returns<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || tx.send(f()).unwrap());
        rx.recv_timeout(Duration::from_secs(30)).expect("the barrier wait did not return")
    }

    fn parked_barrier(n: usize) -> (Arc<PoolBarrier<u32>>, Vec<std::thread::JoinHandle<Option<u32>>>) {
        let b = Arc::new(PoolBarrier::new());
        b.start(1);
        assert!(b.try_lead());
        let followers = (0..n).map(|_| {
            let b = b.clone();
            std::thread::spawn(move || b.park())
        }).collect();
        while b.parked() < n {
            std::thread::yield_now();
        }
        (b, followers)
    }

    #[test]
    fn a_flush_hands_parked_workers_the_new_arena() {
        let (b, followers) = parked_barrier(2);
        assert!(b.wait_for_followers(2));
        b.release(Some(7));
        for f in followers {
            assert_eq!(f.join().unwrap(), Some(7));
        }
        assert_eq!(b.parked(), 0);
        assert!(!b.in_progress());
        assert_eq!(b.arena(), Some(7));
    }

    #[test]
    fn a_seal_only_cycle_resumes_parked_workers_in_place() {
        let (b, followers) = parked_barrier(2);
        b.release(None);
        for f in followers {
            assert_eq!(f.join().unwrap(), None);
        }
        assert_eq!(b.arena(), Some(1));
    }

    /// The live hang: a worker saw the cycle in progress, but the leader
    /// released it before the worker reached the barrier.
    #[test]
    fn park_after_the_cycle_ended_returns_at_once() {
        let b = Arc::new(PoolBarrier::new());
        b.start(1);
        assert!(b.try_lead());
        b.release(None);
        let b2 = b.clone();
        assert_eq!(returns(move || b2.park()), None);
        assert_eq!(b.parked(), 0);
    }

    /// A worker parked when the pool stops, with no leader left to release
    /// it, must come back so `stop()`'s join returns. Nothing times the
    /// wait out: only `stop`'s wakeup gets it out.
    #[test]
    fn a_parked_worker_returns_when_the_pool_stops() {
        let (b, followers) = parked_barrier(1);
        b.stop();
        for f in followers {
            assert_eq!(returns(move || f.join().unwrap()), None);
        }
        assert_eq!(b.parked(), 0);
    }

    /// A leader waiting for a worker that left instead of parking must see
    /// the stop.
    #[test]
    fn a_leader_waiting_for_followers_returns_when_the_pool_stops() {
        let b = Arc::new(PoolBarrier::<u32>::new());
        b.start(1);
        assert!(b.try_lead());
        let leader = {
            let b = b.clone();
            std::thread::spawn(move || b.wait_for_followers(1))
        };
        // Give the leader time to block; the result does not depend on it.
        std::thread::sleep(Duration::from_millis(10));
        b.stop();
        assert!(!returns(move || leader.join().unwrap()));
    }
}
