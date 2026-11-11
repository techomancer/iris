//! A worker's wake-up: state behind a lock, and a condition variable that
//! rings when the state changes.
//!
//! A thread that sleeps until another thread gives it work has to test for
//! that work holding the lock the other thread changes it under, and wait
//! without letting go of the lock in between. Get either half wrong and a
//! notify lands while nobody is waiting, and the sleeper sleeps through work
//! that is already there. Four device threads lost wakeups that way: one
//! waited without testing for work first, one was stopped by a flag stored
//! outside the lock, one tested work kept behind a different lock, and one
//! shared a condition variable between two waiters and woke the wrong one.
//!
//! [`Doorbell`] keeps the rule in one place. The state lives inside it, so
//! the only way to change what a waiter tests is under its lock. A waiter
//! names the condition it waits for and [`DoorbellGuard::wait_until`] tests
//! it before every wait. [`Doorbell::ring`] changes the state and notifies
//! under the lock, and wakes every waiter, so two threads waiting for
//! different things on one doorbell each get to look. [`Doorbell::close`] is
//! the stop: it is part of the state, so it cannot slip past a waiter
//! either.
//!
//! `loom_tests` checks the waits against every interleaving of a ring or a
//! close: run them with
//! `RUSTFLAGS="--cfg loom" cargo test --lib sync::doorbell::loom_tests`.

use std::ops::{Deref, DerefMut};

use super::{Condvar, Mutex, MutexGuard};

/// State `T` that threads wait on until it changes. See the module docs.
pub struct Doorbell<T> {
    state: Mutex<Bell<T>>,
    rung: Condvar,
}

struct Bell<T> {
    value: T,
    /// Set by [`Doorbell::close`]: waits give up instead of sleeping.
    closed: bool,
}

/// The doorbell's state, locked. Changing it through the guard wakes nobody
/// until [`DoorbellGuard::ring`].
pub struct DoorbellGuard<'a, T> {
    guard: MutexGuard<'a, Bell<T>>,
    rung: &'a Condvar,
}

impl<T> Doorbell<T> {
    /// An open doorbell holding `value`.
    pub fn new(value: T) -> Self {
        Self { state: Mutex::new(Bell { value, closed: false }), rung: Condvar::new() }
    }

    /// Lock the state, to read it or to change it without waking anyone.
    pub fn lock(&self) -> DoorbellGuard<'_, T> {
        DoorbellGuard { guard: self.state.lock(), rung: &self.rung }
    }

    /// [`Self::lock`] if no other thread holds the lock, for a reader that
    /// must not wait (the monitor).
    pub fn try_lock(&self) -> Option<DoorbellGuard<'_, T>> {
        Some(DoorbellGuard { guard: self.state.try_lock()?, rung: &self.rung })
    }

    /// Change the state with `f` and wake every waiter, all under the lock.
    pub fn ring<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let mut g = self.lock();
        let r = f(&mut g);
        g.ring();
        r
    }

    /// Lock and wait until `ready`: [`DoorbellGuard::wait_until`].
    pub fn wait_until(&self, ready: impl FnMut(&mut T) -> bool) -> Result<DoorbellGuard<'_, T>, DoorbellGuard<'_, T>> {
        self.lock().wait_until(ready)
    }

    /// Wake every waiter and make waits give up, until [`Self::reopen`].
    pub fn close(&self) {
        self.state.lock().closed = true;
        self.rung.notify_all();
    }

    /// Let waits sleep again after [`Self::close`]. A device's `stop`
    /// reopens once it has joined its thread, ready for the next `start`.
    pub fn reopen(&self) {
        self.state.lock().closed = false;
    }

    /// Whether some thread holds the lock, for the lock registry.
    pub fn is_locked(&self) -> bool {
        self.state.is_locked()
    }
}

impl<'a, T> DoorbellGuard<'a, T> {
    /// Wake every waiter, for a change made through this guard. They run
    /// once the guard is dropped.
    pub fn ring(&self) {
        self.rung.notify_all();
    }

    /// The doorbell has been closed.
    pub fn is_closed(&self) -> bool {
        self.guard.closed
    }

    /// Wait until `ready` holds, testing it under the lock before every
    /// wait. `Ok` with the guard once it holds; `Err` with the guard once
    /// the doorbell is closed, which wins over `ready` (a closing worker that
    /// still has to drain looks at the state through the `Err` guard).
    pub fn wait_until(mut self, mut ready: impl FnMut(&mut T) -> bool) -> Result<Self, Self> {
        loop {
            if self.guard.closed {
                return Err(self);
            }
            if ready(&mut self.guard.value) {
                return Ok(self);
            }
            self.rung.wait(&mut self.guard);
        }
    }
}

impl<T> Deref for DoorbellGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard.value
    }
}

impl<T> DerefMut for DoorbellGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard.value
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::{mpsc, Arc};
    use std::thread;
    use std::time::Duration;

    /// A ring from another thread wakes a waiter, which sees what was rung.
    #[test]
    fn a_ring_wakes_a_waiter() {
        let db = Arc::new(Doorbell::new(0u32));
        let (seen, rx) = mpsc::channel();
        let waiter = {
            let db = Arc::clone(&db);
            thread::spawn(move || seen.send(*db.wait_until(|v| *v != 0).ok().unwrap()).unwrap())
        };
        db.ring(|v| *v = 7);
        assert_eq!(rx.recv_timeout(Duration::from_secs(10)), Ok(7), "the ring was lost");
        waiter.join().unwrap();
    }

    /// A close wakes a waiter whose condition never holds.
    #[test]
    fn a_close_wakes_a_waiter() {
        let db = Arc::new(Doorbell::new(()));
        let (stopped, rx) = mpsc::channel();
        let waiter = {
            let db = Arc::clone(&db);
            thread::spawn(move || stopped.send(db.wait_until(|_| false).is_err()).unwrap())
        };
        db.close();
        assert_eq!(rx.recv_timeout(Duration::from_secs(10)), Ok(true), "the close was lost");
        waiter.join().unwrap();
    }

    /// A closed doorbell refuses waits until it is reopened.
    #[test]
    fn reopen_lets_waits_sleep_again() {
        let db = Doorbell::new(1u32);
        db.close();
        assert!(db.wait_until(|v| *v == 1).err().expect("closed").is_closed());
        db.reopen();
        assert!(db.wait_until(|v| *v == 1).is_ok());
    }
}

/// Every interleaving of a waiter against a ring or a close, checked by
/// loom. Each model deadlocks, and fails, if a wait sleeps before it tests
/// for what it waits for.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use crate::sync::{thread, Arc};

    /// A ring that lands before the waiter first looks is not lost.
    #[test]
    fn a_ring_before_the_wait_is_seen() {
        loom::model(|| {
            let db = Arc::new(Doorbell::new(false));
            db.ring(|v| *v = true);
            let waiter = {
                let db = Arc::clone(&db);
                thread::spawn(move || db.wait_until(|v| *v).is_ok())
            };
            assert!(waiter.join().unwrap());
        });
    }

    /// A ring racing the waiter, before, during or after its wait, wakes it,
    /// and the waiter sees the value that was rung.
    #[test]
    fn a_ring_during_the_wait_wakes_it() {
        loom::model(|| {
            let db = Arc::new(Doorbell::new(0u32));
            let ringer = {
                let db = Arc::clone(&db);
                thread::spawn(move || db.ring(|v| *v = 1))
            };
            let seen = *db.wait_until(|v| *v != 0).ok().unwrap();
            ringer.join().unwrap();
            assert_eq!(seen, 1);
        });
    }

    /// A close racing a waiter that waits for something nobody rings wakes
    /// it, whether it lands before or during the wait.
    #[test]
    fn a_close_while_waiting_stops_the_waiter() {
        loom::model(|| {
            let db = Arc::new(Doorbell::new(()));
            let waiter = {
                let db = Arc::clone(&db);
                thread::spawn(move || db.wait_until(|_| false).is_err())
            };
            db.close();
            assert!(waiter.join().unwrap());
        });
    }

    /// A producer filling a one-slot queue that nobody drains blocks for
    /// room on its second item; a close must release it. It reports what
    /// it queued: one item if the close came after the first, none if
    /// before.
    #[test]
    fn a_close_releases_a_blocked_producer() {
        loom::model(|| {
            let db = Arc::new(Doorbell::new(None::<u32>));
            let producer = {
                let db = Arc::clone(&db);
                thread::spawn(move || {
                    let mut queued = 0;
                    for i in 1..=2 {
                        let Ok(mut slot) = db.wait_until(|s| s.is_none()) else { break };
                        *slot = Some(i);
                        slot.ring();
                        queued += 1;
                    }
                    queued
                })
            };
            db.close();
            let queued = producer.join().unwrap();
            assert_eq!(queued, db.lock().iter().count());
        });
    }

    /// Two threads wait on one doorbell for different things, a producer
    /// for room and a watcher for the end. The ring that makes room must
    /// reach the producer: woken alone, the watcher would wait again and
    /// leave the producer asleep, and the take of its item below would never
    /// return.
    #[test]
    fn a_ring_wakes_every_waiter() {
        loom::model(|| {
            // (slot, done), with the slot full.
            let db = Arc::new(Doorbell::new((Some(0u32), false)));
            let producer = {
                let db = Arc::clone(&db);
                thread::spawn(move || {
                    let mut st = db.wait_until(|st| st.0.is_none()).ok().unwrap();
                    st.0 = Some(1);
                    st.ring();
                })
            };
            let watcher = {
                let db = Arc::clone(&db);
                thread::spawn(move || db.wait_until(|st| st.1).is_ok())
            };
            assert_eq!(db.ring(|st| st.0.take()), Some(0));
            let mut st = db.wait_until(|st| st.0.is_some()).ok().unwrap();
            assert_eq!(st.0.take(), Some(1));
            st.1 = true;
            st.ring();
            drop(st);
            producer.join().unwrap();
            assert!(watcher.join().unwrap());
        });
    }
}
