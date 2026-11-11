//! Locks, condition variables, atomics and threads that loom can check.
//!
//! A normal build re-exports parking_lot's `Mutex` and `Condvar` and std's
//! atomics, `Arc` and `thread` as they are, so code importing them from here
//! compiles to exactly what it did. Under `RUSTFLAGS="--cfg loom"` the same
//! names are loom's, so a loom model drives the real code rather than a copy
//! of its protocol.
//!
//! The API is parking_lot's: `lock()` returns the guard, and a condition
//! variable waits on `&mut guard`. Loom follows std (`lock()` returns a
//! `Result`, `wait` takes the guard and gives it back), so under loom a thin
//! wrapper puts parking_lot's calling convention on top. Loom ignores wait
//! timeouts: `wait_for` and `wait_until` block until notified, so a model
//! reports a lost wakeup as a deadlock even where a timeout would have hidden
//! it.
//!
//! Loom's types only work inside `loom::model`. Tests that build code using
//! them outside a model are `cfg(not(loom))`.

#[cfg(not(loom))]
pub use parking_lot::{Condvar, Mutex, MutexGuard, WaitTimeoutResult};
#[cfg(not(loom))]
pub use std::sync::{atomic, Arc};
#[cfg(not(loom))]
pub use std::thread;

#[cfg(loom)]
pub use loom::sync::{atomic, Arc, WaitTimeoutResult};
#[cfg(loom)]
pub use loom::thread;
#[cfg(loom)]
pub use loom_shim::{Condvar, Mutex, MutexGuard};

/// parking_lot's calling convention over loom's std-shaped types.
#[cfg(loom)]
mod loom_shim {
    use std::ops::{Deref, DerefMut};
    use std::time::{Duration, Instant};

    pub struct Mutex<T>(loom::sync::Mutex<T>);

    /// The loom guard sits in an `Option` so a condition variable can take
    /// it through `&mut` and put it back, and `unlocked` can let go of the
    /// lock and take it again.
    pub struct MutexGuard<'a, T> {
        lock: &'a loom::sync::Mutex<T>,
        guard: Option<loom::sync::MutexGuard<'a, T>>,
    }

    impl<T> Mutex<T> {
        pub fn new(value: T) -> Self {
            Self(loom::sync::Mutex::new(value))
        }

        pub fn lock(&self) -> MutexGuard<'_, T> {
            MutexGuard { lock: &self.0, guard: Some(self.0.lock().unwrap()) }
        }

        pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
            let guard = self.0.try_lock().ok()?;
            Some(MutexGuard { lock: &self.0, guard: Some(guard) })
        }

        pub fn is_locked(&self) -> bool {
            self.try_lock().is_none()
        }
    }

    impl<T> MutexGuard<'_, T> {
        /// Run `f` with the lock let go, and take it again after.
        pub fn unlocked<U>(s: &mut Self, f: impl FnOnce() -> U) -> U {
            drop(s.guard.take());
            let r = f();
            s.guard = Some(s.lock.lock().unwrap());
            r
        }
    }

    impl<T> Deref for MutexGuard<'_, T> {
        type Target = T;
        fn deref(&self) -> &T {
            self.guard.as_ref().unwrap()
        }
    }

    impl<T> DerefMut for MutexGuard<'_, T> {
        fn deref_mut(&mut self) -> &mut T {
            self.guard.as_mut().unwrap()
        }
    }

    #[derive(Default)]
    pub struct Condvar(loom::sync::Condvar);

    impl Condvar {
        pub fn new() -> Self {
            Self(loom::sync::Condvar::new())
        }

        pub fn notify_one(&self) {
            self.0.notify_one();
        }

        pub fn notify_all(&self) {
            self.0.notify_all();
        }

        pub fn wait<T>(&self, guard: &mut MutexGuard<'_, T>) {
            let g = guard.guard.take().unwrap();
            guard.guard = Some(self.0.wait(g).unwrap());
        }

        pub fn wait_for<T>(&self, guard: &mut MutexGuard<'_, T>, timeout: Duration) -> loom::sync::WaitTimeoutResult {
            let g = guard.guard.take().unwrap();
            let (g, r) = self.0.wait_timeout(g, timeout).unwrap();
            guard.guard = Some(g);
            r
        }

        pub fn wait_until<T>(&self, guard: &mut MutexGuard<'_, T>, deadline: Instant) -> loom::sync::WaitTimeoutResult {
            self.wait_for(guard, deadline.saturating_duration_since(Instant::now()))
        }
    }
}

/// The loom wrapper's calling convention, checked by loom.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use std::time::Duration;

    /// A guard let go with `unlocked` lets another thread take the lock, and
    /// a wait on `&mut guard` sees that thread's notify.
    #[test]
    fn the_wrapper_waits_and_unlocks_through_the_guard() {
        loom::model(|| {
            let m = Arc::new((Mutex::new(0u32), Condvar::new()));
            let bumper = {
                let m = Arc::clone(&m);
                thread::spawn(move || {
                    let (lock, cv) = &*m;
                    *lock.lock() += 1;
                    cv.notify_all();
                })
            };
            let (lock, cv) = &*m;
            let mut g = lock.lock();
            MutexGuard::unlocked(&mut g, || ());
            while *g == 0 {
                cv.wait_for(&mut g, Duration::from_millis(1));
            }
            drop(g);
            bumper.join().unwrap();
        });
    }
}
