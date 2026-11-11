//! `Mutex` and `Condvar` that loom can schedule, for code that uses
//! parking_lot.
//!
//! A normal build re-exports parking_lot's types as they are, so code importing
//! them from here compiles to what it did. Under `RUSTFLAGS="--cfg loom"` the
//! same names are loom's, behind a wrapper that keeps parking_lot's calling
//! convention: `lock()` returns the guard, and a condition variable waits on
//! `&mut guard`. Loom follows std, where `lock()` returns a `Result` and a wait
//! takes the guard and gives it back, so an import swap alone cannot put loom
//! under parking_lot code. This is what lets a model drive the real code rather
//! than a copy of its protocol.
//!
//! Only what the loom-modelled code calls is wrapped. Loom ignores wait
//! timeouts: `wait_for` blocks until notified, so a model reports a lost
//! wake-up as a deadlock even where a timeout would have hidden it. Loom's types
//! work only inside `loom::model`, so a `--cfg loom` build runs the `loom_tests`
//! modules and nothing else. loom is a dev-dependency, so the cfg works for
//! `cargo test` builds, not for `cargo build`.
//!
//! Atomics need no wrapper: a file that models them swaps its own import, as
//! `pci/intx.rs` does.

#[cfg(not(loom))]
pub use parking_lot::{Condvar, Mutex, MutexGuard};

#[cfg(loom)]
pub use loom_shim::{Condvar, Mutex, MutexGuard};

/// parking_lot's calling convention over loom's std-shaped types.
#[cfg(loom)]
mod loom_shim {
    use std::ops::{Deref, DerefMut};
    use std::time::Duration;

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

        pub fn is_locked(&self) -> bool {
            self.0.try_lock().is_err()
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

        pub fn wait_for<T>(&self, guard: &mut MutexGuard<'_, T>, timeout: Duration) {
            let g = guard.guard.take().unwrap();
            let (g, _) = self.0.wait_timeout(g, timeout).unwrap();
            guard.guard = Some(g);
        }
    }
}
