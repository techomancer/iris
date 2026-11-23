//! A running device owned by one test, stopped and freed when the test ends.
//!
//! Device threads hold a `'static` reference to their device (`start()`
//! transmutes `&self`), so tests used to `Box::leak` every board they built.
//! A REX3 is ~75 MB of framebuffers and planes and a GR2 ~25 MB, and the lib
//! tests build a few hundred of them: the leaks added up to 12 GB at peak,
//! enough for the 16 GB GitHub runner to kill the test step with SIGTERM
//! (exit 143) or drop the job as "The operation was canceled".
//!
//! `TestBoard` keeps the board at a fixed heap address for the threads, and on
//! drop calls `Device::stop()` — which joins them — before freeing it.

use std::ops::Deref;
use std::sync::Arc;

use crate::clock::Clock;
use crate::hptimer::TimerManager;
use crate::traits::Device;

pub(crate) struct TestBoard<T: Device>(Arc<T>);

impl<T: Device> TestBoard<T> {
    /// Take ownership of a board. Wrap it before `start()`: the threads keep
    /// the address it has from then on.
    pub(crate) fn new(board: Arc<T>) -> Self {
        Self(board)
    }
}

impl<T: Device> Deref for TestBoard<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Device> Drop for TestBoard<T> {
    fn drop(&mut self) {
        // Joining the device threads ends every 'static borrow they hold;
        // the Arc then frees the board. Stop is idempotent for tests that
        // already called it.
        self.0.stop();
    }
}

/// A timer manager on a stepped machine clock. Timers hold their manager
/// weakly, so a test keeps this alive for as long as its devices run.
#[derive(Clone)]
pub(crate) struct Stepped {
    pub tm: Arc<TimerManager>,
    pub clock: Clock,
}

impl Stepped {
    pub(crate) fn new() -> Self {
        let clock = Clock::stepped();
        let tm = Arc::new(TimerManager::with_clock(clock.clone()));
        tm.manual.store(true, std::sync::atomic::Ordering::Relaxed);
        Self { tm, clock }
    }

    /// Move the clock on and run every timer it has reached before returning.
    pub(crate) fn step(&self, ns: u64) {
        self.clock.step(ns);
        self.tm.run_due();
    }
}

impl Deref for Stepped {
    type Target = Clock;
    fn deref(&self) -> &Clock {
        &self.clock
    }
}

/// Wait for another thread to make `cond` true. The minute only catches a
/// hang; a loaded host can starve a thread for seconds.
pub(crate) fn until(mut cond: impl FnMut() -> bool) -> bool {
    let end = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while !cond() {
        if std::time::Instant::now() >= end {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
    true
}
