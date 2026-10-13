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
