//! Priority compile queue for the REX3 shader JIT.
//!
//! The async shader compiler used to take its work from a single bounded
//! `SyncSender`. A full channel dropped the request — a deliberate choice so the
//! GFIFO consumer thread never blocks — but that made no distinction between a
//! one-shot warm-up/prefetch shape and a shape the guest is drawing *right now*.
//! A boot-time flood of profile shapes could fill all 256 slots and starve the
//! shader IRIX was actually asking for, so the hot draw kept missing the JIT
//! and ran on the interpreter until the backlog drained.
//!
//! This queue gives the compiler two lanes:
//!
//! - a bounded **cold** lane for warm-up/prefetch. It is best-effort: when it is
//!   full a request is rejected, the caller drops the `Queued` marker, and a
//!   later draw retries (see `clanker/rex3/rex-jit-queue-retry.md`).
//! - an **hot** lane for whatever the draw path is asking for at dispatch time.
//!   The worker drains it before the cold lane and it is never rejected for a
//!   full queue, so live draw work cannot be starved by prefetch.
//!
//! The worker pops hot-first; [`CompileQueue::recv`] parks on a condvar rather
//! than spinning. The queue carries no Cranelift types, so it is defined outside
//! the `rex-jit` feature gate and its behaviour is unit-tested without a
//! compiler.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

/// Depth of the cold (warm-up/prefetch) lane. A full cold lane drops the
/// request; this is the old single-channel capacity.
pub const COLD_QUEUE_CAPACITY: usize = 256;

/// A compile request: the normalised `(DrawMode0, DrawMode1, clipmode)` triple
/// that keys both the dispatch map and the compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompileRequest {
    pub dm0: u32,
    pub dm1: u32,
    pub cm: u32,
}

#[derive(Default)]
struct State {
    cold: VecDeque<CompileRequest>,
    hot: VecDeque<CompileRequest>,
    closed: bool,
}

/// A two-lane compile queue. See the module docs for the cold/hot split.
pub struct CompileQueue {
    state: Mutex<State>,
    ready: Condvar,
}

impl CompileQueue {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State::default()),
            ready: Condvar::new(),
        }
    }

    /// Enqueue a best-effort cold request. Returns the request back when the
    /// cold lane is full (or the queue is closed) so the caller can drop its
    /// `Queued` marker and let a later draw retry.
    pub fn send_cold(&self, req: CompileRequest) -> Result<(), CompileRequest> {
        let mut st = self.state.lock().unwrap();
        if st.closed || st.cold.len() >= COLD_QUEUE_CAPACITY {
            return Err(req);
        }
        st.cold.push_back(req);
        self.ready.notify_one();
        Ok(())
    }

    /// Enqueue a cold request, waiting for room if the lane is full.
    ///
    /// Only warm-up blocks here: it runs on its own thread before the guest is
    /// drawing and wants the whole profile queued. The draw path must use
    /// [`Self::send_hot`] instead.
    pub fn send_cold_blocking(&self, req: CompileRequest) -> Result<(), CompileRequest> {
        let mut st = self.state.lock().unwrap();
        while !st.closed && st.cold.len() >= COLD_QUEUE_CAPACITY {
            st = self.ready.wait(st).unwrap();
        }
        if st.closed {
            return Err(req);
        }
        st.cold.push_back(req);
        self.ready.notify_one();
        Ok(())
    }

    /// Enqueue a hot request. Only a closed queue rejects it: the whole point of
    /// the hot lane is that a shape being drawn is never dropped for a full cold
    /// lane. The lane is unbounded, but the draw path files at most one request
    /// per uncompiled shape (it marks the shape `Queued`), so it holds at most
    /// one entry per outstanding shape.
    pub fn send_hot(&self, req: CompileRequest) -> Result<(), CompileRequest> {
        let mut st = self.state.lock().unwrap();
        if st.closed {
            return Err(req);
        }
        st.hot.push_back(req);
        self.ready.notify_one();
        Ok(())
    }

    /// The next request, hot lane first, or `None` once the queue is closed and
    /// drained. Blocks while the queue is open and empty.
    pub fn recv(&self) -> Option<CompileRequest> {
        let mut st = self.state.lock().unwrap();
        loop {
            if let Some(req) = st.hot.pop_front() {
                return Some(req);
            }
            if let Some(req) = st.cold.pop_front() {
                // A free slot: wake a producer blocked for room.
                self.ready.notify_all();
                return Some(req);
            }
            if st.closed {
                return None;
            }
            st = self.ready.wait(st).unwrap();
        }
    }

    /// Pop the next request without blocking: hot lane first, then cold. For
    /// tests that inspect lane order.
    #[cfg(test)]
    pub(crate) fn try_pop(&self) -> Option<CompileRequest> {
        let mut st = self.state.lock().unwrap();
        st.hot.pop_front().or_else(|| st.cold.pop_front())
    }

    /// Close the queue. [`Self::recv`] drains whatever is left and then returns
    /// `None`; further sends are rejected.
    pub fn close(&self) {
        let mut st = self.state.lock().unwrap();
        st.closed = true;
        self.ready.notify_all();
    }
}

impl Default for CompileQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full cold lane must not stop a hot request being admitted, and the
    /// worker must pop it before any queued cold work.
    #[test]
    fn hot_request_is_admitted_ahead_of_a_full_cold_queue() {
        let q = CompileQueue::new();
        for i in 0..COLD_QUEUE_CAPACITY as u32 {
            assert!(q.send_cold(CompileRequest { dm0: 0, dm1: i, cm: 0 }).is_ok());
        }
        // The cold lane is full: another cold request is rejected.
        let cold = CompileRequest { dm0: 0, dm1: 0xDEAD, cm: 0 };
        assert_eq!(q.send_cold(cold), Err(cold));

        // A hot request is still admitted...
        let hot = CompileRequest { dm0: 1, dm1: 0xBEEF, cm: 7 };
        assert!(q.send_hot(hot).is_ok());

        // ...and drained ahead of the cold backlog.
        assert_eq!(q.try_pop(), Some(hot));
    }

    /// The hot request must survive to the worker even when the cold lane never
    /// had room for it — the drop-on-full path is cold-only.
    #[test]
    fn hot_request_survives_a_full_queue() {
        let q = CompileQueue::new();
        for i in 0..COLD_QUEUE_CAPACITY as u32 {
            q.send_cold(CompileRequest { dm0: 0, dm1: i, cm: 0 }).unwrap();
        }
        let hot = CompileRequest { dm0: 2, dm1: 7, cm: 3 };
        q.send_hot(hot).unwrap();

        let mut seen = 0;
        while let Some(req) = q.try_pop() {
            if req == hot {
                seen += 1;
            }
        }
        assert_eq!(seen, 1, "hot request was dropped while the cold lane was full");
    }

    /// `recv` prefers hot, drains what is queued after close, then stops.
    #[test]
    fn recv_drains_hot_first_and_stops_after_close() {
        let q = CompileQueue::new();
        let cold = CompileRequest { dm0: 0, dm1: 1, cm: 0 };
        let hot = CompileRequest { dm0: 0, dm1: 2, cm: 0 };
        q.send_cold(cold).unwrap();
        q.send_hot(hot).unwrap();

        assert_eq!(q.recv(), Some(hot));
        q.close();
        // Cold work queued before close is still delivered.
        assert_eq!(q.recv(), Some(cold));
        assert_eq!(q.recv(), None);
        assert_eq!(q.send_hot(hot), Err(hot));
    }

    /// A warm-up profile larger than the cold lane must reach the worker in
    /// full. The producer parks in `send_cold_blocking` once the lane fills;
    /// the worker's pops have to wake it, or both sleep until a guest draw.
    /// The lane is filled first and the worker started only once the
    /// producer has had time to park, so the pops are what must wake it.
    #[test]
    fn blocking_cold_send_resumes_when_the_worker_drains() {
        use std::sync::mpsc;
        use std::sync::Arc;
        use std::time::Duration;

        let req = |i| CompileRequest { dm0: 0, dm1: i, cm: 0 };
        let full = COLD_QUEUE_CAPACITY as u32;
        let total = full + 8;
        let q = Arc::new(CompileQueue::new());
        for i in 0..full {
            q.send_cold(req(i)).unwrap();
        }
        let producer = {
            let q = Arc::clone(&q);
            std::thread::spawn(move || {
                for i in full..total {
                    q.send_cold_blocking(req(i)).unwrap();
                }
            })
        };
        std::thread::sleep(Duration::from_millis(100));
        let (done, rx) = mpsc::channel();
        let worker = {
            let q = Arc::clone(&q);
            std::thread::spawn(move || {
                for _ in 0..total {
                    q.recv().unwrap();
                }
                done.send(()).unwrap();
            })
        };
        let finished = rx.recv_timeout(Duration::from_secs(10)).is_ok();
        // Release a stuck pair so the test does not leak blocked threads.
        q.close();
        producer.join().ok();
        worker.join().ok();
        assert!(finished, "worker never saw the whole profile: producer and worker both slept");
    }
}
