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
//! The worker pops hot-first; [`CompileQueue::recv`] sleeps on the queue's
//! doorbell rather than spinning. The queue carries no Cranelift types, so it
//! is defined outside the `rex-jit` feature gate and its behaviour is
//! unit-tested without a compiler.

use std::collections::VecDeque;

use crate::sync::Doorbell;

/// Depth of the cold (warm-up/prefetch) lane. A full cold lane drops the
/// request; this is the old single-channel capacity.
#[cfg(not(loom))]
pub const COLD_QUEUE_CAPACITY: usize = 256;
/// One slot keeps the loom model's state space small while still filling the
/// lane on the second send.
#[cfg(loom)]
pub const COLD_QUEUE_CAPACITY: usize = 1;

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
}

/// A two-lane compile queue. See the module docs for the cold/hot split.
pub struct CompileQueue {
    /// Rung when a request is queued (the worker waits for one) and when the
    /// worker frees a cold slot (a warm-up producer waits for room). A ring
    /// wakes both, and each looks for what it waits for.
    state: Doorbell<State>,
}

impl CompileQueue {
    pub fn new() -> Self {
        Self { state: Doorbell::new(State::default()) }
    }

    /// Enqueue a best-effort cold request. Returns the request back when the
    /// cold lane is full (or the queue is closed) so the caller can drop its
    /// `Queued` marker and let a later draw retry.
    pub fn send_cold(&self, req: CompileRequest) -> Result<(), CompileRequest> {
        let mut st = self.state.lock();
        if st.is_closed() || st.cold.len() >= COLD_QUEUE_CAPACITY {
            return Err(req);
        }
        st.cold.push_back(req);
        st.ring();
        Ok(())
    }

    /// Enqueue a cold request, waiting for room if the lane is full.
    ///
    /// Only warm-up blocks here: it runs on its own thread before the guest is
    /// drawing and wants the whole profile queued. The draw path must use
    /// [`Self::send_hot`] instead.
    pub fn send_cold_blocking(&self, req: CompileRequest) -> Result<(), CompileRequest> {
        let Ok(mut st) = self.state.wait_until(|st| st.cold.len() < COLD_QUEUE_CAPACITY) else {
            return Err(req);
        };
        st.cold.push_back(req);
        st.ring();
        Ok(())
    }

    /// Enqueue a hot request. Only a closed queue rejects it: the whole point of
    /// the hot lane is that a shape being drawn is never dropped for a full cold
    /// lane. The lane is unbounded, but the draw path files at most one request
    /// per uncompiled shape (it marks the shape `Queued`), so it holds at most
    /// one entry per outstanding shape.
    pub fn send_hot(&self, req: CompileRequest) -> Result<(), CompileRequest> {
        let mut st = self.state.lock();
        if st.is_closed() {
            return Err(req);
        }
        st.hot.push_back(req);
        st.ring();
        Ok(())
    }

    /// The next request, hot lane first, or `None` once the queue is closed and
    /// drained. Blocks while the queue is open and empty.
    pub fn recv(&self) -> Option<CompileRequest> {
        // Closed, the queue still hands out what it holds.
        let mut st = self.state.wait_until(|st| !st.hot.is_empty() || !st.cold.is_empty()).unwrap_or_else(|st| st);
        if let Some(req) = st.hot.pop_front() {
            return Some(req);
        }
        let req = st.cold.pop_front()?;
        // Free slot: let a blocked warm-up producer refill the lane.
        st.ring();
        Some(req)
    }

    /// Pop the next request without blocking: hot lane first, then cold. For
    /// tests that inspect lane order.
    #[cfg(test)]
    pub(crate) fn try_pop(&self) -> Option<CompileRequest> {
        let mut st = self.state.lock();
        st.hot.pop_front().or_else(|| st.cold.pop_front())
    }

    /// Close the queue. [`Self::recv`] drains whatever is left and then returns
    /// `None`; further sends are rejected.
    pub fn close(&self) {
        self.state.close();
    }
}

impl Default for CompileQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, not(loom)))]
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

/// Model-checked under `RUSTFLAGS="--cfg loom" cargo test --lib rex3_jit_queue`.
/// Loom ignores timeouts and reports a lost wakeup as every thread blocked.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use crate::sync::{thread, Arc};

    fn req(i: u32) -> CompileRequest {
        CompileRequest { dm0: 0, dm1: i, cm: 0 }
    }

    /// The warm-up producer fills the one-slot cold lane and blocks for the
    /// second request; the worker must drain both, in order, with no sleeper
    /// left waiting on a wakeup that was never sent.
    #[test]
    fn warm_up_producer_and_worker_both_finish() {
        loom::model(|| {
            let q = Arc::new(CompileQueue::new());
            let producer = {
                let q = Arc::clone(&q);
                thread::spawn(move || {
                    q.send_cold_blocking(req(1)).unwrap();
                    q.send_cold_blocking(req(2)).unwrap();
                })
            };
            assert_eq!(q.recv(), Some(req(1)));
            assert_eq!(q.recv(), Some(req(2)));
            producer.join().unwrap();
        });
    }

    /// A close can land while the producer waits for room. The producer must
    /// wake and either queue or be refused, and the worker must see exactly
    /// what was accepted, then `None`.
    #[test]
    fn close_releases_a_blocked_producer() {
        loom::model(|| {
            let q = Arc::new(CompileQueue::new());
            let producer = {
                let q = Arc::clone(&q);
                thread::spawn(move || {
                    let mut accepted = 0;
                    for i in 1..=2 {
                        if q.send_cold_blocking(req(i)).is_ok() {
                            accepted += 1;
                        }
                    }
                    accepted
                })
            };
            let closer = {
                let q = Arc::clone(&q);
                thread::spawn(move || q.close())
            };
            let mut got = 0;
            while q.recv().is_some() {
                got += 1;
            }
            closer.join().unwrap();
            // Every accepted request was delivered before the queue reported empty.
            assert_eq!(got, producer.join().unwrap());
        });
    }
}
