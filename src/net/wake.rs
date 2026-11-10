//! A wake-up signal between the NIC thread and its network backend.
//!
//! Waiters can block on it two ways: [`Wake::wait`] (a condvar, for threads
//! that only wait for the other side) or [`Wake::poller`], a readiness poller
//! that host sockets join as [`Watched`] (the NAT engine sleeps until either
//! the guest sends or a host socket is ready). `notify` serves both.
//!
//! A notify is never lost: it sets a flag that the next wait consumes, so a
//! signal sent while the other side was busy still ends its next wait at once.

use std::io;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};
use polling::{AsSource, Event, PollMode, Poller};

pub struct Wake {
    lock: Mutex<()>,
    cvar: Condvar,
    pending: AtomicBool,
    poller: Poller,
    next_key: AtomicUsize,
}

impl Wake {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            lock: Mutex::new(()),
            cvar: Condvar::new(),
            pending: AtomicBool::new(false),
            poller: Poller::new().expect("iris: cannot create the network wake poller"),
            next_key: AtomicUsize::new(0),
        })
    }

    /// Wake whoever waits (now or at its next wait).
    pub fn notify(&self) {
        if self.pending.swap(true, Ordering::AcqRel) {
            return; // already signalled and not yet consumed
        }
        // With nobody in the poller, this ends its next wait at once.
        let _ = self.poller.notify();
        // Taking the lock orders this against a waiter between its check of
        // `pending` and its sleep, so the notify can't slip through the gap.
        drop(self.lock.lock());
        self.cvar.notify_all();
    }

    /// Block until notified or `timeout` passes. Returns true if notified.
    pub fn wait(&self, timeout: Duration) -> bool {
        let mut g = self.lock.lock();
        if !self.pending.load(Ordering::Acquire) {
            let _ = self.cvar.wait_for(&mut g, timeout);
        }
        drop(g);
        self.consume()
    }

    /// Clear the signal (after a wait in `poller` returned, or to start
    /// afresh). Returns true if it was set. Clear it after the wait and
    /// before handling the work: a notify landing in between then ends the
    /// next wait early instead of being lost.
    pub fn consume(&self) -> bool {
        self.pending.swap(false, Ordering::AcqRel)
    }

    /// The poller `notify` wakes. One thread waits in it, with the host
    /// sockets it registered as [`Watched`].
    pub fn poller(&self) -> &Poller {
        &self.poller
    }

    /// For the lock-holder diagnostics (`locks` monitor command).
    pub fn is_locked(&self) -> bool {
        self.lock.is_locked()
    }
}

/// A host socket in a [`Wake`]'s poller: while it is ready for what it is
/// watched for, it ends the poller's wait. It leaves the poller when dropped.
pub(crate) struct Watched<S: AsSource> {
    sock: S,
    wake: Arc<Wake>,
    key: usize,
    interest: (bool, bool), // (readable, writable) as last registered
}

impl<S: AsSource> Watched<S> {
    /// Register `sock`, level-triggered, for readability and/or writability.
    pub fn new(sock: S, wake: &Arc<Wake>, readable: bool, writable: bool) -> io::Result<Self> {
        let key = wake.next_key.fetch_add(1, Ordering::Relaxed);
        let ev = Event::new(key, readable, writable);
        // SAFETY: `drop` deletes the socket from the poller before it closes.
        unsafe { wake.poller.add_with_mode(&sock.source(), ev, mode(readable, writable))? };
        Ok(Self { sock, wake: wake.clone(), key, interest: (readable, writable) })
    }

    /// The key the poller reports this socket's events under.
    pub fn key(&self) -> usize {
        self.key
    }

    /// Change what the socket is watched for (a system call only on a change).
    pub fn set_interest(&mut self, readable: bool, writable: bool) {
        if self.interest == (readable, writable) {
            return;
        }
        self.interest = (readable, writable);
        let ev = Event::new(self.key, readable, writable);
        let _ = self.wake.poller.modify_with_mode(&self.sock, ev, mode(readable, writable));
    }
}

/// Level-triggered, except with no interest: epoll still reports a hang-up
/// then, and level-triggered that would end every wait until the socket closes.
fn mode(readable: bool, writable: bool) -> PollMode {
    if readable || writable { PollMode::Level } else { PollMode::Oneshot }
}

impl<S: AsSource> Deref for Watched<S> {
    type Target = S;
    fn deref(&self) -> &S {
        &self.sock
    }
}

impl<S: AsSource> Drop for Watched<S> {
    fn drop(&mut self) {
        let _ = self.wake.poller.delete(&self.sock);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn a_notify_before_the_wait_is_not_lost() {
        let w = Wake::new();
        w.notify();
        let t0 = Instant::now();
        assert!(w.wait(Duration::from_secs(5)));
        assert!(t0.elapsed() < Duration::from_millis(100));
        assert!(!w.wait(Duration::from_millis(10)), "consumed");
    }

    #[test]
    fn notify_from_another_thread_ends_the_wait() {
        let w = Wake::new();
        let w2 = w.clone();
        let h = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            w2.notify();
        });
        let t0 = Instant::now();
        assert!(w.wait(Duration::from_secs(5)));
        assert!(t0.elapsed() < Duration::from_secs(1));
        h.join().unwrap();
    }

    #[test]
    fn a_notify_ends_the_poller_wait() {
        let w = Wake::new();
        let mut events = polling::Events::new();
        w.notify();
        let t0 = Instant::now();
        w.poller().wait(&mut events, Some(Duration::from_secs(5))).unwrap();
        assert!(t0.elapsed() < Duration::from_millis(100));
        assert!(w.consume());
        let t0 = Instant::now();
        w.poller().wait(&mut events, Some(Duration::from_millis(50))).unwrap();
        assert!(t0.elapsed() >= Duration::from_millis(45), "consumed");
    }

    #[test]
    fn a_watched_socket_ends_the_wait_while_ready() {
        let w = Wake::new();
        let rx = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let sock = Watched::new(rx, &w, true, false).unwrap();
        let mut events = polling::Events::new();
        let mut ready = || {
            events.clear();
            w.poller().wait(&mut events, Some(Duration::from_millis(20))).unwrap();
            events.iter().any(|e| e.key == sock.key() && e.readable)
        };
        assert!(!ready());
        let tx = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        tx.send_to(b"x", sock.local_addr().unwrap()).unwrap();
        assert!(ready());
        assert!(ready(), "level-triggered: ready until read");
        sock.recv_from(&mut [0u8; 8]).unwrap();
        assert!(!ready());
    }

    #[test]
    fn a_reset_socket_watched_for_nothing_ends_one_wait_at_most() {
        let w = Wake::new();
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let c = std::net::TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (s, _) = l.accept().unwrap();
        let _c = Watched::new(c, &w, false, false).unwrap();
        socket2::SockRef::from(&s).set_linger(Some(Duration::ZERO)).unwrap();
        drop(s); // the far end resets: a hang-up epoll reports whatever the interest
        std::thread::sleep(Duration::from_millis(20));
        let mut events = polling::Events::new();
        w.poller().wait(&mut events, Some(Duration::from_millis(20))).unwrap();
        let t0 = Instant::now();
        w.poller().wait(&mut events, Some(Duration::from_millis(50))).unwrap();
        assert!(t0.elapsed() >= Duration::from_millis(45), "woken again: {:?}", t0.elapsed());
    }
}
