//! A wake-up signal between the NIC thread and its network backend.
//!
//! Waiters can block on it two ways: [`Wake::wait`] (a condvar, for threads
//! that only wait for the other side) or [`Wake::poller`], a readiness poller
//! that host sockets join as `Watched` (the NAT engine sleeps until either
//! the guest sends or a host socket is ready). `notify` serves both.
//!
//! A notify is never lost: it sets a flag that the next wait consumes, so a
//! signal sent while the other side was busy still ends its next wait at once.

use std::io::{self, Read, Write};
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use parking_lot::{Condvar, Mutex};
use polling::{AsSource, Event, Events, PollMode, Poller};

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
    /// sockets it registered as `Watched`.
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

impl<S: AsSource + Read> Read for Watched<S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> { self.sock.read(buf) }
}

impl<S: AsSource + Write> Write for Watched<S> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { self.sock.write(buf) }
    fn flush(&mut self) -> io::Result<()> { self.sock.flush() }
}

/// One thread that runs every console pump when a [`Watched`] socket is ready or on `notify`,
/// in place of a timer-polling thread per console.
pub struct HostIo {
    pub wake: Arc<Wake>,
    lines: Mutex<Vec<(usize, Pump)>>,
    next_id: AtomicUsize,
    #[cfg(test)]
    passes: AtomicUsize,
}

/// A pump returns whether it moved input.
type Pump = Box<dyn FnMut() -> bool + Send>;

/// Dropping it removes the pump, after any pass that is running it.
pub struct HostPump {
    io: &'static HostIo,
    id: usize,
}

/// The process-wide host-IO thread.
pub fn host_io() -> &'static HostIo {
    static IO: OnceLock<&'static HostIo> = OnceLock::new();
    IO.get_or_init(HostIo::start)
}

impl HostIo {
    fn start() -> &'static HostIo {
        let io: &'static HostIo = Box::leak(Box::new(Self { wake: Wake::new(), lines: Mutex::new(Vec::new()), next_id: AtomicUsize::new(0), #[cfg(test)] passes: AtomicUsize::new(0) }));
        std::thread::Builder::new().name("host-io".into()).spawn(move || io.run()).expect("spawn host-io thread");
        io
    }

    /// Run `pump` whenever the thread wakes. It runs under the lines' lock, so it must
    /// not add a pump or drop a `HostPump`.
    pub fn add(&'static self, pump: impl FnMut() -> bool + Send + 'static) -> HostPump {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.lines.lock().push((id, Box::new(pump)));
        self.wake.notify();
        HostPump { io: self, id }
    }

    fn run(&self) {
        let mut events = Events::new();
        let mut stuck = false;
        loop {
            // With no pump (a stopped machine) nobody reads the ready sockets, so only a notify may wake us.
            if self.lines.lock().is_empty() {
                self.wake.wait(Duration::from_secs(3600));
                continue;
            }
            events.clear();
            let waited = self.wake.poller.wait(&mut events, None);
            self.wake.consume();
            #[cfg(test)]
            self.passes.fetch_add(1, Ordering::Relaxed);
            let moved = self.lines.lock().iter_mut().fold(false, |m, l| (l.1)() | m);
            // Level-triggered: a ready socket the pumps left alone twice running (a full input FIFO)
            // must not end every wait. Once is a guest that read the byte first. A failed wait counts too.
            let blocked = (!events.is_empty() || waited.is_err()) && !moved;
            if blocked && stuck {
                std::thread::sleep(Duration::from_millis(2));
            }
            stuck = blocked;
        }
    }
}

impl Drop for HostPump {
    fn drop(&mut self) {
        // Dropped after the guard: a closure's last reference may take `lines` again.
        let gone = {
            let mut lines = self.io.lines.lock();
            lines.iter().position(|l| l.0 == self.id).map(|i| lines.remove(i))
        };
        drop(gone);
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

    #[test]
    fn with_nothing_ready_no_pump_runs() {
        let io = HostIo::start();
        let n = Arc::new(AtomicUsize::new(0));
        let n2 = n.clone();
        let _pump = io.add(move || n2.fetch_add(1, Ordering::SeqCst) == usize::MAX);
        std::thread::sleep(Duration::from_millis(100));
        assert!(n.load(Ordering::SeqCst) <= 1, "woke with nothing to do");
    }

    #[test]
    fn a_socket_the_pump_leaves_unread_does_not_spin_the_thread() {
        let io = HostIo::start();
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut c = std::net::TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (s, _) = l.accept().unwrap();
        let _w = Watched::new(s, &io.wake, true, false).unwrap();
        c.write_all(b"x").unwrap();
        let n = Arc::new(AtomicUsize::new(0));
        let n2 = n.clone();
        let _pump = io.add(move || n2.fetch_add(1, Ordering::SeqCst) == usize::MAX);
        std::thread::sleep(Duration::from_millis(300));
        // One pass per pause (about 150); a spin would run tens of thousands.
        assert!(n.load(Ordering::SeqCst) < 1000, "{} passes", n.load(Ordering::SeqCst));
    }

    /// With every pump gone, a ready socket nobody reads must not wake the thread.
    #[test]
    fn a_ready_socket_with_no_pump_does_not_wake_the_thread() {
        let io = HostIo::start();
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut c = std::net::TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (s, _) = l.accept().unwrap();
        let _w = Watched::new(s, &io.wake, true, false).unwrap();
        io.add(|| false);
        c.write_all(b"x").unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let before = io.passes.load(Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(300));
        assert!(io.passes.load(Ordering::Relaxed) - before < 10, "spun with no pump");
    }

    /// A pump whose closure holds the last reference to something that drops a pump itself does not deadlock.
    #[test]
    fn a_pump_dropped_with_a_closure_that_drops_a_pump_does_not_deadlock() {
        let io = HostIo::start();
        let inner = io.add(|| false);
        let outer = io.add(move || {
            let _keep = &inner;
            false
        });
        drop(outer);
    }

    #[test]
    fn dropping_a_pump_waits_for_the_pass_running_it_and_none_follows() {
        let io = HostIo::start();
        let (entered_tx, entered) = std::sync::mpsc::channel();
        let (go, go_rx) = std::sync::mpsc::channel::<()>();
        let n = Arc::new(AtomicUsize::new(0));
        let n2 = n.clone();
        let pump = io.add(move || {
            if n2.fetch_add(1, Ordering::SeqCst) == 0 {
                entered_tx.send(()).unwrap();
                go_rx.recv().unwrap();
            }
            false
        });
        entered.recv_timeout(Duration::from_secs(10)).unwrap();
        let done = Arc::new(AtomicBool::new(false));
        let done2 = done.clone();
        let h = std::thread::spawn(move || {
            drop(pump);
            done2.store(true, Ordering::SeqCst);
        });
        std::thread::sleep(Duration::from_millis(100));
        assert!(!done.load(Ordering::SeqCst), "dropped under a running pass");
        go.send(()).unwrap();
        h.join().unwrap();
        io.wake.notify();
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(n.load(Ordering::SeqCst), 1, "ran after it was dropped");
    }
}
