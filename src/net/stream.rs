//! A private Ethernet segment carried over a byte stream, in the framing of
//! QEMU's `-netdev stream`: each frame is a 32-bit big-endian length followed
//! by that many bytes of Ethernet frame, with no FCS.
//!
//! iris connects as the client. The listening peer is whatever serves the
//! segment, for example another emulator or an install server with its own
//! user-space TCP/IP stack, so the guest sees a plain LAN with no NAT gateway
//! on it: no DHCP, no DNS, nothing iris answers itself. Nothing here needs
//! host privileges.

use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use super::{NetBackend, Wake};
use crate::devlog::LogModule;

/// The largest frame the peer may send: an Ethernet frame without FCS, plus
/// room for a VLAN tag. A longer length means the stream lost framing.
pub const MAX_FRAME: usize = 1518;

/// The shortest frame the guest's NIC can be handed: an Ethernet header.
const MIN_FRAME: usize = 14;

/// How long one blocked write or connect may take before the backend looks at
/// `running` again.
const IO_TICK: Duration = Duration::from_millis(250);

/// How long a peer may refuse to read before its session is given up.
const WRITE_STALL: Duration = Duration::from_secs(5);

/// A connected stream: the read half, the write half and a function that
/// shuts the connection down, ending a blocked read.
type Session = (Box<dyn Read + Send>, Box<dyn Write + Send>, Box<dyn Fn() + Send>);

/// Where the segment's server listens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEndpoint {
    Tcp(SocketAddr),
    #[cfg(unix)]
    Unix(PathBuf),
}

impl StreamEndpoint {
    /// `tcp:IP:PORT` (an IPv4 address, or an IPv6 one in brackets) or
    /// `unix:PATH`. A host name is not resolved.
    pub fn parse(s: &str) -> Result<Self, String> {
        if let Some(addr) = s.strip_prefix("tcp:") {
            return addr
                .parse()
                .map(Self::Tcp)
                .map_err(|e| format!("\"{addr}\" is not an IP:PORT address: {e}"));
        }
        #[cfg(unix)]
        if let Some(path) = s.strip_prefix("unix:") {
            if path.is_empty() {
                return Err("unix: needs a socket path".into());
            }
            return Ok(Self::Unix(PathBuf::from(path)));
        }
        Err(format!("\"{s}\" is neither tcp:IP:PORT nor unix:PATH"))
    }

    fn connect(&self) -> io::Result<Session> {
        match self {
            Self::Tcp(addr) => {
                let s = TcpStream::connect_timeout(addr, IO_TICK * 4)?;
                s.set_nodelay(true)?;
                s.set_write_timeout(Some(IO_TICK))?;
                let r = s.try_clone()?;
                let c = s.try_clone()?;
                Ok((Box::new(r), Box::new(s), Box::new(move || { let _ = c.shutdown(Shutdown::Both); })))
            }
            #[cfg(unix)]
            Self::Unix(path) => {
                let s = UnixStream::connect(path)?;
                s.set_write_timeout(Some(IO_TICK))?;
                let r = s.try_clone()?;
                let c = s.try_clone()?;
                Ok((Box::new(r), Box::new(s), Box::new(move || { let _ = c.shutdown(Shutdown::Both); })))
            }
        }
    }
}

impl std::fmt::Display for StreamEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tcp(a) => write!(f, "tcp:{a}"),
            #[cfg(unix)]
            Self::Unix(p) => write!(f, "unix:{}", p.display()),
        }
    }
}

/// Write one frame with its length prefix.
pub fn write_frame(w: &mut impl Write, frame: &[u8]) -> io::Result<()> {
    write_frame_while(w, frame, &|| true)
}

/// [`write_frame`] for a writer with a write timeout: a timed-out write is
/// retried while `keep_going` holds and the peer has not stalled for
/// [`WRITE_STALL`], then fails with `TimedOut`. The length and body go out in
/// one buffer so the peer never sees a split header.
fn write_frame_while(w: &mut impl Write, frame: &[u8], keep_going: &dyn Fn() -> bool) -> io::Result<()> {
    let mut buf = Vec::with_capacity(4 + frame.len());
    buf.extend_from_slice(&(frame.len() as u32).to_be_bytes());
    buf.extend_from_slice(frame);
    let (mut sent, mut stalled) = (0, None::<Instant>);
    while sent < buf.len() {
        match w.write(&buf[sent..]) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) => {
                sent += n;
                stalled = None;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {
                let since = *stalled.get_or_insert_with(Instant::now);
                if !keep_going() || since.elapsed() > WRITE_STALL {
                    return Err(io::ErrorKind::TimedOut.into());
                }
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Read one frame. `Ok(None)` is a clean end of stream between frames.
pub fn read_frame(r: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut len = [0u8; 4];
    let mut got = 0;
    while got < 4 {
        match r.read(&mut len[got..]) {
            Ok(0) if got == 0 => return Ok(None),
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    let len = u32::from_be_bytes(len) as usize;
    if len > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("frame length {len}")));
    }
    let mut frame = vec![0u8; len];
    r.read_exact(&mut frame)?;
    Ok(Some(frame))
}

impl super::GatewayConfig {
    /// The `[network] stream` peer, for a machine whose mode is
    /// [`NetMode::Stream`](crate::config::NetMode::Stream). `validate` has
    /// already parsed it; a missing or bad value here is a bug in the caller.
    pub fn stream_endpoint(&self) -> Result<StreamEndpoint, String> {
        let s = self.stream.as_deref().ok_or("network.mode = \"stream\" needs [network] stream")?;
        StreamEndpoint::parse(s)
    }
}

/// The backend for `[network] mode = "stream"`: connects to `endpoint` and
/// moves frames between it and the NIC's rings. Frames the guest sends
/// arrive on `tx_cons` (with `tx_wake` notified), and frames from the peer go
/// on `rx_prod` with `rx_wake` notified.
///
/// [`run`](NetBackend::run) retries the connection once a second until it
/// succeeds, as QEMU's `reconnect-ms` does, so the server may start after the
/// machine, and connects again if the peer later goes away. Frames the guest
/// sends while there is no peer are dropped, as on an unplugged cable. It
/// returns when `running` clears or the NIC drops its end of the rings, and
/// may be run again.
///
/// It is shared (`Arc<StreamBackend>` is a [`NetBackend`] too): one handle
/// runs on the link's thread while another reports its counters. The rings
/// belong to the running handle alone.
pub struct StreamBackend {
    endpoint: StreamEndpoint,
    io: Mutex<Io>,
    tx_wake: Arc<Wake>,
    rx_wake: Arc<Wake>,
    running: Arc<AtomicBool>,
    retry: Duration,
    stats: Arc<StreamStats>,
    /// The link's count of guest frames, for the GUI.
    frames: Arc<AtomicU64>,
}

/// The rings the running backend owns.
struct Io {
    tx_cons: rtrb::Consumer<Vec<u8>>,
    // Held by the receive thread during a session, which hands it back.
    rx_prod: Option<rtrb::Producer<Vec<u8>>>,
}

const RECONNECT: Duration = Duration::from_secs(1);

/// What the segment has carried, for the adapter's status command.
#[derive(Default)]
pub struct StreamStats {
    connected: AtomicBool,
    to_peer: AtomicU64,
    from_peer: AtomicU64,
    dropped: AtomicU64,
    runts: AtomicU64,
}

impl StreamBackend {
    pub fn new(
        endpoint: StreamEndpoint,
        tx_cons: rtrb::Consumer<Vec<u8>>,
        rx_prod: rtrb::Producer<Vec<u8>>,
        tx_wake: Arc<Wake>,
        rx_wake: Arc<Wake>,
        running: Arc<AtomicBool>,
        frames: Arc<AtomicU64>,
    ) -> Self {
        let stats = Arc::new(StreamStats::default());
        let io = Mutex::new(Io { tx_cons, rx_prod: Some(rx_prod) });
        Self { endpoint, io, tx_wake, rx_wake, running, retry: RECONNECT, stats, frames }
    }

    fn live(&self, io: &Io) -> bool {
        self.running.load(Ordering::SeqCst) && !io.tx_cons.is_abandoned()
    }

    /// Discard the frames the guest has queued, counting them.
    fn drain(&self, io: &mut Io) {
        while io.tx_cons.pop().is_ok() {
            self.frames.fetch_add(1, Ordering::Relaxed);
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Sleep until `until`, or until the backend stops.
    fn pause(&self, io: &Io, until: Instant) {
        while self.live(io) && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(20).min(self.retry));
        }
    }

    /// Connect, retrying until it works. `None` if the backend stopped first.
    fn connect(&self, io: &mut Io) -> Option<Session> {
        let mut said = false;
        while self.live(io) {
            match self.endpoint.connect() {
                Ok(c) => return Some(c),
                Err(e) => {
                    // Once per outage; the connection is logged when it comes.
                    if !std::mem::replace(&mut said, true) {
                        eprintln!("iris: network stream {}: {e}; retrying", self.endpoint);
                    }
                    // Frames the guest sends meanwhile have nowhere to go.
                    self.drain(io);
                    self.pause(io, Instant::now() + self.retry);
                }
            }
        }
        None
    }

    /// The receive thread for one session: peer frames onto the ring. It
    /// hands the producer back when the session ends, so the next session
    /// keeps feeding the same ring.
    fn receive(
        mut rd: Box<dyn Read + Send>,
        mut rx_prod: rtrb::Producer<Vec<u8>>,
        rx_wake: Arc<Wake>,
        ended: Arc<AtomicBool>,
        ended_wake: Arc<Wake>,
        stats: Arc<StreamStats>,
    ) -> rtrb::Producer<Vec<u8>> {
        let mut last_runt_log: Option<Instant> = None;
        loop {
            match read_frame(&mut rd) {
                Ok(Some(frame)) if frame.len() < MIN_FRAME => {
                    // Shorter than an Ethernet header is not a frame; the
                    // NICs would take it for one, so it never reaches them.
                    let n = stats.runts.fetch_add(1, Ordering::Relaxed) + 1;
                    if last_runt_log.map_or(true, |t| t.elapsed() >= Duration::from_secs(1)) {
                        last_runt_log = Some(Instant::now());
                        crate::dlog!(LogModule::Net, "stream: dropped a {} byte frame from the peer ({n} so far)", frame.len());
                    }
                }
                Ok(Some(mut frame)) => {
                    // Counted before the NIC can see it, so a status read after
                    // a delivery includes it.
                    stats.from_peer.fetch_add(1, Ordering::Relaxed);
                    // Wait for the NIC to make room rather than drop: the
                    // peer's TCP retransmits cost far more.
                    loop {
                        match rx_prod.push(frame) {
                            Ok(()) => break,
                            Err(rtrb::PushError::Full(f)) => {
                                if rx_prod.is_abandoned() || ended.load(Ordering::SeqCst) {
                                    return rx_prod;
                                }
                                frame = f;
                                rx_wake.notify();
                                // The NIC does not signal freed room, so look
                                // again shortly; the session ending wakes this
                                // at once.
                                ended_wake.wait(Duration::from_millis(1));
                            }
                        }
                    }
                    rx_wake.notify();
                }
                Ok(None) => {
                    eprintln!("iris: network stream closed by peer");
                    return rx_prod;
                }
                Err(e) => {
                    eprintln!("iris: network stream read: {e}");
                    return rx_prod;
                }
            }
        }
    }
}

impl StreamBackend {
    /// `NetBackend::run` for a shared backend.
    fn run_shared(&self) {
        let mut guard = self.io.lock();
        let io = &mut *guard;
        while let Some((rd, mut wr, close)) = self.connect(io) {
            eprintln!("iris: network stream connected to {}", self.endpoint);
            let started = Instant::now();
            // Frames queued during an outage are stale; the new peer never
            // saw the conversation they belonged to.
            self.drain(io);
            let prod = io.rx_prod.take().expect("receive producer");
            let ended = Arc::new(AtomicBool::new(false));
            let ended_wake = Wake::new();
            let (rw, stats) = (self.rx_wake.clone(), self.stats.clone());
            let (rx_ended, rx_ended_wake) = (ended.clone(), ended_wake.clone());
            self.stats.connected.store(true, Ordering::Relaxed);
            let rx = std::thread::Builder::new()
                .name("net-stream-rx".into())
                .spawn(move || Self::receive(rd, prod, rw, rx_ended, rx_ended_wake, stats))
                .expect("net-stream rx thread");
            loop {
                self.tx_wake.wait(Duration::from_millis(10));
                let mut failed = false;
                while let Ok(frame) = io.tx_cons.pop() {
                    self.frames.fetch_add(1, Ordering::Relaxed);
                    if let Err(e) = write_frame_while(&mut wr, &frame, &|| self.live(io)) {
                        eprintln!("iris: network stream write: {e}");
                        failed = true;
                        break;
                    }
                    self.stats.to_peer.fetch_add(1, Ordering::Relaxed);
                }
                if failed || rx.is_finished() || !self.live(io) {
                    break;
                }
            }
            ended.store(true, Ordering::SeqCst);
            ended_wake.notify();
            close();
            self.stats.connected.store(false, Ordering::Relaxed);
            io.rx_prod = Some(rx.join().expect("net-stream rx thread"));
            // A peer that accepts and hangs up must not be redialled in a
            // tight loop: wait out the rest of the retry interval.
            self.pause(io, started + self.retry);
        }
    }
}

impl NetBackend for StreamBackend {
    fn run(&mut self) {
        self.run_shared();
    }

    fn report(&self, w: &mut dyn Write) -> io::Result<()> {
        let s = &self.stats;
        writeln!(w, "Stream segment {}: {}", self.endpoint,
                 if s.connected.load(Ordering::Relaxed) { "connected" } else { "waiting for the peer" })?;
        writeln!(w, "  {} frames to the peer, {} from it, {} dropped while it was away, {} runts from it dropped",
                 s.to_peer.load(Ordering::Relaxed), s.from_peer.load(Ordering::Relaxed),
                 s.dropped.load(Ordering::Relaxed), s.runts.load(Ordering::Relaxed))
    }
}

impl NetBackend for Arc<StreamBackend> {
    fn run(&mut self) {
        self.run_shared();
    }

    fn report(&self, w: &mut dyn Write) -> io::Result<()> {
        (**self).report(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn a_frame_is_a_big_endian_length_then_its_bytes() {
        let mut buf = Vec::new();
        write_frame(&mut buf, &[1, 2, 3]).unwrap();
        assert_eq!(&buf[..], &[0, 0, 0, 3, 1, 2, 3]);
        let mut r = &buf[..];
        assert_eq!(read_frame(&mut r).unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(read_frame(&mut r).unwrap(), None);
        // Too long for an Ethernet frame, a cut-off length, a cut-off frame.
        for (bad, kind) in [
            (&[0u8, 0, 0x10, 0][..], io::ErrorKind::InvalidData),
            (&[0, 0][..], io::ErrorKind::UnexpectedEof),
            (&[0, 0, 0, 4, 9][..], io::ErrorKind::UnexpectedEof),
        ] {
            let mut r = bad;
            assert_eq!(read_frame(&mut r).unwrap_err().kind(), kind);
        }
    }

    /// A backend on a running flag, run on a thread, and the NIC's ends of
    /// its rings.
    struct Rig {
        to_net: rtrb::Producer<Vec<u8>>,
        from_net: rtrb::Consumer<Vec<u8>>,
        tx_wake: Arc<Wake>,
        rx_wake: Arc<Wake>,
        running: Arc<AtomicBool>,
        frames: Arc<AtomicU64>,
        status: Arc<StreamBackend>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    fn rig(endpoint: StreamEndpoint, retry: Duration) -> Rig {
        let (to_net, tx_cons) = rtrb::RingBuffer::<Vec<u8>>::new(8);
        let (rx_prod, from_net) = rtrb::RingBuffer::<Vec<u8>>::new(8);
        let (tx_wake, rx_wake) = (Wake::new(), Wake::new());
        let running = Arc::new(AtomicBool::new(true));
        let frames = Arc::new(AtomicU64::new(0));
        let mut b = StreamBackend::new(endpoint, tx_cons, rx_prod, tx_wake.clone(), rx_wake.clone(), running.clone(), frames.clone());
        b.retry = retry;
        let mut r = Rig { to_net, from_net, tx_wake, rx_wake, running, frames, status: Arc::new(b), thread: None };
        r.start();
        r
    }

    impl Rig {
        fn start(&mut self) {
            self.running.store(true, Ordering::SeqCst);
            let mut runner = self.status.clone();
            self.thread = Some(std::thread::spawn(move || runner.run()));
        }

        fn stop(&mut self) {
            self.running.store(false, Ordering::SeqCst);
            self.thread.take().unwrap().join().unwrap();
        }

        /// One frame from the guest to `peer`, and one back.
        fn exchange(&mut self, peer: &mut impl ReadWrite, out: u8, back: u8) {
            // A frame queued before the session is up is stale and discarded.
            let deadline = Instant::now() + Duration::from_secs(10);
            while !self.status.stats.connected.load(Ordering::Relaxed) {
                assert!(Instant::now() < deadline, "backend never reported the session");
                std::thread::sleep(Duration::from_millis(2));
            }
            self.to_net.push(vec![out; 60]).unwrap();
            self.tx_wake.notify();
            assert_eq!(read_frame(peer).unwrap(), Some(vec![out; 60]));
            write_frame(peer, &[back; 64]).unwrap();
            let deadline = Instant::now() + Duration::from_secs(10);
            let got = loop {
                if let Ok(f) = self.from_net.pop() {
                    break f;
                }
                assert!(Instant::now() < deadline, "no frame from the peer");
                self.rx_wake.wait(Duration::from_millis(10));
            };
            assert_eq!(got, vec![back; 64]);
        }
    }

    trait ReadWrite: Read + Write {}
    impl<T: Read + Write> ReadWrite for T {}

    /// Frames cross a real socket both ways, TCP and unix, with the server
    /// as the peer, and the status line counts them.
    #[test]
    fn pumps_frames_both_ways() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut r = rig(StreamEndpoint::Tcp(listener.local_addr().unwrap()), RECONNECT);
        let (mut peer, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        r.exchange(&mut peer, 0xaa, 0x55);
        // The backend counts a frame once its write returns, which can be
        // after the peer has read it and answered; wait for the count.
        let deadline = Instant::now() + Duration::from_secs(10);
        while r.status.stats.to_peer.load(Ordering::Relaxed) < 1 {
            assert!(Instant::now() < deadline, "the frame sent to the peer is never counted");
            std::thread::sleep(Duration::from_millis(2));
        }
        let mut text = Vec::new();
        r.status.report(&mut text).unwrap();
        let text = String::from_utf8(text).unwrap();
        assert_eq!(r.frames.load(Ordering::Relaxed), 1, "the guest's frame is counted for the GUI");
        assert!(text.contains(": connected") && text.contains("1 frames to the peer, 1 from it"), "{text}");
        r.stop();

        #[cfg(unix)]
        {
            let path = std::env::temp_dir().join(format!("iris-stream-{}.sock", std::process::id()));
            let _ = std::fs::remove_file(&path);
            let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
            let mut r = rig(StreamEndpoint::Unix(path.clone()), RECONNECT);
            let (mut peer, _) = listener.accept().unwrap();
            peer.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            r.exchange(&mut peer, 7, 8);
            r.stop();
            let _ = std::fs::remove_file(&path);
        }
    }

    /// The server starts after the machine, and comes back after it drops
    /// the connection: the client connects, and connects again.
    #[test]
    fn connects_late_and_reconnects() {
        // Reserve a port, then free it so the first attempts are refused.
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let mut r = rig(StreamEndpoint::Tcp(format!("127.0.0.1:{port}").parse().unwrap()), Duration::from_millis(50));
        std::thread::sleep(Duration::from_millis(200));
        let listener = TcpListener::bind(("127.0.0.1", port)).unwrap();
        for (out, back) in [(1, 2), (3, 4)] {
            let (mut peer, _) = listener.accept().unwrap();
            peer.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            r.exchange(&mut peer, out, back);
        }
        r.stop();
    }

    /// Clearing the running flag ends the backend whether it is waiting for
    /// a peer or connected, and the same backend runs again afterwards, as
    /// the IOC3 does across a stop.
    #[test]
    fn stops_when_running_clears() {
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        rig(StreamEndpoint::Tcp(format!("127.0.0.1:{port}").parse().unwrap()), RECONNECT).stop();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut r = rig(StreamEndpoint::Tcp(listener.local_addr().unwrap()), RECONNECT);
        let (_first, _) = listener.accept().unwrap();
        r.stop();
        r.start();
        let (mut peer, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        r.exchange(&mut peer, 5, 9);
        r.stop();
    }

    /// Frames shorter than an Ethernet header, including an empty one, never
    /// reach the guest's ring; the frames around them do.
    #[test]
    fn drops_runts_from_the_peer() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut r = rig(StreamEndpoint::Tcp(listener.local_addr().unwrap()), RECONNECT);
        let (mut peer, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        for len in [0usize, 1, 13, 14, 60] {
            write_frame(&mut peer, &vec![len as u8; len]).unwrap();
        }
        let mut got = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        while got.len() < 2 {
            assert!(Instant::now() < deadline, "frames from the peer: {got:?}");
            match r.from_net.pop() {
                Ok(f) => got.push(f.len()),
                Err(_) => {
                    r.rx_wake.wait(Duration::from_millis(10));
                }
            }
        }
        assert_eq!(got, [14, 60]);
        assert_eq!(r.status.stats.runts.load(Ordering::Relaxed), 3);
        r.stop();
    }

    /// A peer that accepts and hangs up at once is redialled once per retry
    /// interval, not in a tight loop.
    #[test]
    fn a_peer_that_hangs_up_is_redialled_at_the_retry_rate() {
        const DIALS: u32 = 6;
        let retry = Duration::from_millis(100);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let mut r = rig(StreamEndpoint::Tcp(listener.local_addr().unwrap()), retry);
        // Time the redials against the clock the test reads itself, not a
        // fixed window: a loaded host can stall the backend for part of a
        // window, but it cannot make the dials come closer together.
        let (mut first, mut last, mut accepted) = (None::<Instant>, Instant::now(), 0);
        let deadline = Instant::now() + Duration::from_secs(30);
        while accepted < DIALS {
            assert!(Instant::now() < deadline, "only {accepted} connections in 30 seconds");
            if listener.accept().is_ok() {
                last = Instant::now();
                first.get_or_insert(last);
                accepted += 1;
            } else {
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        r.stop();
        // Each dial waits out a full retry after the previous one. We see
        // them late, never early, so only the first sighting can shrink the
        // span; allow it one whole interval of lateness. A spinning client
        // makes all six in a few milliseconds.
        let span = last - first.unwrap();
        let least = retry * (DIALS - 2);
        assert!(span >= least, "{DIALS} connections in {span:?}, expected at least {least:?}");
    }

    /// Frames queued while no peer was there are not sent to the one that
    /// turns up.
    #[test]
    fn stale_frames_are_not_sent_to_a_new_peer() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut r = rig(StreamEndpoint::Tcp(listener.local_addr().unwrap()), Duration::from_millis(500));
        let wait_for = |want: bool, r: &Rig| {
            let deadline = Instant::now() + Duration::from_secs(10);
            while r.status.stats.connected.load(Ordering::Relaxed) != want {
                assert!(Instant::now() < deadline, "session never became {want}");
                std::thread::sleep(Duration::from_millis(2));
            }
        };
        let (first, _) = listener.accept().unwrap();
        wait_for(true, &r);
        drop(first);
        // The session is over and the backend is waiting out its retry
        // interval; queue frames for the gap.
        wait_for(false, &r);
        for _ in 0..3 {
            r.to_net.push(vec![0xee; 60]).unwrap();
        }
        let (mut peer, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        r.exchange(&mut peer, 1, 2);
        r.stop();
    }

    /// A peer that accepts and never reads fills the socket; stopping the
    /// backend still returns promptly.
    #[test]
    fn stop_does_not_wait_for_a_peer_that_never_reads() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut r = rig(StreamEndpoint::Tcp(listener.local_addr().unwrap()), RECONNECT);
        let (_peer, _) = listener.accept().unwrap();
        // Queue until the backend stops taking frames: the socket is full
        // and its write is stuck.
        let (mut queued, mut last_ok) = (0, Instant::now());
        let deadline = Instant::now() + Duration::from_secs(30);
        while last_ok.elapsed() < Duration::from_millis(500) {
            assert!(Instant::now() < deadline, "the socket never filled");
            if r.to_net.push(vec![0x5a; 1500]).is_ok() {
                queued += 1;
                last_ok = Instant::now();
                r.tx_wake.notify();
            } else {
                std::thread::sleep(Duration::from_micros(200));
            }
        }
        assert!(queued > 1000, "only {queued} frames went in before the backend stalled");
        let t = Instant::now();
        r.stop();
        assert!(t.elapsed() < Duration::from_secs(2), "stop took {:?}", t.elapsed());
    }
}
