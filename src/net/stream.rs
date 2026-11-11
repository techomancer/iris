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

use super::{NetBackend, NetReport, Wake};
use crate::devlog::LogModule;

/// The largest frame the peer may send: an Ethernet frame without FCS, plus
/// room for a VLAN tag. A longer frame is read and discarded.
pub const MAX_FRAME: usize = 1518;

/// The longest frame read through to discard. A peer with a jumbo MTU sends
/// frames up to about 9 KB, and an IP packet is at most 64 KiB. A longer
/// length is no frame of any peer's: the stream lost framing, so the session
/// ends rather than swallow gigabytes.
const MAX_DISCARD: usize = 1 << 16;

/// The shortest frame the guest's NIC can be handed: an Ethernet header.
const MIN_FRAME: usize = 14;

/// How long one blocked read, write or connect may take before the backend
/// looks at `running` again.
const IO_TICK: Duration = Duration::from_millis(250);

/// How long a peer may refuse to read before its session is given up.
const WRITE_STALL: Duration = Duration::from_secs(5);

/// A connected stream: the read half, the write half and a function that
/// shuts the connection down. The shutdown tells the peer; it does not end a
/// blocked read everywhere, so the read half times out instead.
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
                // Windows does not end a blocked recv when another handle shuts
                // the socket down, so the reader must wake to see the session
                // end. It does not do that with a receive timeout either: a
                // receive that times out leaves the socket in an undefined
                // state there, and the reader resumes a frame after one.
                #[cfg(not(windows))]
                s.set_read_timeout(Some(IO_TICK))?;
                let r = s.try_clone()?;
                #[cfg(windows)]
                let r = PolledRead::new(r)?;
                let c = s.try_clone()?;
                Ok((Box::new(r), Box::new(s), Box::new(move || { let _ = c.shutdown(Shutdown::Both); })))
            }
            #[cfg(unix)]
            Self::Unix(path) => {
                let s = UnixStream::connect(path)?;
                s.set_write_timeout(Some(IO_TICK))?;
                s.set_read_timeout(Some(IO_TICK))?;
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

/// A TCP read half that waits for the socket to be readable instead of
/// relying on a receive timeout: a read that finds nothing within
/// [`IO_TICK`] fails with `TimedOut` without having touched the socket.
#[cfg_attr(not(windows), allow(dead_code))]
struct PolledRead {
    sock: TcpStream,
    poller: polling::Poller,
    events: polling::Events,
}

#[cfg_attr(not(windows), allow(dead_code))]
impl PolledRead {
    fn new(sock: TcpStream) -> io::Result<Self> {
        let poller = polling::Poller::new()?;
        // SAFETY: `drop` removes the socket from the poller before it closes.
        unsafe { poller.add_with_mode(&sock, polling::Event::readable(0), polling::PollMode::Level)? };
        Ok(Self { sock, poller, events: polling::Events::new() })
    }
}

impl Read for PolledRead {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.events.clear();
        self.poller.wait(&mut self.events, Some(IO_TICK))?;
        if self.events.iter().next().is_none() {
            return Err(io::ErrorKind::TimedOut.into());
        }
        self.sock.read(buf)
    }
}

impl Drop for PolledRead {
    fn drop(&mut self) {
        let _ = self.poller.delete(&self.sock);
    }
}

/// Write one frame with its length prefix.
pub fn write_frame(w: &mut impl Write, frame: &[u8]) -> io::Result<()> {
    write_frame_while(w, frame, &|| true)
}

/// [`write_frame`] for a writer with a write timeout: a timed-out write is
/// retried while `keep_going` holds and the peer has not stalled for
/// [`WRITE_STALL`], then fails with `TimedOut`. The length and body go out in
/// one vectored write, so a socket peer never sees a split header.
fn write_frame_while(w: &mut impl Write, frame: &[u8], keep_going: &dyn Fn() -> bool) -> io::Result<()> {
    let len = (frame.len() as u32).to_be_bytes();
    let mut parts = [io::IoSlice::new(&len), io::IoSlice::new(frame)];
    let mut left = &mut parts[..];
    let mut stalled = None::<Instant>;
    while !left.is_empty() {
        match w.write_vectored(left) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) => {
                io::IoSlice::advance_slices(&mut left, n);
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

/// What [`read_frame_while`] found next on the stream.
enum Incoming {
    End,
    Frame(Vec<u8>),
    /// A frame longer than [`MAX_FRAME`], whose bytes have been read and dropped.
    Oversize(usize),
}

/// Read one frame. `Ok(None)` is a clean end of stream between frames. A frame
/// longer than [`MAX_FRAME`] is an `InvalidData` error.
pub fn read_frame(r: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    match read_frame_while(r, &|| false)? {
        Incoming::End => Ok(None),
        Incoming::Frame(f) => Ok(Some(f)),
        Incoming::Oversize(len) => Err(io::Error::new(io::ErrorKind::InvalidData, format!("frame length {len}"))),
    }
}

/// [`read_frame`] for a reader with a read timeout: a timed-out read is
/// retried while `keep_going` holds, keeping the part of the frame already
/// read, and otherwise fails as the read did.
fn read_frame_while(r: &mut impl Read, keep_going: &dyn Fn() -> bool) -> io::Result<Incoming> {
    let mut len = [0u8; 4];
    match fill(r, &mut len, keep_going)? {
        0 => return Ok(Incoming::End),
        4 => {}
        _ => return Err(io::ErrorKind::UnexpectedEof.into()),
    }
    let len = u32::from_be_bytes(len) as usize;
    if len > MAX_DISCARD {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("frame length {len}")));
    }
    if len > MAX_FRAME {
        let mut skip = [0u8; MAX_FRAME];
        let mut left = len;
        while left > 0 {
            let n = left.min(skip.len());
            if fill(r, &mut skip[..n], keep_going)? < n {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            left -= n;
        }
        return Ok(Incoming::Oversize(len));
    }
    let mut frame = vec![0u8; len];
    if fill(r, &mut frame, keep_going)? < len {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(Incoming::Frame(frame))
}

/// Read until `buf` is full or the stream ends, returning how much arrived.
fn fill(r: &mut impl Read, buf: &mut [u8], keep_going: &dyn Fn() -> bool) -> io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) && keep_going() => {}
            Err(e) => return Err(e),
        }
    }
    Ok(got)
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
pub struct StreamBackend {
    tx_cons: rtrb::Consumer<Vec<u8>>,
    // Held by the receive thread during a session, which hands it back.
    rx_prod: Option<rtrb::Producer<Vec<u8>>>,
    tx_wake: Arc<Wake>,
    rx_wake: Arc<Wake>,
    running: Arc<AtomicBool>,
    retry: Duration,
    stats: Arc<StreamStats>,
    /// The link's count of guest frames, for the GUI.
    frames: Arc<AtomicU64>,
}

/// How long the adapter's backend waits between attempts to reach the peer.
pub(super) const RECONNECT: Duration = Duration::from_secs(1);

/// What the segment has carried, for the adapter's status command.
pub struct StreamStats {
    endpoint: StreamEndpoint,
    connected: AtomicBool,
    to_peer: AtomicU64,
    from_peer: AtomicU64,
    dropped: AtomicU64,
    runts: AtomicU64,
    oversize: AtomicU64,
}

impl StreamStats {
    fn new(endpoint: StreamEndpoint) -> Self {
        Self {
            endpoint,
            connected: AtomicBool::new(false),
            to_peer: AtomicU64::new(0),
            from_peer: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            runts: AtomicU64::new(0),
            oversize: AtomicU64::new(0),
        }
    }
}

impl NetReport for StreamStats {
    fn report(&self, w: &mut dyn Write) -> io::Result<()> {
        writeln!(w, "Stream segment {}: {}", self.endpoint,
                 if self.connected.load(Ordering::Relaxed) { "connected" } else { "waiting for the peer" })?;
        writeln!(w, "  {} frames to the peer, {} from it, {} dropped while it was away, {} runts and {} oversize frames from it dropped",
                 self.to_peer.load(Ordering::Relaxed), self.from_peer.load(Ordering::Relaxed),
                 self.dropped.load(Ordering::Relaxed), self.runts.load(Ordering::Relaxed), self.oversize.load(Ordering::Relaxed))
    }
}

impl StreamBackend {
    /// `retry` is how long to wait between attempts to reach the peer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        endpoint: StreamEndpoint,
        tx_cons: rtrb::Consumer<Vec<u8>>,
        rx_prod: rtrb::Producer<Vec<u8>>,
        tx_wake: Arc<Wake>,
        rx_wake: Arc<Wake>,
        running: Arc<AtomicBool>,
        frames: Arc<AtomicU64>,
        retry: Duration,
    ) -> Self {
        let stats = Arc::new(StreamStats::new(endpoint));
        Self { tx_cons, rx_prod: Some(rx_prod), tx_wake, rx_wake, running, retry, stats, frames }
    }

    /// The counters and status report, which outlive the backend's thread.
    pub fn stats(&self) -> Arc<StreamStats> {
        self.stats.clone()
    }

    fn live(&self) -> bool {
        self.running.load(Ordering::SeqCst) && !self.tx_cons.is_abandoned()
    }

    /// Discard the frames the guest has queued, counting them.
    fn drain(&mut self) {
        while self.tx_cons.pop().is_ok() {
            self.frames.fetch_add(1, Ordering::Relaxed);
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Sleep until `until`, or until the backend stops.
    fn pause(&self, until: Instant) {
        while self.live() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(20).min(self.retry));
        }
    }

    /// Connect, retrying until it works. `None` if the backend stopped first.
    fn connect(&mut self) -> Option<Session> {
        let mut said = false;
        while self.live() {
            match self.stats.endpoint.connect() {
                Ok(c) => return Some(c),
                Err(e) => {
                    // Once per outage; the connection is logged when it comes.
                    if !std::mem::replace(&mut said, true) {
                        eprintln!("iris: network stream {}: {e}; retrying", self.stats.endpoint);
                    }
                    // Frames the guest sends meanwhile have nowhere to go.
                    self.drain();
                    self.pause(Instant::now() + self.retry);
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
        let mut said_oversize = false;
        loop {
            match read_frame_while(&mut rd, &|| !ended.load(Ordering::SeqCst)) {
                Ok(Incoming::Oversize(len)) => {
                    // The NICs cannot take it, and the next frame is intact.
                    stats.oversize.fetch_add(1, Ordering::Relaxed);
                    if !std::mem::replace(&mut said_oversize, true) {
                        crate::dlog!(LogModule::Net, "stream: dropped a {len} byte frame from the peer (more are counted in net status)");
                    }
                }
                Ok(Incoming::Frame(frame)) if frame.len() < MIN_FRAME => {
                    // Shorter than an Ethernet header is not a frame; the
                    // NICs would take it for one, so it never reaches them.
                    let n = stats.runts.fetch_add(1, Ordering::Relaxed) + 1;
                    if last_runt_log.map_or(true, |t| t.elapsed() >= Duration::from_secs(1)) {
                        last_runt_log = Some(Instant::now());
                        crate::dlog!(LogModule::Net, "stream: dropped a {} byte frame from the peer ({n} so far)", frame.len());
                    }
                }
                Ok(Incoming::Frame(mut frame)) => {
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
                Ok(Incoming::End) => {
                    eprintln!("iris: network stream closed by peer");
                    return rx_prod;
                }
                // The session ended while the read waited.
                Err(_) if ended.load(Ordering::SeqCst) => return rx_prod,
                Err(e) => {
                    eprintln!("iris: network stream read: {e}");
                    return rx_prod;
                }
            }
        }
    }
}

impl NetBackend for StreamBackend {
    fn run(&mut self) {
        while let Some((rd, mut wr, close)) = self.connect() {
            eprintln!("iris: network stream connected to {}", self.stats.endpoint);
            let started = Instant::now();
            // Frames queued during an outage are stale; the new peer never
            // saw the conversation they belonged to.
            self.drain();
            let prod = self.rx_prod.take().expect("receive producer");
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
                while let Ok(frame) = self.tx_cons.pop() {
                    self.frames.fetch_add(1, Ordering::Relaxed);
                    if let Err(e) = write_frame_while(&mut wr, &frame, &|| self.live()) {
                        eprintln!("iris: network stream write: {e}");
                        failed = true;
                        break;
                    }
                    self.stats.to_peer.fetch_add(1, Ordering::Relaxed);
                }
                if failed || rx.is_finished() || !self.live() {
                    break;
                }
            }
            ended.store(true, Ordering::SeqCst);
            ended_wake.notify();
            close();
            self.stats.connected.store(false, Ordering::Relaxed);
            self.rx_prod = Some(rx.join().expect("net-stream rx thread"));
            // A peer that accepts and hangs up must not be redialled in a
            // tight loop: wait out the rest of the retry interval.
            self.pause(started + self.retry);
        }
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
            (&[0u8, 0, 0x10, 0, 0][..], io::ErrorKind::UnexpectedEof),
            (&[0u8, 0, 0x06, 0xdc][..], io::ErrorKind::UnexpectedEof),
            (&[0u8, 1, 0, 1][..], io::ErrorKind::InvalidData),
            (&[0, 0][..], io::ErrorKind::UnexpectedEof),
            (&[0, 0, 0, 4, 9][..], io::ErrorKind::UnexpectedEof),
        ] {
            let mut r = bad;
            assert_eq!(read_frame(&mut r).unwrap_err().kind(), kind);
        }
    }

    /// A writer that takes one byte per write still gets whole frames, the
    /// empty one included.
    #[test]
    fn a_frame_survives_short_writes() {
        struct OneByte(Vec<u8>);
        impl Write for OneByte {
            fn write(&mut self, b: &[u8]) -> io::Result<usize> {
                self.0.extend_from_slice(&b[..b.len().min(1)]);
                Ok(b.len().min(1))
            }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }
        let mut w = OneByte(Vec::new());
        write_frame(&mut w, &[7, 8, 9]).unwrap();
        write_frame(&mut w, &[]).unwrap();
        assert_eq!(w.0, [0, 0, 0, 3, 7, 8, 9, 0, 0, 0, 0]);
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
        stats: Arc<StreamStats>,
        /// The backend while it is not running; a run takes it away on its
        /// thread and the thread hands it back.
        idle: Option<StreamBackend>,
        thread: Option<std::thread::JoinHandle<StreamBackend>>,
    }

    fn rig(endpoint: StreamEndpoint, retry: Duration) -> Rig {
        let (to_net, tx_cons) = rtrb::RingBuffer::<Vec<u8>>::new(8);
        let (rx_prod, from_net) = rtrb::RingBuffer::<Vec<u8>>::new(8);
        let (tx_wake, rx_wake) = (Wake::new(), Wake::new());
        let running = Arc::new(AtomicBool::new(true));
        let frames = Arc::new(AtomicU64::new(0));
        let b = StreamBackend::new(endpoint, tx_cons, rx_prod, tx_wake.clone(), rx_wake.clone(), running.clone(), frames.clone(), retry);
        let mut r = Rig { to_net, from_net, tx_wake, rx_wake, running, frames, stats: b.stats(), idle: Some(b), thread: None };
        r.start();
        r
    }

    impl Rig {
        fn start(&mut self) {
            self.running.store(true, Ordering::SeqCst);
            let mut backend = self.idle.take().expect("the backend is not running");
            self.thread = Some(std::thread::spawn(move || {
                backend.run();
                backend
            }));
        }

        fn stop(&mut self) {
            self.running.store(false, Ordering::SeqCst);
            self.idle = Some(self.thread.take().unwrap().join().unwrap());
        }

        /// One frame from the guest to `peer`, and one back.
        fn exchange(&mut self, peer: &mut impl ReadWrite, out: u8, back: u8) {
            // A frame queued before the session is up is stale and discarded.
            let deadline = Instant::now() + Duration::from_secs(10);
            while !self.stats.connected.load(Ordering::Relaxed) {
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
        while r.stats.to_peer.load(Ordering::Relaxed) < 1 {
            assert!(Instant::now() < deadline, "the frame sent to the peer is never counted");
            std::thread::sleep(Duration::from_millis(2));
        }
        let mut text = Vec::new();
        r.stats.report(&mut text).unwrap();
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
        assert_eq!(r.stats.runts.load(Ordering::Relaxed), 3);
        r.stop();
    }

    /// A frame longer than an Ethernet frame is skipped, not the end of the
    /// session: the frame after it still arrives.
    #[test]
    fn an_oversize_frame_from_the_peer_is_skipped() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut r = rig(StreamEndpoint::Tcp(listener.local_addr().unwrap()), RECONNECT);
        let (mut peer, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        write_frame(&mut peer, &[0x11; 2000]).unwrap();
        write_frame(&mut peer, &[0x22; 60]).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let got = loop {
            if let Ok(f) = r.from_net.pop() {
                break f;
            }
            assert!(Instant::now() < deadline, "no frame after the oversize one");
            r.rx_wake.wait(Duration::from_millis(10));
        };
        assert_eq!(got, vec![0x22; 60]);
        assert!(r.stats.connected.load(Ordering::Relaxed), "the session ended");
        assert_eq!(r.stats.oversize.load(Ordering::Relaxed), 1);
        r.stop();
    }

    /// A reader that polls instead of using a receive timeout keeps a frame
    /// whole across a pause longer than its tick, and reports a quiet socket
    /// as a timeout.
    #[test]
    fn a_polled_reader_keeps_a_frame_across_a_pause() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut peer, _) = listener.accept().unwrap();
        let mut rd = PolledRead::new(client).unwrap();
        assert_eq!(rd.read(&mut [0u8; 4]).unwrap_err().kind(), io::ErrorKind::TimedOut);
        let mut buf = Vec::new();
        write_frame(&mut buf, &[0x3c; 60]).unwrap();
        let (head, tail) = (buf[..10].to_vec(), buf[10..].to_vec());
        let writer = std::thread::spawn(move || {
            peer.write_all(&head).unwrap();
            std::thread::sleep(IO_TICK * 2);
            peer.write_all(&tail).unwrap();
            peer
        });
        match read_frame_while(&mut rd, &|| true).unwrap() {
            Incoming::Frame(f) => assert_eq!(f, vec![0x3c; 60]),
            _ => panic!("no frame"),
        }
        let peer = writer.join().unwrap();
        drop(peer);
        assert!(matches!(read_frame_while(&mut rd, &|| true).unwrap(), Incoming::End));
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
            while r.stats.connected.load(Ordering::Relaxed) != want {
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
        // More than the 8-frame ring went in, so the socket took frames until it filled.
        assert!(queued > 64, "only {queued} frames went in before the backend stalled");
        let t = Instant::now();
        r.stop();
        assert!(t.elapsed() < Duration::from_secs(2), "stop took {:?}", t.elapsed());
    }

    /// The receive thread keeps a frame whole across its read timeouts, and
    /// ends when the session does even mid-frame and with no shutdown: Windows
    /// does not let a shutdown end a blocked read, as Linux does.
    #[test]
    fn the_reader_ends_with_its_session_without_a_shutdown() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let (rd, _wr, _close) = StreamEndpoint::Tcp(listener.local_addr().unwrap()).connect().unwrap();
        let (mut peer, _) = listener.accept().unwrap();
        let (prod, mut cons) = rtrb::RingBuffer::<Vec<u8>>::new(8);
        let ended = Arc::new(AtomicBool::new(false));
        let (rx_ended, rx_wake) = (ended.clone(), Wake::new());
        let rx = std::thread::spawn(move || {
            StreamBackend::receive(rd, prod, rx_wake, rx_ended, Wake::new(), Arc::new(StreamStats::new(StreamEndpoint::Tcp("127.0.0.1:1".parse().unwrap()))))
        });
        // A frame split by a pause longer than a read timeout.
        let mut buf = Vec::new();
        write_frame(&mut buf, &[0x3c; 60]).unwrap();
        peer.write_all(&buf[..10]).unwrap();
        std::thread::sleep(IO_TICK * 2);
        peer.write_all(&buf[10..]).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let got = loop {
            if let Ok(f) = cons.pop() {
                break f;
            }
            assert!(Instant::now() < deadline, "the split frame never arrived");
            std::thread::sleep(Duration::from_millis(2));
        };
        assert_eq!(got, vec![0x3c; 60]);
        // Half a frame, then the session ends while the reader waits for the rest.
        peer.write_all(&buf[..10]).unwrap();
        std::thread::sleep(IO_TICK / 2);
        ended.store(true, Ordering::SeqCst);
        let t = Instant::now();
        while !rx.is_finished() {
            assert!(t.elapsed() < Duration::from_secs(2), "the reader is still blocked");
            std::thread::sleep(Duration::from_millis(2));
        }
        rx.join().unwrap();
        assert!(cons.pop().is_err(), "half a frame reached the ring");
    }
}
