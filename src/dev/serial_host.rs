//! The host end of a serial line: where a guest UART's bytes go and where
//! its input comes from.
//!
//! A [`SerialBackend`] is a TCP console, a Unix-socket console, the in-process CI backend, a file
//! tee, or nothing. Every machine's UARTs (the Indy's and Indigo2's Z85C30,
//! the O2's MACE 16550s, the IOC3's and IOC4's ports, the L1 console) take
//! one, and [`console_backends`] picks them for the machines whose consoles
//! are configured the same way.
//!
//! A backend never refuses a byte and never makes a UART wait. The log tee
//! and the CI capture take every byte first; each socket client then has its
//! own [`ClientQueue`], which keeps what it has not read, drops the oldest
//! past [`CLIENT_BACKLOG`], and is written out by [`SerialBackend::flush`]
//! from the UART's existing poll. A backend never calls back into the UART.

use std::fs::OpenOptions;
use std::collections::VecDeque;
use std::io::{self, Read, Write};
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use polling::AsSource;

use crate::config::MachineConfig;
use crate::dev::serial_log::SerialLog;
use crate::hptimer::TimerManager;
use crate::net::telnet::IAC;
use crate::net::wake::{host_io, Watched};

/// Parity of a serial line's characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Parity {
    #[default]
    None,
    Odd,
    Even,
}

/// How the guest has framed and clocked a serial line, as its UART registers say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineConfig {
    /// Bits per second, or `None` when the UART is clocked from a pin whose
    /// rate the emulator cannot see (an external clock, as a Macintosh MIDI
    /// interface supplies).
    pub baud: Option<u32>,
    /// 5 to 8.
    pub data_bits: u8,
    pub parity: Parity,
    /// 1 or 2 (1.5 is reported as 2).
    pub stop_bits: u8,
}

impl Default for LineConfig {
    fn default() -> Self {
        Self { baud: Some(9600), data_bits: 8, parity: Parity::None, stop_bits: 1 }
    }
}

impl std::fmt::Display for LineConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.baud {
            Some(b) => write!(f, "{b}")?,
            None => write!(f, "ext-clock")?,
        }
        let p = match self.parity {
            Parity::None => 'N',
            Parity::Odd => 'O',
            Parity::Even => 'E',
        };
        write!(f, " {}{}{}", self.data_bits, p, self.stop_bits)
    }
}

/// The modem-control outputs the guest drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModemOut {
    pub dtr: bool,
    pub rts: bool,
    /// Sending a break (the line held at space).
    pub brk: bool,
}

/// The modem-status inputs the guest reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModemIn {
    pub cts: bool,
    pub dcd: bool,
    pub dsr: bool,
    pub ri: bool,
}

impl ModemIn {
    /// What a line with nothing that drives the inputs shows: a socket or a
    /// log is always "connected and clear to send".
    pub const IDLE: ModemIn = ModemIn { cts: true, dcd: true, dsr: true, ri: false };
}

pub trait SerialBackend: Send + Sync {
    /// Send a byte. Never blocks and never refuses.
    fn send_byte(&self, byte: u8);
    /// The guest changed the line's rate or framing. A backend wired to a
    /// real port programs the port to match; others ignore it.
    fn set_line(&self, _line: LineConfig) {}
    /// The guest changed DTR, RTS or break.
    fn set_modem(&self, _out: ModemOut) {}
    /// The modem-status inputs, polled from the UART's receive loop.
    fn modem_in(&self) -> ModemIn {
        ModemIn::IDLE
    }
    /// The machine's timers, for a backend that polls the host on the machine clock.
    fn bind_timers(&self, _tm: &Arc<TimerManager>) {}
    /// True for a line that carries a device's own bytes (a host serial port,
    /// an emulated peripheral), which the UART must pass through untouched.
    /// False for a console a person types at, where the UART may apply its
    /// console conveniences.
    fn raw(&self) -> bool {
        false
    }
    /// What the host end is, for the monitor's `serial`: where it listens and
    /// who is connected, or the device and the line it is set to.
    fn describe(&self) -> String {
        "a host end".into()
    }
    /// Write out what socket clients have queued, one write each. The UART's
    /// receive thread or service loop calls it every poll.
    fn flush(&self) {}
    /// True while a flush has output left to write, so a port can skip it when idle.
    fn queued(&self) -> bool {
        false
    }
    /// Bytes dropped in all for clients that did not read, for the monitor.
    fn dropped(&self) -> u64 {
        0
    }
    fn recv_byte(&self) -> io::Result<u8>;
}

/// The most output a socket client keeps unread before the oldest goes.
pub const CLIENT_BACKLOG: usize = 1 << 20;

/// Output a socket client has not read yet. Past [`CLIENT_BACKLOG`] the oldest whole units
/// go and are counted, and a telnet client is told how many.
pub(crate) struct ClientQueue {
    buf: VecDeque<u8>,
    telnet: bool,
    /// Leading bytes left of a unit the socket took only part of.
    keep: usize,
    /// Dropped since the client was last told.
    gap: u64,
    /// The notice being written.
    note: Vec<u8>,
    total: u64,
}

impl ClientQueue {
    pub(crate) fn new(telnet: bool) -> Self {
        Self { buf: VecDeque::new(), telnet, keep: 0, gap: 0, note: Vec::new(), total: 0 }
    }

    /// The length of the unit at `i`: a data byte, an escaped 0xff or a
    /// three-byte telnet reply.
    fn unit(&self, i: usize) -> usize {
        match (self.telnet, self.buf[i], self.buf.get(i + 1)) {
            (true, IAC, Some(251..=254)) => 3,
            (true, IAC, _) => 2,
            _ => 1,
        }
    }

    pub(crate) fn push(&mut self, unit: &[u8]) {
        self.buf.extend(unit);
        while self.buf.len() > CLIENT_BACKLOG {
            let n = self.unit(self.keep).min(self.buf.len() - self.keep);
            self.buf.drain(self.keep..self.keep + n);
            if self.gap == 0 && self.note.is_empty() {
                log::warn!("serial: a client is not reading; dropping its oldest output");
            }
            self.gap += n as u64;
            self.total += n as u64;
        }
    }

    /// One pass of non-blocking writes: a split unit's rest, a dropped-bytes notice, the queue.
    /// An error means the client has gone.
    pub(crate) fn flush(&mut self, w: &mut impl Write) -> io::Result<()> {
        if self.keep > 0 {
            let n = {
                let head = self.buf.as_slices().0;
                write_some(w, &head[..self.keep.min(head.len())])?
            };
            self.buf.drain(..n);
            self.keep -= n;
            if self.keep > 0 {
                return Ok(());
            }
        }
        if self.gap > 0 && self.note.is_empty() {
            if self.telnet {
                // Until the socket takes some of it, later drops join the same count.
                let note = format!("\r\n[iris: {} bytes dropped]\r\n", self.gap).into_bytes();
                let n = write_some(w, &note)?;
                if n == 0 {
                    return Ok(());
                }
                self.note = note[n..].to_vec();
            }
            self.gap = 0;
        }
        if !self.note.is_empty() {
            let n = write_some(w, &self.note)?;
            self.note.drain(..n);
            if !self.note.is_empty() {
                return Ok(());
            }
        }
        let n = write_some(w, self.buf.as_slices().0)?;
        let mut at = 0;
        while at < n {
            at += self.unit(at);
        }
        self.buf.drain(..n);
        self.keep = at - n;
        Ok(())
    }

    pub(crate) fn dropped(&self) -> u64 {
        self.total
    }

    /// True while anything is left to write.
    fn queued(&self) -> bool {
        !self.buf.is_empty() || !self.note.is_empty() || self.gap > 0
    }
}

/// One non-blocking write: how much the socket took, 0 when it is full.
fn write_some(w: &mut impl Write, bytes: &[u8]) -> io::Result<usize> {
    if bytes.is_empty() {
        return Ok(0);
    }
    match w.write(bytes) {
        Ok(0) => Err(io::ErrorKind::WriteZero.into()),
        Ok(n) => Ok(n),
        Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted) => Ok(0),
        Err(e) => Err(e),
    }
}

/// Drops TX bytes and never yields RX. Used as a placeholder when a channel
/// isn't wired to a host I/O source (e.g. CI mode unused channel).
pub(crate) struct NullBackend;

impl SerialBackend for NullBackend {
    fn send_byte(&self, _byte: u8) {}
    fn describe(&self) -> String {
        "nothing".into()
    }
    fn recv_byte(&self) -> io::Result<u8> {
        Err(io::Error::new(io::ErrorKind::WouldBlock, "null"))
    }
}

/// Tees guest-emitted bytes to a host file in addition to delegating to an
/// inner backend. Used by `serial_log` (`--serial-log <file>`) to capture the
/// console transcript while the console's own listener keeps working
/// unchanged.
pub struct TeeBackend {
    inner: Arc<dyn SerialBackend>,
    log: Arc<SerialLog>,
}

impl TeeBackend {
    pub fn new(inner: Arc<dyn SerialBackend>, log: Arc<SerialLog>) -> Self {
        Self { inner, log }
    }
}

impl SerialBackend for TeeBackend {
    fn send_byte(&self, byte: u8) {
        self.log.write(&[byte]);
        self.inner.send_byte(byte);
    }
    fn set_line(&self, line: LineConfig) {
        self.inner.set_line(line);
    }
    fn set_modem(&self, out: ModemOut) {
        self.inner.set_modem(out);
    }
    fn modem_in(&self) -> ModemIn {
        self.inner.modem_in()
    }
    fn bind_timers(&self, tm: &Arc<TimerManager>) {
        self.inner.bind_timers(tm);
    }
    fn raw(&self) -> bool {
        self.inner.raw()
    }
    fn describe(&self) -> String {
        format!("{}; log {}", self.inner.describe(), self.log.name())
    }
    fn flush(&self) {
        self.inner.flush();
    }
    fn queued(&self) -> bool {
        self.inner.queued()
    }
    fn dropped(&self) -> u64 {
        self.inner.dropped()
    }
    fn recv_byte(&self) -> io::Result<u8> {
        self.inner.recv_byte()
    }
}

/// Bind a Unix socket at `path`. A socket file left by an earlier iris is
/// replaced; one that something still answers on, or any other kind of
/// file, is an error and left alone.
#[cfg(unix)]
pub(crate) fn bind_unix_socket(path: &Path) -> io::Result<UnixListener> {
    use std::os::unix::fs::FileTypeExt;
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_socket() => {
            if UnixStream::connect(path).is_ok() {
                return Err(io::Error::new(io::ErrorKind::AddrInUse, "another process is listening on it"));
            }
            std::fs::remove_file(path)?;
        }
        Ok(_) => return Err(io::Error::new(io::ErrorKind::AlreadyExists, "exists and is not a socket")),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    UnixListener::bind(path)
}


/// A listening socket and what its clients are.
pub(crate) trait Listen: Send + Sync {
    type Stream: Read + Write + Send + AsSource;
    /// Whether clients speak telnet (offered character mode, output escaped)
    /// or raw bytes.
    const TELNET: bool;
    /// Clients served at once; others wait in the listen queue.
    const MAX: usize;
    /// Take one waiting client, non-blocking.
    fn accept(&self) -> io::Result<Self::Stream>;
    /// Where it listens, for the monitor.
    fn describe_addr(&self) -> String;
}

impl Listen for Watched<TcpListener> {
    type Stream = TcpStream;
    const TELNET: bool = true;
    const MAX: usize = usize::MAX;

    fn accept(&self) -> io::Result<TcpStream> {
        let (socket, _) = TcpListener::accept(self)?;
        socket.set_nonblocking(true)?;
        // The console is latency-sensitive: without TCP_NODELAY, Nagle
        // holds a keystroke or a log line back until the previous
        // segment is acked, which shows up as a stutter on every
        // interaction. Set it on accept, before any byte is written.
        let _ = socket.set_nodelay(true);
        Ok(socket)
    }

    fn describe_addr(&self) -> String {
        self.local_addr().map_or_else(|_| "tcp ?".into(), |a| format!("tcp {a}"))
    }
}

/// A Unix socket's listener and the file it removes when it goes.
#[cfg(unix)]
pub(crate) struct UnixLine {
    path: std::path::PathBuf,
    listener: Watched<UnixListener>,
}

#[cfg(unix)]
impl Listen for UnixLine {
    type Stream = UnixStream;
    const TELNET: bool = false;
    const MAX: usize = 1;

    fn accept(&self) -> io::Result<UnixStream> {
        let (socket, _) = self.listener.accept()?;
        socket.set_nonblocking(true)?;
        Ok(socket)
    }

    fn describe_addr(&self) -> String {
        format!("unix {}", self.path.display())
    }
}

#[cfg(unix)]
impl Drop for UnixLine {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        crate::workstation::keep_at_exit(&self.path);
    }
}

/// One socket client.
struct Conn<S: AsSource> {
    /// Joined to the host-IO thread's poller, so input wakes it.
    stream: Watched<S>,
    /// Set for a telnet client.
    telnet: Option<crate::net::telnet::TelnetFilter>,
    /// Output and telnet replies, in order, that the socket has not taken.
    queue: ClientQueue,
}

impl<S: AsSource + Write> Conn<S> {
    /// One flush pass; while output is left, the host-IO thread retries when the socket can take more.
    fn flush(&mut self) -> io::Result<()> {
        let r = self.queue.flush(&mut self.stream);
        self.stream.set_interest(true, self.queue.queued());
        r
    }
}

/// A serial line served on a listening socket. Every connected client (up to
/// `L::MAX`) sees all the output and may type, so a person at the console and
/// a script can share it.
pub(crate) struct SocketBackend<L: Listen> {
    /// The socket's entry in `ports`.
    _port: Option<crate::ports::PortGuard>,
    pub(crate) listener: L,
    conns: Mutex<Vec<Conn<L::Stream>>>,
    /// Bytes dropped for clients that have since gone.
    dropped: AtomicU64,
}

pub(crate) type TcpSocketBackend = SocketBackend<Watched<TcpListener>>;
/// A Unix socket line: raw bytes, no telnet, one client at a time (the next
/// waits in the listen queue until the first hangs up), for scripts.
#[cfg(unix)]
pub(crate) type UnixSocketBackend = SocketBackend<UnixLine>;

/// Several host ends of one serial line (a TCP port and a Unix socket):
/// the guest's output goes to all of them and its input comes from
/// whichever has some.
struct FanoutBackend(Vec<Arc<dyn SerialBackend>>);

impl SerialBackend for FanoutBackend {
    fn send_byte(&self, byte: u8) {
        for b in &self.0 {
            b.send_byte(byte);
        }
    }
    fn set_line(&self, line: LineConfig) {
        for b in &self.0 {
            b.set_line(line);
        }
    }
    fn set_modem(&self, out: ModemOut) {
        for b in &self.0 {
            b.set_modem(out);
        }
    }
    fn describe(&self) -> String {
        self.0.iter().map(|b| b.describe()).collect::<Vec<_>>().join(" + ")
    }
    fn flush(&self) {
        for b in &self.0 {
            b.flush();
        }
    }
    fn queued(&self) -> bool {
        self.0.iter().any(|b| b.queued())
    }
    fn dropped(&self) -> u64 {
        self.0.iter().map(|b| b.dropped()).sum()
    }
    fn recv_byte(&self) -> io::Result<u8> {
        for b in &self.0 {
            if let Ok(byte) = b.recv_byte() {
                return Ok(byte);
            }
        }
        Err(io::Error::new(io::ErrorKind::WouldBlock, "WouldBlock"))
    }
}

impl<L: Listen> SocketBackend<L> {
    fn with(listener: L, port: Option<crate::ports::PortGuard>) -> Self {
        Self { _port: port, listener, conns: Mutex::new(Vec::new()), dropped: AtomicU64::new(0) }
    }

    /// Take any clients waiting to connect, offering telnet ones character
    /// mode. One whose handshake cannot be written is dropped; it can
    /// reconnect.
    fn accept_waiting(&self, conns: &mut Vec<Conn<L::Stream>>) {
        while conns.len() < L::MAX {
            let Ok(stream) = self.listener.accept() else { break };
            let Ok(stream) = Watched::new(stream, &host_io().wake, true, false) else { continue };
            let mut c = Conn {
                stream,
                telnet: L::TELNET.then(crate::net::telnet::TelnetFilter::new),
                queue: ClientQueue::new(L::TELNET),
            };
            if !L::TELNET || c.stream.write_all(&crate::net::telnet::TelnetFilter::initial_handshake()).is_ok() {
                conns.push(c);
            }
        }
    }

    /// Write out what every client has queued; one that has gone is dropped.
    fn flush_all(&self, conns: &mut Vec<Conn<L::Stream>>) {
        conns.retain_mut(|c| {
            let kept = c.flush().is_ok();
            if !kept {
                self.dropped.fetch_add(c.queue.dropped(), Ordering::Relaxed);
            }
            kept
        });
    }
}

impl SocketBackend<Watched<TcpListener>> {
    /// Bind a TCP serial backend. Returns `None` (rather than panicking) if the
    /// port can't be bound — most commonly because a stale emulator process from
    /// an earlier crash is still holding it. Callers fall back to a NullBackend
    /// so the machine still boots; that serial channel is simply unavailable
    /// until the port frees.
    pub(crate) fn new<A: std::net::ToSocketAddrs>(addr: A) -> Option<Self> {
        let listener = match TcpListener::bind(addr) {
            Ok(l) => l,
            Err(e) => {
                log::warn!("serial TCP backend disabled: failed to bind socket: {e}");
                return None;
            }
        };
        if let Err(e) = listener.set_nonblocking(true) {
            log::warn!("serial TCP backend disabled: failed to set nonblocking: {e}");
            return None;
        }
        let port = listener.local_addr().map(|a| crate::ports::register("tcp", a, "serial port")).ok();
        // Joined to the host-IO thread's poller, so a connecting client wakes it.
        let listener = Watched::new(listener, &host_io().wake, true, false)
            .inspect_err(|e| log::warn!("serial TCP backend disabled: {e}"))
            .ok()?;
        Some(Self::with(listener, port))
    }
}

#[cfg(unix)]
impl SocketBackend<UnixLine> {
    /// Bind `path`, as `bind_unix_socket` does.
    pub(crate) fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let listener = bind_unix_socket(&path)?;
        listener.set_nonblocking(true)?;
        let listener = Watched::new(listener, &host_io().wake, true, false)?;
        crate::workstation::remove_at_exit(&path);
        let port = crate::ports::register("unix", path.display(), "serial port");
        Ok(Self::with(UnixLine { path, listener }, Some(port)))
    }
}

impl<L: Listen> SerialBackend for SocketBackend<L> {
    /// With no client the byte goes nowhere: nobody waits for it.
    fn send_byte(&self, byte: u8) {
        // A telnet client gets 0xff doubled.
        let pair = [IAC, byte];
        let unit = if L::TELNET && byte == IAC { &pair[..] } else { &pair[1..] };
        for c in self.conns.lock().iter_mut() {
            if !c.queue.queued() {
                host_io().wake.notify(); // the host-IO thread writes it out
            }
            c.queue.push(unit);
        }
    }

    fn flush(&self) {
        self.flush_all(&mut self.conns.lock());
    }

    fn queued(&self) -> bool {
        self.conns.lock().iter().any(|c| c.queue.queued())
    }

    fn dropped(&self) -> u64 {
        let live: u64 = self.conns.lock().iter().map(|c| c.queue.dropped()).sum();
        self.dropped.load(Ordering::Relaxed) + live
    }

    fn describe(&self) -> String {
        let n = self.conns.lock().len();
        format!("{}, {n} client{}", self.listener.describe_addr(), if n == 1 { "" } else { "s" })
    }

    fn recv_byte(&self) -> io::Result<u8> {
        let mut conns = self.conns.lock();
        self.accept_waiting(&mut conns);
        let mut lost = false;
        // Each client in turn: pull bytes through its telnet filter until a
        // real data byte falls out or it has nothing more to give. A client
        // that has gone away is dropped.
        let mut i = 0;
        while i < conns.len() {
            let mut gone = false;
            loop {
                let c = &mut conns[i];
                let mut buf = [0u8; 1];
                match c.stream.read(&mut buf) {
                    Ok(1) => {
                        let data = match c.telnet.as_mut() {
                            Some(t) => {
                                let mut reply = Vec::new();
                                let data = t.feed(buf[0], &mut reply);
                                // Behind queued output, so a reply never lands
                                // inside an escaped data byte.
                                if !reply.is_empty() {
                                    c.queue.push(&reply);
                                    // Nothing else flushes it while the guest is idle.
                                    let _ = c.flush();
                                }
                                data
                            }
                            None => Some(buf[0]),
                        };
                        if let Some(d) = data {
                            return Ok(d);
                        }
                        // Telnet command consumed; keep reading.
                    }
                    Ok(_) => {
                        gone = true;
                        break;
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(_) => {
                        gone = true;
                        break;
                    }
                }
            }
            if gone {
                let c = conns.remove(i);
                self.dropped.fetch_add(c.queue.dropped(), Ordering::Relaxed);
                lost = true;
            } else {
                i += 1;
            }
        }
        let kind = if lost { io::ErrorKind::NotConnected } else { io::ErrorKind::WouldBlock };
        Err(kind.into())
    }
}

/// A TCP serial console on 127.0.0.1:`port` (0 takes any free port), in
/// telnet character mode, for any device with a serial line: every client
/// connected sees the output and may type. Returns the backend and the port
/// it bound, or `None` when the port is taken.
pub fn tcp_serial_backend(port: u16) -> Option<(Arc<dyn SerialBackend>, u16)> {
    let b = TcpSocketBackend::new(("127.0.0.1", port))?;
    let bound = b.listener.local_addr().ok()?.port();
    Some((Arc::new(b), bound))
}

/// A serial line wired to nothing: output is dropped, nothing arrives.
pub fn null_serial_backend() -> Arc<dyn SerialBackend> {
    Arc::new(NullBackend)
}

// ============================================================================
// CiSerialBackend — in-process serial backend used by --ci mode.
// ============================================================================

/// Serial backend that the CI control socket reads from and writes to. The
/// guest sees this as channel A (the IRIX console). Host pushes bytes into
/// `host_to_guest` via `push_host`; the existing RX thread drains them into
/// `channel_a.rx_queue`. Guest output reaches `send_byte`, which pushes into
/// `guest_to_host` and wakes anyone waiting in `wait_for`.
pub struct CiSerialBackend {
    host_to_guest: Mutex<VecDeque<u8>>,
    guest_to_host: Mutex<Vec<u8>>,
    /// Counts the times the buffer was emptied (a drain, a reset or a wait
    /// taking its match), changed under the buffer's lock. A `wait_for` that
    /// sees it change searches from the start again, as what it had searched
    /// is gone.
    drains: AtomicU64,
    cv: Condvar,
    /// Optional file mirror: every byte that flows through ttyd1 in either
    /// direction is appended here. Host-injected bytes are echoed by IRIX
    /// anyway, so writing only guest output already captures the full
    /// transcript without duplication.
    log: Mutex<Option<Arc<SerialLog>>>,
}

impl CiSerialBackend {
    pub fn new() -> Self {
        Self {
            host_to_guest: Mutex::new(VecDeque::new()),
            guest_to_host: Mutex::new(Vec::new()),
            drains: AtomicU64::new(0),
            cv: Condvar::new(),
            log: Mutex::new(None),
        }
    }

    /// Attach the shared serial-log sink. The caller opens the file (and
    /// picks the timestamp mode), so the CI and TCP paths write the same bytes.
    pub fn set_log(&self, log: Arc<SerialLog>) {
        *self.log.lock() = Some(log);
    }

    /// Inject bytes from host to guest (the harness typing on the console).
    pub fn push_host(&self, data: &[u8]) {
        let mut q = self.host_to_guest.lock();
        q.extend(data.iter().copied());
        host_io().wake.notify();
    }

    /// Drain everything the guest has produced since the last call. Empties
    /// the buffer; the returned Vec is the guest output as raw bytes.
    pub fn drain_guest(&self) -> Vec<u8> {
        let mut q = self.guest_to_host.lock();
        self.drains.fetch_add(1, Ordering::Relaxed);
        std::mem::take(&mut *q)
    }

    /// Block until `needle` is seen in guest output, or `timeout` expires.
    /// On success returns the consumed bytes up to and including the match;
    /// bytes that arrived after the match stay in the buffer for the next
    /// `serial-read`. On timeout returns `None` without consuming anything.
    pub fn wait_for(&self, needle: &[u8], timeout: Duration) -> Option<Vec<u8>> {
        if needle.is_empty() {
            return Some(Vec::new());
        }
        let deadline = Instant::now() + timeout;
        let mut q = self.guest_to_host.lock();
        // How much of the buffer earlier passes searched, so a pass after a
        // wakeup looks at the new bytes only. The guest's transmit thread
        // waits on the lock this holds. A match can straddle the old end,
        // hence the needle's length less one is searched again.
        let mut scanned = 0usize;
        let mut seen = self.drains.load(Ordering::Relaxed);
        let mut last = false;
        loop {
            // Another reader emptied the buffer under us, and it may have
            // filled again past `scanned`: start over.
            let drains = self.drains.load(Ordering::Relaxed);
            if drains != seen {
                (scanned, seen) = (0, drains);
            }
            let from = scanned.saturating_sub(needle.len() - 1);
            if let Some(pos) = find_subseq(&q[from..], needle) {
                let end = from + pos + needle.len();
                let consumed: Vec<u8> = q.drain(..end).collect();
                self.drains.fetch_add(1, Ordering::Relaxed);
                return Some(consumed);
            }
            // One more pass after the wait times out, in case bytes arrived
            // between the last pass and the timeout.
            if last || Instant::now() >= deadline {
                return None;
            }
            scanned = q.len();
            last = self.cv.wait_until(&mut q, deadline).timed_out();
        }
    }

    /// Clear both queues. Called on `restore`/`rollback` so stale serial
    /// output from the previous run doesn't leak into the next test.
    pub fn reset(&self) {
        self.host_to_guest.lock().clear();
        let mut q = self.guest_to_host.lock();
        self.drains.fetch_add(1, Ordering::Relaxed);
        q.clear();
    }
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

impl SerialBackend for CiSerialBackend {
    fn describe(&self) -> String {
        match self.log.lock().as_ref() {
            Some(log) => format!("CI control socket; log {}", log.name()),
            None => "CI control socket".into(),
        }
    }
    fn send_byte(&self, byte: u8) {
        self.guest_to_host.lock().push(byte);
        if let Some(log) = self.log.lock().as_ref() {
            log.write(&[byte]);
        }
        self.cv.notify_all();
    }

    fn recv_byte(&self) -> io::Result<u8> {
        let mut q = self.host_to_guest.lock();
        match q.pop_front() {
            Some(b) => Ok(b),
            None => Err(io::Error::new(io::ErrorKind::WouldBlock, "empty")),
        }
    }
}

/// Open `cfg.serial_log`, if one is set, as the sink the machine's CI and
/// TCP consoles both write through, so the two logs are the same bytes
/// (timestamps included). A file that cannot be opened is reported and the
/// machine boots without a log. `tag` names the machine in the messages. An
/// existing file is added to, or started over with `serial_log_overwrite`.
pub fn open_log(cfg: &MachineConfig, tag: &str) -> Option<Arc<SerialLog>> {
    open_log_at(cfg, cfg.serial_log.as_deref(), tag)
}

/// As [`open_log`] for the transcript at `path` (a line's own log, when it
/// is not the console's), with `cfg`'s timestamp setting.
pub fn open_log_at(cfg: &MachineConfig, path: Option<&str>, tag: &str) -> Option<Arc<SerialLog>> {
    let path = path?;
    let p = Path::new(path);
    let opened = if path == "-" {
        Ok(SerialLog::stdout(cfg.serial_log_timestamps))
    } else {
        if cfg.serial_log_overwrite {
            SerialLog::create(p, cfg.serial_log_timestamps)
        } else {
            SerialLog::open(p, cfg.serial_log_timestamps)
        }
    };
    match opened {
        Ok(log) => {
            eprintln!("iris: {tag} serial console mirroring to {path}");
            Some(log)
        }
        Err(e) => {
            eprintln!("iris: {tag} serial log {path}: {e}");
            None
        }
    }
}

/// A serial console on a Unix socket at `path`; see `UnixSocketBackend`.
/// Returns `None` when the socket cannot be made, with the reason on stderr,
/// as a TCP port that is taken does.
#[cfg(unix)]
pub fn unix_serial_backend(path: &str, label: &str, setting: &str) -> Option<Arc<dyn SerialBackend>> {
    match UnixSocketBackend::new(path) {
        Ok(b) => {
            eprintln!("iris: {label} on socket {path}");
            Some(Arc::new(b))
        }
        Err(e) => {
            eprintln!("iris: {label}: socket {path}: {e} ({setting})");
            None
        }
    }
}

#[cfg(not(unix))]
pub fn unix_serial_backend(path: &str, label: &str, setting: &str) -> Option<Arc<dyn SerialBackend>> {
    eprintln!("iris: {label}: socket {path}: Unix sockets need a Unix host ({setting})");
    None
}

/// Where one serial line is served on the host.
pub struct SerialLine<'a> {
    /// Names the line in messages, e.g. "Octane serial port 1".
    pub label: &'a str,
    /// The TCP port, or `None` for a line with no listener.
    pub tcp_port: Option<u16>,
    /// The setting that chose `tcp_port`, for the message when it is taken.
    pub tcp_setting: &'a str,
    /// A Unix socket to serve the line on as well, from the `socket_setting`
    /// key.
    pub socket: Option<&'a str>,
    pub socket_setting: &'a str,
}

/// Wire one serial line: TCP on `tcp_port` plus the Unix socket when one is
/// set, teed to `log`. The line that carries the CI control socket's console
/// (`ci`) is not served on TCP (so several `--ci` instances coexist); with a
/// socket, its output goes to both the control socket's buffer and the
/// socket, and either may type. Returns the backend, and whether a host end
/// could be made; without one the backend only logs.
pub fn serial_line(
    line: &SerialLine,
    log: Option<Arc<SerialLog>>,
    ci: Option<&Arc<CiSerialBackend>>,
) -> (Arc<dyn SerialBackend>, bool) {
    let socket = line.socket.and_then(|path| unix_serial_backend(path, line.label, line.socket_setting));
    if let Some(ci) = ci {
        if let Some(log) = log {
            ci.set_log(log);
        }
        return match socket {
            Some(socket) => (Arc::new(FanoutBackend(vec![ci.clone(), socket])), true),
            None => (ci.clone(), true),
        };
    }
    let mut ends: Vec<Arc<dyn SerialBackend>> = Vec::new();
    if let Some(port) = line.tcp_port {
        match tcp_serial_backend(port) {
            Some((b, p)) => {
                eprintln!("iris: {} on 127.0.0.1:{p}", line.label);
                ends.push(b);
            }
            None => eprintln!(
                "iris: {}: port {port} unavailable; is another iris using it? ({})",
                line.label, line.tcp_setting
            ),
        }
    }
    ends.extend(socket);
    let served = !ends.is_empty();
    let backend: Arc<dyn SerialBackend> = match ends.len() {
        0 => null_serial_backend(),
        1 => ends.remove(0),
        _ => Arc::new(FanoutBackend(ends)),
    };
    match log {
        Some(log) => (Arc::new(TeeBackend::new(backend, log)), served),
        None => (backend, served),
    }
}

/// The host ends of a machine's two serial ports: `(console, second)`,
/// each in the monitor's `serial` map under the board's names for them
/// (`ports`, as `["IOC3 A", "IOC3 B"]`).
///
/// The console is the CI backend in CI mode, otherwise TCP on
/// `serial_port_a` (nothing, if the port is taken), either teed to
/// `serial_log`. The second port serves TCP on `serial_port_b`, teed to
/// `serial_log_b`, or nothing when nothing could serve it. In CI mode neither
/// port is served on TCP, so several `--ci` instances coexist. Each also
/// takes a Unix socket, `serial_socket_a` and `serial_socket_b`. A
/// `[hostserial]`, `[spaceball]` or `[wacom]` section takes its port instead.
pub fn console_backends(
    cfg: &MachineConfig,
    tag: &str,
    ports: [&str; 2],
    ci: Option<&Arc<CiSerialBackend>>,
) -> Result<(Arc<dyn SerialBackend>, Arc<dyn SerialBackend>), String> {
    let [port_a, port_b] = cfg.serial_ports();
    let log = open_log(cfg, tag);
    let label = format!("{tag} serial port 1");
    let console = match crate::dev::serial_dev::peripheral(cfg, 1, log.clone())? {
        Some(b) => {
            log::warn!("{tag}: serial1 is the console port; the PROM and IRIX console now go to the peripheral");
            b
        }
        None => {
            serial_line(
                &SerialLine {
                    label: &label,
                    tcp_port: Some(port_a),
                    tcp_setting: "serial_port_a",
                    socket: cfg.serial_socket_a.as_deref(),
                    socket_setting: "serial_socket_a",
                },
                log,
                ci,
            )
            .0
        }
    };
    let console = crate::dev::serial_map::mapped(tag, ports[0], Some(1), true, cfg, console);
    let label = format!("{tag} serial port 2");
    let log = open_log_at(cfg, cfg.serial_log_b.as_deref(), &label);
    let second = match crate::dev::serial_dev::peripheral(cfg, 2, log.clone())? {
        Some(b) => b,
        None => {
            let (second, served) = serial_line(
                &SerialLine {
                    label: &label,
                    tcp_port: ci.is_none().then_some(port_b),
                    tcp_setting: "serial_port_b",
                    socket: cfg.serial_socket_b.as_deref(),
                    socket_setting: "serial_socket_b",
                },
                log,
                None,
            );
            if served { second } else { null_serial_backend() }
        }
    };
    Ok((console, crate::dev::serial_map::mapped(tag, ports[1], Some(2), false, cfg, second)))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A byte typed at a TCP console reaches a pump, so the socket woke the host-IO thread.
    #[test]
    fn a_byte_typed_at_a_tcp_console_wakes_the_host_io_thread() {
        let (b, port) = tcp_serial_backend(0).unwrap();
        let mut client = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let _ = b.recv_byte(); // accepts the client
        let (tx, rx) = std::sync::mpsc::channel();
        let b2 = b.clone();
        let _pump = host_io().add(move || b2.recv_byte().is_ok_and(|x| tx.send(x).is_ok()));
        client.write_all(b"x").unwrap();
        assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), b'x');
    }

    /// Output the guest queues reaches a client with nothing polling, and what a stalled client left
    /// behind follows once it reads, with no more output to prompt it.
    #[test]
    fn queued_output_is_written_out_by_the_host_io_thread() {
        use socket2::{Domain, SockRef, Socket, Type};
        let b = Arc::new(TcpSocketBackend::new(("127.0.0.1", 0)).unwrap());
        let c = Socket::new(Domain::IPV4, Type::STREAM, None).unwrap();
        c.set_recv_buffer_size(4096).unwrap();
        c.connect(&b.listener.local_addr().unwrap().into()).unwrap();
        let mut client: TcpStream = c.into();
        client.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let _ = b.recv_byte();
        SockRef::from(&*b.conns.lock()[0].stream).set_send_buffer_size(4096).unwrap();
        let b2 = b.clone();
        let _pump = host_io().add(move || {
            b2.flush();
            false
        });
        let sent: Vec<u8> = (0..CLIENT_BACKLOG / 2).map(|i| (i % 200) as u8).collect();
        let head = crate::net::telnet::TelnetFilter::initial_handshake().len();
        sent.iter().for_each(|&x| b.send_byte(x));
        let mut got = vec![0; head + sent.len()];
        client.read_exact(&mut got).unwrap();
        assert_eq!(got[head..], sent[..]);
    }

    /// The console is the CI backend in CI mode and TCP otherwise, either
    /// teed to the same log, and the second port comes up on its own outside
    /// CI mode only. The log adds to what is there, or starts over with
    /// `serial_log_overwrite`.
    #[test]
    fn the_console_is_ci_or_tcp_and_both_log_the_same_bytes() {
        let dir = std::env::temp_dir().join(format!("iris-serial-host-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("serial.log");
        std::fs::write(&log, "stale").unwrap();
        let mut cfg = MachineConfig::default();
        cfg.serial_log = Some(log.to_string_lossy().into_owned());
        cfg.serial_port_a = Some(0);
        cfg.serial_port_b = Some(0);

        let ci = Arc::new(CiSerialBackend::new());
        let (console, second) = console_backends(&cfg, "Test", ["port 1", "port 2"], Some(&ci)).unwrap();
        console.send_byte(b'c');
        assert_eq!(ci.drain_guest(), b"c");
        assert_eq!(std::fs::read(&log).unwrap(), b"stalec", "added to what was there");
        drop(console);
        cfg.serial_log_overwrite = true;
        let (console, _) = console_backends(&cfg, "Test", ["port 1", "port 2"], Some(&ci)).unwrap();
        console.send_byte(b'c');
        assert_eq!(std::fs::read(&log).unwrap(), b"c", "started over");
        assert_eq!(second.describe(), "nothing", "CI mode serves no TCP port, so the second port is wired to nothing");

        drop(console);
        let (console, second) = console_backends(&cfg, "Test", ["port 1", "port 2"], None).unwrap();
        assert!(second.describe().starts_with("tcp 127.0.0.1:"), "{}", second.describe());
        console.send_byte(b't');
        assert_eq!(std::fs::read(&log).unwrap(), b"t");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A TCP line carries the guest's output to a client and into the log;
    /// a line with no listener only logs.
    #[test]
    fn a_tcp_line_is_teed_to_its_log_and_a_line_without_ends_only_logs() {
        let dir = std::env::temp_dir().join(format!("iris-serial-host-line-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let log_path = dir.join("tcp.log");
        let log = SerialLog::create(&log_path, Default::default()).unwrap();
        let (backend, served) = serial_line(
            &SerialLine { label: "Test", tcp_port: Some(0), tcp_setting: "port", socket: None, socket_setting: "socket" },
            Some(log),
            None,
        );
        assert!(served);
        backend.send_byte(b't');
        assert_eq!(std::fs::read(&log_path).unwrap(), b"t");

        let log_path = dir.join("none.log");
        let log = SerialLog::create(&log_path, Default::default()).unwrap();
        let (backend, served) = serial_line(
            &SerialLine { label: "Test", tcp_port: None, tcp_setting: "port", socket: None, socket_setting: "socket" },
            Some(log),
            None,
        );
        assert!(!served);
        backend.send_byte(b'n');
        assert_eq!(std::fs::read(&log_path).unwrap(), b"n");
        assert!(backend.recv_byte().is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A needle that arrives a byte at a time, so no wakeup sees all of it,
    /// is still found, and only up to its end is consumed.
    #[test]
    fn wait_for_finds_a_match_split_across_wakeups() {
        let ci = Arc::new(CiSerialBackend::new());
        let waiter = {
            let ci = ci.clone();
            std::thread::spawn(move || ci.wait_for(b"login: ", Duration::from_secs(10)))
        };
        for &b in b"boot\r\nlogin: more" {
            ci.send_byte(b);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(waiter.join().unwrap().as_deref(), Some(&b"boot\r\nlogin: "[..]));
        assert_eq!(ci.drain_guest(), b"more");
    }

    /// A drain and a refill that land together, past the point a parked
    /// waiter last searched, must not hide the needle at the start of the new
    /// output. The refill is made without a wake-up and one follows it, so the
    /// waiter first looks at it after both.
    #[test]
    fn wait_for_sees_output_that_refilled_a_drained_buffer() {
        let ci = Arc::new(CiSerialBackend::new());
        ci.guest_to_host.lock().extend_from_slice(b"0123456789");
        let waiter = {
            let ci = ci.clone();
            std::thread::spawn(move || ci.wait_for(b"NEEDLE", Duration::from_secs(10)))
        };
        // notify_all reports how many threads it woke, so a non-zero count
        // proves the waiter has searched the ten bytes and parked. Nothing
        // changed, so it searches again and parks again with the same offset.
        let deadline = Instant::now() + Duration::from_secs(10);
        while ci.cv.notify_all() == 0 {
            assert!(Instant::now() < deadline, "waiter never parked");
            std::thread::yield_now();
        }
        assert_eq!(ci.drain_guest(), b"0123456789");
        ci.guest_to_host.lock().extend_from_slice(b"NEEDLE and more bytes");
        ci.cv.notify_all();
        assert_eq!(waiter.join().unwrap().as_deref(), Some(&b"NEEDLE"[..]));
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("iris-serial-host-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Read from a socket until `want` has arrived, a second at most.
    #[cfg(unix)]
    fn read_until(stream: &mut UnixStream, want: &[u8]) -> Vec<u8> {
        stream.set_read_timeout(Some(Duration::from_millis(50))).unwrap();
        let mut got = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !got.windows(want.len()).any(|w| w == want) && Instant::now() < deadline {
            let mut buf = [0u8; 64];
            if let Ok(n) = stream.read(&mut buf) {
                got.extend_from_slice(&buf[..n]);
            }
        }
        got
    }

    /// A console on a Unix socket sends the guest's bytes to the client and
    /// takes the client's, alongside its TCP port, and both go to the log.
    #[cfg(unix)]
    #[test]
    fn a_console_is_served_on_a_unix_socket_beside_its_tcp_port() {
        let dir = temp_dir("socket");
        let sock = dir.join("console.sock");
        let log_path = dir.join("console.log");
        let log = SerialLog::create(&log_path, Default::default()).unwrap();
        let path = sock.to_string_lossy().into_owned();
        let (backend, served) = serial_line(
            &SerialLine { label: "Test", tcp_port: Some(0), tcp_setting: "port", socket: Some(&path), socket_setting: "socket" },
            Some(log),
            None,
        );
        assert!(served);

        let mut client = UnixStream::connect(&sock).unwrap();
        // The socket accepts on the guest's receive poll.
        let _ = backend.recv_byte();
        client.write_all(b"x").unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let typed = loop {
            match backend.recv_byte() {
                Ok(b) => break b,
                Err(_) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
                Err(e) => panic!("no input from the socket: {e}"),
            }
        };
        assert_eq!(typed, b'x');

        for b in b"login: " {
            backend.send_byte(*b);
        }
        backend.flush();
        assert_eq!(read_until(&mut client, b"login: "), b"login: ");
        assert_eq!(std::fs::read(&log_path).unwrap(), b"login: ");

        drop(backend);
        assert!(!sock.exists(), "the socket file goes with the backend");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A socket file an earlier run left is replaced; a live one, and a file
    /// that is not a socket, are refused and left alone.
    #[cfg(unix)]
    #[test]
    fn a_unix_serial_socket_replaces_a_stale_one_and_spares_other_files() {
        let dir = temp_dir("stale");
        let sock = dir.join("s.sock");
        let stale = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        drop(stale);
        assert!(sock.exists(), "a dead listener leaves its file");
        let live = UnixSocketBackend::new(&sock).expect("a stale socket is replaced");
        assert_eq!(UnixSocketBackend::new(&sock).err().map(|e| e.kind()), Some(io::ErrorKind::AddrInUse));
        drop(live);

        let plain = dir.join("plain");
        std::fs::write(&plain, "keep").unwrap();
        assert_eq!(UnixSocketBackend::new(&plain).err().map(|e| e.kind()), Some(io::ErrorKind::AlreadyExists));
        assert_eq!(std::fs::read(&plain).unwrap(), b"keep");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With the control socket and a console socket both in play, the guest's
    /// output reaches the socket, the log and the control socket's buffer,
    /// and typing works from either.
    #[cfg(unix)]
    #[test]
    fn a_ci_console_on_a_socket_still_feeds_the_control_buffer() {
        let dir = temp_dir("ci");
        let sock = dir.join("console.sock");
        let log_path = dir.join("console.log");
        let mut cfg = MachineConfig::default();
        cfg.serial_log = Some(log_path.to_string_lossy().into_owned());
        cfg.serial_port_a = Some(0);
        cfg.serial_port_b = Some(0);
        cfg.serial_socket_a = Some(sock.to_string_lossy().into_owned());

        let ci = Arc::new(CiSerialBackend::new());
        let (console, _) = console_backends(&cfg, "Test", ["port 1", "port 2"], Some(&ci)).unwrap();
        let mut client = UnixStream::connect(&sock).unwrap();
        let _ = console.recv_byte();
        console.send_byte(b'k');
        console.flush();
        assert_eq!(read_until(&mut client, b"k"), b"k");
        assert_eq!(std::fs::read(&log_path).unwrap(), b"k");
        assert_eq!(ci.drain_guest(), b"k");

        ci.push_host(b"c");
        assert_eq!(console.recv_byte().unwrap(), b'c');
        client.write_all(b"s").unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let typed = loop {
            match console.recv_byte() {
                Ok(b) => break b,
                Err(_) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
                Err(e) => panic!("no input from the socket: {e}"),
            }
        };
        assert_eq!(typed, b's');
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The second port has its own log and socket; with neither TCP nor a
    /// socket to serve it, it is wired to nothing.
    #[cfg(unix)]
    #[test]
    fn the_second_port_has_its_own_socket_and_log() {
        let dir = temp_dir("second");
        let sock = dir.join("b.sock");
        let log_path = dir.join("b.log");
        let busy = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let mut cfg = MachineConfig::default();
        cfg.serial_port_a = Some(0);
        cfg.serial_port_b = Some(busy.local_addr().unwrap().port());
        let (_, second) = console_backends(&cfg, "Test", ["port 1", "port 2"], None).unwrap();
        assert_eq!(second.describe(), "nothing", "a taken port with no socket leaves the port wired to nothing");

        cfg.serial_socket_b = Some(sock.to_string_lossy().into_owned());
        cfg.serial_log_b = Some(log_path.to_string_lossy().into_owned());
        let (_, second) = console_backends(&cfg, "Test", ["port 1", "port 2"], None).unwrap();
        assert!(second.describe().starts_with("unix "), "the socket serves the port");
        let mut client = UnixStream::connect(&sock).unwrap();
        let _ = second.recv_byte();
        second.send_byte(b'b');
        second.flush();
        assert_eq!(read_until(&mut client, b"b"), b"b");
        assert_eq!(std::fs::read(&log_path).unwrap(), b"b");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A client that takes bytes only while open.
    #[derive(Default)]
    pub(crate) struct Gate {
        pub(crate) open: std::sync::atomic::AtomicBool,
        pub(crate) out: Mutex<Vec<u8>>,
    }

    impl Write for &Gate {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if !self.open.load(Ordering::SeqCst) {
                return Err(io::ErrorKind::WouldBlock.into());
            }
            self.out.lock().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// A host end with one raw client, a [`Gate`], queued as the sockets are.
    pub(crate) struct GatedLine {
        pub(crate) gate: Gate,
        queue: Mutex<ClientQueue>,
    }

    impl GatedLine {
        pub(crate) fn new() -> Arc<GatedLine> {
            Arc::new(GatedLine { gate: Gate::default(), queue: Mutex::new(ClientQueue::new(false)) })
        }
        pub(crate) fn open(&self, yes: bool) {
            self.gate.open.store(yes, Ordering::SeqCst);
        }
    }

    impl SerialBackend for GatedLine {
        fn send_byte(&self, byte: u8) {
            self.queue.lock().push(&[byte]);
        }
        fn flush(&self) {
            self.queue.lock().flush(&mut &self.gate).unwrap();
        }
        fn queued(&self) -> bool {
            self.queue.lock().queued()
        }
        fn dropped(&self) -> u64 {
            self.queue.lock().dropped()
        }
        fn recv_byte(&self) -> io::Result<u8> {
            Err(io::ErrorKind::WouldBlock.into())
        }
    }

    /// A paused client loses nothing under the cap; past it the oldest whole
    /// units go, counted exactly, and the rest arrives in order.
    #[test]
    fn a_client_queue_drops_its_oldest_whole_units_past_the_cap() {
        let gate = Gate::default();
        let mut q = ClientQueue::new(false);
        let sent: Vec<u8> = (0..CLIENT_BACKLOG + 1000).map(|i| i as u8).collect();
        for b in &sent[..CLIENT_BACKLOG] {
            q.push(&[*b]);
        }
        q.flush(&mut &gate).unwrap();
        assert_eq!(q.dropped(), 0, "a whole backlog fits");
        for b in &sent[CLIENT_BACKLOG..] {
            q.push(&[*b]);
        }
        assert_eq!(q.dropped(), 1000);
        gate.open.store(true, Ordering::SeqCst);
        q.flush(&mut &gate).unwrap();
        assert_eq!(*gate.out.lock(), sent[1000..], "a raw client gets no notice");
    }

    /// A telnet client's escapes are dropped whole, and it is told how many
    /// bytes went before the ones that follow, in one notice however many
    /// polls passed while its socket was full.
    #[test]
    fn a_telnet_client_is_told_what_was_dropped_and_no_escape_is_split() {
        let gate = Gate::default();
        let mut q = ClientQueue::new(true);
        let mut wire = Vec::new();
        for _ in 0..CLIENT_BACKLOG / 2 + 1 {
            crate::net::telnet::escape_byte(IAC, &mut wire);
            q.push(&[IAC, IAC]);
        }
        q.flush(&mut &gate).unwrap();
        q.push(b"x");
        q.flush(&mut &gate).unwrap();
        assert_eq!(q.dropped(), 4, "two escapes, whole");
        gate.open.store(true, Ordering::SeqCst);
        q.flush(&mut &gate).unwrap();
        let out = gate.out.lock();
        let note = b"\r\n[iris: 4 bytes dropped]\r\n";
        assert!(out.starts_with(note));
        assert_eq!(out[note.len()..out.len() - 1], wire[4..]);
        assert_eq!(out.last(), Some(&b'x'));
    }

    /// A socket that takes half an escape keeps the other half for the next
    /// write, and a drop never takes it.
    #[test]
    fn a_split_escape_goes_out_whole_even_past_the_cap() {
        struct Half(Vec<u8>, usize);
        impl Write for Half {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                let n = buf.len().min(self.1);
                self.1 -= n;
                self.0.extend_from_slice(&buf[..n]);
                if n == 0 { Err(io::ErrorKind::WouldBlock.into()) } else { Ok(n) }
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut w = Half(Vec::new(), 1);
        let mut q = ClientQueue::new(true);
        q.push(&[IAC, IAC]);
        q.flush(&mut w).unwrap();
        for _ in 0..CLIENT_BACKLOG {
            q.push(b"y");
        }
        w.1 = usize::MAX;
        q.flush(&mut w).unwrap();
        assert_eq!(&w.0[..2], [IAC, IAC], "the escape's second half went first");
        assert!(w.0[2..].starts_with(b"\r\n[iris: 1 bytes dropped]\r\n"));
    }

    /// A TCP client that stops reading never holds the guest, another client
    /// or the log, and once it reads it gets everything under the cap.
    #[test]
    fn a_stalled_tcp_client_holds_nothing_and_loses_nothing_under_the_cap() {
        use socket2::{Domain, SockRef, Socket, Type};
        let b = Arc::new(TcpSocketBackend::new(("127.0.0.1", 0)).unwrap());
        let addr = b.listener.local_addr().unwrap();
        let connect = |rcvbuf: usize| {
            let c = Socket::new(Domain::IPV4, Type::STREAM, None).unwrap();
            c.set_recv_buffer_size(rcvbuf).unwrap();
            c.connect(&addr.into()).unwrap();
            let c: TcpStream = c.into();
            c.set_read_timeout(Some(Duration::from_millis(20))).unwrap();
            c
        };
        let (mut slow, mut fast) = (connect(4096), connect(1 << 20));
        let t0 = Instant::now();
        while b.conns.lock().len() < 2 {
            assert!(t0.elapsed() < Duration::from_secs(10), "never accepted");
            let _ = b.recv_byte();
        }
        let slow_end = slow.local_addr().unwrap();
        for c in b.conns.lock().iter() {
            if c.stream.peer_addr().unwrap() == slow_end {
                SockRef::from(&*c.stream).set_send_buffer_size(4096).unwrap();
            }
        }
        let dir = temp_dir("stall");
        let log = SerialLog::create(&dir.join("log"), Default::default()).unwrap();
        let tee = TeeBackend::new(b.clone(), log);
        let sent: Vec<u8> = (0..CLIENT_BACKLOG / 2).map(|i| (i % 200) as u8).collect();
        let head = crate::net::telnet::TelnetFilter::initial_handshake().len();
        let read_all = |c: &mut TcpStream, want: usize| {
            let mut got = Vec::new();
            let t0 = Instant::now();
            while got.len() < want {
                assert!(t0.elapsed() < Duration::from_secs(20), "only {} of {want}", got.len());
                let mut buf = [0u8; 65536];
                if let Ok(n) = c.read(&mut buf) {
                    got.extend_from_slice(&buf[..n]);
                }
                tee.flush();
            }
            got
        };
        for (i, &x) in sent.iter().enumerate() {
            tee.send_byte(x);
            if i % 4096 == 0 {
                tee.flush();
            }
        }
        assert_eq!(read_all(&mut fast, head + sent.len())[head..], sent[..], "the fast client is not held back");
        assert_eq!(std::fs::read(dir.join("log")).unwrap(), sent, "the log has every byte");
        assert_eq!(read_all(&mut slow, head + sent.len())[head..], sent[..], "the slow one loses nothing under the cap");
        assert_eq!(tee.dropped(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A Unix-socket client that stops reading is queued the same way, and a
    /// CI capture on the same line takes every byte at once.
    #[cfg(unix)]
    #[test]
    fn a_stalled_unix_client_and_the_ci_capture_on_one_line() {
        let dir = temp_dir("unix-stall");
        let socket = Arc::new(UnixSocketBackend::new(dir.join("s.sock")).unwrap());
        let mut client = UnixStream::connect(dir.join("s.sock")).unwrap();
        client.set_read_timeout(Some(Duration::from_millis(20))).unwrap();
        let _ = socket.recv_byte();
        let ci = Arc::new(CiSerialBackend::new());
        let f = FanoutBackend(vec![ci.clone(), socket]);
        let sent: Vec<u8> = (0..CLIENT_BACKLOG + 5000).map(|i| (i % 251) as u8).collect();
        for &x in &sent {
            f.send_byte(x);
        }
        assert_eq!(ci.drain_guest(), sent, "the CI capture is complete");
        let mut got = Vec::new();
        let want = sent.len() - 5000;
        assert!(
            crate::test_board::until(|| {
                f.flush();
                let mut buf = [0u8; 65536];
                if let Ok(n) = client.read(&mut buf) {
                    got.extend_from_slice(&buf[..n]);
                }
                got.len() >= want
            }),
            "only {} of {want}",
            got.len()
        );
        assert_eq!(f.dropped(), 5000, "only the bytes past the cap went");
        assert_eq!(got, sent[5000..], "the oldest went, the rest in order");
        drop(f);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A client that goes away with output unread is let go with its count
    /// kept, and the next one gets output again from a clean queue.
    #[cfg(unix)]
    #[test]
    fn a_unix_client_that_reconnects_gets_output_again() {
        let dir = temp_dir("unix-reconnect");
        let b = UnixSocketBackend::new(dir.join("s.sock")).unwrap();
        let old = UnixStream::connect(dir.join("s.sock")).unwrap();
        let _ = b.recv_byte();
        for i in 0..CLIENT_BACKLOG + 1000 {
            b.send_byte(i as u8);
        }
        assert_eq!(b.dropped(), 1000, "the guest was never held, the oldest went");
        drop(old);
        assert!(
            crate::test_board::until(|| {
                matches!(b.recv_byte(), Err(e) if e.kind() == io::ErrorKind::NotConnected)
            }),
            "the gone client was never let go"
        );
        assert_eq!(b.dropped(), 1000, "its count outlives it");
        let mut client = UnixStream::connect(dir.join("s.sock")).unwrap();
        client.set_read_timeout(Some(Duration::from_millis(20))).unwrap();
        let _ = b.recv_byte();
        b.send_byte(b'o');
        b.send_byte(b'k');
        let mut got = Vec::new();
        assert!(
            crate::test_board::until(|| {
                b.flush();
                let mut buf = [0u8; 16];
                if let Ok(n) = client.read(&mut buf) {
                    got.extend_from_slice(&buf[..n]);
                }
                got.len() >= 2
            }),
            "the new client got nothing"
        );
        assert_eq!(got, b"ok", "none of the old client's queue");
        assert_eq!(b.dropped(), 1000);
        drop(b);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A TCP client that leaves is reported once as `NotConnected`, as a Unix
    /// one is, and the next one gets only what comes after it connects.
    #[test]
    fn a_tcp_client_that_leaves_reads_as_not_connected() {
        let b = TcpSocketBackend::new(("127.0.0.1", 0)).unwrap();
        let addr = b.listener.local_addr().unwrap();
        let old = TcpStream::connect(addr).unwrap();
        assert!(crate::test_board::until(|| {
            let _ = b.recv_byte();
            b.conns.lock().len() == 1
        }));
        drop(old);
        assert!(
            crate::test_board::until(|| {
                matches!(b.recv_byte(), Err(e) if e.kind() == io::ErrorKind::NotConnected)
            }),
            "the gone client was never reported"
        );
        assert!(b.conns.lock().is_empty());
        let _new = TcpStream::connect(addr).unwrap();
        assert!(crate::test_board::until(|| {
            let _ = b.recv_byte();
            b.conns.lock().len() == 1
        }));
        assert!(matches!(b.recv_byte(), Err(e) if e.kind() == io::ErrorKind::WouldBlock), "a live client reads as idle");
    }

    /// The TCP counterpart: `flush_all` removing a gone client keeps its
    /// count, and a client that connects after gets only what comes next.
    #[test]
    fn a_tcp_client_that_reconnects_gets_output_again() {
        let b = TcpSocketBackend::new(("127.0.0.1", 0)).unwrap();
        let addr = b.listener.local_addr().unwrap();
        let old = TcpStream::connect(addr).unwrap();
        assert!(crate::test_board::until(|| {
            let _ = b.recv_byte();
            b.conns.lock().len() == 1
        }));
        for i in 0..CLIENT_BACKLOG + 1000 {
            b.send_byte(i as u8);
        }
        let held = b.dropped();
        assert!(held > 0, "the guest was never held, the oldest went");
        drop(old);
        assert!(
            crate::test_board::until(|| {
                b.flush();
                b.conns.lock().is_empty()
            }),
            "the gone client was never removed"
        );
        assert_eq!(b.dropped(), held, "its count outlives it");
        let mut client = TcpStream::connect(addr).unwrap();
        client.set_read_timeout(Some(Duration::from_millis(20))).unwrap();
        assert!(crate::test_board::until(|| {
            let _ = b.recv_byte();
            b.conns.lock().len() == 1
        }));
        b.send_byte(b'o');
        b.send_byte(b'k');
        let head = crate::net::telnet::TelnetFilter::initial_handshake().len();
        let mut got = Vec::new();
        assert!(
            crate::test_board::until(|| {
                b.flush();
                let mut buf = [0u8; 64];
                if let Ok(n) = client.read(&mut buf) {
                    got.extend_from_slice(&buf[..n]);
                }
                got.len() >= head + 2
            }),
            "the new client got nothing"
        );
        assert_eq!(got[head..], *b"ok", "none of the old client's queue");
        assert_eq!(b.dropped(), held);
    }

    /// A UART as the contract drives it, through its own registers.
    pub(crate) trait Port {
        /// Write a byte to the transmit register.
        fn write(&mut self, b: u8);
        /// The transmitter will take a byte, from the chip's status register.
        fn tx_ready(&mut self) -> bool;
        /// One pass of what the chip does between guest accesses.
        fn poll(&mut self);
    }

    type MakePort = fn(Arc<dyn SerialBackend>) -> Box<dyn Port>;

    /// No UART waits on a stalled client, and the client gets every byte
    /// once it reads, under the cap.
    fn check_port(name: &str, make: MakePort) {
        let line = GatedLine::new();
        let mut port = make(line.clone());
        let sent: Vec<u8> = (0..4000u32).map(|i| (i % 251) as u8).collect();
        for &b in &sent {
            let t0 = Instant::now();
            while !port.tx_ready() {
                assert!(t0.elapsed() < Duration::from_secs(1), "{name}: the transmitter waited on the client");
                port.poll();
            }
            port.write(b);
        }
        line.open(true);
        let t0 = Instant::now();
        while line.gate.out.lock().len() < sent.len() {
            assert!(t0.elapsed() < Duration::from_secs(10), "{name}: the queued bytes never went");
            port.poll();
        }
        assert_eq!(*line.gate.out.lock(), sent, "{name}: every byte once, in order");
        assert_eq!(line.dropped(), 0, "{name}");
    }

    #[test]
    fn no_uart_waits_on_a_stalled_client_or_loses_its_bytes_under_the_cap() {
        let ports: [(&str, MakePort); 4] = [
            ("O2 MACE 16550", crate::dev::mace_uart::tests::contract_port),
            ("IOC3 16550", crate::dev::ioc3::tests::contract_port),
            ("IOC4 16550", crate::dev::ioc4::tests::contract_port),
            ("Z85C30", crate::dev::z85c30::tests::contract_port),
        ];
        for (name, make) in ports {
            check_port(name, make);
        }
    }
}
