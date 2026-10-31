//! The host end of a serial line: where a guest UART's bytes go and where
//! its input comes from.
//!
//! A [`SerialBackend`] is a TCP console, the in-process CI backend, a file
//! tee, or nothing. Every machine's UARTs (the Indy's Z85C30, the O2's MACE
//! 16550s, the Octane's IOC3) take one, and [`console_backends`] picks them
//! for the machines whose consoles are configured the same way.

use std::fs::OpenOptions;
use std::collections::VecDeque;
use std::io::{self, Read, Write};
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::config::MachineConfig;
use crate::dev::serial_log::SerialLog;

pub trait SerialBackend: Send + Sync {
    fn send_byte(&self, byte: u8);
    fn recv_byte(&self) -> io::Result<u8>;
}

/// Drops TX bytes and never yields RX. Used as a placeholder when a channel
/// isn't wired to a host I/O source (e.g. CI mode unused channel).
pub(crate) struct NullBackend;

impl SerialBackend for NullBackend {
    fn send_byte(&self, _byte: u8) {}
    fn recv_byte(&self) -> io::Result<u8> {
        Err(io::Error::new(io::ErrorKind::WouldBlock, "null"))
    }
}

/// Tees guest-emitted bytes to a host file in addition to delegating to an
/// inner backend.  Used by `--serial-log <file>` to capture the serial
/// console transcript while the real TCP listener on port 8881 keeps
/// working unchanged.
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
    fn recv_byte(&self) -> io::Result<u8> {
        self.inner.recv_byte()
    }
}

#[cfg(unix)]
struct UnixSocketBackend {
    listener: UnixListener,
    stream: Mutex<Option<UnixStream>>,
}

#[cfg(unix)]
impl UnixSocketBackend {
    fn new<P: AsRef<Path>>(path: P) -> Self {
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(path).expect("Failed to bind serial socket");
        listener.set_nonblocking(true).expect("Failed to set nonblocking");
        Self {
            listener,
            stream: Mutex::new(None),
        }
    }
}

#[cfg(unix)]
impl SerialBackend for UnixSocketBackend {
    fn send_byte(&self, byte: u8) {
        let mut stream_guard = self.stream.lock();
        if let Some(ref mut stream) = *stream_guard {
            if let Err(e) = stream.write_all(&[byte]) {
                if e.kind() != io::ErrorKind::WouldBlock {
                    *stream_guard = None;
                }
            }
        }
    }

    fn recv_byte(&self) -> io::Result<u8> {
        let mut guard = self.stream.lock();
        
        if guard.is_none() {
            match self.listener.accept() {
                Ok((socket, _)) => {
                    socket.set_nonblocking(true)?;
                    *guard = Some(socket);
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                    return Err(io::Error::new(io::ErrorKind::WouldBlock, "No connection"));
                }
                Err(e) => return Err(e),
            }
        }

        if let Some(ref mut stream) = *guard {
            let mut buf = [0u8; 1];
            match stream.read(&mut buf) {
                Ok(1) => Ok(buf[0]),
                Ok(_) => { // EOF
                    *guard = None;
                    Err(io::Error::new(io::ErrorKind::NotConnected, "EOF"))
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Err(io::Error::new(io::ErrorKind::WouldBlock, "WouldBlock")),
                Err(e) => {
                    *guard = None;
                    Err(e)
                }
            }
        } else {
            Err(io::Error::new(io::ErrorKind::WouldBlock, "No connection"))
        }
    }
}

struct TcpConn {
    stream: TcpStream,
    telnet: crate::net::telnet::TelnetFilter,
}

pub(crate) struct TcpSocketBackend {
    pub(crate) listener: TcpListener,
    /// Every connected client: each sees all the output and may type, so a
    /// person at the console and a script can share it.
    conns: Mutex<Vec<TcpConn>>,
}

impl TcpSocketBackend {
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
        Some(Self {
            listener,
            conns: Mutex::new(Vec::new()),
        })
    }

    /// Take any clients waiting to connect, offering each telnet character
    /// mode. One whose handshake cannot be written is dropped; it can
    /// reconnect.
    fn accept_waiting(&self, conns: &mut Vec<TcpConn>) {
        while let Ok((socket, _)) = self.listener.accept() {
            if socket.set_nonblocking(true).is_err() {
                continue;
            }
            // The console is latency-sensitive: without TCP_NODELAY, Nagle
            // holds a keystroke or a log line back until the previous
            // segment is acked, which shows up as a stutter on every
            // interaction. Set it on accept, before any byte is written.
            let _ = socket.set_nodelay(true);
            let mut c = TcpConn { stream: socket, telnet: crate::net::telnet::TelnetFilter::new() };
            if c.stream.write_all(&crate::net::telnet::TelnetFilter::initial_handshake()).is_ok() {
                conns.push(c);
            }
        }
    }
}

impl SerialBackend for TcpSocketBackend {
    fn send_byte(&self, byte: u8) {
        let mut conns = self.conns.lock();
        if conns.is_empty() {
            return;
        }
        let mut buf = Vec::with_capacity(2);
        crate::net::telnet::escape_byte(byte, &mut buf);
        conns.retain_mut(|c| match c.stream.write_all(&buf) {
            Ok(()) => true,
            Err(e) => e.kind() == io::ErrorKind::WouldBlock,
        });
    }

    fn recv_byte(&self) -> io::Result<u8> {
        let mut conns = self.conns.lock();
        self.accept_waiting(&mut conns);
        if conns.is_empty() {
            return Err(io::Error::new(io::ErrorKind::WouldBlock, "No connection"));
        }
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
                        let mut reply = Vec::new();
                        let data = c.telnet.feed(buf[0], &mut reply);
                        if !reply.is_empty() {
                            if let Err(e) = c.stream.write_all(&reply) {
                                if e.kind() != io::ErrorKind::WouldBlock {
                                    gone = true;
                                    break;
                                }
                            }
                        }
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
                conns.remove(i);
            } else {
                i += 1;
            }
        }
        Err(io::Error::new(io::ErrorKind::WouldBlock, "WouldBlock"))
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
    }

    /// Drain everything the guest has produced since the last call. Empties
    /// the buffer; the returned Vec is the guest output as raw bytes.
    pub fn drain_guest(&self) -> Vec<u8> {
        let mut q = self.guest_to_host.lock();
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
        loop {
            if let Some(pos) = find_subseq(&q, needle) {
                let end = pos + needle.len();
                let consumed: Vec<u8> = q.drain(..end).collect();
                return Some(consumed);
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            if self.cv.wait_until(&mut q, deadline).timed_out() {
                // One more scan in case bytes arrived between the last check
                // and the timeout.
                if let Some(pos) = find_subseq(&q, needle) {
                    let end = pos + needle.len();
                    let consumed: Vec<u8> = q.drain(..end).collect();
                    return Some(consumed);
                }
                return None;
            }
        }
    }

    /// Clear both queues. Called on `restore`/`rollback` so stale serial
    /// output from the previous run doesn't leak into the next test.
    pub fn reset(&self) {
        self.host_to_guest.lock().clear();
        self.guest_to_host.lock().clear();
    }
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

impl SerialBackend for CiSerialBackend {
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

/// How [`open_log`] treats a transcript that is already there.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LogMode {
    /// One boot's transcript: truncate, so scripts tailing the file see only
    /// this boot's prompts.
    Fresh,
    /// Keep what an earlier run wrote and add to it.
    Append,
}

/// Open `cfg.serial_log`, if one is set, as the sink the machine's CI and
/// TCP consoles both write through, so the two logs are the same bytes
/// (timestamps included). A file that cannot be opened is reported and the
/// machine boots without a log. `tag` names the machine in the messages.
pub fn open_log(cfg: &MachineConfig, tag: &str, mode: LogMode) -> Option<Arc<SerialLog>> {
    open_log_at(cfg, cfg.serial_log.as_deref(), tag, mode)
}

/// As [`open_log`] for the transcript at `path` (a line's own log, when it
/// is not the console's), with `cfg`'s timestamp setting.
pub fn open_log_at(cfg: &MachineConfig, path: Option<&str>, tag: &str, mode: LogMode) -> Option<Arc<SerialLog>> {
    let path = path?;
    let p = Path::new(path);
    let opened = match mode {
        LogMode::Fresh => SerialLog::create(p, cfg.serial_log_timestamps),
        LogMode::Append => SerialLog::open(p, cfg.serial_log_timestamps),
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

/// Where one serial line is served on the host.
pub struct SerialLine<'a> {
    /// Names the line in messages, e.g. "Octane serial port 1".
    pub label: &'a str,
    /// The TCP port, or `None` for a line with no listener.
    pub tcp_port: Option<u16>,
    /// The setting that chose `tcp_port`, for the message when it is taken.
    pub tcp_setting: &'a str,
}

/// Wire one serial line: TCP on `tcp_port`, teed to `log`. The line that
/// carries the CI control socket's console (`ci`) is not served on TCP (so
/// several `--ci` instances coexist). Returns the backend, and whether a
/// host end could be made; without one the backend only logs.
pub fn serial_line(
    line: &SerialLine,
    log: Option<Arc<SerialLog>>,
    ci: Option<&Arc<CiSerialBackend>>,
) -> (Arc<dyn SerialBackend>, bool) {
    if let Some(ci) = ci {
        if let Some(log) = log {
            ci.set_log(log);
        }
        return (ci.clone(), true);
    }
    let mut backend = null_serial_backend();
    let mut served = false;
    if let Some(port) = line.tcp_port {
        match tcp_serial_backend(port) {
            Some((b, p)) => {
                eprintln!("iris: {} on 127.0.0.1:{p}", line.label);
                backend = b;
                served = true;
            }
            None => eprintln!(
                "iris: {}: port {port} unavailable; is another iris using it? ({})",
                line.label, line.tcp_setting
            ),
        }
    }
    match log {
        Some(log) => (Arc::new(TeeBackend::new(backend, log)), served),
        None => (backend, served),
    }
}

/// The host ends of a machine's two serial ports: `(console, second)`.
///
/// The console is the CI backend in CI mode, otherwise TCP on
/// `serial_port_a` (nothing, if the port is taken), either teed to
/// `serial_log`. The second port serves TCP on `serial_port_b`, or is `None`
/// when that port is taken, so the caller leaves it unwired.
pub fn console_backends(
    cfg: &MachineConfig,
    tag: &str,
    ci: Option<&Arc<CiSerialBackend>>,
) -> (Arc<dyn SerialBackend>, Option<Arc<dyn SerialBackend>>) {
    let [port_a, port_b] = cfg.serial_ports();
    let log = open_log(cfg, tag, LogMode::Fresh);
    let label = format!("{tag} serial port 1");
    let (console, _) = serial_line(
        &SerialLine { label: &label, tcp_port: Some(port_a), tcp_setting: "serial_port_a" },
        log,
        ci,
    );
    let label = format!("{tag} serial port 2");
    let (second, served) = serial_line(
        &SerialLine { label: &label, tcp_port: Some(port_b), tcp_setting: "serial_port_b" },
        None,
        None,
    );
    (console, served.then_some(second))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The console is the CI backend in CI mode and TCP otherwise, either
    /// teed to the same fresh log, and the second port comes up on its own.
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
        let (console, second) = console_backends(&cfg, "Test", Some(&ci));
        console.send_byte(b'c');
        assert_eq!(ci.drain_guest(), b"c");
        assert_eq!(std::fs::read(&log).unwrap(), b"c", "a fresh log, not the stale one");
        assert!(second.is_some());

        let (console, _) = console_backends(&cfg, "Test", None);
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
            &SerialLine { label: "Test", tcp_port: Some(0), tcp_setting: "port" },
            Some(log),
            None,
        );
        assert!(served);
        backend.send_byte(b't');
        assert_eq!(std::fs::read(&log_path).unwrap(), b"t");

        let log_path = dir.join("none.log");
        let log = SerialLog::create(&log_path, Default::default()).unwrap();
        let (backend, served) = serial_line(
            &SerialLine { label: "Test", tcp_port: None, tcp_setting: "port" },
            Some(log),
            None,
        );
        assert!(!served);
        backend.send_byte(b'n');
        assert_eq!(std::fs::read(&log_path).unwrap(), b"n");
        assert!(backend.recv_byte().is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
