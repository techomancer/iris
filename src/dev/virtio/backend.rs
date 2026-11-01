//! The host end of a virtio-serial port: raw bytes in and out.
//!
//! Unlike the UARTs' [`SerialBackend`](crate::dev::serial_host::SerialBackend)
//! these are 8-bit clean: no telnet negotiation, no escaping. A port carries
//! whatever the guest program on it speaks (a console, a file transfer, a
//! protocol of its own).

use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

pub trait PortBackend: Send + Sync {
    /// Guest output. May drop it when nobody is listening.
    fn write(&self, data: &[u8]);
    /// Host input for the guest, without blocking. 0 means nothing now.
    fn read(&self, buf: &mut [u8]) -> usize;
    /// Whether anything is attached on the host side. The guest is told
    /// through `VIRTIO_CONSOLE_PORT_OPEN`.
    fn connected(&self) -> bool {
        true
    }
}

/// Drops output, never produces input, and reports itself unconnected.
pub struct NullPort;

impl PortBackend for NullPort {
    fn write(&self, _data: &[u8]) {}
    fn read(&self, _buf: &mut [u8]) -> usize {
        0
    }
    fn connected(&self) -> bool {
        false
    }
}

/// In-memory port: what the guest wrote accumulates for the host to take, and
/// the host queues what the guest will read. For tests and embedders.
#[derive(Default)]
pub struct MemPort {
    tx: Mutex<Vec<u8>>,
    rx: Mutex<VecDeque<u8>>,
}

impl MemPort {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
    /// Queue input for the guest.
    pub fn push_rx(&self, data: &[u8]) {
        self.rx.lock().extend(data.iter().copied());
    }
    /// Take everything the guest has written so far.
    pub fn take_tx(&self) -> Vec<u8> {
        std::mem::take(&mut *self.tx.lock())
    }
}

impl PortBackend for MemPort {
    fn write(&self, data: &[u8]) {
        self.tx.lock().extend_from_slice(data);
    }
    fn read(&self, buf: &mut [u8]) -> usize {
        let mut rx = self.rx.lock();
        let n = buf.len().min(rx.len());
        for b in buf[..n].iter_mut() {
            *b = rx.pop_front().unwrap();
        }
        n
    }
}

trait Conn: Read + Write + Send {}
impl<T: Read + Write + Send> Conn for T {}

type Accept = Box<dyn Fn() -> io::Result<Box<dyn Conn>> + Send + Sync>;

/// A listening socket serving one client at a time: the next waits in the
/// listen queue until the first hangs up. Guest output with no client is
/// dropped.
struct SocketPort {
    accept: Accept,
    client: Mutex<Option<Box<dyn Conn>>>,
}

impl SocketPort {
    fn with_client<R>(&self, f: impl FnOnce(&mut Box<dyn Conn>) -> R) -> Option<R> {
        let mut c = self.client.lock();
        if c.is_none() {
            *c = (self.accept)().ok();
        }
        c.as_mut().map(f)
    }
}

impl PortBackend for SocketPort {
    fn write(&self, data: &[u8]) {
        let mut gone = false;
        self.with_client(|c| {
            let deadline = Instant::now() + Duration::from_millis(500);
            let mut rest = data;
            while !rest.is_empty() {
                match c.write(rest) {
                    Ok(0) => { gone = true; break }
                    Ok(n) => rest = &rest[n..],
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        // A slow reader: wait briefly, then drop the rest
                        // rather than stall the guest's queue.
                        if Instant::now() > deadline { break }
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(_) => { gone = true; break }
                }
            }
        });
        if gone {
            *self.client.lock() = None;
        }
    }

    fn read(&self, buf: &mut [u8]) -> usize {
        let mut gone = false;
        let n = self
            .with_client(|c| match c.read(buf) {
                Ok(0) => { gone = true; 0 }
                Ok(n) => n,
                Err(_) => 0,
            })
            .unwrap_or(0);
        if gone {
            *self.client.lock() = None;
        }
        n
    }

    fn connected(&self) -> bool {
        self.with_client(|_| ()).is_some()
    }
}

/// A raw TCP port on 127.0.0.1:`port` (0 takes any free port). Returns the
/// backend and the port it bound.
pub fn tcp_port(port: u16) -> io::Result<(Arc<dyn PortBackend>, u16)> {
    let l = TcpListener::bind(("127.0.0.1", port))?;
    l.set_nonblocking(true)?;
    let bound = l.local_addr()?.port();
    let accept: Accept = Box::new(move || {
        let (s, _) = l.accept()?;
        s.set_nonblocking(true)?;
        let _ = s.set_nodelay(true);
        Ok(Box::new(s) as Box<dyn Conn>)
    });
    Ok((Arc::new(SocketPort { accept, client: Mutex::new(None) }), bound))
}

/// A raw Unix-socket port at `path`. A stale socket file is replaced; a live
/// one, or any other kind of file, is an error and left alone.
#[cfg(unix)]
pub fn unix_port(path: &str) -> io::Result<Arc<dyn PortBackend>> {
    use std::os::unix::net::{UnixListener, UnixStream};
    if let Ok(meta) = std::fs::symlink_metadata(path) {
        use std::os::unix::fs::FileTypeExt;
        if !meta.file_type().is_socket() {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{path} exists and is not a socket")));
        }
        if UnixStream::connect(path).is_ok() {
            return Err(io::Error::new(io::ErrorKind::AddrInUse, format!("{path} is in use")));
        }
        std::fs::remove_file(path)?;
    }
    let l = UnixListener::bind(path)?;
    l.set_nonblocking(true)?;
    let accept: Accept = Box::new(move || {
        let (s, _) = l.accept()?;
        s.set_nonblocking(true)?;
        Ok(Box::new(s) as Box<dyn Conn>)
    });
    Ok(Arc::new(SocketPort { accept, client: Mutex::new(None) }))
}

#[cfg(not(unix))]
pub fn unix_port(path: &str) -> io::Result<Arc<dyn PortBackend>> {
    Err(io::Error::new(io::ErrorKind::Unsupported, format!("{path}: Unix sockets need a Unix host")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_port_round_trips() {
        let p = MemPort::new();
        p.write(b"abc");
        assert_eq!(p.take_tx(), b"abc");
        assert!(p.take_tx().is_empty());
        p.push_rx(b"xyz12");
        let mut b = [0u8; 3];
        assert_eq!(p.read(&mut b), 3);
        assert_eq!(&b, b"xyz");
        assert_eq!(p.read(&mut b), 2);
        assert_eq!(p.read(&mut b), 0);
    }

    #[test]
    fn tcp_port_is_eight_bit_clean() {
        let (p, port) = tcp_port(0).unwrap();
        assert!(!p.connected());
        let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        // The accept happens on first use.
        let deadline = Instant::now() + Duration::from_secs(2);
        while !p.connected() {
            assert!(Instant::now() < deadline, "never accepted");
            std::thread::sleep(Duration::from_millis(5));
        }
        let bytes = [0xff, 0xfb, 0x01, 0x00, 0x0d, 0x0a];
        p.write(&bytes);
        let mut got = [0u8; 6];
        c.read_exact(&mut got).unwrap();
        assert_eq!(got, bytes);
        c.write_all(&bytes).unwrap();
        let mut back = Vec::new();
        let mut buf = [0u8; 16];
        while back.len() < 6 {
            assert!(Instant::now() < deadline, "input never arrived");
            let n = p.read(&mut buf);
            back.extend_from_slice(&buf[..n]);
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(back, bytes);
    }
}
