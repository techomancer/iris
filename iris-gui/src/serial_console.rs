//! In-app IRIX serial-console viewer.
//!
//! The emulated SGI Indy exposes its serial console (ttyd1) as a loopback TCP
//! server on `127.0.0.1:8881` (see `iris::dev::z85c30`). This viewer connects to it
//! as a client and shows the live console stream, and lets the user type back
//! into it — so the serial console works inside the app without an external
//! terminal. It is also the visible demonstration of the app's network
//! entitlements: the emulator *listens* (network.server) and this viewer
//! *connects* (network.client), both on loopback.
//!
//! A background thread owns the socket, strips inbound telnet negotiation via
//! `iris::net::telnet::TelnetFilter`, and parks decoded text in a shared buffer.

use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::Mutex;

use iris::net::telnet::{self, TelnetFilter};

/// The loopback address the emulator binds for ttyd1 (IRIX serial console).
pub const SERIAL_ADDR: &str = "127.0.0.1:8881";
/// Cap on retained scrollback so a long boot doesn't grow the buffer forever.
const MAX_TEXT: usize = 128 * 1024;

#[derive(Default)]
struct Shared {
    /// Decoded console text (telnet stripped, bare CR dropped).
    text: String,
    /// True once the TCP connection is established.
    connected: bool,
    /// Set if the connection could not be made / was lost.
    error: Option<String>,
    /// Bumped on every change so the GUI can decide whether to re-scroll.
    seq: u64,
}

pub struct SerialConsole {
    shared: Arc<Mutex<Shared>>,
    running: Arc<AtomicBool>,
    /// Write half (a clone of the socket) for sending typed input.
    write: Arc<Mutex<Option<TcpStream>>>,
    worker: Option<JoinHandle<()>>,
}

impl SerialConsole {
    /// Connect to the loopback serial console and start streaming.
    pub fn connect() -> Self {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let running = Arc::new(AtomicBool::new(true));
        let write = Arc::new(Mutex::new(None));
        let (s2, r2, w2) = (shared.clone(), running.clone(), write.clone());
        let worker = std::thread::Builder::new()
            .name("iris-gui-serial".into())
            .spawn(move || run(s2, r2, w2))
            .expect("spawn serial-console worker");
        Self { shared, running, write, worker: Some(worker) }
    }

    /// (text, connected, error, seq) snapshot for rendering.
    pub fn snapshot(&self) -> (String, bool, Option<String>, u64) {
        let g = self.shared.lock();
        (g.text.clone(), g.connected, g.error.clone(), g.seq)
    }

    pub fn clear(&self) {
        let mut g = self.shared.lock();
        g.text.clear();
        g.seq = g.seq.wrapping_add(1);
    }

    /// Send raw bytes to the guest console (telnet-escaping 0xFF).
    pub fn send(&self, bytes: &[u8]) {
        let mut esc = Vec::with_capacity(bytes.len());
        for &b in bytes {
            telnet::escape_byte(b, &mut esc);
        }
        if let Some(s) = self.write.lock().as_mut() {
            let _ = s.write_all(&esc);
            let _ = s.flush();
        }
    }
}

impl Drop for SerialConsole {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        // Shutting the socket down unblocks the reader's blocking read so the
        // worker exits promptly.
        if let Some(s) = self.write.lock().take() {
            let _ = s.shutdown(Shutdown::Both);
        }
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

fn run(shared: Arc<Mutex<Shared>>, running: Arc<AtomicBool>, write: Arc<Mutex<Option<TcpStream>>>) {
    let addr: SocketAddr = SERIAL_ADDR.parse().expect("valid loopback addr");
    let stream = match TcpStream::connect_timeout(&addr, Duration::from_millis(800)) {
        Ok(s) => s,
        Err(e) => {
            shared.lock().error = Some(format!(
                "could not connect to {SERIAL_ADDR}: {e}\nStart the emulator first, then reopen."
            ));
            return;
        }
    };
    let wclone = match stream.try_clone() {
        Ok(c) => c,
        Err(e) => {
            shared.lock().error = Some(format!("socket clone failed: {e}"));
            return;
        }
    };
    *write.lock() = Some(wclone);
    // Short read timeout so the loop can observe `running` for shutdown.
    let _ = stream.set_read_timeout(Some(Duration::from_millis(200)));
    {
        let mut g = shared.lock();
        g.connected = true;
        g.error = None;
        g.seq = g.seq.wrapping_add(1);
    }

    // Client-side telnet handling: decline negotiation, strip IAC. The guest's
    // tty echoes typed characters, so no telnet-layer echo is needed.
    let mut filter = TelnetFilter::new_passive();
    let mut buf = [0u8; 2048];
    let mut rstream = stream;

    while running.load(Ordering::Relaxed) {
        match rstream.read(&mut buf) {
            Ok(0) => break, // EOF — server closed
            Ok(n) => {
                let mut replies = Vec::new();
                let mut data = Vec::with_capacity(n);
                for &b in &buf[..n] {
                    if let Some(d) = filter.feed(b, &mut replies) {
                        data.push(d);
                    }
                }
                if !replies.is_empty() {
                    if let Some(s) = write.lock().as_mut() {
                        let _ = s.write_all(&replies);
                        let _ = s.flush();
                    }
                }
                if !data.is_empty() {
                    append_text(&shared, &data);
                }
            }
            Err(ref e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                continue;
            }
            Err(_) => break,
        }
    }

    *write.lock() = None;
    let mut g = shared.lock();
    g.connected = false;
    g.seq = g.seq.wrapping_add(1);
}

fn append_text(shared: &Arc<Mutex<Shared>>, data: &[u8]) {
    let chunk = String::from_utf8_lossy(data);
    let mut g = shared.lock();
    for ch in chunk.chars() {
        // Drop bare CR; egui handles \n line breaks.
        if ch != '\r' {
            g.text.push(ch);
        }
    }
    if g.text.len() > MAX_TEXT {
        let cut = g.text.len() - MAX_TEXT;
        let mut idx = cut;
        while !g.text.is_char_boundary(idx) {
            idx += 1;
        }
        g.text.drain(..idx);
    }
    g.seq = g.seq.wrapping_add(1);
}
