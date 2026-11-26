//! Buffered file writer for device traces and recordings, with the disk I/O
//! on a background thread.
//!
//! The producer (a device's bus path or engine thread) appends into a plain
//! `Vec<u8>`: a record costs a bounds check and a copy, and formatted lines a
//! `write_fmt` into memory. Every `CHUNK` bytes the buffer is handed to the
//! writer thread through a bounded channel and a recycled buffer takes its
//! place, so the producer never makes a syscall. When the disk falls more than
//! `IN_FLIGHT` chunks behind, `put` blocks: traces and recordings are
//! lossless, and a recording with a hole cannot be replayed.
//!
//! Not synchronised: the owner keeps it behind whatever lock orders its
//! records (`Rex3::rec`, `Rex3::trace`).

use std::fs::File;
use std::io::{self, Write};
use std::sync::mpsc::{sync_channel, Receiver, Sender, SyncSender};
use std::thread::JoinHandle;
use std::time::Instant;

/// Bytes collected before a buffer is handed to the writer thread.
const CHUNK: usize = 1 << 20;
/// Full buffers queued for the writer before `put` blocks.
const IN_FLIGHT: usize = 4;

enum Msg {
    Data(Vec<u8>),
    /// Everything sent before this is in the file: answer on the sender.
    Sync(Sender<()>),
}

pub struct TraceWriter {
    buf: Vec<u8>,
    tx: Option<SyncSender<Msg>>,
    /// Buffers the writer thread has emptied, for reuse.
    spare: Receiver<Vec<u8>>,
    thread: Option<JoinHandle<io::Result<()>>>,
    /// Bytes produced so far (written or queued).
    bytes: u64,
    pub path: String,
}

impl TraceWriter {
    pub fn create(path: &str) -> io::Result<Self> {
        let mut file = File::create(path)?;
        let (tx, rx) = sync_channel::<Msg>(IN_FLIGHT);
        let (spare_tx, spare) = sync_channel::<Vec<u8>>(IN_FLIGHT + 1);
        let thread = std::thread::Builder::new().name("trace-writer".into()).spawn(move || {
            let mut res = Ok(());
            for msg in rx {
                match msg {
                    Msg::Data(mut v) => {
                        // After an error keep draining, so the producer never
                        // blocks on a writer that stopped listening.
                        if res.is_ok() {
                            res = file.write_all(&v);
                        }
                        v.clear();
                        let _ = spare_tx.try_send(v);
                    }
                    Msg::Sync(ack) => {
                        if res.is_ok() {
                            res = file.flush();
                        }
                        let _ = ack.send(());
                    }
                }
            }
            res.and_then(|_| file.flush())
        })?;
        Ok(Self { buf: Vec::with_capacity(CHUNK + 4096), tx: Some(tx), spare, thread: Some(thread), bytes: 0, path: path.to_string() })
    }

    /// Append raw bytes.
    #[inline]
    pub fn put(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        self.bytes += bytes.len() as u64;
        if self.buf.len() >= CHUNK {
            self.ship();
        }
    }

    /// Append formatted text and a newline.
    pub fn line(&mut self, text: std::fmt::Arguments) {
        let before = self.buf.len();
        let _ = self.buf.write_fmt(text);
        self.buf.push(b'\n');
        self.bytes += (self.buf.len() - before) as u64;
        if self.buf.len() >= CHUNK {
            self.ship();
        }
    }

    /// Bytes produced so far.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Hand the current buffer to the writer thread.
    fn ship(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        let next = self.spare.try_recv().unwrap_or_else(|_| Vec::with_capacity(CHUNK + 4096));
        let full = std::mem::replace(&mut self.buf, next);
        if let Some(tx) = &self.tx {
            let _ = tx.send(Msg::Data(full));
        }
    }

    /// Send what has been collected without waiting for it to reach the disk.
    pub fn flush(&mut self) {
        self.ship();
    }

    /// Wait until everything put so far is in the file.
    pub fn sync(&mut self) {
        self.ship();
        let (ack_tx, ack) = std::sync::mpsc::channel();
        if let Some(tx) = &self.tx {
            if tx.send(Msg::Sync(ack_tx)).is_ok() {
                let _ = ack.recv();
            }
        }
    }

    /// Write out everything and close the file, reporting any write error.
    pub fn finish(mut self) -> io::Result<u64> {
        self.close().map(|_| self.bytes)
    }

    fn close(&mut self) -> io::Result<()> {
        self.ship();
        self.tx = None;
        match self.thread.take() {
            Some(t) => t.join().unwrap_or_else(|_| Err(io::Error::other("trace writer panicked"))),
            None => Ok(()),
        }
    }
}

impl Drop for TraceWriter {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// The writer, start time and flush throttle the graphics traces share. A
/// board's sink keeps its own sequence counters and collapsed access run.
pub struct TraceFile {
    out: Option<TraceWriter>,
    start: Option<Instant>,
    last_flush: Option<Instant>,
}

impl TraceFile {
    pub const fn new() -> Self {
        Self { out: None, start: None, last_flush: None }
    }

    pub fn is_open(&self) -> bool {
        self.out.is_some()
    }

    /// The file's path, or "" when none is open.
    pub fn path(&self) -> &str {
        self.out.as_ref().map_or("", |f| f.path.as_str())
    }

    /// Start a trace into `path`, replacing one that runs.
    pub fn open(&mut self, path: &str) -> io::Result<()> {
        self.out = Some(TraceWriter::create(path)?);
        self.start = Some(Instant::now());
        self.last_flush = None;
        Ok(())
    }

    /// Append `text` stamped with the seconds since `open`.
    pub fn line(&mut self, text: std::fmt::Arguments) {
        let t = self.start.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);
        if let Some(f) = self.out.as_mut() {
            f.line(format_args!("{t:12.6} {text}"));
        }
    }

    /// Append `text` as it is, with no stamp.
    pub fn raw(&mut self, text: std::fmt::Arguments) {
        if let Some(f) = self.out.as_mut() {
            f.line(text);
        }
    }

    /// Whether `flush` should run: forced, or 500 ms since the last one. The
    /// caller closes its pending access run first.
    pub fn due(&self, force: bool) -> bool {
        force || self.last_flush.map_or(true, |l| l.elapsed().as_millis() >= 500)
    }

    /// Hand what is collected to the writer thread; when `force`, wait until
    /// it is in the file.
    pub fn flush(&mut self, force: bool) {
        if let Some(f) = self.out.as_mut() {
            if force {
                f.sync();
            } else {
                f.flush();
            }
        }
        self.last_flush = Some(Instant::now());
    }

    /// Write out everything and close: the path and the bytes written, or
    /// None when no trace was open.
    pub fn close(&mut self) -> Option<(String, io::Result<u64>)> {
        let f = self.out.take()?;
        let path = f.path.clone();
        Some((path, f.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_everything_in_order_across_chunks() {
        let path = std::env::temp_dir().join(format!("trace_writer_{}.bin", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        let mut w = TraceWriter::create(&path).unwrap();
        let mut want = Vec::new();
        // Enough to cross several chunk boundaries and recycle buffers.
        for i in 0..(3 * CHUNK / 4 + 77) as u32 {
            let b = i.to_le_bytes();
            w.put(&b);
            want.extend_from_slice(&b);
        }
        w.line(format_args!("end {}", 42));
        want.extend_from_slice(b"end 42\n");
        w.sync();
        assert_eq!(std::fs::read(&path).unwrap(), want, "sync must leave everything put so far in the file");
        assert_eq!(w.finish().unwrap(), want.len() as u64);
        assert_eq!(std::fs::read(&path).unwrap(), want);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_forced_flush_leaves_the_stamped_lines_in_the_file() {
        let path = std::env::temp_dir().join(format!("trace_file_{}.txt", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        let mut f = TraceFile::new();
        assert!(!f.is_open() && f.close().is_none());
        f.open(&path).unwrap();
        f.raw(format_args!("# header"));
        f.line(format_args!("GF  #{:06} x", 1));
        assert!(f.due(false), "nothing has flushed yet");
        f.flush(true);
        assert!(!f.due(false), "a flush starts the 500 ms throttle");
        assert!(f.due(true));
        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("# header"));
        let l = lines.next().unwrap();
        assert!(l.ends_with(" GF  #000001 x") && l.as_bytes()[..12].iter().all(|b| b.is_ascii_digit() || *b == b'.' || *b == b' '), "{l}");
        let (p, bytes) = f.close().unwrap();
        assert_eq!((p.as_str(), bytes.unwrap()), (path.as_str(), text.len() as u64));
        let _ = std::fs::remove_file(&path);
    }
}
