//! The `--serial-log` sink: a file the guest's console output is mirrored to.
//!
//! Raw mode (`timestamps = off`, the default) writes every byte as it arrives,
//! exactly as before timestamps existed — a transcript a test can diff.
//! Wall-clock mode prefixes each completed line with a UTC ISO-8601 stamp so a
//! harness can tell a slow boot from a missing line. A partial line is held
//! until its newline (a stamp must belong to the line it starts) and written
//! by `flush`/`Drop` when the machine stops.
//!
//! ECS mode (`timestamps = ecs`) turns each completed, non-empty line into one
//! Elastic Common Schema JSON document with logger `irix.guest`, for a log
//! collector. Control characters are dropped and a trailing CR is trimmed. The
//! path `-` writes to standard output instead of a file.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// How `--serial-log` stamps its lines.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
#[clap(rename_all = "lower")]
pub enum SerialLogTimestamps {
    /// Raw bytes, byte-for-byte a transcript (the default).
    #[default]
    Off,
    /// `[YYYY-MM-DDTHH:MM:SS.ffffffZ] ` before each completed line.
    Wall,
    /// One ECS JSON line per completed, non-empty line (logger `irix.guest`).
    Ecs,
}

struct Inner {
    file: Box<dyn Write + Send>,
    /// Bytes of a completed-line's prefix not yet written (wall mode only).
    pending: Vec<u8>,
}

/// A serial-log file shared by the backends. Cloned through an `Arc`, so the
/// CI and TCP paths write through one sink.
pub struct SerialLog {
    timestamps: SerialLogTimestamps,
    inner: Mutex<Inner>,
}

impl SerialLog {
    /// Open for append: an existing transcript is preserved.
    pub fn open(path: &Path, timestamps: SerialLogTimestamps) -> io::Result<std::sync::Arc<Self>> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self::with_writer(Box::new(file), timestamps))
    }

    /// Create or truncate: one boot's transcript, for callers that tail it.
    pub fn create(path: &Path, timestamps: SerialLogTimestamps) -> io::Result<std::sync::Arc<Self>> {
        let file = File::create(path)?;
        Ok(Self::with_writer(Box::new(file), timestamps))
    }

    /// Standard output, for a container whose log collector reads it.
    pub fn stdout(timestamps: SerialLogTimestamps) -> std::sync::Arc<Self> {
        Self::with_writer(Box::new(io::stdout()), timestamps)
    }

    fn with_writer(file: Box<dyn Write + Send>, timestamps: SerialLogTimestamps) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self { timestamps, inner: Mutex::new(Inner { file, pending: Vec::new() }) })
    }

    pub fn timestamps(&self) -> SerialLogTimestamps {
        self.timestamps
    }

    pub fn write(&self, bytes: &[u8]) {
        let mut guard = match self.inner.lock() { Ok(g) => g, Err(p) => p.into_inner() };
        let Inner { file, pending } = &mut *guard;
        match self.timestamps {
            SerialLogTimestamps::Ecs => {
                pending.extend_from_slice(bytes);
                let mut start = 0;
                while let Some(pos) = pending[start..].iter().position(|&b| b == b'\n') {
                    let end = start + pos + 1;
                    if let Some(line) = ecs_line(&pending[start..end]) {
                        let _ = file.write_all(line.as_bytes());
                    }
                    start = end;
                }
                if start > 0 {
                    pending.drain(..start);
                }
                let _ = file.flush();
            }
            SerialLogTimestamps::Off => {
                let _ = file.write_all(bytes);
                let _ = file.flush();
            }
            SerialLogTimestamps::Wall => {
                pending.extend_from_slice(bytes);
                let mut start = 0;
                while let Some(pos) = pending[start..].iter().position(|&b| b == b'\n') {
                    let end = start + pos + 1;
                    let stamp = format_wall();
                    let _ = file.write_all(stamp.as_bytes());
                    let _ = file.write_all(&pending[start..end]);
                    start = end;
                }
                if start > 0 {
                    pending.drain(..start);
                }
                let _ = file.flush();
            }
        }
    }

    /// Write any held partial line (wall mode) and flush the file.
    pub fn flush(&self) {
        let mut guard = match self.inner.lock() { Ok(g) => g, Err(p) => p.into_inner() };
        let Inner { file, pending } = &mut *guard;
        if self.timestamps == SerialLogTimestamps::Wall && !pending.is_empty() {
            let stamp = format_wall();
            let _ = file.write_all(stamp.as_bytes());
            let bytes = std::mem::take(pending);
            let _ = file.write_all(&bytes);
        } else if self.timestamps == SerialLogTimestamps::Ecs && !pending.is_empty() {
            let bytes = std::mem::take(pending);
            if let Some(line) = ecs_line(&bytes) {
                let _ = file.write_all(line.as_bytes());
            }
        }
        let _ = file.flush();
    }
}

impl Drop for SerialLog {
    fn drop(&mut self) {
        self.flush();
    }
}

/// One ECS JSON document (with its newline) for a line of guest output, or
/// `None` if nothing printable is left once control characters are dropped.
fn ecs_line(raw: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(raw);
    let msg: String = text.chars().filter(|c| *c == '\t' || !c.is_control()).collect();
    let msg = msg.trim();
    if msg.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(msg.len() + 128);
    out.push_str("{\"@timestamp\":\"");
    out.push_str(&iso_wall());
    out.push_str("\",\"log.level\":\"info\",\"message\":\"");
    for c in msg.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push_str("\",\"log.logger\":\"irix.guest\",\"event.dataset\":\"irix.guest\",\"ecs.version\":\"1.6.0\"}\n");
    Some(out)
}

/// `[YYYY-MM-DDTHH:MM:SS.ffffffZ] ` for the current UTC time.
pub fn format_wall() -> String {
    format!("[{}] ", iso_wall())
}

/// `YYYY-MM-DDTHH:MM:SS.ffffffZ` for the current UTC time.
fn iso_wall() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = now.as_secs() as i64;
    let micros = now.subsec_micros();
    let (y, mo, d, h, mi, s) = civil_from_unix(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{micros:06}Z")
}

/// Days-to-civil conversion (Howard Hinnant's algorithm), with the Unix epoch
/// as day 0. Returns (year, month, day, hour, minute, second) in UTC.
fn civil_from_unix(unix: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = unix.div_euclid(86_400);
    let secs = unix.rem_euclid(86_400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (
        y,
        m,
        d,
        (secs / 3_600) as u32,
        ((secs % 3_600) / 60) as u32,
        (secs % 60) as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("iris-serial-log-{}-{name}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn read_all(path: &Path) -> String {
        let mut s = String::new();
        File::open(path).unwrap().read_to_string(&mut s).unwrap();
        s
    }

    /// Raw mode is a transcript: the bytes on disk are the guest's bytes,
    /// including a partial line, with no injected prefix.
    #[test]
    fn raw_mode_is_byte_identical() {
        let path = temp_path("raw");
        let log = SerialLog::open(&path, SerialLogTimestamps::Off).unwrap();
        log.write(b"kernel: up\r\n");
        log.write(b"partial");
        log.flush();
        drop(log);
        assert_eq!(read_all(&path).as_bytes(), b"kernel: up\r\npartial");
        let _ = std::fs::remove_file(path);
    }

    /// Wall mode stamps each completed line, and holds a partial line until
    /// its newline (or the final flush).
    #[test]
    fn wall_mode_stamps_completed_lines_and_flushes_partial() {
        let path = temp_path("wall");
        let log = SerialLog::open(&path, SerialLogTimestamps::Wall).unwrap();
        log.write(b"first\nsec");
        assert_eq!(read_all(&path).lines().count(), 1, "only the completed line is written");
        log.write(b"ond\n");
        log.write(b"last, no newline");
        log.flush();
        drop(log);
        let text = read_all(&path);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        for line in &lines {
            assert!(line.starts_with('[') && line.contains('T') && line.contains('Z'),
                "line is stamped: {line:?}");
        }
        assert!(lines[0].ends_with("first"));
        assert!(lines[1].ends_with("second"));
        assert!(lines[2].ends_with("last, no newline"));
        let _ = std::fs::remove_file(path);
    }

    /// ECS mode: one JSON document per non-empty line, control characters
    /// dropped, quotes escaped, logger `irix.guest`.
    #[test]
    fn ecs_mode_emits_json_lines() {
        let path = temp_path("ecs");
        let log = SerialLog::open(&path, SerialLogTimestamps::Ecs).unwrap();
        log.write(b"\x1b[2Jkernel \"up\"\r\n\r\nsec");
        log.write(b"ond\n");
        log.write(b"tail, no newline");
        log.flush();
        drop(log);
        let text = read_all(&path);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "{text:?}");
        assert!(lines[0].contains(r#""message":"[2Jkernel \"up\"""#), "{}", lines[0]);
        assert!(lines[1].contains(r#""message":"second""#));
        assert!(lines[2].contains(r#""message":"tail, no newline""#));
        for l in &lines {
            assert!(l.starts_with(r#"{"@timestamp":""#) && l.contains(r#""log.logger":"irix.guest""#), "{l}");
        }
        let _ = std::fs::remove_file(path);
    }

    /// The stamp is an ISO-8601 UTC instant with microseconds: 27 characters
    /// plus the trailing space, e.g. `[2026-10-25T05:38:58.123456Z] `.
    #[test]
    fn stamp_shape() {
        let stamp = format_wall();
        assert_eq!(stamp.len(), 30, "{stamp:?}");
        assert!(stamp.starts_with('['));
        assert!(stamp.ends_with("Z] "));
        assert_eq!(&stamp[1..5], "2026");
    }
}
