//! Graphics-board annotated traces share this sink: one file written by a
//! `TraceWriter` thread, with per-class counters and collapsed runs.

use std::fmt::{self, Display};
use std::io::{self, Write};
use std::time::Instant;

use crate::trace_writer::TraceWriter;

/// A run of similar accesses a trace collapses into one line.
pub trait TraceRun {
    /// Fold `next` into this run if it continues it.
    fn merge(&mut self, next: &Self) -> bool;
    /// Write the collapsed line, numbered `seq`.
    fn emit(&self, seq: u64, line: impl FnOnce(fmt::Arguments));
}

/// A board's trace state behind its lock: the file, the clock, one
/// counter per line class and the pending collapsed run.
pub struct TraceFile<R, const N: usize> {
    out: Option<TraceWriter>,
    start: Instant,
    last_flush: Instant,
    run: Option<R>,
    /// The counter in `seq` that numbers collapsed runs.
    run_seq: usize,
    names: [&'static str; N],
    pub seq: [u64; N],
}

impl<R: TraceRun, const N: usize> TraceFile<R, N> {
    pub fn new(names: [&'static str; N], run_seq: usize) -> Self {
        let now = Instant::now();
        Self { out: None, start: now, last_flush: now, run: None, run_seq, names, seq: [0; N] }
    }

    pub fn is_open(&self) -> bool {
        self.out.is_some()
    }

    pub fn path(&self) -> &str {
        self.out.as_ref().map_or("", |o| &o.path)
    }

    /// Start a trace into `path`, replacing any running one.
    pub fn open(&mut self, path: &str) -> io::Result<()> {
        self.out = Some(TraceWriter::create(path)?);
        self.start = Instant::now();
        self.last_flush = self.start;
        self.seq = [0; N];
        self.run = None;
        Ok(())
    }

    /// A line with no time stamp.
    pub fn raw(&mut self, text: impl Display) {
        if let Some(f) = self.out.as_mut() {
            f.line(format_args!("{text}"));
        }
    }

    /// A line stamped with the seconds since the trace started.
    pub fn line(&mut self, text: impl Display) {
        let t = self.start.elapsed().as_secs_f64();
        self.raw(format_args!("{t:12.6} {text}"));
    }

    /// The next line of `class`, numbered, after any pending run.
    pub fn event(&mut self, class: usize, tag: &str, text: impl Display) {
        self.close_run();
        self.seq[class] += 1;
        let seq = self.seq[class];
        self.line(format_args!("{tag} #{seq:06} {text}"));
    }

    /// One access: it extends the pending run or starts the next one.
    pub fn access(&mut self, next: R) {
        if self.run.as_mut().is_some_and(|r| r.merge(&next)) {
            return;
        }
        self.close_run();
        self.seq[self.run_seq] += 1;
        self.run = Some(next);
    }

    /// Emit the pending run, if any.
    pub fn close_run(&mut self) {
        if let Some(r) = self.run.take() {
            r.emit(self.seq[self.run_seq], |a| self.line(a));
        }
    }

    /// Hand the lines to the writer thread, at most twice a second. A forced
    /// flush waits until they are in the file.
    pub fn flush(&mut self, force: bool) {
        if !force && self.last_flush.elapsed().as_millis() < 500 {
            return;
        }
        self.close_run();
        if let Some(f) = self.out.as_mut() {
            if force {
                f.sync();
            } else {
                f.flush();
            }
        }
        self.last_flush = Instant::now();
    }

    /// "hq 12 / re3 7": lines so far per counter.
    pub fn counts(&self) -> String {
        let n = self.names.iter().zip(self.seq);
        n.map(|(n, c)| format!("{n} {c}")).collect::<Vec<_>>().join(" / ")
    }

    /// Close the file, writing out everything. Returns a summary, or None
    /// when no trace was running.
    pub fn stop(&mut self) -> Option<String> {
        self.close_run();
        let f = self.out.take()?;
        let path = f.path.clone();
        Some(match f.finish() {
            Ok(bytes) => format!("{path} ({} entries, {bytes} bytes)", self.counts()),
            Err(e) => format!("{path}: write failed: {e}"),
        })
    }
}

/// The classes of a board's trace, named in `trace <file> [classes]`.
pub struct Classes {
    pub names: &'static [(&'static str, u32)],
    /// What `all` means.
    pub all: u32,
}

impl Classes {
    /// A pattern over the class names, as `log` takes (`gfifo,cpu`, `all`, `-cpu`).
    pub fn parse(&self, spec: &str) -> Result<u32, String> {
        crate::devlog::select(spec, self.names, self.all)
    }
}

/// `<board> trace <file> [classes]`, `trace flush` and `trace off`, over the
/// board's own start, stop and flush. `name` leads an error.
pub fn trace_command(
    args: &[&str],
    name: &str,
    classes: &Classes,
    w: &mut dyn Write,
    flush: impl FnOnce(),
    stop: impl FnOnce() -> Option<String>,
    start: impl FnOnce(&str, u32) -> io::Result<()>,
) -> Result<(), String> {
    let text = match args {
        ["flush"] => {
            flush();
            "trace flushed".to_string()
        }
        [] | ["off"] => match stop() {
            Some(s) => format!("trace closed: {s}"),
            None => "trace not active".to_string(),
        },
        [path, rest @ ..] => {
            let mask = classes.parse(rest.first().copied().unwrap_or("all")).map_err(|e| format!("{name}: {e}"))?;
            start(path, mask).map_err(|e| format!("{name} {path}: {e}"))?;
            format!("tracing to {path} (mask {mask:#x})")
        }
    };
    writeln!(w, "{text}").map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Poll {
        val: u32,
        count: u64,
    }

    impl TraceRun for Poll {
        fn merge(&mut self, n: &Self) -> bool {
            self.count += (self.val == n.val) as u64;
            self.val == n.val
        }

        fn emit(&self, seq: u64, line: impl FnOnce(fmt::Arguments)) {
            line(format_args!("POLL #{seq:06} {} x{}", self.val, self.count));
        }
    }

    #[test]
    fn lines_are_numbered_collapsed_and_readable_after_a_forced_flush() {
        let path = std::env::temp_dir().join(format!("trace_sink_{}.txt", std::process::id()));
        let path = path.to_str().unwrap();
        let mut t = TraceFile::<Poll, 2>::new(["a", "b"], 1);
        t.open(path).unwrap();
        t.raw("# header");
        t.event(0, "A  ", "one");
        for val in [1, 1, 2] {
            t.access(Poll { val, count: 1 });
        }
        t.event(0, "A  ", "two");
        t.flush(true);
        let text = std::fs::read_to_string(path).unwrap();
        let lines: Vec<_> = text.lines().enumerate().map(|(i, l)| if i == 0 { l } else { &l[13..] }).collect();
        assert_eq!(lines, ["# header", "A   #000001 one", "POLL #000001 1 x2", "POLL #000002 2 x1", "A   #000002 two"]);
        let summary = t.stop().unwrap();
        assert!(summary.contains("(a 2 / b 2 entries, "), "{summary}");
        assert!(t.stop().is_none());
        let _ = std::fs::remove_file(path);
    }
}
