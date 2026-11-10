//! Running a bare-metal guest under the `iris` this `cargo test` built.
//!
//! The emulator runs in a child process, not in the test binary: a guest that
//! hangs is killed at its timeout, and an emulator that crashes fails its own
//! case instead of taking every other case in the binary down with it. Cargo
//! builds the binaries for an integration test with the test's own profile
//! and features, so `cargo test --release --features jitv2` runs the release
//! jitv2 emulator, exactly as `cargo build --release --features jitv2` would
//! have built it.

#![allow(dead_code)] // each suite uses its own half of this

pub mod cputest;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// The engine this build runs, for messages: a cargo feature, so one test
/// binary covers one engine and every CPU.
pub fn engine() -> &'static str {
    if cfg!(feature = "jitv2") { "jitv2" } else { "interp" }
}

/// Repo-relative path, wherever cargo runs the test from.
pub fn repo(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(p)
}

/// A debug build runs these suites far too slowly to be worth waiting for,
/// so asking for one (`--ignored` without `--release`) fails at once rather
/// than looking hung.
pub fn require_optimised_build() {
    assert!(
        !cfg!(debug_assertions),
        "the bare-metal suites need an optimised emulator: cargo test --release"
    );
}

/// Seconds before a guest is declared hung. A hang detector, not a performance
/// budget; `IRIS_SUITE_TIMEOUT` overrides it.
pub fn timeout(default_s: u64) -> Duration {
    let s = std::env::var("IRIS_SUITE_TIMEOUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_s);
    Duration::from_secs(s)
}

/// A fresh directory for one run: the emulator's working directory, its
/// machine config and whatever it writes. Under cargo's per-target scratch
/// directory and named for the run and this process, so cases running in
/// parallel, and two `cargo test`s sharing a target directory, never share
/// a file. Kept afterwards: it holds the guest's console log.
pub fn run_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("bare-metal")
        .join(format!("{}-{}-{}", name, engine(), std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("run")).expect("create the run directory");
    dir
}

/// The suite's machine config with every listening port moved to one the OS
/// picks. The defaults are fixed (monitor 8888, serial 8880/8881), so a second
/// emulator running at the same time would lose them to the first, and a
/// developer's own emulator would lose them to the test. Written to
/// `<dir>/run/bare.toml`, mirroring `<suite>/run/bare.toml`.
pub fn machine_config(suite_config: &str, dir: &Path) -> PathBuf {
    let base = std::fs::read_to_string(repo(suite_config))
        .unwrap_or_else(|e| panic!("{}: {}", suite_config, e));
    // Top-level keys, so they go before the file's first table.
    let text = format!(
        "monitor_port = 0\nserial_port_a = 0\nserial_port_b = 0\n\n{}",
        base
    );
    let path = dir.join("run").join("bare.toml");
    std::fs::write(&path, text).expect("write the machine config");
    path
}

/// What a finished child left behind.
pub struct Output {
    pub status: ExitStatus,
    pub stdout: String,
    pub stderr: String,
}

impl Output {
    /// Describes a status that means the emulator itself died, not that the
    /// guest reported failures: killed by a signal on Unix. The guest's exit
    /// code is its own business (cpu-tests exits with its failure count).
    pub fn crash(&self) -> Option<String> {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            if let Some(sig) = self.status.signal() {
                return Some(format!(
                    "the emulator was killed by signal {}{}",
                    sig,
                    if self.status.core_dumped() { " (core dumped)" } else { "" }
                ));
            }
        }
        None
    }
}

/// Kills the child if the test unwinds before it is reaped, so a failed
/// assertion cannot leave an emulator running.
struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

/// Run `cmd` to completion or until `limit`, whichever is first. A timeout is
/// an error naming the limit and the tail of what the child printed.
pub fn run(mut cmd: Command, limit: Duration, log: &Path) -> Result<Output, String> {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = Guard(cmd.spawn().map_err(|e| format!("spawn {:?}: {}", cmd, e))?);

    // Drain both pipes on their own threads: a child that fills a pipe while
    // this thread sleeps would block instead of finishing or timing out.
    let drain = |mut r: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut v = Vec::new();
            let _ = r.read_to_end(&mut v);
            v
        })
    };
    let out = drain(Box::new(child.0.stdout.take().unwrap()));
    let err = drain(Box::new(child.0.stderr.take().unwrap()));

    let started = Instant::now();
    let status = loop {
        if let Some(s) = child.0.try_wait().map_err(|e| e.to_string())? {
            break Some(s);
        }
        if started.elapsed() >= limit {
            let _ = child.0.kill();
            let _ = child.0.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };

    let stdout = String::from_utf8_lossy(&out.join().unwrap_or_default()).into_owned();
    let stderr = String::from_utf8_lossy(&err.join().unwrap_or_default()).into_owned();
    let _ = std::fs::write(log, format!("{}\n--- stderr ---\n{}", stdout, stderr));

    match status {
        Some(status) => Ok(Output { status, stdout, stderr }),
        None => Err(format!(
            "timed out after {}s: the guest never finished (log: {})\n--- last 30 lines ---\n{}",
            limit.as_secs(),
            log.display(),
            tail(&stdout, 30)
        )),
    }
}

pub fn tail(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// One thing a suite found wrong with a run: about one guest test or kernel
/// (`test`), or about the run as a whole.
#[derive(Debug)]
pub struct Finding {
    pub test: Option<String>,
    pub what: String,
}

impl Finding {
    pub fn run(what: impl Into<String>) -> Self {
        Finding { test: None, what: what.into() }
    }
    pub fn test(name: &str, what: impl Into<String>) -> Self {
        Finding { test: Some(name.to_string()), what: what.into() }
    }
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match &self.test {
            Some(t) => write!(f, "{t} {}", self.what),
            None => f.write_str(&self.what),
        }
    }
}

impl PartialEq<&str> for Finding {
    fn eq(&self, other: &&str) -> bool {
        self.to_string() == *other
    }
}

/// The first line of a failed cell's panic: the guest tests that failed,
/// by name, so whatever shows one line shows them.
pub fn headline(cell: &str, findings: &[Finding]) -> String {
    let names: Vec<&str> = findings.iter().filter_map(|f| f.test.as_deref()).collect();
    match (names.len(), findings.first()) {
        (0, Some(f)) => format!("{cell}: {f}"),
        (0, None) => format!("{cell}: failed"),
        (n, _) if n <= 10 => format!("{cell}: {n} failing: {}", names.join(", ")),
        (n, _) => format!("{cell}: {n} failing: {}, and {} more", names[..10].join(", "), n - 10),
    }
}

/// Where the per-guest-test JUnit report `<name>.xml` goes, when
/// `IRIS_SUITE_JUNIT` names a directory (CI points it beside nextest's own).
pub fn junit_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("IRIS_SUITE_JUNIT").map(|d| Path::new(&d).join(format!("{name}.xml")))
}

