//! Running cpu-tests and judging its TAP stream: a cell passes when exactly
//! the tests listed for it in `cpu-tests/expected-failures.txt` fail and every
//! planned test reported.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use iris::junit::{self, Duration, TestCase, TestSuite};

use super::*;

/// Seconds before a hung guest is killed. A healthy run takes about a second;
/// the margin is for a loaded machine.
const TIMEOUT_S: u64 = 900;

/// `IRIS_CPUTEST_ELF`, else what `make -C cpu-tests` builds.
pub fn guest_elf_path() -> PathBuf {
    std::env::var_os("IRIS_CPUTEST_ELF")
        .map(Into::into)
        .unwrap_or_else(|| repo("cpu-tests/build/cputest.elf"))
}

/// The tests a cell is expected to fail: its `[cell]` section of
/// expected-failures.txt, or `None` when the file has no such section. A
/// header may name several cells, `[a, b]`. `#` starts a comment anywhere on
/// a line.
pub fn expected_failures(cell: &str) -> Option<BTreeSet<String>> {
    let text = std::fs::read_to_string(repo("cpu-tests/expected-failures.txt"))
        .expect("read cpu-tests/expected-failures.txt");
    parse_expected_failures(&text, cell)
}

/// `expected_failures` on the file's text.
pub fn parse_expected_failures(text: &str, cell: &str) -> Option<BTreeSet<String>> {
    let mut want = None;
    let mut section = None;
    for line in text.lines() {
        let line = line.split('#').next().unwrap().trim();
        if line.starts_with('[') {
            let names = line.trim_matches(|c| c == '[' || c == ']');
            section = Some(names.split(',').any(|n| n.trim() == cell));
            if section == Some(true) {
                want.get_or_insert_with(BTreeSet::new);
            }
        } else if !line.is_empty() && section == Some(true) {
            want.get_or_insert_with(BTreeSet::new).insert(line.to_string());
        }
    }
    want
}

/// What the guest's TAP stream says: the planned count, each reported test's
/// name and whether it passed, and whether the guest got to the end of its
/// run. A skipped test reports `ok`.
pub struct Tap {
    pub plan: Option<usize>,
    pub results: Vec<(String, bool)>,
    /// Parallel to `results`: the guest skipped it on this CPU (`# SKIP`).
    pub skipped: Vec<bool>,
    /// The first result whose number is not its position, as (position,
    /// number): a repeated or missing line.
    pub misnumbered: Option<(usize, usize)>,
    /// The guest printed `IRIS-CPUTEST-DONE`, after its last test.
    pub done: bool,
}

/// `1..N` on a line of its own, `ok N - name` / `not ok N - name` at the
/// start of a line, where the name runs to the first space (`# SKIP` follows),
/// and `IRIS-CPUTEST-DONE rc=N` at the end.
pub fn parse_tap(log: &str) -> Tap {
    let mut tap = Tap {
        plan: None,
        results: Vec::new(),
        skipped: Vec::new(),
        misnumbered: None,
        done: false,
    };
    // The console ends lines with CR LF. A bare CR ends one too.
    let log = log.replace("\r\n", "\n").replace('\r', "\n");
    for line in log.split('\n') {
        if line.starts_with("IRIS-CPUTEST-DONE rc=") {
            tap.done = true;
        }
        if tap.plan.is_none() {
            if let Some(n) = line.strip_prefix("1..") {
                if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) {
                    tap.plan = n.parse().ok();
                }
            }
        }
        let (ok, rest) = match line.strip_prefix("not ok ") {
            Some(r) => (false, r),
            None => match line.strip_prefix("ok ") {
                Some(r) => (true, r),
                None => continue,
            },
        };
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        let Some(tail) = (digits > 0).then(|| &rest[digits..]).and_then(|r| r.strip_prefix(" - "))
        else {
            continue;
        };
        let name: String = tail.chars().take_while(|c| !c.is_whitespace()).collect();
        if !name.is_empty() {
            let number: usize = rest[..digits].parse().unwrap_or(0);
            if tap.misnumbered.is_none() && number != tap.results.len() + 1 {
                tap.misnumbered = Some((tap.results.len() + 1, number));
            }
            tap.skipped.push(tail[name.len()..].trim_start().starts_with("# SKIP"));
            tap.results.push((name, ok));
        }
    }
    tap
}

/// Every way the run disagrees with expected-failures.txt, or is not a whole
/// run. A truncated or crashed run fails even when every result it printed
/// was expected.
pub fn judge(cell: &str, tap: &Tap, want: &BTreeSet<String>) -> Vec<Finding> {
    let failed: BTreeSet<String> =
        tap.results.iter().filter(|(_, ok)| !ok).map(|(n, _)| n.clone()).collect();
    let Some(plan) = tap.plan else {
        return vec![Finding::run("no plan line: the guest never reached its first test")];
    };
    let mut errs = Vec::new();
    for n in failed.difference(want) {
        errs.push(Finding::test(n, format!("failed and is not an expected failure on {cell}")));
    }
    if let Some((pos, number)) = tap.misnumbered {
        errs.push(Finding::run(format!(
            "result {pos} is numbered {number}: a result line is repeated or missing"
        )));
    }
    if !tap.done {
        errs.push(Finding::run("no IRIS-CPUTEST-DONE line: the guest never finished its run"));
    }
    let reported = tap.results.len();
    if reported < plan {
        errs.push(Finding::run(format!("{reported} of {plan} tests reported: the run stopped early")));
    } else if reported > plan {
        errs.push(Finding::run(format!("{reported} tests reported against a plan of {plan}")));
    } else {
        for n in want.difference(&failed) {
            errs.push(Finding::test(
                n,
                format!("now passes on {cell}: remove it from [{cell}] in expected-failures.txt"),
            ));
        }
    }
    errs
}

/// The run as JUnit, filed under `cpu-tests.<cpu>.<engine>`: a case per
/// reported test, failed when a finding names it, skipped when the guest
/// skipped it or it is an expected failure. A finding about a test the guest
/// never reported, or about the whole run (as `run`), is a failed case too.
pub fn junit_suite(cell: &str, tap: &Tap, errs: &[Finding]) -> TestSuite {
    let found = |name: Option<&str>| -> String {
        let f: Vec<&str> =
            errs.iter().filter(|e| e.test.as_deref() == name).map(|e| e.what.as_str()).collect();
        f.join("; ")
    };
    let mut cases = Vec::new();
    for ((name, ok), &skipped) in tap.results.iter().zip(&tap.skipped) {
        let f = found(Some(name));
        cases.push(if !f.is_empty() {
            TestCase::failure(name, Duration::ZERO, "cpu-tests", &f)
        } else if skipped {
            TestCase::skipped_with_cause(name, "skip", "skipped on this CPU")
        } else if !ok {
            TestCase::skipped_with_cause(name, "expected", "expected failure (expected-failures.txt)")
        } else {
            TestCase::success(name, Duration::ZERO)
        });
    }
    for e in errs {
        let name = e.test.as_deref().unwrap_or("run");
        if !cases.iter().any(|c: &TestCase| c.name == name) {
            cases.push(TestCase::failure(name, Duration::ZERO, "cpu-tests", &found(e.test.as_deref())));
        }
    }
    junit::suite(&junit::classname("cpu-tests", cell), cases)
}

/// One finished run of the guest.
pub struct GuestRun {
    pub tap: Tap,
    pub crash: Option<String>,
    pub stdout: String,
    pub log: PathBuf,
}

impl GuestRun {
    /// The cell's verdict: empty when it passes. A cell with no section in
    /// expected-failures.txt fails, so a misspelt cell name cannot pass as
    /// one that expects nothing to fail.
    pub fn errors(&self, cell: &str) -> Vec<Finding> {
        let Some(want) = expected_failures(cell) else {
            return vec![Finding::run(format!(
                "no [{cell}] section in cpu-tests/expected-failures.txt: add one, empty if nothing is expected to fail"))];
        };
        let mut errs = judge(cell, &self.tap, &want);
        if let Some(c) = &self.crash {
            errs.insert(0, Finding::run(c.clone()));
        }
        errs
    }
}

/// Run the guest on `cpu` under this build's `iris`, with the cache model
/// `cache` (`auto`, `full` or `shadow`). An error is a run that never
/// finished (a timeout or a failure to start); a crash is reported by the run
/// and judged with it.
pub fn run_guest(cpu: &str, cache: &str, elf: &Path) -> Result<GuestRun, String> {
    let dir = run_dir(&format!("cpu-tests-{cpu}-{cache}"));
    let config = machine_config("cpu-tests/run/bare.toml", &dir);
    let log = dir.join("serial.log");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_iris"));
    cmd.current_dir(&dir)
        .arg("--config").arg(&config)
        .arg("--load-elf").arg(elf)
        .arg("--test-device")
        .arg("--test-device-dump").arg(dir.join("dump.json"))
        .args(["--headless", "--noaudio", "--cpu", cpu, "--cache", cache]);
    let out = run(cmd, timeout(TIMEOUT_S), &log)?;
    Ok(GuestRun { tap: parse_tap(&out.stdout), crash: out.crash(), stdout: out.stdout, log })
}
