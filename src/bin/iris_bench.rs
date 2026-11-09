//! `iris-bench` — drive the benchmark suite and turn its output into a report.
//!
//! The suite itself is `bench/`: a bare-metal MIPS binary that runs under IRIS
//! with no operating system, and the same C compiled natively for the host.
//! Both print the same machine-readable block. This program runs them, parses
//! it, and answers the three questions worth asking:
//!
//!   - **how fast** is a given build of IRIS, per kernel and overall, in guest
//!     instructions per host second;
//!   - **how correct** is it, as the share of kernels whose result checksum
//!     matched an independently computed golden value;
//!   - **where does the time go**, both as share of wall clock and as
//!     emulation efficiency, which are different lists.
//!
//! `matrix` builds each engine (a compile-time cargo feature) and runs every
//! CPU x engine combination, passing the CPU model as `--cpu` — so comparing
//! engines means comparing binaries, and comparing CPUs means comparing flags.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use serde::Deserialize;

// The report's data model, its parser and the reference table live in the
// library: this binary, the in-process runner and the GUI all need them, and
// "what accuracy means" should have exactly one definition.
use iris::bench_report::{
    find_features, fmt_rate, host_info, parse_block, reference_entry, MachineInfo, ReferenceTable, Row, Run,
    RunSettings, EXPECT_EXC,
};
use iris::bench_runner::{self, BenchOptions};
use iris::junit;
use iris::cpu::parts::{self, Part};

// A bin crate root resolves `mod` relative to src/bin/, so say where it is.
#[path = "iris_bench/ab.rs"]
mod ab;

// ─── running ─────────────────────────────────────────────────────────────────

/// Everything `bench/` produces lives under one directory so a report can be
/// assembled from whatever happens to be there.
fn default_out() -> PathBuf { PathBuf::from("bench/build/results") }

fn repo_relative(p: &str) -> PathBuf {
    // Run from the repo root by preference, but tolerate being run from
    // bench/ — every path in this program is repo-relative.
    let direct = PathBuf::from(p);
    if direct.exists() { return direct; }
    let up = PathBuf::from("..").join(p);
    if up.exists() { return up; }
    direct
}

/// The cargo features an emulator binary was built with, from the banner it
/// prints on stderr before it reads any configuration. `--help` gets that far
/// and exits.
fn probe_features(iris: &Path) -> Result<Vec<String>, String> {
    let mut cmd = Command::new(iris);
    cmd.arg("--help");
    let out = run_with_timeout(cmd, 60)?;
    find_features(&String::from_utf8_lossy(&out.stderr)).ok_or_else(|| format!(
        "{} printed no `build features:` line for --help, so its build cannot be checked. \
         Is it an iris binary?", iris.display()))
}

/// The emulator arguments that make a run deterministic for a binary built
/// with `features`, or why it cannot be.
fn deterministic_args(features: &[String]) -> Result<Vec<String>, String> {
    let has = |f: &str| features.iter().any(|x| x == f);
    if !has("ci_clock") {
        return Err(format!(
            "--deterministic needs a `ci_clock` build: its test device clock counts retired \
             instructions, which is what makes the suite pick the same work every run. This \
             binary was built with [{}], so the run would be timed and its counts would vary. \
             Rebuild with --features ci_clock.", features.join(" ")));
    }
    if has("lightning") && has("jitv2") {
        return Err("--deterministic needs `--jitv2-compile sync`, which a `lightning` build \
                    rejects because it always compiles on the pool, so a build with both \
                    `lightning` and `jitv2` cannot run it. Use lightning without jitv2, or \
                    jitv2 without lightning.".into());
    }
    // Lightning without the JIT has nothing to compile, and refuses the option.
    Ok(if has("lightning") { Vec::new() } else { ["--jitv2-compile", "sync"].map(String::from).to_vec() })
}

fn run_guest(
    iris: &Path,
    elf: &Path,
    config: &Path,
    label: &str,
    timeout_s: u64,
    extra: &[String],
) -> Result<Run, String> {
    if !iris.exists() { return Err(format!("no emulator at {}", iris.display())); }
    if !elf.exists() {
        return Err(format!("no suite binary at {} — run `make -C bench`", elf.display()));
    }

    // Run from bench/, where the suite's own relative paths resolve and where
    // the emulator's stray output files belong. Everything handed to the
    // emulator is absolute, so it does not matter what those paths looked like
    // on the way in — a --elf pointing somewhere else entirely still works.
    let abs = |p: &Path| -> Result<PathBuf, String> {
        std::fs::canonicalize(p).map_err(|e| format!("{}: {}", p.display(), e))
    };
    let (iris, elf, config) = (abs(iris)?, abs(elf)?, abs(config)?);
    let suite_id = suite_id_of(&elf)?;
    let cwd = config.parent().and_then(|p| p.parent()).map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    let mut cmd = Command::new(&iris);
    cmd.current_dir(&cwd)
        .arg("--config").arg(&config)
        .arg("--load-elf").arg(&elf)
        .arg("--test-device")
        .arg("--headless")
        .arg("--noaudio");
    for e in extra { cmd.arg(e); }

    let started = Instant::now();
    let out = run_with_timeout(cmd, timeout_s)?;
    let wall_s = started.elapsed().as_secs_f64();

    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let p = parse_block(&stdout).map_err(|e| {
        format!("{}\n{}", e, failure_context(&out.status, &stdout, &stderr))
    })?;

    Ok(Run {
        cell: label.to_string(),
        features: iris::bench_report::parse_features(&stderr),
        machine: p.machine,
        host: host_info(),
        rows: p.rows,
        checked: p.checked,
        matched: p.matched,
        total_ns: p.total_ns,
        total_icount: p.total_icount,
        wall_s,
        suite_id,
        settings: p.settings,
    })
}

fn suite_id_of(elf: &Path) -> Result<String, String> {
    let bytes = std::fs::read(elf).map_err(|e| format!("{}: {}", elf.display(), e))?;
    Ok(iris::benchsuite::suite_id_of(&bytes))
}

fn run_host(exe: &Path, timeout_s: u64) -> Result<Run, String> {
    if !exe.exists() {
        return Err(format!("no host build at {} — run `make -C bench hostbench`", exe.display()));
    }
    let started = Instant::now();
    let out = run_with_timeout(Command::new(exe), timeout_s)?;
    let wall_s = started.elapsed().as_secs_f64();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let mut p = parse_block(&stdout)?;
    p.machine.cpu = "host".to_string();
    Ok(Run {
        cell: "host".to_string(),
        features: Vec::new(),
        machine: p.machine,
        host: host_info(),
        rows: p.rows,
        checked: p.checked,
        matched: p.matched,
        total_ns: p.total_ns,
        total_icount: p.total_icount,
        wall_s,
        suite_id: String::new(),
        settings: p.settings,
    })
}

/// `--cpu`: a part or a family of the ones the suite measures.
fn bench_cpu(name: &str) -> Result<&'static Part, String> {
    parts::find(bench_runner::BENCH_CPUS, name).ok_or_else(|| format!("unknown cpu {name:?}"))
}

/// Run the suite inside this process, streaming the guest's console as it
/// arrives.
///
/// The guest prints its table a row at a time so that a run which shows nothing
/// for a minute is not mistaken for a hang, and that property is worth keeping
/// at the command line — so this echoes every line rather than waiting for the
/// end and printing the parsed summary.
fn run_embedded(label: &str, timeout_s: u64, quick: bool, cpu: &'static Part,
                cache: iris::config::CacheModel, deterministic: bool) -> Result<Run, String> {
    let opts = BenchOptions {
        quick,
        label: label.to_string(),
        timeout: Duration::from_secs(timeout_s),
        cpu,
        cache,
        deterministic,
        ..Default::default()
    };
    bench_runner::run(&opts, |p| {
        if let bench_runner::Progress::Line(l) = p {
            println!("{}", l);
        }
    })
}

/// A run's kernels as JUnit, filed under `bench.<cpu>.<engine>`: failed
/// where the checksum missed its golden value, skipped where there was none
/// to check.
fn junit_suite(cell: &str, rows: &[Row]) -> junit::TestSuite {
    use junit::{Duration, TestCase};
    let cases = rows.iter().map(|r| {
        let time = Duration::nanoseconds(r.ns as i64);
        match r.status.as_str() {
            "OK" => TestCase::success(&r.name, time),
            "MISMATCH" => TestCase::failure(&r.name, time, "checksum",
                &format!("checksum {} != golden {}", r.checksum, r.golden)),
            other => TestCase::skipped_with_cause(&r.name, "unchecked", &other.to_lowercase()),
        }
    });
    junit::suite(&junit::classname("bench", cell), cases)
}

/// What to show when an emulator run produced no report: how it ended, and
/// the tail of both streams. The reason is usually on stderr (a configuration
/// error, a refused option, a panic).
fn failure_context(status: &impl std::fmt::Display, stdout: &str, stderr: &str) -> String {
    format!(
        "emulator {}\n--- last 20 lines of emulator stderr ---\n{}\n\
         --- last 20 lines of emulator stdout ---\n{}",
        status, tail(stderr, 20), tail(stdout, 20))
}

/// Wait for a child, killing it after `timeout_s`. A benchmark that hangs is a
/// finding, not a reason to block a matrix run forever.
fn run_with_timeout(mut cmd: Command, timeout_s: u64) -> Result<std::process::Output, String> {
    use std::process::Stdio;
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("spawn: {}", e))?;

    // Drain both pipes on their own threads: a child that fills a pipe buffer
    // while we sleep would deadlock instead of timing out.
    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let t_out = std::thread::spawn(move || { let mut v = Vec::new(); let _ = std::io::copy(&mut so, &mut v); v });
    let t_err = std::thread::spawn(move || { let mut v = Vec::new(); let _ = std::io::copy(&mut se, &mut v); v });

    let deadline = Instant::now() + std::time::Duration::from_secs(timeout_s);
    let status = loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(s) => break s,
            None => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let status = child.wait().map_or("was killed".into(), |s| s.to_string());
                    // Killing closes the pipes, so the readers finish: what the
                    // child said before it hung is the best clue why.
                    let stdout = String::from_utf8_lossy(&t_out.join().unwrap_or_default()).into_owned();
                    let stderr = String::from_utf8_lossy(&t_err.join().unwrap_or_default()).into_owned();
                    return Err(format!("timed out after {}s\n{}", timeout_s,
                                       failure_context(&status, &stdout, &stderr)));
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    };
    let stdout = t_out.join().unwrap_or_default();
    let stderr = t_err.join().unwrap_or_default();
    Ok(std::process::Output { status, stdout, stderr })
}

fn tail(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// Write a result without replacing an earlier one: `<label>.json`, then
/// `<label>.2.json`, `<label>.3.json`. A rerun is another sample of that cell
/// and `report` combines them. `create_new` keeps two runs finishing together
/// from picking one name.
fn save(run: &Run, dir: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
    let json = serde_json::to_string_pretty(run).map_err(|e| e.to_string())?;
    for n in 1u32.. {
        let name = if n == 1 { format!("{}.json", run.cell) } else { format!("{}.{}.json", run.cell, n) };
        let path = dir.join(name);
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut f) => {
                f.write_all(json.as_bytes()).map_err(|e| format!("{}: {}", path.display(), e))?;
                return Ok(path);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("{}: {}", path.display(), e)),
        }
    }
    unreachable!()
}

/// Most recently modified `*.json` in `dir`, skipping the host run (it has no
/// suite_id and is not a machine anyone compares against).
fn newest_result(dir: &Path) -> Result<PathBuf, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("json") { continue; }
        // `host.json`, or `host.2.json` for a repeat.
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if stem.split('.').next() == Some("host") { continue; }
        let Ok(m) = e.metadata().and_then(|m| m.modified()) else { continue };
        if best.as_ref().map_or(true, |(t, _)| m > *t) { best = Some((m, p)); }
    }
    best.map(|(_, p)| p)
        .ok_or_else(|| format!("no result files in {} — run `iris-bench run` first", dir.display()))
}

fn load_all(dir: &Path) -> Result<Vec<Run>, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
    let mut paths: Vec<PathBuf> = rd.flatten().map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json")).collect();
    // Directory order is arbitrary; a report of the same files should read the same.
    paths.sort();
    let mut runs = Vec::new();
    for p in paths {
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {}", p.display(), e))?;
        match serde_json::from_str::<Run>(&text) {
            Ok(r) => runs.push(r),
            Err(e) => eprintln!("iris-bench: skipping {}: {}", p.display(), e),
        }
    }
    if runs.is_empty() { return Err(format!("no result files in {}", dir.display())); }
    // Emulated cells first, host last: the host is the reference, not a peer.
    runs.sort_by(|a, b| (a.cell == "host", &a.cell).cmp(&(b.cell == "host", &b.cell)));
    Ok(runs)
}

// ─── the matrix ──────────────────────────────────────────────────────────────

/// A cell is a cargo feature set plus a CPU. Only the feature set needs a build:
/// the CPU is a runtime setting, so the r4400 and r5000 cells of an engine share
/// one binary. `expect_cpu` is still checked, because a cell silently running
/// the wrong CPU is a mistake this repo has made before.
struct Cell {
    name: &'static str,
    features: &'static str,
    /// Passed as `--cpu`. Runtime since 96e5ddd, so cells that differ only by
    /// CPU share one emulator binary instead of each needing its own build.
    cpu: &'static str,
    /// What the guest must print in `#machine cpu=`. Checked, because an
    /// overwritten target/release/iris silently turning an "R4400" cell into
    /// an R5000 run is a mistake this repo has made before — see
    /// clanker/testing/benchmark-suite-gotchas.md.
    expect_cpu: &'static str,
}

const CELLS: &[Cell] = &[
    Cell { name: "r4400-interp",    features: "",           cpu: "r4400", expect_cpu: "R4400" },
    Cell { name: "r5000-interp",    features: "",           cpu: "r5000", expect_cpu: "R5000" },
    Cell { name: "r4400-jitv2",     features: "jitv2",      cpu: "r4400", expect_cpu: "R4400" },
    Cell { name: "r5000-jitv2",     features: "jitv2",      cpu: "r5000", expect_cpu: "R5000" },
    Cell { name: "r4400-lightning", features: "lightning",  cpu: "r4400", expect_cpu: "R4400" },
    Cell { name: "r4400-jitv2-lightning", features: "jitv2,lightning", cpu: "r4400", expect_cpu: "R4400" },
    // The R10000 family. It always runs the shadow caches, so the default
    // build covers it. Full chip names: a bare family name would pick the
    // chip of the first machine that lists it.
    Cell { name: "r10000-interp",   features: "",           cpu: "r10000_195", expect_cpu: "R10000" },
    Cell { name: "r12000-interp",   features: "",           cpu: "r12000_360", expect_cpu: "R12000" },
    Cell { name: "r14000-interp",   features: "",           cpu: "r14000_600", expect_cpu: "R14000" },
];

fn build_cell(cell: &Cell, root: &Path, force: bool) -> Result<PathBuf, String> {
    // Keyed by feature set: cells differing only in --cpu share a binary.
    let slug = if cell.features.is_empty() { "default".to_string() }
               else { cell.features.replace(',', "-") };
    let dest = root.join("bench/build").join(format!("iris-{}", slug));
    if dest.exists() && !force {
        println!("  reusing {}", dest.display());
        return Ok(dest);
    }
    println!("  building {} {}", cell.name,
             if cell.features.is_empty() { "(default features)".to_string() }
             else { format!("--features {}", cell.features) });

    let mut cmd = Command::new("cargo");
    cmd.current_dir(root).args(["build", "--release", "--bin", "iris"]);
    if !cell.features.is_empty() { cmd.args(["--features", cell.features]); }
    let st = cmd.status().map_err(|e| format!("cargo: {}", e))?;
    if !st.success() { return Err(format!("cargo build failed for {}", cell.name)); }

    let src = root.join("target/release/iris");
    std::fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
    // Copy rather than run in place: the next cell's build overwrites
    // target/release/iris, and a matrix that races its own artefacts produces
    // results labelled with the wrong build.
    std::fs::copy(&src, &dest).map_err(|e| format!("copy {}: {}", src.display(), e))?;
    Ok(dest)
}


// ─── the guest-OS level suite ────────────────────────────────────────────────
//
// bench/ measures the emulated CPU with no operating system in the way. This
// measures the machine as a user meets it: a filesystem on an emulated SCSI
// disk, IRIX's buffer cache and syscall path, the tools that shipped in the
// box, and the X server driving REX3. None of that is visible to a bare-metal
// kernel, and all of it is what "is the emulator fast enough to use" means.
//
// Every step is timed on the host around one `iris-ci run`, with the measured
// no-op round trip subtracted, so nothing depends on the guest having a usable
// clock or a working `time`.

/// The `suite_id` of a guest-OS run, which has no suite image to hash. It keeps
/// these runs out of the kernel tables and out of the reference table.
const GUEST_SUITE: &str = "guest-os";

/// Directories an IRIX 6.5 install actually puts programs in. `which` is a csh
/// script there and `command -v` is not in its Bourne shell, so a program is
/// probed by testing for it directly.
const IRIX_PATH: &str =
    "/bin:/usr/bin:/usr/sbin:/usr/bsd:/usr/etc:/usr/bin/X11:/usr/local/bin:/sbin:/usr/gfx";

#[derive(Debug, Deserialize)]
struct StepFile {
    step: Vec<Step>,
}

#[derive(Debug, Deserialize)]
struct Step {
    name: String,
    unit: String,
    /// Work units the command performs. 0 means "not a fixed quantity" (a
    /// `find` over /usr, an x11perf run) — the duration is still comparable
    /// between runs on the same disk image, the rate is not.
    #[serde(default)]
    work: u64,
    /// Untimed preparation.
    #[serde(default)]
    setup: Option<String>,
    cmd: String,
    /// Program that must exist, or the step is skipped rather than failed.
    #[serde(default)]
    requires: Option<String>,
    /// Run it, but do not record a row (cleanup).
    #[serde(default)]
    skip_timing: bool,
    /// Seconds the setup and the command may each take. A step that overruns
    /// is FAILED. Defaults to the `--timeout` of the run.
    #[serde(default)]
    timeout: Option<u64>,
}

/// Wrap a guest command so it runs under a Bourne shell with a sane PATH,
/// whatever the login shell is. Consequence, documented in steps.toml: a step
/// command may not contain a single quote.
fn wrap(cmd: &str) -> String {
    format!("sh -c 'PATH={}; export PATH; {}'", IRIX_PATH, cmd)
}

/// Shell that prints `IRISBENCH-HAVE` when `prog` is an executable the guest can
/// run: a name is looked up in `$PATH`, a path is tested as given. PATH is split
/// with IFS; `tr : \" \"` inside backquotes hands IRIX's tr literal `\"`
/// arguments, so it prints usage and no program was ever found.
///
/// The marker is typed as `HAV\E` so the tty's echo of the command line cannot
/// itself contain the word the output is searched for.
fn probe_command(prog: &str) -> String {
    if prog.contains('/') {
        format!("test -x {prog} && echo IRISBENCH-HAV\\E; true")
    } else {
        format!("IFS=:; for d in $PATH; do test -x $d/{prog} && echo IRISBENCH-HAV\\E; done; true")
    }
}

struct Ci {
    bin: PathBuf,
    socket: Option<PathBuf>,
    shell: String,
    timeout: u64,
}

/// How one command run in the guest ended, from `iris-ci run`'s exit status.
enum Outcome {
    /// `secs` is host wall time around the `iris-ci run`.
    Done { out: String, secs: f64 },
    /// The command ran and exited nonzero. `said` is iris-ci's message, which
    /// names the status, for the console.
    Failed { out: String, said: String },
    /// The guest did not finish in time. It may still be running the command.
    TimedOut,
}

impl Outcome {
    /// For commands whose failure means the harness is broken, not the step.
    fn done(self) -> Result<(String, f64), String> {
        match self {
            Outcome::Done { out, secs } => Ok((out, secs)),
            Outcome::Failed { said, .. } => Err(format!("guest command failed: {said}")),
            Outcome::TimedOut => Err("guest command timed out".into()),
        }
    }
}

/// `iris-ci run`'s exit statuses when the guest command exited nonzero and
/// when the guest did not finish in time (see src/iris_ci_main.rs).
const IRIS_CI_GUEST_FAILED: i32 = 4;
const IRIS_CI_TIMED_OUT: i32 = 5;

/// Read how an `iris-ci run` ended from its exit status. Any failure other
/// than the guest command's (no emulator, no socket) is an error: the run
/// cannot go on.
fn classify(status: Option<i32>, out: String, stderr: &str, secs: f64) -> Result<Outcome, String> {
    let said = stderr.trim().to_string();
    match status {
        Some(0) => Ok(Outcome::Done { out, secs }),
        Some(IRIS_CI_GUEST_FAILED) => Ok(Outcome::Failed { out, said }),
        Some(IRIS_CI_TIMED_OUT) => Ok(Outcome::TimedOut),
        _ => Err(format!("iris-ci run failed: {said}")),
    }
}

impl Ci {
    /// Run `guest_cmd`, waiting `secs` for it (`self.timeout` when `None`). An
    /// `Err` means the guest could not be asked; a command that ran and failed
    /// is an `Ok`.
    fn run(&self, guest_cmd: &str, secs: Option<u64>) -> Result<Outcome, String> {
        let secs = secs.unwrap_or(self.timeout);
        let mut cmd = Command::new(&self.bin);
        if let Some(s) = &self.socket { cmd.arg("--socket").arg(s); }
        cmd.arg("run")
            .arg(guest_cmd)
            .arg("--shell").arg(&self.shell)
            .arg("--timeout").arg(secs.to_string());
        let started = Instant::now();
        // iris-ci gets 30 s past the guest's deadline to report it, an
        // allowance with room to spare rather than a measured figure.
        let out = run_with_timeout(cmd, secs + 30)?;
        classify(
            out.status.code(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            &String::from_utf8_lossy(&out.stderr),
            started.elapsed().as_secs_f64(),
        )
    }

    /// Get the shell back after a step timed out: Ctrl-C is a raw byte on the
    /// serial line, with no carriage return. Then check that a no-op answers;
    /// if not, every later timing would be measured against a wedged tty. The
    /// waits (30 s to send, 2 s to settle, 30 s for the no-op) are generous
    /// guesses, not measured times.
    fn recover(&self) -> Result<(), String> {
        let mut cmd = Command::new(&self.bin);
        if let Some(s) = &self.socket { cmd.arg("--socket").arg(s); }
        cmd.arg("serial-send").arg("--no-cr").arg("\u{3}");
        if !run_with_timeout(cmd, 30)?.status.success() { return Err("iris-ci serial-send failed".into()); }
        std::thread::sleep(Duration::from_secs(2)); // let the tty flush and the prompt redraw
        match self.run(&wrap(":"), Some(30))? {
            Outcome::Done { .. } => Ok(()),
            _ => Err("the guest shell did not answer after Ctrl-C".into()),
        }
    }

    fn has(&self, prog: &str) -> Result<bool, String> {
        let (out, _) = self.run(&wrap(&probe_command(prog)), None)?.done()?;
        Ok(out.contains("IRISBENCH-HAVE"))
    }

    /// Smallest round trip we can measure: everything a step's timing has in
    /// common with doing nothing at all — serial latency, prompt matching,
    /// process spawn. Subtracted from every step, so a one-second step is a
    /// second of work rather than a second of work plus the harness.
    fn baseline(&self) -> Result<f64, String> {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let (_, s) = self.run(&wrap(":"), None)?.done()?;
            if s < best { best = s; }
        }
        Ok(best)
    }
}

fn row(step: &Step, ns: u64, status: &str) -> Row {
    Row {
        name: step.name.clone(),
        unit: step.unit.clone(),
        iters: 1,
        work: step.work,
        ns,
        icount: 0,
        count: 0,
        exc: 0,
        checksum: "0x0000000000000000".into(),
        golden: "0x0000000000000000".into(),
        status: status.into(),
    }
}

fn load_steps(paths: &[PathBuf]) -> Result<Vec<Step>, String> {
    let mut steps = Vec::new();
    for p in paths {
        let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {}", p.display(), e))?;
        let file: StepFile = toml::from_str(&text).map_err(|e| format!("{}: {}", p.display(), e))?;
        steps.extend(file.step);
    }
    Ok(steps)
}

/// How a guest step ended, for its JUnit case.
enum StepEnd<'a> {
    Timed(f64),
    Failed(&'a str),
    Skipped(&'a str),
}

/// A guest step as a JUnit case: passed with its time, failed with why, or
/// skipped with the program it needs.
fn step_case(name: &str, end: StepEnd) -> junit::TestCase {
    use junit::{Duration, TestCase};
    match end {
        StepEnd::Timed(secs) => TestCase::success(name, Duration::seconds_f64(secs)),
        StepEnd::Failed(why) => TestCase::failure(name, Duration::ZERO, "guest step", why),
        StepEnd::Skipped(why) => TestCase::skipped_with_cause(name, "not installed", why),
    }
}

/// The run, and a JUnit case for each of its rows.
fn run_irix(ci: &Ci, steps: &[Step], label: &str) -> Result<(Run, Vec<junit::TestCase>), String> {

    // Identify the guest before anything else: a result recorded against the
    // wrong disk image is worse than no result.
    let (uname, _) = ci.run(&wrap("uname -a"), None)?.done()?;
    let cpu = uname.lines().find(|l| l.contains("IRIX")).unwrap_or("IRIX").trim().to_string();
    println!("  guest: {}", cpu);

    let baseline = ci.baseline()?;
    println!("  round-trip floor: {:.0} ms", baseline * 1e3);

    let mut rows = Vec::new();
    let mut cases = Vec::new();
    let mut total_ns = 0u64;

    // A step that fails or is skipped gets a row of its own and the suite goes
    // on: one missing tool or broken workload says nothing about the next step.
    for step in steps {
        if let Some(prog) = &step.requires {
            if !ci.has(prog)? {
                println!("  {:<22} skipped ({} not installed)", step.name, prog);
                rows.push(row(step, 0, "SKIP"));
                cases.push(step_case(&step.name, StepEnd::Skipped(&format!("{prog} not installed"))));
                continue;
            }
        }
        let mut o = match &step.setup {
            Some(setup) => ci.run(&wrap(setup), step.timeout)?,
            None => Outcome::Done { out: String::new(), secs: 0.0 },
        };
        let in_setup = !matches!(o, Outcome::Done { .. });
        if !in_setup { o = ci.run(&wrap(&step.cmd), step.timeout)?; }

        let (secs, last) = match o {
            Outcome::Done { out, secs } => (secs, out.trim_end().lines().last().unwrap_or("").to_string()),
            bad => {
                let hung = matches!(bad, Outcome::TimedOut);
                let (why, out) = match bad {
                    Outcome::Failed { out, said } => (said.trim_start_matches("iris-ci: ").to_string(), out),
                    _ => (format!("did not finish in {} s", step.timeout.unwrap_or(ci.timeout)), String::new()),
                };
                let why = format!("{}{}", if in_setup { "setup: " } else { "" }, why);
                println!("  {:<22} FAILED ({why})", step.name);
                for l in tail(out.trim_end(), 3).lines() { println!("      {l}"); }
                rows.push(row(step, 0, "FAILED"));
                cases.push(step_case(&step.name, StepEnd::Failed(&why)));
                if hung { ci.recover()?; }
                continue;
            }
        };
        if step.skip_timing { continue; }

        let ns = ((secs - baseline).max(0.0) * 1e9) as u64;
        total_ns += ns;
        println!(
            "  {:<22} {:>8.2} s{}",
            step.name,
            ns as f64 / 1e9,
            if step.work > 0 {
                format!("   {}/s", fmt_rate(step.work as f64 * 1e9 / ns.max(1) as f64))
            } else {
                String::new()
            }
        );
        // Programs like Dhrystone print their own figure last.
        if !last.is_empty() { println!("      {last}"); }
        // Nothing here is checksummed: these are IRIX's own tools against
        // IRIX's own filesystem, and their output is not ours to predict.
        rows.push(row(step, ns, "UNCHECKED"));
        cases.push(step_case(&step.name, StepEnd::Timed(ns as f64 / 1e9)));
    }

    Ok((Run {
        cell: label.to_string(),
        features: Vec::new(),
        machine: MachineInfo { cpu, ..Default::default() },
        host: host_info(),
        rows,
        checked: 0,
        matched: 0,
        total_ns,
        total_icount: 0,
        wall_s: total_ns as f64 / 1e9,
        // The guest-OS suite runs IRIX's own tools, not the bare-metal binary,
        // so there is no suite hash and its numbers never join the reference
        // table — they are only comparable against runs on the same disk image.
        suite_id: GUEST_SUITE.into(),
        // Not the bare-metal harness, so its run configuration does not apply.
        settings: RunSettings::default(),
    }, cases))
}

// ─── reports ─────────────────────────────────────────────────────────────────

fn fmt_ratio(a: f64, b: f64) -> String {
    if b <= 0.0 || a <= 0.0 { return "-".into(); }
    let r = a / b;
    if r >= 100.0 { format!("{:.0}x", r) } else { format!("{:.2}x", r) }
}

/// A cell and its runs: a rerun is another sample of the same cell, so the
/// report combines them. Cells keep the order in which they first appear.
type Group<'a> = (&'a str, Vec<&'a Run>);

fn groups(runs: &[Run]) -> Vec<Group<'_>> {
    let mut g: Vec<Group<'_>> = Vec::new();
    for r in runs {
        match g.iter_mut().find(|(c, _)| *c == r.cell) {
            Some((_, v)) => v.push(r),
            None => g.push((&r.cell, vec![r])),
        }
    }
    g
}

/// Median, minimum, maximum and count of some samples.
type Stat = (f64, f64, f64, usize);

fn stat(mut v: Vec<f64>) -> Option<Stat> {
    if v.is_empty() { return None; }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let median = if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 };
    Some((median, v[0], v[n - 1], n))
}

/// `stat` of `f` over a group's runs; a run without a value is not a sample.
fn over(g: &Group, f: impl Fn(&Run) -> Option<f64>) -> Option<Stat> {
    stat(g.1.iter().filter_map(|r| f(r)).collect())
}

/// A group's median rate for one kernel, 0 where it has none.
fn median(g: &Group, kernel: &str) -> f64 { kernel_rate(g, kernel).map_or(0.0, |s| s.0) }

/// One kernel's rate over a group's runs; a run that measured nothing for it is not a sample.
fn kernel_rate(g: &Group, kernel: &str) -> Option<Stat> {
    over(g, |r| r.row(kernel).map(|x| x.rate()).filter(|v| *v > 0.0))
}

/// `median`, and `[min-max]` after it when there is more than one sample.
fn fmt_stat(s: Option<Stat>, f: impl Fn(f64) -> String) -> String {
    match s {
        None => "-".into(),
        Some((m, _, _, 1)) => f(m),
        Some((m, lo, hi, _)) => format!("{} [{}-{}]", f(m), f(lo), f(hi)),
    }
}

fn label(g: &Group) -> String {
    if g.1.len() > 1 { format!("{} (n={})", g.0, g.1.len()) } else { g.0.to_string() }
}

/// The run whose overall speed is the group's median, for the sections that
/// need one run's rows.
fn typical<'a>(g: &Group<'a>) -> &'a Run {
    let mut v = g.1.clone();
    v.sort_by(|a, b| a.mips().total_cmp(&b.mips()));
    v[(v.len() - 1) / 2]
}

/// A group is as correct as its worst run.
fn least_accurate<'a>(g: &Group<'a>) -> &'a Run {
    g.1.iter().copied().min_by(|a, b| a.accuracy().total_cmp(&b.accuracy())).unwrap()
}

fn mips_of(r: &Run) -> Option<f64> { Some(r.mips()).filter(|v| *v > 0.0) }

/// A cell's rate as a fraction of the host's own on the same kernel: `1/12`
/// when the emulator is twelve times slower, `1.50x` if it were faster.
fn fmt_native(rate: f64, host: f64) -> String {
    if rate <= 0.0 || host <= 0.0 { return "-".into(); }
    let slower = host / rate;
    if slower < 1.0 { format!("{:.2}x", rate / host) }
    else if slower < 10.0 { format!("1/{:.1}", slower) }
    else { format!("1/{:.0}", slower) }
}

/// The first 12 characters of a suite ID, enough to tell two images apart.
fn short_suite(id: &str) -> &str { if id.is_empty() { "-" } else { &id[..id.len().min(12)] } }

/// A `ci_clock` build's test device reads retired instructions, so every rate
/// it prints is a constant of the build, not a speed.
fn is_counted(r: &Run) -> bool { r.features.iter().any(|f| f == "ci_clock") }

/// A way two runs can differ that makes their rates incomparable: its name,
/// and a run's value on it. None where the axis does not apply: the host run
/// has no emulator, hence no clock or features, and no guest image.
type Axis = (&'static str, fn(&Run) -> Option<String>);

const AXES: [Axis; 5] = [
    // Rates and DMIPS count iterations of compiled guest code: another
    // compiler changed the instructions per Dhrystone iteration by 37%.
    ("guest image", |r| Some(short_suite(&r.suite_id).to_string()).filter(|_| !r.suite_id.is_empty())),
    ("run length", |r| Some(r.settings.shortened_because().map_or("full".into(), |w| format!("quick: {w}")))),
    ("clock", |r| (r.cell != "host").then(|| if is_counted(r) { "ci_clock" } else { "timed" }.into())),
    ("host CPU", |r| Some(format!("{} ({} {}, {} cores)", r.host.cpu_model, r.host.os, r.host.arch, r.host.cores))),
    // ci_clock is the clock axis, not a second difference.
    ("build features", |r| (r.cell != "host").then(|| {
        let f: Vec<&str> = r.features.iter().map(|f| f.as_str()).filter(|f| *f != "ci_clock").collect();
        if f.is_empty() { "(none)".into() } else { f.join(" ") }
    })),
];

/// The axes a report checks. Cells built with different features are meant to
/// be compared by `matrix`, so it drops the last one.
fn axes(across_features: bool) -> &'static [Axis] {
    if across_features { &AXES[..4] } else { &AXES }
}

/// The names of the axes on which `runs` do not all agree.
fn differing<'a>(runs: impl Iterator<Item = &'a Run> + Clone, axes: &[Axis]) -> Vec<&'static str> {
    axes.iter().filter(|(_, value)| {
        let mut v = runs.clone().filter_map(|r| value(r));
        v.next().map_or(false, |first| v.any(|x| x != first))
    }).map(|a| a.0).collect()
}

/// One warning per axis on which the runs disagree: a sentence, then each
/// value with the cells that had it. Both renderers print these.
fn warnings(runs: &[Run], axes: &[Axis]) -> Vec<(String, Vec<String>)> {
    axes.iter().filter(|a| !differing(runs.iter(), std::slice::from_ref(a)).is_empty()).map(|(name, value)| {
        let mut by_value: Vec<(String, Vec<&str>)> = Vec::new();
        for r in runs {
            let Some(v) = value(r) else { continue };
            match by_value.iter_mut().find(|(x, _)| *x == v) {
                Some((_, cells)) => if !cells.contains(&r.cell.as_str()) { cells.push(&r.cell) },
                None => by_value.push((v, vec![&r.cell])),
            }
        }
        (format!("These runs differ in {name}, so their rates are not comparable; \
                  ratios between cells that differ are shown as n/c."),
         by_value.iter().map(|(v, cells)| format!("{v}: {}", cells.join(", "))).collect())
    }).collect()
}

/// Every step of each guest run, in the order it ran. The kernel tables have
/// nothing to say about these: no accuracy, and steps like `find` have no rate.
fn guest_steps(guest: &[Run]) -> String {
    if guest.is_empty() { return String::new(); }
    let mut o = String::from("## Guest steps\n\nTimes are host wall clock around each step, \
                              less the round trip of an empty command.\n\n");
    for r in guest {
        o.push_str(&format!("### {}: {}\n\n| step | status | time | rate |\n|---|---|---:|---:|\n",
                            r.cell, r.machine.cpu));
        for row in &r.rows {
            let (status, timed) = match row.status.as_str() {
                "UNCHECKED" => ("ok", true),
                "SKIP" => ("skipped", false),
                other => (other, false),
            };
            let time = if timed { format!("{:.3} s", row.ns as f64 / 1e9) } else { "-".into() };
            let rate = if timed && row.rate() > 0.0 { format!("{} {}/s", fmt_rate(row.rate()), row.unit) } else { "-".into() };
            o.push_str(&format!("| {} | {} | {} | {} |\n", row.name, status, time, rate));
        }
        o.push('\n');
    }
    o
}

fn markdown(all: &[Run], baseline: Option<&str>, across_features: bool) -> String {
    let (guest, runs): (Vec<Run>, Vec<Run>) = all.iter().cloned().partition(|r| r.suite_id == GUEST_SUITE);
    let runs = &runs[..];
    let mut o = String::new();
    let (groups, axes) = (groups(runs), axes(across_features));
    let emulated: Vec<&Group> = groups.iter().filter(|g| g.0 != "host").collect();
    let host = groups.iter().find(|g| g.0 == "host");
    let base = baseline
        .and_then(|b| groups.iter().find(|g| g.0 == b))
        .or_else(|| emulated.first().copied());

    o.push_str("# IRIS benchmark report\n\n");
    if let Some(h) = all.first() {
        o.push_str(&format!(
            "Host: {} {} / {} / {} cores\n\n",
            h.host.os, h.host.arch, h.host.cpu_model, h.host.cores
        ));
    }
    if runs.is_empty() { return o + &guest_steps(&guest); }

    for (what, values) in warnings(runs, axes) {
        o.push_str(&format!("> **Warning:** {what}\n>\n"));
        for v in values { o.push_str(&format!("> - {v}\n")); }
        o.push('\n');
    }

    // ── per-cell summary ────────────────────────────────────────────────────
    o.push_str("## Cells\n\n");
    o.push_str("A cell run more than once shows the median, with `[min-max]` after it.\n\n");
    o.push_str("| cell | n | features | CPU | suite | accuracy | guest MIPS | DMIPS | whet/s | LINPACK MFLOPS | timed | wall |\n");
    o.push_str("|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for g in &groups {
        let r = g.1[0];
        let feats = if r.features.is_empty() { "-".to_string() } else { r.features.join(" ") };
        let worst = least_accurate(g);
        o.push_str(&format!(
            "| {} | {} | {} | {} | `{}` | {:.1}% ({}/{}) | {} | {} | {} | {} | {} | {} |\n",
            g.0, g.1.len(), feats, r.machine.cpu, short_suite(&r.suite_id),
            worst.accuracy(), worst.matched, worst.checked,
            fmt_stat(over(g, mips_of), |v| format!("{:.1}", v)),
            fmt_stat(over(g, |r| r.dmips()), |v| format!("{:.1}", v)),
            fmt_stat(over(g, |r| r.whet_loops()), |v| format!("{:.0}", v)),
            fmt_stat(over(g, |r| r.linpack_mflops()), |v| format!("{:.2}", v)),
            fmt_stat(over(g, |r| Some(r.total_ns as f64 / 1e9)), |v| format!("{:.1} s", v)),
            fmt_stat(over(g, |r| Some(r.wall_s)), |v| format!("{:.1} s", v)),
        ));
    }
    o.push('\n');

    // ── accuracy detail ─────────────────────────────────────────────────────
    let mut any_bad = false;
    for r in runs {
        for row in &r.rows {
            if row.status == "MISMATCH" {
                if !any_bad {
                    o.push_str("## Checksum mismatches\n\n");
                    o.push_str("A kernel whose result differs from the independently computed \
                                golden value. Either the emulator computed something wrong, or \
                                the kernel is not as deterministic as it claims — both are worth \
                                chasing.\n\n");
                    o.push_str("| cell | benchmark | got | want |\n|---|---|---|---|\n");
                    any_bad = true;
                }
                o.push_str(&format!("| {} | {} | `{}` | `{}` |\n",
                                    r.cell, row.name, row.checksum, row.golden));
            }
        }
    }
    if any_bad { o.push('\n'); }

    let mut any_exc = false;
    for r in runs {
        for row in &r.rows {
            if row.exc > 0 && !EXPECT_EXC.contains(&row.name.as_str()) {
                if !any_exc {
                    o.push_str("## Unexpected exceptions\n\n");
                    o.push_str("The harness steps over a faulting instruction and carries on, so a \
                                kernel that faults still reports a throughput — for doing something \
                                other than what it claims. Anything listed here is measuring the \
                                exception path.\n\n");
                    o.push_str("| cell | benchmark | exceptions |\n|---|---|---:|\n");
                    any_exc = true;
                }
                o.push_str(&format!("| {} | {} | {} |\n", r.cell, row.name, row.exc));
            }
        }
    }
    if any_exc { o.push('\n'); }

    // ── per-kernel rates ────────────────────────────────────────────────────
    o.push_str("## Per-kernel throughput\n\n");
    o.push_str("Work units per second, in each kernel's own unit. Each `vs base` column \
                is that cell's median rate over the median rate of **");
    o.push_str(base.map(|b| b.0).unwrap_or("-"));
    o.push_str("**");
    if host.is_some() {
        o.push_str("; each `native` column is that cell's median rate as a fraction of the host's \
                    own on the identical kernel");
    }
    o.push_str(".\n\n");

    let versus: Vec<&Group> = emulated.iter().copied()
        .filter(|g| base.map_or(false, |b| g.0 != b.0)).collect();
    o.push_str("| benchmark | unit |");
    for g in &emulated { o.push_str(&format!(" {} |", label(g))); }
    for g in &versus { o.push_str(&format!(" {} vs base |", g.0)); }
    if let Some(h) = host {
        o.push_str(&format!(" {} |", label(h)));
        for g in &emulated { o.push_str(&format!(" {} native |", g.0)); }
    }
    o.push('\n');
    o.push_str("|---|---|");
    let columns = emulated.len() + versus.len() + host.map_or(0, |_| 1 + emulated.len());
    for _ in 0..columns { o.push_str("---:|"); }
    o.push('\n');

    // A ratio between cells that differ on an axis is withheld, and a note
    // below the table says which. Against the host only the axes both have apply.
    let mut notes: Vec<String> = Vec::new();
    let mut ratio = |g: &Group, other: &Group, what: &str, f: fn(f64, f64) -> String, k: &str| {
        let d = differing(g.1.iter().chain(&other.1).copied(), axes);
        if d.is_empty() { return f(median(g, k), median(other, k)); }
        let note = format!("{} {what}: differs from {} in {}", g.0, other.0, d.join(", "));
        if !notes.contains(&note) { notes.push(note); }
        "n/c".into()
    };

    let names = ordered_names(runs);
    for name in &names {
        let unit = runs.iter().find_map(|r| r.row(name).map(|x| x.unit.clone()))
            .unwrap_or_default();
        o.push_str(&format!("| {} | {} |", name, unit));
        for g in &emulated { o.push_str(&format!(" {} |", fmt_stat(kernel_rate(g, name), fmt_rate))); }
        if let Some(b) = base {
            for g in &versus { o.push_str(&format!(" {} |", ratio(g, b, "vs base", fmt_ratio, name))); }
        }
        if let Some(h) = host {
            o.push_str(&format!(" {} |", fmt_stat(kernel_rate(h, name), fmt_rate)));
            for g in &emulated { o.push_str(&format!(" {} |", ratio(g, h, "native", fmt_native, name))); }
        }
        o.push('\n');
    }
    o.push('\n');
    if !notes.is_empty() {
        o.push_str("`n/c` means not comparable, and no ratio is shown:\n\n");
        for n in &notes { o.push_str(&format!("- {n}\n")); }
        o.push('\n');
    }

    // ── efficiency ──────────────────────────────────────────────────────────
    for g in &emulated {
        let r = typical(g);
        if r.mips() <= 0.0 { continue; }
        o.push_str(&format!("## Where {} spends its time\n\n", g.0));
        if g.1.len() > 1 {
            o.push_str(&format!("The median-speed run of {}.\n\n", g.1.len()));
        }

        let mut by_time: Vec<&Row> = r.rows.iter().filter(|x| x.ns > 0).collect();
        by_time.sort_by(|a, b| b.ns.cmp(&a.ns));
        o.push_str("| benchmark | share of wall clock | guest MIPS |\n|---|---:|---:|\n");
        for row in by_time.iter().take(8) {
            o.push_str(&format!("| {} | {:.1}% | {:.1} |\n",
                row.name, row.ns as f64 * 100.0 / r.total_ns.max(1) as f64, row.mips()));
        }
        o.push('\n');

        let mut by_mips: Vec<&Row> = r.rows.iter().filter(|x| x.icount > 0).collect();
        by_mips.sort_by(|a, b| a.mips().partial_cmp(&b.mips()).unwrap());
        o.push_str("Least efficient — the kernels where the emulator does the most host work \
                    per guest instruction:\n\n");
        o.push_str("| benchmark | guest MIPS | vs this cell's average |\n|---|---:|---:|\n");
        let avg = r.mips();
        for row in by_mips.iter().take(8) {
            o.push_str(&format!("| {} | {:.1} | {:.2}x |\n",
                row.name, row.mips(), if avg > 0.0 { row.mips() / avg } else { 0.0 }));
        }
        o.push('\n');
    }

    o + &guest_steps(&guest)
}

/// Every kernel name that appears anywhere, in the order the first run lists
/// them — the suite's own order, which groups related kernels together.
fn ordered_names(runs: &[Run]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for r in runs {
        for row in &r.rows {
            if !names.iter().any(|n| n == &row.name) { names.push(row.name.clone()); }
        }
    }
    names
}

fn text_summary(all: &[Run], across_features: bool) -> String {
    let (guest, runs): (Vec<Run>, Vec<Run>) = all.iter().cloned().partition(|r| r.suite_id == GUEST_SUITE);
    let runs = &runs[..];
    let mut o = String::new();
    for r in &guest {
        let n = |status: &str| r.rows.iter().filter(|x| x.status == status).count();
        o.push_str(&format!("{:<24} guest steps: {} timed, {} skipped, {} failed   {:>7.1} s timed\n",
                            r.cell, n("UNCHECKED"), n("SKIP"), n("FAILED"), r.total_ns as f64 / 1e9));
    }
    for g in groups(runs) {
        o.push_str(&format!(
            "{:<24} {:>6.1}% accuracy   {:>8}   {:>7} DMIPS   {:>7} timed\n",
            label(&g), least_accurate(&g).accuracy(),
            match over(&g, mips_of) {
                Some(s) => format!("{} MIPS", fmt_stat(Some(s), |v| format!("{:.1}", v))),
                None => "n/a".into(),
            },
            fmt_stat(over(&g, |r| r.dmips()), |v| format!("{:.1}", v)),
            fmt_stat(over(&g, |r| Some(r.total_ns as f64 / 1e9)), |v| format!("{:.1} s", v)),
        ));
    }
    for (what, values) in warnings(runs, axes(across_features)) {
        o.push_str(&format!("warning: {what}\n"));
        for v in values { o.push_str(&format!("  {v}\n")); }
    }
    o
}



// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "iris-bench",
    about = "Run the IRIS benchmark suite and report on it.",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run the suite once.
    ///
    /// In-process by default: this binary *is* an emulator, and the guest image
    /// is linked into it, so there is no subprocess, no ELF on disk and nothing
    /// platform-specific. Pass `--iris` to measure a different emulator binary
    /// instead, which is what `matrix` does.
    Run {
        /// Measure this emulator binary in a subprocess instead of this one
        /// in-process. Needs --elf too, or a built bench/build/irisbench.elf.
        #[arg(long)]
        iris: Option<PathBuf>,
        /// Suite binary for --iris. Defaults to bench/build/irisbench.elf.
        #[arg(long)]
        elf: Option<PathBuf>,
        /// Machine config for --iris. Defaults to bench/run/bare.toml.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Name this result. Defaults to "local".
        #[arg(long, default_value = "local")]
        label: String,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 1800)]
        timeout: u64,
        /// Shorter timed runs and one pass per kernel instead of best of two.
        /// Every kernel still runs and still verifies, so accuracy means the
        /// same thing; only the rates are noisier. Not accepted with --iris —
        /// the register that carries it is set at machine construction.
        #[arg(long)]
        quick: bool,
        /// Emulated CPU for the in-process run. With --iris, forward it to the
        /// emulator instead: `-- --cpu r5000`.
        #[arg(long, default_value = "r4400", value_parser = bench_cpu)]
        cpu: &'static Part,
        /// Cache model for the in-process run: auto (shadow in a jitv2
        /// build), full or shadow. With --iris: `-- --cache shadow`.
        #[arg(long, default_value = "auto")]
        cache: iris::config::CacheModel,
        /// Make two runs do the same work: JIT compiles on the CPU thread and,
        /// in a `ci_clock` build (required in process), the suite reads a clock
        /// driven by retired instructions, so it picks the same iteration
        /// counts every time. The rates it prints are then
        /// constant by construction; compare instruction counts and host cost.
        /// The result is labelled `<label>+deterministic`. With --iris the
        /// emulator's `build features:` line is read first and the run is
        /// refused unless it has `ci_clock`; `--jitv2-compile sync` is passed
        /// where the build allows it (not on a lightning build).
        #[arg(long)]
        deterministic: bool,
        /// Also write a JUnit report here: one case per kernel, failed when
        /// its checksum does not match the golden value.
        #[arg(long)]
        junit: Option<PathBuf>,
        /// Extra arguments passed through to --iris.
        #[arg(last = true)]
        extra: Vec<String>,
    },

    /// Measure the machine the emulator runs on, with the same kernels.
    Host {
        /// Host build of the suite. Defaults to bench/build/irisbench-host.
        #[arg(long)]
        exe: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 600)]
        timeout: u64,
    },

    /// Build every CPU x engine cell and run all of them.
    Matrix {
        /// Comma-separated cell names. Default: the four core cells.
        #[arg(long)]
        cells: Option<String>,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Rebuild even when a cached emulator binary exists.
        #[arg(long)]
        force_build: bool,
        /// Skip the host baseline run.
        #[arg(long)]
        no_host: bool,
        #[arg(long, default_value_t = 1800)]
        timeout: u64,
    },

    /// Turn saved results into a report.
    Report {
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Cell to express speedups against. Defaults to the first emulated cell.
        #[arg(long)]
        baseline: Option<String>,
        #[arg(long, value_parser = ["md", "json", "text"], default_value = "md")]
        format: String,
        /// Show ratios between cells built with different cargo features. By
        /// default they are withheld, since a feature change is a different
        /// emulator; `matrix` is the case where that is the point.
        #[arg(long)]
        across_features: bool,
        /// Write here instead of stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },

    /// Run the guest-OS level suite against a booted, logged-in IRIX.
    ///
    /// Needs an emulator started with `--ci`, a guest sitting at a shell
    /// prompt, and `iris-ci` on hand. Unlike every other subcommand this one
    /// measures the machine as a user meets it — filesystem, buffer cache,
    /// syscalls, IRIX's own tools, and X on REX3.
    Irix {
        /// CI socket. Defaults to iris-ci's own default.
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Step definitions; repeat to run several files in order. Defaults to
        /// bench/irix/steps.toml, then bench/irix/benchies.toml.
        #[arg(long)]
        steps: Vec<PathBuf>,
        /// iris-ci binary. Defaults to target/release/iris-ci.
        #[arg(long)]
        ci: Option<PathBuf>,
        /// Guest login shell, passed through to `iris-ci run`.
        #[arg(long, default_value = "csh")]
        shell: String,
        #[arg(long, default_value = "irix")]
        label: String,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Per-command timeout inside the guest.
        #[arg(long, default_value_t = 300)]
        timeout: u64,
        /// Also write a JUnit report here: one case per step, failed where
        /// the step failed or timed out, skipped where its program is missing.
        #[arg(long)]
        junit: Option<PathBuf>,
    },

    /// Print a `data/bench_reference.json` row for a saved result.
    ///
    /// The reference table is a static file updated by hand — run the suite on
    /// a machine, run this, paste the row in, commit. There is no upload and no
    /// user-writable override; a machine with no row simply has no comparison.
    Reference {
        /// Result JSON to convert. Defaults to the newest in the results dir.
        #[arg(long)]
        from: Option<PathBuf>,
        /// Short stable identifier, e.g. `m1-max-interp`.
        #[arg(long)]
        id: String,
        /// Human label. Defaults to "<host cpu> — <engine>".
        #[arg(long)]
        label: Option<String>,
        /// Measurement date, `YYYY-MM-DD`. Defaults to today.
        #[arg(long)]
        measured: Option<String>,
        /// Merge into this table in place instead of printing the row.
        #[arg(long)]
        into: Option<PathBuf>,
    },

    /// List the cells `matrix` knows about.
    Cells,

    /// Compare two emulator builds with paired runs.
    ///
    /// Runs the same guest image under A and B for N rounds and flags the
    /// kernels whose paired B/A ratio is clearly past the threshold.
    Ab(ab::AbArgs),
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = dispatch(cli.cmd) {
        eprintln!("iris-bench: {}", e);
        std::process::exit(1);
    }
}

fn dispatch(cmd: Cmd) -> Result<(), String> {
    match cmd {
        Cmd::Ab(args) => ab::run(args),

        Cmd::Cells => {
            println!("{:<24} {:<22} {}", "cell", "cargo features", "--cpu");
            for c in CELLS {
                println!("{:<24} {:<22} {}", c.name,
                         if c.features.is_empty() { "(default)" } else { c.features }, c.cpu);
            }
            Ok(())
        }

        Cmd::Run { iris, elf, config, label, out, timeout, quick, cpu, cache, deterministic, junit, mut extra } => {
            let out = out.unwrap_or_else(default_out);
            let label = if deterministic { format!("{label}+deterministic") } else { label };
            let run = match iris {
                Some(iris) => {
                    if quick {
                        return Err("--quick needs the in-process runner; drop --iris".into());
                    }
                    let elf = elf.unwrap_or_else(|| repo_relative("bench/build/irisbench.elf"));
                    let config = config.unwrap_or_else(|| repo_relative("bench/run/bare.toml"));
                    if deterministic {
                        // Refuse before the run: a binary without ci_clock
                        // gives a timed run, whatever the label says.
                        let args = deterministic_args(&probe_features(&iris)?)
                            .map_err(|e| format!("{}: {}", iris.display(), e))?;
                        extra.extend(args);
                    }
                    run_guest(&iris, &elf, &config, &label, timeout, &extra)?
                }
                None => {
                    // Silently ignoring these would be worse than refusing:
                    // they all describe a subprocess that is not being started,
                    // and a run that quietly measured the wrong thing is the
                    // failure mode this whole suite exists to avoid.
                    let stray = [
                        elf.is_some().then_some("--elf"),
                        config.is_some().then_some("--config"),
                        (!extra.is_empty()).then_some("trailing emulator arguments"),
                    ];
                    let stray: Vec<&str> = stray.into_iter().flatten().collect();
                    if !stray.is_empty() {
                        let (verb, obj) = if stray.len() == 1 { ("applies", "it") }
                                          else { ("apply", "them") };
                        return Err(format!(
                            "{} only {} to --iris, and `run` is in-process by default. \
                             Add --iris PATH, or drop {}.", stray.join(" and "), verb, obj));
                    }
                    run_embedded(&label, timeout, quick, cpu, cache, deterministic)?
                }
            };
            let path = save(&run, &out)?;
            if let Some(junit) = junit {
                junit::write(junit_suite(&run.cell, &run.rows), &junit)?;
            }
            print!("{}", text_summary(std::slice::from_ref(&run), false));
            println!("wrote {}", path.display());
            Ok(())
        }

        Cmd::Host { exe, out, timeout } => {
            let exe = exe.unwrap_or_else(|| repo_relative("bench/build/irisbench-host"));
            let out = out.unwrap_or_else(default_out);
            let run = run_host(&exe, timeout)?;
            let path = save(&run, &out)?;
            print!("{}", text_summary(std::slice::from_ref(&run), false));
            println!("wrote {}", path.display());
            Ok(())
        }

        Cmd::Matrix { cells, out, force_build, no_host, timeout } => {
            let root = if PathBuf::from("Cargo.toml").exists() { PathBuf::from(".") }
                       else { PathBuf::from("..") };
            let out = out.unwrap_or_else(default_out);
            let wanted: Vec<&Cell> = match &cells {
                Some(list) => {
                    let names: Vec<&str> = list.split(',').map(str::trim).collect();
                    let sel: Vec<&Cell> = CELLS.iter().filter(|c| names.contains(&c.name)).collect();
                    for n in &names {
                        if !CELLS.iter().any(|c| &c.name == n) {
                            return Err(format!("unknown cell '{}' (try `iris-bench cells`)", n));
                        }
                    }
                    sel
                }
                // The four that answer "which CPU, which engine". The
                // lightning cells are opt-in: they trade away breakpoints and
                // the traceback buffer, so they are a release-build question
                // rather than a default comparison.
                None => CELLS.iter().take(4).collect(),
            };

            let elf = root.join("bench/build/irisbench.elf");
            if !elf.exists() {
                return Err(format!("no suite binary at {} — run `make -C bench` first", elf.display()));
            }
            let config = root.join("bench/run/bare.toml");

            let mut failures = Vec::new();
            for cell in &wanted {
                println!("== {} ==", cell.name);
                let iris = match build_cell(cell, &root, force_build) {
                    Ok(p) => p,
                    Err(e) => { eprintln!("  {}", e); failures.push(cell.name); continue; }
                };
                let cpu_arg = ["--cpu".to_string(), cell.cpu.to_string()];
                match run_guest(&iris, &elf, &config, cell.name, timeout, &cpu_arg) {
                    Ok(run) => {
                        // The guest reads PRId, so its banner is the authority
                        // on which CPU actually ran.
                        if run.machine.cpu != cell.expect_cpu {
                            eprintln!("  FAIL {} — guest reports cpu={}, expected {}",
                                      cell.name, run.machine.cpu, cell.expect_cpu);
                            failures.push(cell.name);
                            continue;
                        }
                        let path = save(&run, &out)?;
                        print!("  ");
                        print!("{}", text_summary(std::slice::from_ref(&run), false));
                        println!("  wrote {}", path.display());
                    }
                    Err(e) => { eprintln!("  FAIL {} — {}", cell.name, e); failures.push(cell.name); }
                }
            }

            if !no_host {
                println!("== host ==");
                let exe = root.join("bench/build/irisbench-host");
                match run_host(&exe, timeout) {
                    Ok(run) => { let p = save(&run, &out)?; println!("  wrote {}", p.display()); }
                    Err(e) => eprintln!("  host baseline skipped: {}", e),
                }
            }

            println!();
            if failures.is_empty() { println!("matrix: every cell completed"); }
            else { println!("matrix: {} cell(s) failed: {}", failures.len(), failures.join(" ")); }

            let runs = load_all(&out)?;
            // Its cells differ in features by design.
            let md = markdown(&runs, None, true);
            let report = out.join("report.md");
            std::fs::write(&report, &md).map_err(|e| format!("{}: {}", report.display(), e))?;
            println!("report: {}", report.display());
            if !failures.is_empty() { return Err("some cells failed".into()); }
            Ok(())
        }

        Cmd::Irix { socket, steps, ci, shell, label, out, timeout, junit } => {
            let ci_bin = ci.unwrap_or_else(|| repo_relative(
                if cfg!(windows) { "target/release/iris-ci.exe" } else { "target/release/iris-ci" }));
            if !ci_bin.exists() {
                return Err(format!("no iris-ci at {} — cargo build --release --bin iris-ci",
                                   ci_bin.display()));
            }
            let paths = if steps.is_empty() {
                vec![repo_relative("bench/irix/steps.toml"), repo_relative("bench/irix/benchies.toml")]
            } else { steps };
            let steps = load_steps(&paths)?;
            let out = out.unwrap_or_else(default_out);
            let ci = Ci { bin: ci_bin, socket, shell, timeout };
            let (run, cases) = run_irix(&ci, &steps, &label)?;
            let path = save(&run, &out)?;
            if let Some(junit) = junit {
                junit::write(junit::suite(&junit::classname("irix", &label), cases), &junit)?;
            }
            println!("wrote {}", path.display());
            Ok(())
        }

        Cmd::Reference { from, id, label, measured, into } => {
            let path = match from {
                Some(p) => p,
                None => newest_result(&default_out())?,
            };
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("{}: {}", path.display(), e))?;
            let run: Run = serde_json::from_str(&text)
                .map_err(|e| format!("{}: {}", path.display(), e))?;
            if run.suite_id == GUEST_SUITE {
                return Err(format!(
                    "{} is a guest-OS run. Its numbers depend on the disk image and are \
                     never reference rows.", path.display()));
            }
            if run.suite_id.is_empty() {
                return Err(format!(
                    "{} has no suite_id — it predates the field, or it is a host run. \
                     Re-run the suite to record one.", path.display()));
            }
            // A ci_clock run reports constant rates, so a row made from one
            // would compare every machine against a number that is not a speed.
            if is_counted(&run) {
                return Err(format!(
                    "{} is a ci_clock run: its rates are constants of the build, not \
                     speeds. Reference rows must come from a timed run.", path.display()));
            }
            // A shortened run is accurate but imprecise, and the table is what
            // every other machine gets compared against. Refuse rather than
            // quietly enshrine a noisy row.
            if let Some(why) = run.settings.shortened_because() {
                return Err(format!(
                    "{} was not a full run ({}). Reference rows must be full runs — \
                     re-run without --quick.", path.display(), why));
            }
            let entry = reference_entry(&run, &id, label.as_deref(), measured.as_deref());

            let Some(table_path) = into else {
                println!("{}", serde_json::to_string_pretty(&entry).map_err(|e| e.to_string())?);
                eprintln!("\n// paste into data/bench_reference.json \"entries\" \
                           (suite_id {})", run.suite_id);
                return Ok(());
            };

            let mut table: ReferenceTable = match std::fs::read_to_string(&table_path) {
                Ok(t) => serde_json::from_str(&t).map_err(|e| format!("{}: {}", table_path.display(), e))?,
                Err(_) => ReferenceTable { schema: 1, suite_id: run.suite_id.clone(), entries: Vec::new() },
            };
            // An empty table adopts the incoming suite; a populated one must
            // agree, or the merged file would hold two different workloads'
            // numbers under one name.
            if table.entries.is_empty() {
                table.suite_id = run.suite_id.clone();
            } else if table.suite_id != run.suite_id {
                return Err(format!(
                    "suite mismatch: {} holds {} but this result is {}. \
                     The suite changed — regenerate every row, or start a new table.",
                    table_path.display(), table.suite_id, run.suite_id));
            }
            table.entries.retain(|e| e.id != entry.id);
            table.entries.push(entry);
            table.entries.sort_by(|a, b| a.id.cmp(&b.id));
            std::fs::write(&table_path, serde_json::to_string_pretty(&table).map_err(|e| e.to_string())? + "\n")
                .map_err(|e| format!("{}: {}", table_path.display(), e))?;
            println!("{}: {} entries (suite {})", table_path.display(), table.entries.len(), table.suite_id);
            Ok(())
        }

        Cmd::Report { dir, baseline, format, across_features, out } => {
            let dir = dir.unwrap_or_else(default_out);
            let runs = load_all(&dir)?;
            let text = match format.as_str() {
                "md" => markdown(&runs, baseline.as_deref(), across_features),
                "text" => text_summary(&runs, across_features),
                "json" => serde_json::to_string_pretty(&runs).map_err(|e| e.to_string())?,
                _ => unreachable!(),
            };
            match out {
                Some(p) => {
                    std::fs::write(&p, &text).map_err(|e| format!("{}: {}", p.display(), e))?;
                    println!("wrote {}", p.display());
                }
                None => { let mut so = std::io::stdout(); let _ = so.write_all(text.as_bytes()); }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_guest_step_is_a_passed_failed_or_skipped_case() {
        let s = junit::suite(&junit::classname("irix", "indy"), [
            step_case("disk/write", StepEnd::Timed(1.5)),
            step_case("gfx/xwd_root", StepEnd::Failed("did not finish in 60 s")),
            step_case("gfx/x11perf_rects", StepEnd::Skipped("x11perf not installed")),
        ]);
        assert_eq!((s.tests(), s.failures(), s.skipped()), (3, 1, 1));
        assert_eq!(s.testcases[0].time, junit::Duration::seconds_f64(1.5));
        assert_eq!(s.testcases[1].classname.as_deref(), Some("irix.indy"));
        assert!(matches!(&s.testcases[1].result, junit::TestResult::Failure { message, .. }
            if message == "did not finish in 60 s"));
    }

    #[test]
    fn junit_fails_a_mismatched_kernel_and_skips_an_unchecked_one() {
        let row = |name: &str, status: &str| Row {
            name: name.into(), unit: "op".into(), iters: 1, work: 1, ns: 1_500_000_000,
            icount: 1, count: 1, exc: 0, checksum: "0x1".into(), golden: "0x2".into(),
            status: status.into(),
        };
        let rows = [row("int/alu", "OK"), row("int/muldiv", "MISMATCH"), row("sys/x", "UNCHECKED")];
        let suite = junit_suite("r4400-interp", &rows);
        assert_eq!((suite.tests(), suite.failures(), suite.skipped()), (3, 1, 1));
        let muldiv = &suite.testcases[1];
        assert_eq!(muldiv.classname.as_deref(), Some("bench.r4400.interp"));
        assert!(matches!(&muldiv.result, junit::TestResult::Failure { message, .. }
            if message == "checksum 0x1 != golden 0x2"));
    }

    /// Only what is local to this binary. The report model, its parser and the
    /// derived figures moved to `iris::bench_report` and are tested there —
    /// keeping a second copy here is how two parsers drift apart.
    #[test]
    fn every_cell_name_is_unique_and_maps_to_a_cpu() {
        for (i, a) in CELLS.iter().enumerate() {
            // `expect_cpu` is what the `matrix` guard compares the guest's
            // `#machine cpu=` line against, so it must describe the `--cpu`
            // this cell actually passes. CPU is a runtime flag (see `Cell::cpu`),
            // not a cargo feature, so the two agreeing is the whole contract.
            let want = a.cpu.split('_').next().unwrap().to_uppercase();
            assert_eq!(a.expect_cpu, want, "{}", a.name);
            assert!(bench_cpu(a.cpu).is_ok());
            for b in CELLS.iter().skip(i + 1) { assert_ne!(a.name, b.name); }
        }
    }

    fn row(name: &str, work: u64, ns: u64) -> Row {
        Row {
            name: name.into(), unit: "ops".into(), iters: 1, work, ns,
            icount: 0, count: 0, exc: 0,
            checksum: "0x0".into(), golden: "0x0".into(), status: "OK".into(),
        }
    }

    /// A run of `work` units of int/alu in `ns`, on a fixed image.
    fn run_rates(cell: &str, work: u64, ns: u64) -> Run {
        Run {
            cell: cell.into(),
            features: Vec::new(),
            machine: Default::default(),
            host: Default::default(),
            rows: vec![row("int/alu", work, ns)],
            checked: 0,
            matched: 0,
            total_ns: 0,
            total_icount: 0,
            wall_s: 0.0,
            suite_id: "blake3:3bc1c0000000".into(),
            settings: Default::default(),
        }
    }

    /// The cells of the per-kernel table's row for int/alu.
    fn alu_cells(md: &str) -> Vec<String> {
        let line = md.lines().find(|l| l.starts_with("| int/alu |"))
            .unwrap_or_else(|| panic!("no int/alu row in:\n{md}"));
        line.trim_matches('|').split('|').map(|c| c.trim().to_string()).collect()
    }

    /// "vs base" and "native" divided the best of all cells by the base or the
    /// host, so a cell slower than the base still read 1.00x.
    #[test]
    fn each_cell_is_compared_with_the_base_and_the_host() {
        let s = 1_000_000_000;
        let runs = [
            run_rates("base", 100, s), run_rates("faster", 125, s),
            run_rates("slower", 86, s), run_rates("host", 1000, s),
        ];
        // int/alu, unit, three cells, two vs base, host, three native
        let c = alu_cells(&markdown(&runs, Some("base"), false));
        assert_eq!(&c[5..7], ["1.25x", "0.86x"]);
        assert_eq!(&c[8..], ["1/10", "1/8.0", "1/12"]);
        assert_eq!(fmt_native(1500.0, 1000.0), "1.50x");
        // no base named: the first emulated cell is the base
        assert_eq!(alu_cells(&markdown(&runs[..2], None, false))[4], "1.25x");
    }

    /// A scratch directory under the system temp dir, removed on drop.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let p = std::env::temp_dir().join(format!("iris-bench-test-{}-{}", std::process::id(), name));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }

    /// `save` wrote `<label>.json`, so a rerun replaced the earlier result.
    #[test]
    fn a_rerun_is_kept_and_host_repeats_are_not_results() {
        let dir = TempDir::new("rerun");
        let names: Vec<_> = (1..=3).map(|n| {
            let p = save(&run_rates("cell", 100, n * 1_000_000_000), &dir.0).unwrap();
            p.file_name().unwrap().to_str().unwrap().to_string()
        }).collect();
        assert_eq!(names, ["cell.json", "cell.2.json", "cell.3.json"]);
        assert_eq!(load_all(&dir.0).unwrap().len(), 3);
        let want = save(&run_rates("emu", 1, 1), &dir.0).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        save(&run_rates("host", 1, 1), &dir.0).unwrap();
        save(&run_rates("host", 1, 1), &dir.0).unwrap();
        assert_eq!(newest_result(&dir.0).unwrap(), want);
    }

    /// Three samples of one cell are one column, not three with one heading,
    /// and ratios use each cell's median.
    #[test]
    fn repeated_runs_report_median_min_max_and_count() {
        assert_eq!(stat(vec![3.0, 1.0, 2.0]), Some((2.0, 1.0, 3.0, 3)));
        assert_eq!(stat(vec![4.0, 1.0, 2.0, 3.0]).unwrap().0, 2.5);
        assert_eq!(stat(Vec::new()), None);
        let s = 1_000_000_000;
        let runs = [
            run_rates("base", 100, s), run_rates("base", 100, s), run_rates("base", 40, s),
            run_rates("new", 80, s), run_rates("new", 90, s), run_rates("new", 120, s),
        ];
        let md = markdown(&runs, Some("base"), false);
        assert_eq!(md.matches("base (n=3)").count(), 1, "{md}");
        let c = alu_cells(&md);
        assert_eq!((c[2].as_str(), c[3].as_str(), c[4].as_str()),
                   ("100.0 [40.0-100.0]", "90.0 [80.0-120.0]", "0.90x"), "{md}");
        assert!(text_summary(&runs, false).starts_with("base (n=3)"));
    }

    /// A run of int/alu at `rate` units/s, differing in what a report compares.
    fn run_with(cell: &str, rate: u64, host: &str, features: &[&str], quick: bool) -> Run {
        let mut r = run_rates(cell, rate, 1_000_000_000);
        r.host.cpu_model = host.into();
        r.features = features.iter().map(|s| s.to_string()).collect();
        if quick { r.settings.time_pct = 30; r.settings.repeats = 1; }
        r
    }

    /// Each axis on which two runs can differ must warn in both renderers and
    /// replace the ratio with n/c, and runs that agree must stay quiet.
    #[test]
    fn runs_that_differ_on_an_axis_warn_and_withhold_their_ratio() {
        let a = || run_with("a", 100, "cpu", &[], false);
        let mut other_image = run_with("b", 80, "cpu", &[], false);
        other_image.suite_id = "blake3:be5200000000".into();
        let cases = [
            ("guest image", other_image),
            ("run length", run_with("b", 80, "cpu", &[], true)),
            ("host CPU", run_with("b", 80, "other cpu", &[], false)),
            ("build features", run_with("b", 80, "cpu", &["jitv2"], false)),
            ("clock", run_with("b", 100, "cpu", &["ci_clock"], false)),
        ];
        for (axis, b) in cases {
            let md = markdown(&[a(), b.clone()], Some("a"), false);
            assert_eq!(alu_cells(&md)[4], "n/c", "{axis}: {md}");
            assert!(md.contains(&format!("**Warning:** These runs differ in {axis}")), "{axis}: {md}");
            assert!(md.contains(&format!("b vs base: differs from a in {axis}")), "{axis}: {md}");
            let text = text_summary(&[a(), b], false);
            assert!(text.contains(&format!("warning: These runs differ in {axis}")), "{axis}: {text}");
        }
        let md = markdown(&[a(), run_with("b", 90, "cpu", &[], false)], Some("a"), false);
        assert_eq!(alu_cells(&md)[4], "0.90x");
        assert!(!md.contains("Warning"), "{md}");
        // --across-features and matrix compare feature sets on purpose
        let jit = run_with("b", 400, "cpu", &["jitv2"], false);
        assert_eq!(alu_cells(&markdown(&[a(), jit], Some("a"), true))[4], "4.00x");
    }

    /// The host run has no image, clock or features to differ on, but its
    /// CPU has to be the emulated cells' CPU.
    #[test]
    fn a_warning_names_each_value_and_the_host_run_is_checked_on_cpu_only() {
        let host = |cpu| [run_with("a", 100, "epyc 7443", &["tlbvmap"], false), run_with("host", 1000, cpu, &[], false)];
        assert_eq!(alu_cells(&markdown(&host("epyc 7443"), None, false)).last().unwrap(), "1/10");
        let md = markdown(&host("epyc 7663"), None, false);
        assert_eq!(alu_cells(&md).last().unwrap(), "n/c", "{md}");
        assert!(md.contains("a native: differs from host in host CPU"), "{md}");
        assert!(md.contains("epyc 7443 (") && md.contains("epyc 7663 ("), "{md}");
        assert!(is_counted(&run_with("x", 1, "cpu", &["lightning", "ci_clock"], false)));
        assert!(!is_counted(&run_with("x", 1, "cpu", &["lightning"], false)));
    }

    /// A failed run showed only "no IRIS-BENCH-BEGIN" and stdout's tail, though
    /// the reason (a refused option, a panic) was on stderr.
    #[test]
    fn a_failed_run_shows_the_status_and_both_streams() {
        let e = failure_context(&"exit status: 3", "some stdout", "Configuration error: no sync\n");
        assert!(e.contains("emulator exit status: 3") && e.contains("some stdout"), "{e}");
        assert!(e.contains("Configuration error: no sync"), "{e}");
    }

    /// `run --deterministic --iris` on a binary without ci_clock ran, and saved
    /// a timed result labelled "+deterministic".
    #[test]
    fn deterministic_needs_ci_clock_and_picks_arguments_by_build() {
        let f = |list: &str| -> Vec<String> { list.split_whitespace().map(String::from).collect() };
        let e = deterministic_args(&f("tlbvmap")).unwrap_err();
        assert!(e.contains("ci_clock") && e.contains("[tlbvmap]"), "{e}");
        assert!(deterministic_args(&[]).is_err());
        assert_eq!(deterministic_args(&f("jitv2 ci_clock")).unwrap(), ["--jitv2-compile", "sync"]);
        // lightning refuses --jitv2-compile sync, and has no JIT to need it
        assert!(deterministic_args(&f("lightning ci_clock")).unwrap().is_empty());
        // with the JIT it needs the sync compile that lightning rejects
        assert!(deterministic_args(&f("lightning jitv2 ci_clock")).is_err());
    }

    /// Run a guest command line through the local Bourne shell, the way the
    /// guest's `sh -c` does after `iris-ci` types it.
    #[cfg(unix)]
    fn local_sh(cmd: &str) -> bool {
        let out = Command::new("sh").arg("-c").arg(cmd).output().unwrap();
        String::from_utf8_lossy(&out.stdout).contains("IRISBENCH-HAVE")
    }

    #[cfg(unix)]
    #[test]
    fn probe_finds_a_program_by_name_or_path_and_not_a_missing_one() {
        assert!(local_sh(&probe_command("sh")));
        assert!(!local_sh(&probe_command("irisbench-no-such-program")));
        let sh = ["/bin/sh", "/usr/bin/sh"].into_iter().find(|p| Path::new(p).exists()).unwrap();
        assert!(local_sh(&probe_command(sh)));
        assert!(!local_sh(&probe_command("/nonexistent/irisbench")));
    }

    /// The tty echoes the typed line into the captured output, so the line must
    /// not contain the word it is searched for.
    #[test]
    fn the_probe_command_line_does_not_contain_its_own_answer() {
        assert!(!probe_command("dd").contains("IRISBENCH-HAVE"));
        assert!(!probe_command("/usr/bin/dd").contains("IRISBENCH-HAVE"));
    }

    fn shipped_steps() -> Vec<Step> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/irix");
        load_steps(&[dir.join("steps.toml"), dir.join("benchies.toml")]).unwrap()
    }

    /// IRIX's dd has no `m` suffix and its ping refuses `-i 0`; either made a
    /// step fail. The buffer cache still holds what disk/write just wrote, so
    /// the cold read must read a file its own setup wrote.
    #[test]
    fn the_shipped_steps_stay_valid_on_irix() {
        let steps = shipped_steps();
        for s in &steps {
            for c in std::iter::once(&s.cmd).chain(s.setup.iter()) {
                assert!(!c.contains('\''), "{}: the sh -c wrapper has no room for a quote", s.name);
                assert!(!c.contains("bs=1m") && !c.contains(" -i 0"), "{}: {c}", s.name);
            }
        }
        for (i, s) in steps.iter().enumerate() {
            assert!(steps[..i].iter().all(|t| t.name != s.name), "{} repeats", s.name);
        }
        let cold = steps.iter().find(|s| s.name == "disk/read_cold").unwrap();
        assert!(!cold.cmd.contains("irisbench.dat"));
        assert!(cold.setup.as_deref().unwrap().contains("hinv -c memory"));
    }

    /// The cold read's setup, run by the local shell with hinv, dd, rm and
    /// sync stubbed out: its exit status and what it ran.
    #[cfg(unix)]
    fn cold_setup(hinv: &str, dd_status: u8) -> (bool, String) {
        let steps = shipped_steps();
        let setup = steps.iter().find(|s| s.name == "disk/read_cold").unwrap().setup.clone().unwrap();
        // The setup sends dd's output to /dev/null, so the stubs report on fd 3.
        let stubs = format!("exec 3>&1; hinv() {{ echo \"{hinv}\"; }}; \
                             dd() {{ echo dd \"$@\" >&3; return {dd_status}; }}; \
                             rm() {{ echo rm \"$@\" >&3; }}; sync() {{ echo sync >&3; }}; ");
        let out = Command::new("sh").arg("-c").arg(stubs + &setup).output().unwrap();
        (out.status.success(), String::from_utf8_lossy(&out.stdout).into_owned())
    }

    #[cfg(unix)]
    #[test]
    fn the_cold_read_setup_writes_one_and_a_half_memories_or_fails_cleanly() {
        let (ok, out) = cold_setup("Main memory size: 256 Mbytes", 0);
        assert!(ok && out.contains("count=384") && out.contains("sync"), "{out}");
        // no number from hinv: fail before writing anything
        let (ok, out) = cold_setup("", 0);
        assert!(!ok && !out.contains("dd ") && out.contains("no memory size"), "{out}");
        let (ok, out) = cold_setup("Main memory size: lots", 0);
        assert!(!ok && !out.contains("dd "), "{out}");
        // a write that fails takes its partial file with it
        let (ok, out) = cold_setup("Main memory size: 256 Mbytes", 1);
        assert!(!ok && out.contains("rm -f /tmp/irisbench2.dat") && !out.contains("sync"), "{out}");
    }

    #[test]
    fn iris_ci_s_exit_status_says_how_a_run_ended() {
        assert!(matches!(classify(Some(0), String::new(), "", 1.0), Ok(Outcome::Done { .. })));
        match classify(Some(4), "x\n".into(), "iris-ci: guest exit 3\n", 1.0) {
            Ok(Outcome::Failed { out, said }) => assert_eq!((out.as_str(), said.as_str()), ("x\n", "iris-ci: guest exit 3")),
            _ => panic!("status 4 is a failed step"),
        }
        assert!(matches!(classify(Some(5), String::new(), "iris-ci: wait-serial: timeout after 5000ms\n", 5.0),
                         Ok(Outcome::TimedOut)));
        // The status decides, not the words.
        assert!(classify(Some(2), String::new(), "iris-ci: iris error: guest exit 3\n", 1.0).is_err());
        assert!(classify(Some(2), String::new(), "iris-ci: iris error: wait-serial: timeout after 5000ms\n", 1.0).is_err());
        // No emulator to talk to: the run stops, it is not a step result.
        assert!(classify(Some(1), String::new(), "iris-ci: connect /x: refused\n", 1.0).is_err());
        assert!(classify(None, String::new(), "", 1.0).is_err());
    }

    fn guest_run() -> Run {
        let step = |name: &str| Step {
            name: name.into(), unit: "B".into(), work: 1 << 20, setup: None,
            cmd: String::new(), requires: None, skip_timing: false, timeout: None,
        };
        Run {
            cell: "indy".into(), features: Vec::new(), machine: MachineInfo::default(),
            host: host_info(), checked: 0, matched: 0, total_ns: 500_000_000, total_icount: 0,
            wall_s: 0.5, suite_id: GUEST_SUITE.into(), settings: RunSettings::default(),
            rows: vec![super::row(&step("disk/write"), 500_000_000, "UNCHECKED"),
                       super::row(&step("gfx/xwd_root"), 0, "FAILED"),
                       super::row(&step("gfx/x11perf_rects"), 0, "SKIP")],
        }
    }

    #[test]
    fn the_report_shows_guest_steps_apart_from_the_kernel_tables() {
        let md = markdown(&[guest_run()], None, false);
        assert!(md.contains("| disk/write | ok | 0.500 s |"), "{md}");
        assert!(md.contains("| gfx/xwd_root | FAILED | - |"), "{md}");
        assert!(md.contains("| gfx/x11perf_rects | skipped | - |"), "{md}");
        assert!(!md.contains("## Cells"), "{md}");
        assert!(text_summary(&[guest_run()], false).contains("1 timed, 1 skipped, 1 failed"));
    }
}
