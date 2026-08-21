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
//! `matrix` builds each CPU x engine combination and runs all of them, because
//! the CPU model and the JIT are compile-time cargo features — comparing them
//! means comparing binaries, not flags.

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
    fmt_rate, host_info, parse_block, reference_entry, MachineInfo, ReferenceTable, Row, Run,
    RunSettings, EXPECT_EXC,
};
use iris::bench_runner::{self, BenchOptions};

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
        format!("{}\n--- last 20 lines of emulator output ---\n{}", e, tail(&stdout, 20))
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

/// Run the suite inside this process, streaming the guest's console as it
/// arrives.
///
/// The guest prints its table a row at a time so that a run which shows nothing
/// for a minute is not mistaken for a hang, and that property is worth keeping
/// at the command line — so this echoes every line rather than waiting for the
/// end and printing the parsed summary.
fn run_embedded(label: &str, timeout_s: u64, quick: bool) -> Result<Run, String> {
    let opts = BenchOptions {
        quick,
        label: label.to_string(),
        timeout: Duration::from_secs(timeout_s),
        ..Default::default()
    };
    bench_runner::run(&opts, |p| {
        if let bench_runner::Progress::Line(l) = p {
            println!("{}", l);
        }
    })
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
                    let _ = child.wait();
                    return Err(format!("timed out after {}s", timeout_s));
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

fn save(run: &Run, dir: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
    let path = dir.join(format!("{}.json", run.cell));
    let json = serde_json::to_string_pretty(run).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| format!("{}: {}", path.display(), e))?;
    Ok(path)
}

/// Most recently modified `*.json` in `dir`, skipping the host run (it has no
/// suite_id and is not a machine anyone compares against).
fn newest_result(dir: &Path) -> Result<PathBuf, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("json") { continue; }
        if p.file_stem().and_then(|s| s.to_str()) == Some("host") { continue; }
        let Ok(m) = e.metadata().and_then(|m| m.modified()) else { continue };
        if best.as_ref().map_or(true, |(t, _)| m > *t) { best = Some((m, p)); }
    }
    best.map(|(_, p)| p)
        .ok_or_else(|| format!("no result files in {} — run `iris-bench run` first", dir.display()))
}

fn load_all(dir: &Path) -> Result<Vec<Run>, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
    let mut runs = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("json") { continue; }
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

/// A cell is a cargo feature set plus the CPU the guest must report. The CPU
/// model and the JIT are compile-time features, so each cell is a separate
/// build of the emulator — there is no runtime switch to flip.
struct Cell {
    name: &'static str,
    features: &'static str,
    /// What the guest must print in `#machine cpu=`. Checked, because an
    /// overwritten target/release/iris silently turning an "R4400" cell into
    /// an R5000 run is a mistake this repo has made before — see
    /// cpu-tests/run/matrix.sh.
    expect_cpu: &'static str,
}

const CELLS: &[Cell] = &[
    Cell { name: "r4400-interp",    features: "",                   expect_cpu: "R4400" },
    Cell { name: "r5000-interp",    features: "r5k",                expect_cpu: "R5000" },
    Cell { name: "r4400-jitv2",     features: "jitv2",              expect_cpu: "R4400" },
    Cell { name: "r5000-jitv2",     features: "r5k,jitv2",          expect_cpu: "R5000" },
    Cell { name: "r4400-lightning", features: "lightning",          expect_cpu: "R4400" },
    Cell { name: "r4400-jitv2-lightning", features: "jitv2,lightning", expect_cpu: "R4400" },
];

fn build_cell(cell: &Cell, root: &Path, force: bool) -> Result<PathBuf, String> {
    let dest = root.join("bench/build").join(format!("iris-{}", cell.name));
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
}

/// Wrap a guest command so it runs under a Bourne shell with a sane PATH,
/// whatever the login shell is. Consequence, documented in steps.toml: a step
/// command may not contain a single quote.
fn wrap(cmd: &str) -> String {
    format!("sh -c 'PATH={}; export PATH; {}'", IRIX_PATH, cmd)
}

struct Ci {
    bin: PathBuf,
    socket: Option<PathBuf>,
    shell: String,
    timeout: u64,
}

impl Ci {
    /// Returns the guest's stdout. An iris-ci failure is an error; a nonzero
    /// exit status inside the guest is not — several steps end on a command
    /// that legitimately returns nonzero.
    fn run(&self, guest_cmd: &str) -> Result<(String, f64), String> {
        let mut cmd = Command::new(&self.bin);
        if let Some(s) = &self.socket { cmd.arg("--socket").arg(s); }
        cmd.arg("run")
            .arg(guest_cmd)
            .arg("--shell").arg(&self.shell)
            .arg("--timeout").arg(self.timeout.to_string());
        let started = Instant::now();
        let out = run_with_timeout(cmd, self.timeout + 30)?;
        let secs = started.elapsed().as_secs_f64();
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        if !out.status.success() && stdout.trim().is_empty() {
            return Err(format!(
                "iris-ci run failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok((stdout, secs))
    }

    fn has(&self, prog: &str) -> Result<bool, String> {
        let probe = format!(
            "for d in `echo {} | tr : \\\" \\\"`; do test -x $d/{} && echo IRISBENCH-HAVE; done",
            IRIX_PATH, prog
        );
        let (out, _) = self.run(&wrap(&probe))?;
        Ok(out.contains("IRISBENCH-HAVE"))
    }

    /// Smallest round trip we can measure: everything a step's timing has in
    /// common with doing nothing at all — serial latency, prompt matching,
    /// process spawn. Subtracted from every step, so a one-second step is a
    /// second of work rather than a second of work plus the harness.
    fn baseline(&self) -> Result<f64, String> {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let (_, s) = self.run(&wrap(":"))?;
            if s < best { best = s; }
        }
        Ok(best)
    }
}

fn run_irix(ci: &Ci, steps_path: &Path, label: &str) -> Result<Run, String> {
    let text = std::fs::read_to_string(steps_path)
        .map_err(|e| format!("{}: {}", steps_path.display(), e))?;
    let file: StepFile = toml::from_str(&text)
        .map_err(|e| format!("{}: {}", steps_path.display(), e))?;

    // Identify the guest before anything else: a result recorded against the
    // wrong disk image is worse than no result.
    let (uname, _) = ci.run(&wrap("uname -a"))?;
    let cpu = uname.lines().find(|l| l.contains("IRIX")).unwrap_or("IRIX").trim().to_string();
    println!("  guest: {}", cpu);

    let baseline = ci.baseline()?;
    println!("  round-trip floor: {:.0} ms", baseline * 1e3);

    let mut rows = Vec::new();
    let mut total_ns = 0u64;

    for step in &file.step {
        if let Some(prog) = &step.requires {
            if !ci.has(prog)? {
                println!("  {:<22} skipped ({} not installed)", step.name, prog);
                continue;
            }
        }
        if let Some(setup) = &step.setup { ci.run(&wrap(setup))?; }

        let (_, secs) = ci.run(&wrap(&step.cmd))?;
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
        rows.push(Row {
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
            // Nothing here is checksummed: these are IRIX's own tools against
            // IRIX's own filesystem, and their output is not ours to predict.
            status: "UNCHECKED".into(),
        });
    }

    Ok(Run {
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
        suite_id: String::new(),
        // Not the bare-metal harness, so its run configuration does not apply.
        settings: RunSettings::default(),
    })
}

// ─── reports ─────────────────────────────────────────────────────────────────

fn fmt_ratio(a: f64, b: f64) -> String {
    if b <= 0.0 || a <= 0.0 { return "-".into(); }
    let r = a / b;
    if r >= 100.0 { format!("{:.0}x", r) } else { format!("{:.2}x", r) }
}

fn markdown(runs: &[Run], baseline: Option<&str>) -> String {
    let mut o = String::new();
    let emulated: Vec<&Run> = runs.iter().filter(|r| r.cell != "host").collect();
    let host = runs.iter().find(|r| r.cell == "host");
    let base = baseline
        .and_then(|b| runs.iter().find(|r| r.cell == b))
        .or_else(|| emulated.first().copied());

    o.push_str("# IRIS benchmark report\n\n");
    if let Some(h) = runs.first() {
        o.push_str(&format!(
            "Host: {} {} / {} / {} cores\n\n",
            h.host.os, h.host.arch, h.host.cpu_model, h.host.cores
        ));
    }

    // ── per-cell summary ────────────────────────────────────────────────────
    o.push_str("## Cells\n\n");
    o.push_str("| cell | features | CPU | accuracy | guest MIPS | DMIPS | whet/s | LINPACK MFLOPS | timed | wall |\n");
    o.push_str("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for r in runs {
        let feats = if r.features.is_empty() { "-".to_string() } else { r.features.join(" ") };
        o.push_str(&format!(
            "| {} | {} | {} | {:.1}% ({}/{}) | {} | {} | {} | {} | {:.1} s | {:.1} s |\n",
            r.cell, feats, r.machine.cpu,
            r.accuracy(), r.matched, r.checked,
            if r.mips() > 0.0 { format!("{:.1}", r.mips()) } else { "n/a".into() },
            r.dmips().map(|v| format!("{:.1}", v)).unwrap_or_else(|| "-".into()),
            r.whet_loops().map(|v| format!("{:.0}", v)).unwrap_or_else(|| "-".into()),
            r.linpack_mflops().map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".into()),
            r.total_ns as f64 / 1e9, r.wall_s,
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
    o.push_str("Work units per second, in each kernel's own unit. `vs base` is \
                relative to **");
    o.push_str(base.map(|b| b.cell.as_str()).unwrap_or("-"));
    o.push_str("**");
    if host.is_some() {
        o.push_str("; `native` is the fraction of the host's own rate on the identical kernel");
    }
    o.push_str(".\n\n");

    o.push_str("| benchmark | unit |");
    for r in &emulated { o.push_str(&format!(" {} |", r.cell)); }
    if base.is_some() && emulated.len() > 1 { o.push_str(" vs base |"); }
    if host.is_some() { o.push_str(" host | native |"); }
    o.push('\n');
    o.push_str("|---|---|");
    for _ in &emulated { o.push_str("---:|"); }
    if base.is_some() && emulated.len() > 1 { o.push_str("---:|"); }
    if host.is_some() { o.push_str("---:|---:|"); }
    o.push('\n');

    let names = ordered_names(runs);
    for name in &names {
        let unit = runs.iter().find_map(|r| r.row(name).map(|x| x.unit.clone()))
            .unwrap_or_default();
        o.push_str(&format!("| {} | {} |", name, unit));
        for r in &emulated {
            o.push_str(&format!(" {} |", r.row(name).map(|x| fmt_rate(x.rate()))
                                             .unwrap_or_else(|| "-".into())));
        }
        if base.is_some() && emulated.len() > 1 {
            let b = base.unwrap().row(name).map(|x| x.rate()).unwrap_or(0.0);
            let best = emulated.iter().filter_map(|r| r.row(name)).map(|x| x.rate())
                .fold(0.0f64, f64::max);
            o.push_str(&format!(" {} |", fmt_ratio(best, b)));
        }
        if let Some(h) = host {
            let hv = h.row(name).map(|x| x.rate()).unwrap_or(0.0);
            let best = emulated.iter().filter_map(|r| r.row(name)).map(|x| x.rate())
                .fold(0.0f64, f64::max);
            o.push_str(&format!(" {} |", fmt_rate(hv)));
            o.push_str(&format!(" {} |", if hv > 0.0 && best > 0.0 {
                format!("1/{:.0}", hv / best)
            } else { "-".into() }));
        }
        o.push('\n');
    }
    o.push('\n');

    // ── efficiency ──────────────────────────────────────────────────────────
    for r in &emulated {
        if r.mips() <= 0.0 { continue; }
        o.push_str(&format!("## Where {} spends its time\n\n", r.cell));

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

    o
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

fn text_summary(runs: &[Run]) -> String {
    let mut o = String::new();
    for r in runs {
        o.push_str(&format!(
            "{:<24} {:>6.1}% accuracy   {:>8}   {:>7} DMIPS   {:>7.1} s timed\n",
            r.cell, r.accuracy(),
            if r.mips() > 0.0 { format!("{:.1} MIPS", r.mips()) } else { "n/a".into() },
            r.dmips().map(|v| format!("{:.1}", v)).unwrap_or_else(|| "-".into()),
            r.total_ns as f64 / 1e9,
        ));
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
        /// Step definitions. Defaults to bench/irix/steps.toml.
        #[arg(long)]
        steps: Option<PathBuf>,
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
        Cmd::Cells => {
            println!("{:<24} {}", "cell", "cargo features");
            for c in CELLS {
                println!("{:<24} {}", c.name,
                         if c.features.is_empty() { "(default)" } else { c.features });
            }
            Ok(())
        }

        Cmd::Run { iris, elf, config, label, out, timeout, quick, extra } => {
            let out = out.unwrap_or_else(default_out);
            let run = match iris {
                Some(iris) => {
                    if quick {
                        return Err("--quick needs the in-process runner; drop --iris".into());
                    }
                    let elf = elf.unwrap_or_else(|| repo_relative("bench/build/irisbench.elf"));
                    let config = config.unwrap_or_else(|| repo_relative("bench/run/bare.toml"));
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
                    run_embedded(&label, timeout, quick)?
                }
            };
            let path = save(&run, &out)?;
            print!("{}", text_summary(std::slice::from_ref(&run)));
            println!("wrote {}", path.display());
            Ok(())
        }

        Cmd::Host { exe, out, timeout } => {
            let exe = exe.unwrap_or_else(|| repo_relative("bench/build/irisbench-host"));
            let out = out.unwrap_or_else(default_out);
            let run = run_host(&exe, timeout)?;
            let path = save(&run, &out)?;
            print!("{}", text_summary(std::slice::from_ref(&run)));
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
                match run_guest(&iris, &elf, &config, cell.name, timeout, &[]) {
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
                        print!("{}", text_summary(std::slice::from_ref(&run)));
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
            let md = markdown(&runs, None);
            let report = out.join("report.md");
            std::fs::write(&report, &md).map_err(|e| format!("{}: {}", report.display(), e))?;
            println!("report: {}", report.display());
            if !failures.is_empty() { return Err("some cells failed".into()); }
            Ok(())
        }

        Cmd::Irix { socket, steps, ci, shell, label, out, timeout } => {
            let ci_bin = ci.unwrap_or_else(|| repo_relative(
                if cfg!(windows) { "target/release/iris-ci.exe" } else { "target/release/iris-ci" }));
            if !ci_bin.exists() {
                return Err(format!("no iris-ci at {} — cargo build --release --bin iris-ci",
                                   ci_bin.display()));
            }
            let steps = steps.unwrap_or_else(|| repo_relative("bench/irix/steps.toml"));
            let out = out.unwrap_or_else(default_out);
            let ci = Ci { bin: ci_bin, socket, shell, timeout };
            let run = run_irix(&ci, &steps, &label)?;
            let path = save(&run, &out)?;
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
            if run.suite_id.is_empty() {
                return Err(format!(
                    "{} has no suite_id — it predates the field, or it is a host run. \
                     Re-run the suite to record one.", path.display()));
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

        Cmd::Report { dir, baseline, format, out } => {
            let dir = dir.unwrap_or_else(default_out);
            let runs = load_all(&dir)?;
            let text = match format.as_str() {
                "md" => markdown(&runs, baseline.as_deref()),
                "text" => text_summary(&runs),
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

    const SAMPLE: &str = "\
noise before the block
IRIS-BENCH-BEGIN v1
#machine cpu=R4400 prid=0x00000440 fir=0x00000500 config=0x00c08483 l2=1 testdev=1 timebase=1
#timebase count_hz=33000000 measured=1
#work base=0x88300000 bytes=25165824
#cols name unit iters work ns icount count exc checksum golden status
int/alu ops 16384 262144 250000000 12500000 8250 0 0x1111 0x1111 OK
int/dhrystone dhry 1000 1757000 1000000000 50000000 33000 0 0x2222 0x3333 MISMATCH
sys/tlb_hit xlat 256 24576 250000000 5000000 8250 4096 0x0000 0x0000 UNCHECKED
#totals benches=3 checked=2 matched=1 ns=1500000000 icount=67500000
IRIS-BENCH-END
";

    #[test]
    fn parses_a_report_block_out_of_surrounding_noise() {
        let (m, rows, checked, matched, ns, ic) = parse_block(SAMPLE).unwrap();
        assert_eq!(m.cpu, "R4400");
        assert_eq!(m.count_hz, 33_000_000);
        assert!(m.timebase && m.testdev && m.l2);
        assert_eq!(m.work_bytes, 25_165_824);
        assert_eq!(rows.len(), 3);
        assert_eq!((checked, matched), (2, 1));
        assert_eq!((ns, ic), (1_500_000_000, 67_500_000));
    }

    #[test]
    fn rates_and_derived_units() {
        let (_, rows, checked, matched, ns, ic) = parse_block(SAMPLE).unwrap();
        let run = Run {
            cell: "t".into(), features: vec![], machine: Machine::default(),
            host: HostInfo::default(), rows, checked, matched,
            total_ns: ns, total_icount: ic, wall_s: 2.0,
            suite_id: "blake3:0000000000000000".into(),
        };
        // 262144 work units in 0.25 s
        assert!((run.row("int/alu").unwrap().rate() - 1_048_576.0).abs() < 1.0);
        // 12.5M instructions in 0.25 s = 50 MIPS
        assert!((run.row("int/alu").unwrap().mips() - 50.0).abs() < 0.01);
        // 1,757,000 dhrystones in 1 s / 1757 = 1000 DMIPS
        assert!((run.dmips().unwrap() - 1000.0).abs() < 0.01);
        assert!(run.whet_loops().is_none(), "the sample block has no whetstone row");
        assert!((run.accuracy() - 50.0).abs() < 0.01);
        assert!((run.mips() - 45.0).abs() < 0.01);
    }

    #[test]
    fn a_truncated_run_is_an_error_not_an_empty_report() {
        let cut = SAMPLE.split(END).next().unwrap();
        assert!(parse_block(cut).unwrap_err().contains("IRIS-BENCH-END"));
        assert!(parse_block("nothing here").unwrap_err().contains("IRIS-BENCH-BEGIN"));
    }

    #[test]
    fn features_come_from_the_emulator_banner() {
        assert_eq!(parse_features("iris: build features: r5k jitv2 tlbvmap\n"),
                   vec!["r5k", "jitv2", "tlbvmap"]);
        assert!(parse_features("iris: build features: (none)\n").is_empty());
        assert!(parse_features("no banner at all").is_empty());
    }

    #[test]
    fn every_cell_name_is_unique_and_maps_to_a_cpu() {
        for (i, a) in CELLS.iter().enumerate() {
            assert!(a.expect_cpu == "R4400" || a.expect_cpu == "R5000");
            // An r5k cell must actually ask for the r5k feature, or the guard
            // in `matrix` would reject its own build.
            assert_eq!(a.expect_cpu == "R5000", a.features.contains("r5k"), "{}", a.name);
            for b in CELLS.iter().skip(i + 1) { assert_ne!(a.name, b.name); }
        }
    }
}
