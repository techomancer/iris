//! Run the benchmark suite in-process and hand back a parsed report.
//!
//! The developer path spawns two processes — `iris-bench` spawns `iris`, which
//! loads a guest binary off disk and prints to a pipe. None of that survives an
//! application sandbox, and none of it is necessary: `iris` is a library, the
//! guest image is linked into it (`crate::benchsuite`), and the test device can
//! deliver the guest's console and exit code to a caller instead of to stdout
//! and `process::exit` (`TestDevice::new_embedded`). So the whole run is a
//! `Machine` on a worker thread, and it behaves identically on macOS, Windows
//! and Linux because there is nothing platform-specific left in it.
//!
//! `iris-bench run` uses this too, so "run the suite and parse the answer" has
//! one implementation rather than two. What still spawns processes is
//! `iris-bench matrix`, and necessarily: the CPU model and the JIT are
//! compile-time cargo features, so comparing them means comparing binaries.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::bench_report::{host_info, parse_block, Run};
use crate::benchsuite;
use crate::config::{CpuModel, MachineConfig};
use crate::machine::Machine;
use crate::testdev::{RunConfig, TestDevice};

/// RAM for the bare-metal machine, in MB per bank — the same 256 MB in two
/// banks as `bench/run/bare.toml`. The suite probes for up to 24 MB of working
/// set above its image, and the DRAM-latency and stream kernels are only
/// measuring DRAM if the buffers genuinely do not fit in any cache.
pub const BENCH_BANKS: [u32; 4] = [128, 128, 0, 0];

/// Quick mode: shorter timed runs and one pass instead of best-of-two.
///
/// It deliberately does **not** drop groups. Every kernel still runs and still
/// verifies against its golden checksum, so a quick run's accuracy score means
/// exactly what a full run's does — which matters, because accuracy is the
/// number the shipping build leads with. What it gives up is measurement
/// precision, and only that.
///
/// The floor is each kernel's verification pass: one exact workload, because
/// that is what the golden checksum was computed against, so it cannot be
/// scaled. That is about 11 s of a full 45 s interpreted run, and it is why
/// quick mode lands near 20 s rather than near zero.
const QUICK: RunConfig = RunConfig { groups: 0, time_pct: 30, repeats: 1 };

#[derive(Debug, Clone)]
pub struct BenchOptions {
    /// Fewer groups and shorter timed runs. Accuracy is unaffected: every
    /// kernel that runs still verifies against its golden checksum.
    pub quick: bool,
    /// Name for the result. `iris-bench` uses the cell name; the GUI uses the
    /// host CPU.
    pub label: String,
    pub banks: [u32; 4],
    /// A hang detector, not a performance budget.
    pub timeout: Duration,
    /// Emulated CPU. Runtime config since 96e5ddd, so the in-process runner picks
    /// it like any other machine setting rather than needing its own build.
    pub cpu: CpuModel,
    /// Set this to abort. The machine is stopped and `run` returns an error —
    /// there is no partial report, because the suite prints its block only at
    /// the end. Exists so an interactive caller's Stop button does something
    /// other than wait out the timeout.
    pub cancel: Option<Arc<AtomicBool>>,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            quick: false,
            label: "local".to_string(),
            banks: BENCH_BANKS,
            cpu: CpuModel::default(),
            timeout: Duration::from_secs(1800),
            cancel: None,
        }
    }
}

/// What the runner reports while the guest is working.
///
/// The suite streams its human table a line at a time — deliberately, so a run
/// that prints nothing for a minute is not mistaken for a hang. The runner has
/// to read that stream anyway to know where it has got to, so the progress
/// events *are* the parsed console lines: a caller showing a progress bar and a
/// caller showing a console are looking at the same data at two levels of
/// detail, not at two separate channels.
#[derive(Debug, Clone)]
pub enum Progress {
    /// The guest is up and has announced how many kernels it intends to run.
    /// `total` is 0 if it did not say (an older guest image).
    Started { total: usize },
    /// A kernel finished and printed its row.
    Kernel { name: String, index: usize },
    /// One line of guest console output, verbatim.
    Line(String),
}

/// Run the suite and return the parsed report.
///
/// Blocks until the guest exits, the timeout expires, or startup fails. Safe to
/// call from any thread; it does its own work on a worker with a large stack,
/// because `Machine::new` puts a >1 MB device map on the stack and Windows
/// gives a thread 1 MB by default.
pub fn run(
    opts: &BenchOptions,
    progress: impl FnMut(Progress) + Send + 'static,
) -> Result<Run, String> {
    let opts = opts.clone();
    std::thread::Builder::new()
        .name("bench-runner".to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || run_inner(&opts, progress))
        .map_err(|e| format!("bench runner thread: {}", e))?
        .join()
        .map_err(|_| "the benchmark runner panicked".to_string())?
}

fn run_inner(
    opts: &BenchOptions,
    mut progress: impl FnMut(Progress),
) -> Result<Run, String> {
    let sink: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::with_capacity(64 * 1024)));
    let done = Arc::new(AtomicBool::new(false));
    let code = Arc::new(AtomicU32::new(0));

    let cfg = if opts.quick { QUICK } else { RunConfig::ALL };

    let testdev = {
        let (done, code) = (done.clone(), code.clone());
        // Runs on the CPU thread inside the guest's store. Two relaxed atomic
        // stores and a return — see `TestDevice::exit` for why returning is
        // both safe and required.
        let on_exit = Box::new(move |c: u32| {
            code.store(c, Ordering::Relaxed);
            done.store(true, Ordering::Release);
        });
        Arc::new(TestDevice::new_embedded(sink.clone(), on_exit, cfg))
    };

    let started = Instant::now();
    let mut machine = Box::new(Machine::new_with_testdev(
        bench_config(opts.banks, opts.cpu),
        Some(testdev),
    ));
    // Deliberately no `register_system_controller`: it hands a raw pointer to
    // this Machine to a thread that outlives the call, and this one is dropped
    // when the run finishes. Nothing here needs `reset`/`save`/`load` anyway.

    machine.load_elf_bytes(benchsuite::SUITE_ELF, "irisbench.elf")
        .map_err(|e| format!("loading the embedded suite: {}", e))?;

    machine.start();
    // `Machine::start` autostarts the CPU only in a non-developer release
    // build; ask for it explicitly so a debug build (`cargo test`) runs too.
    // `MipsCpu::start` is a no-op when it is already running.
    machine.cpu_start();

    let outcome = pump(&sink, &done, opts, started, &mut progress);
    machine.stop();
    // Anything the guest printed between the last poll and the stop.
    let text = String::from_utf8_lossy(&sink.lock()).into_owned();
    outcome?;

    let wall_s = started.elapsed().as_secs_f64();
    let p = parse_block(&text).map_err(|e| {
        format!("{}\n--- last 20 lines of guest output ---\n{}", e, tail(&text, 20))
    })?;

    Ok(Run {
        cell: opts.label.clone(),
        features: crate::build_features::enabled().iter().map(|s| s.to_string()).collect(),
        machine: p.machine,
        host: host_info(),
        rows: p.rows,
        checked: p.checked,
        matched: p.matched,
        total_ns: p.total_ns,
        total_icount: p.total_icount,
        wall_s,
        suite_id: benchsuite::suite_id(),
        settings: p.settings,
    })
}

/// Drain the guest console until it exits, reporting progress a line at a time.
fn pump(
    sink: &Arc<Mutex<Vec<u8>>>,
    done: &Arc<AtomicBool>,
    opts: &BenchOptions,
    started: Instant,
    progress: &mut impl FnMut(Progress),
) -> Result<(), String> {
    let mut read = 0usize;      // bytes of the sink already turned into lines
    let mut partial = String::new();
    let mut state = Table::default();

    loop {
        // Read the flag *before* the buffer, not after. The guest's last act is
        // to print its DONE line and then store to EXIT, so a flag read after
        // the drain could go true in between and lose those lines from the
        // progress stream. This way the final drain is guaranteed to be later
        // than the store. (The parsed report never depended on this — the whole
        // sink is re-read once the machine has stopped — but a console that
        // drops the last line looks like a crash.)
        let finished = done.load(Ordering::Acquire);

        // Take whatever is new. The lock is held only for the copy, never
        // across the callback: `progress` is caller code and the CPU thread
        // writes into this buffer one byte per guest `putc`.
        let chunk = {
            let buf = sink.lock();
            let chunk = buf[read.min(buf.len())..].to_vec();
            read = buf.len();
            chunk
        };

        partial.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(nl) = partial.find('\n') {
            let line: String = partial.drain(..=nl).collect();
            let line = line.trim_end_matches(['\n', '\r']).to_string();
            state.classify(&line, progress);
            progress(Progress::Line(line));
        }

        if finished {
            return Ok(());
        }
        if opts.cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err("stopped".to_string());
        }
        if started.elapsed() >= opts.timeout {
            return Err(format!("the guest never finished — gave up after {}s",
                               opts.timeout.as_secs()));
        }
        std::thread::sleep(POLL);
    }
}

/// 50 ms is well under the ~250 ms a kernel's timed run takes, so no row is
/// ever more than a poll behind, and it costs one uncontended lock per poll.
const POLL: Duration = Duration::from_millis(50);

/// Where the guest's output has got to.
///
/// The human table is bracketed by two horizontal rules, and *only* rows
/// between them are kernels. Bracketing rather than pattern-matching a row is
/// what keeps the count honest: the "where the time went" list further down
/// also leads with `codec/lz`-style names, and counting those too ran the
/// progress bar six past its own total.
#[derive(Default)]
struct Table {
    rules: usize,
    kernels: usize,
}

impl Table {
    fn classify(&mut self, line: &str, progress: &mut impl FnMut(Progress)) {
        if let Some(rest) = line.trim().strip_prefix("IRIS-BENCH-PLAN") {
            let total = rest
                .split_whitespace()
                .filter_map(|t| t.strip_prefix("benches="))
                .filter_map(|v| v.parse().ok())
                .next()
                .unwrap_or(0);
            progress(Progress::Started { total });
            return;
        }

        let trimmed = line.trim();
        if trimmed.len() >= 8 && trimmed.chars().all(|c| c == '-') {
            self.rules += 1;
            return;
        }
        if self.rules != 1 {
            return;
        }

        // "int/alu   ops   61247926   103.36   ok" — group/kernel, then unit,
        // then numbers. A row for a kernel skipped on this CPU has fewer.
        let mut fields = line.split_whitespace();
        if let Some(name) = fields.next() {
            if name.contains('/') && fields.next().is_some() {
                self.kernels += 1;
                progress(Progress::Kernel { name: name.to_string(), index: self.kernels });
            }
        }
    }
}

/// The bare-metal machine the suite runs on: RAM, a test device, and nothing
/// else. No SCSI (there is no disk image and no filesystem to find one on), no
/// graphics, no audio.
pub fn bench_config(banks: [u32; 4], cpu: CpuModel) -> MachineConfig {
    let mut cfg = MachineConfig {
        banks,
        headless: true,
        no_audio: true,
        ..Default::default()
    };
    // `MachineConfig::default` attaches scsi1.raw, which makes startup fatal
    // when the file is absent — and here it always is. Same reason
    // bench/run/bare.toml carries a present-but-empty `[scsi]`.
    cfg.scsi.clear();
    cfg.machine.cpu = cpu;
    cfg
}

fn tail(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole embedded path end to end: build a machine, load the suite out
    /// of the binary, run it, parse what it printed. Quick mode so it is a test
    /// rather than a coffee break.
    ///
    /// Accuracy is asserted at 100%, which makes this a correctness net for the
    /// emulator and not just for the plumbing: every kernel checksums its
    /// result against a golden value compiled into the guest image.
    #[test]
    #[ignore = "runs the emulator for ~30s; run with --ignored"]
    fn the_embedded_suite_runs_and_scores_100_percent() {
        use std::sync::atomic::AtomicUsize;

        let (lines, kernels, planned) = (
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        );
        let run = {
            let (l, k, p) = (lines.clone(), kernels.clone(), planned.clone());
            run(
                &BenchOptions { quick: true, label: "test".into(), ..Default::default() },
                move |ev| {
                    match ev {
                        Progress::Line(_) => &l,
                        Progress::Kernel { .. } => &k,
                        Progress::Started { total } => {
                            p.store(total, Ordering::Relaxed);
                            return;
                        }
                    }
                    .fetch_add(1, Ordering::Relaxed);
                },
            )
        }
        .expect("the embedded suite must run");

        eprintln!(
            "quick run: {:.1}s wall, {:.1}s timed, {:.1} guest MIPS, {}/{} matched, {} rows",
            run.wall_s, run.total_ns as f64 / 1e9, run.mips(),
            run.matched, run.checked, run.rows.len()
        );

        assert!(!run.rows.is_empty(), "no kernels reported");
        assert!(run.checked > 0, "nothing was checked against a golden value");
        assert_eq!(
            run.matched, run.checked,
            "{} of {} checksums matched — the emulator computed a wrong answer",
            run.matched, run.checked
        );
        assert!(run.total_icount > 0, "no retired-instruction count — no test device?");
        assert!(run.machine.timebase, "no host time base: every timing would be a guess");
        assert_eq!(run.suite_id, benchsuite::suite_id());

        // Quick mode narrows precision, not coverage: it must still have run
        // every group, and the result must say it was shortened.
        assert_eq!(run.settings.groups, crate::bench_report::BG_ALL);
        assert!(!run.settings.is_full(), "a quick run must record that it was one");

        // Progress must be usable for a progress bar: an up-front total that
        // the rows actually reach.
        let (planned, kernels) = (planned.load(Ordering::Relaxed), kernels.load(Ordering::Relaxed));
        assert!(planned > 0, "the guest never announced how many kernels it would run");
        assert_eq!(planned, kernels, "progress reported {} of a planned {}", kernels, planned);
        assert_eq!(planned, run.rows.len(), "planned count disagrees with the rows reported");
        assert!(lines.load(Ordering::Relaxed) > kernels, "no console lines beyond the rows");
    }

    #[test]
    fn the_bench_machine_has_no_disks() {
        let cfg = bench_config(BENCH_BANKS, CpuModel::default());
        assert!(cfg.scsi.is_empty(), "a bench machine must not try to open a disk image");
        assert!(cfg.headless && cfg.no_audio);
    }

    #[test]
    fn quick_mode_narrows_precision_and_nothing_else() {
        // Accuracy is what the shipping build leads with, so quick mode must
        // not quietly reduce what the score covers: every group still runs and
        // every kernel still verifies.
        assert_eq!(QUICK.groups, 0, "quick mode must run every group");
        assert!(QUICK.time_pct > 0 && QUICK.time_pct < 100);
        assert_eq!(QUICK.repeats, 1);
    }
}
