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
use crate::config::{JitCompile, MachineConfig};
use crate::cpu::parts::{self, Part};
use crate::machine::Machine;
use crate::dev::testdev::{RunConfig, TestDevice};

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
    pub cpu: &'static Part,
    /// Cache model, as `[machine] cache`.
    pub cache: crate::config::CacheModel,
    /// Make two runs do the same work: compile on the CPU thread
    /// (`jitv2.compile = "sync"`). Needs a `ci_clock` build, whose test device
    /// clock counts retired instructions, so the iteration counts the suite
    /// picks by time are the same every run. The rates it reports are then constant
    /// by construction; compare instruction counts and host cost instead.
    pub deterministic: bool,
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
            cpu: &parts::R4400_150,
            cache: crate::config::CacheModel::default(),
            deterministic: false,
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
        // Rex3Context embeds the HOSTRW data-port array (HOSTRW_BUF_QWORDS u64s,
            // 1 MiB), and construction has several context-sized temporaries in
            // flight before the machine is boxed. This is virtual address space,
            // lazily committed, so the large reservation has no real cost.
            .stack_size(64 * 1024 * 1024)
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

    if opts.deterministic && !cfg!(feature = "ci_clock") {
        return Err("--deterministic needs a build with the `ci_clock` feature: the suite sizes its work by the test device's clock, which only counts retired instructions there".into());
    }
    let started = Instant::now();
    let mut cfg = bench_config(opts.banks, opts.cpu);
    cfg.machine.cache = opts.cache;
    if opts.deterministic {
        cfg.jitv2.compile = JitCompile::Sync;
    }
    let mut machine = Box::new(Machine::new_with_testdev(cfg, Some(testdev)));
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

/// The parts the suite measures. It runs on an Indy with the test device on,
/// which takes any part.
pub const BENCH_CPUS: &[&Part] = &[
    &parts::R4400_150,
    &parts::R5000_180_EARLY,
    &parts::R10000_195,
    &parts::R12000_360,
    &parts::R14000_600,
    &parts::R16000_700_EARLY,
    &parts::R16000_800,
    &parts::R16000_900,
    &parts::R16000_1000,
];

/// The bare-metal machine the suite runs on: RAM, a test device, and nothing
/// else. No SCSI (there is no disk image and no filesystem to find one on), no
/// graphics, no audio.
pub fn bench_config(banks: [u32; 4], cpu: &Part) -> MachineConfig {
    let mut cfg = MachineConfig {
        banks: crate::config::banks_from4(banks),
        headless: true,
        ..Default::default()
    };
    cfg.machine.audio = Some(false);
    // `MachineConfig::default` attaches scsi1.raw, which makes startup fatal
    // when the file is absent — and here it always is. Same reason
    // bench/run/bare.toml carries a present-but-empty `[scsi]`.
    cfg.scsi.clear();
    cfg.set_cpu(cpu);
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

    /// The suite runs with no PROM, so nothing but the harness sets KSEG0's
    /// cacheability, and the R10000 resets with Config.K0 = 2 (uncached).
    /// If the harness stops choosing K0 = 3, every R10000 load and store goes
    /// to the bus and the R10000 cells measure uncached memory, about half the
    /// R4400's rate on the memory kernels, while still scoring 100%.
    #[test]
    #[ignore = "runs the emulator for ~30s; run with --ignored"]
    fn the_r10000_runs_the_suite_with_kseg0_cached() {
        let run = run(
            &BenchOptions {
                quick: true,
                label: "test".into(),
                cpu: &parts::R10000_195,
                ..Default::default()
            },
            |_| {},
        )
        .expect("the embedded suite must run on the R10000");
        assert_eq!(run.matched, run.checked, "the R10000 computed a wrong answer");
        let config = u32::from_str_radix(run.machine.config.trim_start_matches("0x"), 16)
            .expect("the guest reports Config in hex");
        assert_eq!(config & 7, 3, "KSEG0 not cacheable: Config {:#x}", config);
    }

    /// The R14000 and the R16000 share PRId implementation 0x0f; the harness
    /// names the R16000 by its major revision (3 and up), as Linux does.
    #[test]
    #[ignore = "runs the emulator for ~1 min; run with --ignored"]
    fn the_r14000_and_the_r16000_are_named_apart() {
        for (cpu, name) in [(&parts::R14000_600, "R14000"), (&parts::R16000_700_EARLY, "R16000"), (&parts::R16000_1000, "R16000")] {
            let run = run(&BenchOptions { quick: true, label: "test".into(), cpu, ..Default::default() }, |_| {})
                .expect("the embedded suite must run");
            assert_eq!(run.machine.cpu, name, "{}", cpu.name());
        }
    }

    /// A bare-metal program that sizes its work by the test device's clock, as
    /// the suite does: count loop passes until 20 us have gone by, print the
    /// count (big-endian) and exit.
    fn clock_spin_elf() -> Vec<u8> {
        const T0: u32 = 8; const T1: u32 = 9; const T2: u32 = 10; const T3: u32 = 11;
        const T4: u32 = 12; const T5: u32 = 13; const S0: u32 = 16;
        let i = |op: u32, rs: u32, rt: u32, imm: u32| (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF);
        let srl = |rd: u32, rt: u32, sa: u32| (rt << 16) | (rd << 11) | (sa << 6) | 0x02;
        let ns = crate::dev::testdev::REG_HOST_NS_LO;
        let code = [
            i(0x0F, 0, T0, 0xBF40),         // lui   t0, 0xbf40       testdev, uncached
            i(0x23, T0, T1, ns),            // lw    t1, ns(t0)       start
            (S0 << 11) | 0x21,              // move  s0, zero
            i(0x09, S0, S0, 1),             // loop: addiu s0, s0, 1
            i(0x23, T0, T2, ns),            //       lw    t2, ns(t0)
            (T2 << 21) | (T1 << 16) | (T3 << 11) | 0x23, // subu t3, t2, t1
            i(0x0B, T3, T4, 20_000),        //       sltiu t4, t3, 20000
            i(0x05, T4, 0, (-5i32) as u32), //       bne   t4, zero, loop
            0,                              //       nop
            srl(T5, S0, 24), i(0x2B, T0, T5, 4), // sw count byte 3 to PUTC
            srl(T5, S0, 16), i(0x2B, T0, T5, 4),
            srl(T5, S0, 8), i(0x2B, T0, T5, 4),
            i(0x2B, T0, S0, 4),
            i(0x2B, T0, 0, 0xC),            // sw    zero, EXIT(t0)
            (0x02 << 26) | (0x0820_1000 >> 2), // j     spin, on the next page
            0,
        ];
        // After the exit store the guest spins on a page of its own, through
        // a register jump. A compiled loop never returns to the CPU thread to
        // notice a stop; a jump through a register leaves the compiled
        // region every time round.
        let mut payload: Vec<u8> = code.iter().flat_map(|w| w.to_be_bytes()).collect();
        payload.resize(0x1000, 0);
        const T6: u32 = 14;
        for w in [i(0x0F, 0, T6, 0x8820), i(0x0D, T6, T6, 0x1008), (T6 << 21) | 0x08, 0] {
            payload.extend_from_slice(&w.to_be_bytes());
        }
        let vaddr: u32 = 0x8820_0000;
        let mut v = vec![0u8; 84];
        v[0..4].copy_from_slice(b"\x7fELF");
        v[4] = 1; v[5] = 2; v[6] = 1;
        v[16..18].copy_from_slice(&2u16.to_be_bytes());
        v[18..20].copy_from_slice(&8u16.to_be_bytes());
        v[24..28].copy_from_slice(&vaddr.to_be_bytes());
        v[28..32].copy_from_slice(&52u32.to_be_bytes());
        v[42..44].copy_from_slice(&32u16.to_be_bytes());
        v[44..46].copy_from_slice(&1u16.to_be_bytes());
        v[52..56].copy_from_slice(&1u32.to_be_bytes());
        v[56..60].copy_from_slice(&84u32.to_be_bytes());
        v[60..64].copy_from_slice(&vaddr.to_be_bytes());
        v[68..72].copy_from_slice(&(payload.len() as u32).to_be_bytes());
        v[72..76].copy_from_slice(&(payload.len() as u32).to_be_bytes());
        v[76..80].copy_from_slice(&5u32.to_be_bytes());
        v.extend_from_slice(&payload);
        v
    }

    /// Run `clock_spin_elf` once on a machine built the way the runner builds
    /// one. Returns what it printed.
    fn run_clock_spin(deterministic: bool) -> Vec<u8> {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let sink: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
                let done = Arc::new(AtomicBool::new(false));
                let on_exit = { let done = done.clone(); Box::new(move |_| done.store(true, Ordering::Release)) };
                let testdev = Arc::new(
                    TestDevice::new_embedded(sink.clone(), on_exit, RunConfig::ALL));
                let mut cfg = bench_config([32, 0, 0, 0], &parts::R4400_150);
                if deterministic {
                    cfg.jitv2.compile = JitCompile::Sync;
                }
                let mut machine = Box::new(Machine::new_with_testdev(cfg, Some(testdev)));
                machine.load_elf_bytes(&clock_spin_elf(), "clock-spin.elf").expect("load");
                machine.start();
                machine.cpu_start();
                let started = Instant::now();
                while !done.load(Ordering::Acquire) {
                    assert!(started.elapsed() < Duration::from_secs(60), "the guest never exited");
                    std::thread::sleep(Duration::from_millis(5));
                }
                machine.stop();
                let out = sink.lock().clone();
                out
            })
            .expect("spawn")
            .join()
            .expect("the machine panicked")
    }

    /// Deterministic mode makes a guest that sizes its work by time do the
    /// same work every run: the same loop count, whatever else the host is
    /// doing.
    #[test]
    #[cfg(feature = "ci_clock")]
    fn ci_clock_deterministic_mode_repeats_a_time_sized_guest_exactly() {
        let a = run_clock_spin(true);
        let b = run_clock_spin(true);
        assert_eq!(a.len(), 4, "the guest prints its loop count: {a:?}");
        assert_eq!(a, b, "two deterministic runs must agree");
        // A pass is six instructions at NS_PER_GUEST_CYCLE each, so 20 us is
        // about 333 of them, give or take the setup and the reads at the edges.
        let passes = u32::from_be_bytes([a[0], a[1], a[2], a[3]]);
        assert!((331..=336).contains(&passes), "loop count {passes} does not follow the guest clock");
    }

    #[test]
    fn the_bench_machine_has_no_disks() {
        let cfg = bench_config(BENCH_BANKS, &parts::R4400_150);
        assert!(cfg.scsi.is_empty(), "a bench machine must not try to open a disk image");
        assert!(cfg.headless && !cfg.machine.has_audio());
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
