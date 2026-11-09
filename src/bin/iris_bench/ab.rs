//! `iris-bench ab` — a paired comparison of two emulator builds.
//!
//! "Is B slower than A?" has to be asked on a host that is busy with other
//! work, so a stored baseline or two back-to-back runs say little. Each round
//! runs A and B once on one guest image, in a random order (or together, with
//! `--concurrent`), and gives every kernel one B/A ratio of nanoseconds per
//! guest instruction. A kernel is judged from the median of its ratios and an
//! interval for that median, so one spoiled round in thirty barely matters and
//! a result the rounds cannot support is reported as inconclusive.
//!
//! Each side runs through `run_guest`, as `iris-bench run --iris` does, so
//! every run is saved as an ordinary result that `iris-bench report` can read.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Args;
use iris::bench_report::Run;

/// Fewest rounds that can give a verdict: the fewest whose interval reaches
/// CONFIDENCE. Up to 14 rounds the interval is the whole range of the ratios,
/// and the range misses the median only when every round lands on one side of
/// it, which happens with probability 2^(1-n): 0.2% at 10 rounds, 0.098% at
/// 11. Below 11 an A/A run would flag rows more often than CONFIDENCE allows.
const MIN_ROUNDS: usize = 11;
/// 99.9% rather than 95% because a table has 46 kernels: at 95% an A/A run
/// would flag about two of them by chance.
const CONFIDENCE: f64 = 0.999;
/// How far guest instructions per iteration may differ between A and B.
const IPI_TOLERANCE: f64 = 0.005;

fn median(v: &[f64]) -> f64 { super::stat(v.to_vec()).map_or(f64::NAN, |s| s.0) }

/// The interval for the median of `x` that the sign test inverts: the order
/// statistics `x[k]` and `x[n-1-k]` for the largest `k` whose binomial
/// coverage is at least `conf`. With too few samples for any `k`, the range.
/// It needs no random numbers, so the same rounds always print the same
/// interval.
fn median_ci(x: &[f64], conf: f64) -> Option<(f64, f64)> {
    let n = x.len();
    if n < 3 || x.iter().any(|v| !v.is_finite()) { return None; }
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    // P(Bin(n, 1/2) <= i) is the chance the median lies below x[i+1].
    let (mut p, mut cdf, mut k) = (0.5f64.powi(n as i32), 0.0, 0);
    for i in 0..n / 2 {
        cdf += p;
        if 1.0 - 2.0 * cdf < conf { break; }
        k = i;
        p *= (n - i) as f64 / (i + 1) as f64;
    }
    Some((s[k], s[n - 1 - k]))
}

/// SLOWER when the interval lies above 1 and the median is past 1 + the
/// threshold, and `faster` the mirror image. `same` when the whole interval
/// is within the threshold either way, so a change that size would have
/// shown. Otherwise `inconclusive`, which never means "no change".
fn decide(ratios: &[f64], ci: Option<(f64, f64)>, thr: f64) -> &'static str {
    let Some((lo, hi)) = ci.filter(|_| ratios.len() >= MIN_ROUNDS) else { return "inconclusive" };
    let (m, up) = (median(ratios), 1.0 + thr);
    if lo > 1.0 && m > up { "SLOWER" }
    else if hi < 1.0 && m < 1.0 / up { "faster" }
    else if lo >= 1.0 / up && hi <= up { "same" }
    else { "inconclusive" }
}

/// One row: a kernel, or the whole suite.
struct Line {
    name: String,
    /// Median guest MIPS per side.
    mips: [f64; 2],
    /// Median per-round B/A cost ratio, its interval, and its scatter
    /// (1.4826 x MAD / median), which is what an A/A run would show.
    ratio: f64,
    ci: Option<(f64, f64)>,
    noise: f64,
    verdict: &'static str,
    /// Guest instructions per iteration per side, when they differ in every round.
    ipi: Option<[f64; 2]>,
}

/// A kernel's (nanoseconds per guest instruction, instructions per iteration)
/// in one run, if it has usable figures.
fn kernel_cost(run: &Run, name: &str) -> Option<(f64, f64)> {
    let k = run.row(name)?;
    (k.icount > 0 && k.ns > 0 && k.status != "SKIP")
        .then(|| (k.ns as f64 / k.icount as f64, k.icount as f64 / k.iters.max(1) as f64))
}

fn line(name: &str, rounds: &[[Run; 2]], thr: f64, cost: impl Fn(&Run) -> Option<(f64, f64)>) -> Line {
    let side = |s: usize| -> Vec<Option<(f64, f64)>> { rounds.iter().map(|r| cost(&r[s])).collect() };
    let (a, b) = (side(0), side(1));
    let ratios: Vec<f64> = a.iter().zip(&b).filter_map(|(x, y)| Some(y.as_ref()?.0 / x.as_ref()?.0)).collect();
    let col = |v: &[Option<(f64, f64)>], f: fn(&(f64, f64)) -> f64| -> Vec<f64> { v.iter().flatten().map(f).collect() };
    let (ia, ib) = (col(&a, |x| x.1), col(&b, |x| x.1));
    // The medians must differ by more than the tolerance and the two sides'
    // ranges must not overlap: run to run, a kernel's fixed setup moves its
    // per-iteration figure by up to 4% on one binary, while a different image
    // shifts every round.
    let range = |v: &[f64]| (v.iter().copied().fold(f64::NAN, f64::min), v.iter().copied().fold(f64::NAN, f64::max));
    let ((lo_a, hi_a), (lo_b, hi_b)) = (range(&ia), range(&ib));
    let drift = !ia.is_empty() && !ib.is_empty()
        && (median(&ia) - median(&ib)).abs() > IPI_TOLERANCE * median(&ia).abs().max(1.0)
        && (lo_b > hi_a || lo_a > hi_b);
    let ci = median_ci(&ratios, CONFIDENCE);
    let m = median(&ratios);
    let dev: Vec<f64> = ratios.iter().map(|r| (r - m).abs()).collect();
    Line {
        name: name.to_string(),
        mips: [1e3 / median(&col(&a, |x| x.0)), 1e3 / median(&col(&b, |x| x.0))],
        ratio: m,
        ci,
        noise: 1.4826 * median(&dev) / m.abs(),
        verdict: decide(&ratios, ci, thr),
        ipi: drift.then(|| [median(&ia), median(&ib)]),
    }
}

/// Every kernel's line, then the whole suite's.
fn summarise(rounds: &[[Run; 2]], thr: f64) -> Vec<Line> {
    let first = &rounds[0][0];
    let mut v: Vec<Line> = first.rows.iter().filter(|k| kernel_cost(first, &k.name).is_some())
        .map(|k| line(&k.name, rounds, thr, |r| kernel_cost(r, &k.name))).collect();
    v.push(line("(suite)", rounds, thr, |r| Some((r.total_ns as f64 / r.total_icount.max(1) as f64, 0.0))));
    v
}

fn pct(ratio: f64) -> String { format!("{:+.1}%", (ratio - 1.0) * 100.0) }

/// The table. Rates are guest MIPS (higher is better). The ratio is of cost,
/// B over A, so +5% means B takes 5% longer per guest instruction.
fn table(lines: &[Line], la: &str, lb: &str, thr: f64) -> String {
    let mut o = format!("| kernel | {la} MIPS | {lb} MIPS | paired B/A [{:.1}% interval] | verdict |\n|---|---:|---:|---|---|\n",
                        CONFIDENCE * 100.0);
    for l in lines {
        let ci = l.ci.map_or("n/a".to_string(), |(lo, hi)| format!("{} [{}, {}]", pct(l.ratio), pct(lo), pct(hi)));
        o.push_str(&format!("| {} | {:.1} | {:.1} | {} | {} |\n", l.name, l.mips[0], l.mips[1], ci, l.verdict));
    }
    o.push_str(&format!(
        "\nRates are the median round's guest MIPS. The verdict is SLOWER or faster when the interval \
         excludes no change and the median passes {t:.1}%, same when the whole interval is within {t:.1}% \
         either way, and inconclusive otherwise: more rounds narrow the interval.\n", t = thr * 100.0));
    o
}

/// Reasons not to trust the table as it stands.
fn warnings(rounds: &[[Run; 2]], lines: &[Line], thr: f64) -> Vec<String> {
    let mut w = Vec::new();
    if rounds.len() < MIN_ROUNDS {
        w.push(format!("only {} round(s): too few to judge anything; use at least {MIN_ROUNDS}, and 30 or more on a busy host", rounds.len()));
    }
    let drift: Vec<String> = lines.iter().filter_map(|l| l.ipi.map(|[a, b]| format!("{} ({a:.1} vs {b:.1})", l.name))).collect();
    if !drift.is_empty() {
        w.push(format!("guest instructions per iteration differ by more than {:.1}% in every round for {}: the \
                        guest code is not the same, or one emulator miscounts, so the ratios do not compare like with like",
                       IPI_TOLERANCE * 100.0, drift.join(", ")));
    }
    let mut bad: Vec<String> = Vec::new();
    for r in rounds {
        for (side, run) in ["A", "B"].iter().zip(r) {
            for k in run.rows.iter().filter(|k| k.status == "MISMATCH") {
                let m = format!("{} ({side})", k.name);
                if !bad.contains(&m) { bad.push(m); }
            }
        }
    }
    if !bad.is_empty() { w.push(format!("checksum mismatch: {}", bad.join(", "))); }
    let suite = lines.last().expect("a summary has a suite line");
    if suite.noise > thr {
        w.push(format!("the whole-suite B/A ratio scattered by {:.1}% between rounds, so this host was noisy; \
                        small changes need more rounds or a quieter host", suite.noise * 100.0));
    }
    w
}

/// Whether A and B may be timed against each other. A `ci_clock` build's
/// guest clock counts retired instructions, so every ratio would read as no
/// change. Builds with different features differ in more than the change
/// under test, so a flagged row would not say which caused it; with
/// `across_features` the comparison is the point, to measure what a feature
/// costs.
fn check_builds(a: &[String], b: &[String], across_features: bool) -> Result<(), String> {
    for (side, f) in [("A", a), ("B", b)] {
        if f.iter().any(|x| x == "ci_clock") {
            return Err(format!("{side} is a ci_clock build. Its guest clock counts instructions, not host \
                                time, so it cannot be timed; build it without ci_clock"));
        }
    }
    let show = |f: &[String]| if f.is_empty() { "(none)".to_string() } else { f.join(" ") };
    if a == b || across_features { return Ok(()); }
    Err(format!("A has features {}, B has {}. Builds with different features differ in more than the \
                 change under test; build both alike, or pass --across-features to measure what the \
                 features cost", show(a), show(b)))
}

/// The emulator option that publishes a run configuration to the guest.
const RUN_CONFIG_FLAG: &str = "--test-device-run-config";

/// Whether an emulator's `--help` lists RUN_CONFIG_FLAG.
fn help_lists_run_config(help: &str) -> bool {
    help.split_whitespace().any(|w| w == RUN_CONFIG_FLAG)
}

fn takes_run_config(iris: &Path) -> Result<bool, String> {
    let mut cmd = Command::new(iris);
    cmd.arg("--help");
    let out = super::run_with_timeout(cmd, 60).map_err(|e| format!("{}: {e}", iris.display()))?;
    Ok(help_lists_run_config(&String::from_utf8_lossy(&out.stdout)))
}

/// Give both builds the same shortened run, and return the arguments that
/// do it. When both take RUN_CONFIG_FLAG, that is passed and `image` is left
/// alone. Otherwise the run length is written into `image`, which both run,
/// and no argument is needed: the patch is a compatibility path for builds
/// older than the flag, to be removed once comparisons against them no longer
/// matter.
fn run_length_args(image: &mut [u8], takes_flag: [bool; 2], time_pct: u32, repeats: u32)
                   -> Result<Vec<String>, String> {
    if takes_flag == [true, true] {
        return Ok(vec![RUN_CONFIG_FLAG.into(), format!("time_pct={time_pct},repeats={repeats}")]);
    }
    set_run_length(image, time_pct, repeats)?;
    Ok(Vec::new())
}

/// Set the suite's run length in a copy of its image, for a build older than
/// RUN_CONFIG_FLAG: such a build publishes no run configuration, so the guest
/// runs the defaults compiled into the image. The guest reports the run length
/// it used and every run is checked against the asked-for values, so a patch
/// that didn't take is caught.
fn set_run_length(image: &mut [u8], time_pct: u32, repeats: u32) -> Result<(), String> {
    for (name, val) in [("bench_time_pct", time_pct), ("bench_repeats", repeats)] {
        let at = iris::elf::symbol_offset(image, name)
            .ok_or_else(|| format!("the image has no initialised 4-byte `{name}`"))?;
        image[at..at + 4].copy_from_slice(&val.to_be_bytes());
    }
    Ok(())
}

#[derive(Args, Debug)]
pub struct AbArgs {
    /// Emulator binary A, the baseline. An `iris` build, not an `iris-bench`.
    #[arg(long)]
    a: PathBuf,
    /// Emulator binary B, the candidate.
    #[arg(long)]
    b: PathBuf,
    /// Guest suite image both builds run. Defaults to the image linked into
    /// this `iris-bench`. Hold it fixed when the builds span a rebuild of the
    /// image, which moves the rates on its own.
    #[arg(long)]
    elf: Option<PathBuf>,
    /// Machine config. Defaults to bench/run/bare.toml.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Rounds; each runs A and B once.
    #[arg(long, default_value_t = 30)]
    rounds: usize,
    /// Each kernel's timed run, as a percentage of the suite's default (about
    /// 250 ms), from 10, the shortest the guest runs, to 4095. Shorter runs are
    /// less often hit by another process.
    #[arg(long, default_value_t = 10)]
    time_pct: u32,
    /// Timed runs per kernel, 1 to 15; the guest keeps the fastest.
    #[arg(long, default_value_t = 3)]
    repeats: u32,
    /// Compare builds with different features, to measure what a feature
    /// costs. Without it, `ab` refuses them.
    #[arg(long)]
    across_features: bool,
    /// Start A and B at the same time each round instead of one after the
    /// other, so what the host does to one lands on both. iris-bench pins
    /// nothing: pin it by wrapping the command (`taskset -c 2-9 iris-bench ab
    /// --concurrent ...` on Linux, `start /wait /affinity MASK iris-bench ab
    /// --concurrent ...` on Windows; macOS has no affinity API).
    #[arg(long)]
    concurrent: bool,
    /// Emulated CPU, forwarded to the emulator as `--cpu`.
    #[arg(long)]
    cpu: Option<String>,
    /// Effect size, percent: a row is flagged when its interval excludes no
    /// change and its median ratio passes this.
    #[arg(long, default_value_t = 3.0)]
    threshold: f64,
    /// Names for the table columns and the saved results' cells.
    #[arg(long, default_value = "A")]
    label_a: String,
    #[arg(long, default_value = "B")]
    label_b: String,
    /// Per-run timeout, seconds.
    #[arg(long, default_value_t = 1800)]
    timeout: u64,
    /// Results directory; each comparison gets an `ab-<time>` directory in it.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Extra arguments passed to both emulators.
    #[arg(last = true)]
    extra: Vec<String>,
}

pub fn run(args: AbArgs) -> Result<(), String> {
    if args.rounds == 0 { return Err("--rounds must be at least 1".into()); }
    if args.label_a == args.label_b { return Err("--label-a and --label-b must differ".into()); }
    // The guest raises a shorter time to its floor, and the run configuration
    // has 12 bits for the time and 4 for the repeats.
    if !(10..=4095).contains(&args.time_pct) { return Err("--time-pct must be from 10 to 4095".into()); }
    if !(1..=15).contains(&args.repeats) { return Err("--repeats must be from 1 to 15".into()); }
    let abs = |p: &Path| std::fs::canonicalize(p).map_err(|e| format!("{}: {e}", p.display()));
    let (bin_a, bin_b) = (abs(&args.a)?, abs(&args.b)?);
    check_builds(&super::probe_features(&bin_a)?, &super::probe_features(&bin_b)?, args.across_features)?;

    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_secs());
    let dir = args.out.clone().unwrap_or_else(super::default_out).join(format!("ab-{secs}"));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;

    let (mut image, source) = match &args.elf {
        Some(p) => (std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?, "--elf"),
        None => (iris::benchsuite::SUITE_ELF.to_vec(), "the image linked into this iris-bench"),
    };
    let suite_id = iris::benchsuite::suite_id_of(&image);
    let takes_flag = [takes_run_config(&bin_a)?, takes_run_config(&bin_b)?];
    let mut extra = Vec::new();
    // Every run is checked to report these, whichever way they were given.
    let (expect, how) = match run_length_args(&mut image, takes_flag, args.time_pct, args.repeats) {
        Ok(a) => {
            let how = if a.is_empty() { "written into the image: a build predates --test-device-run-config" }
                      else { "--test-device-run-config" };
            extra.extend(a);
            (Some((args.time_pct, args.repeats)), how)
        }
        Err(e) => {
            println!("note: running the image's own settings, not time_pct {} repeats {}: {e}", args.time_pct, args.repeats);
            (None, "the image's own")
        }
    };
    let elf = dir.join("suite.elf");
    std::fs::write(&elf, &image).map_err(|e| format!("{}: {e}", elf.display()))?;
    let elf = abs(&elf)?;
    let config = abs(&args.config.clone().unwrap_or_else(|| super::repo_relative("bench/run/bare.toml")))?;
    if let Some(c) = &args.cpu { extra.extend(["--cpu".to_string(), c.clone()]); }
    extra.extend(args.extra.iter().cloned());

    let one = |bin: &Path, label: &str| -> Result<Run, String> {
        let run = super::run_guest(bin, &elf, &config, label, args.timeout, &extra)
            .map_err(|e| format!("{}: {e}", bin.display()))?;
        match expect {
            Some((pct, reps)) if (run.settings.time_pct, run.settings.repeats) != (pct, reps) => Err(format!(
                "{}: the guest ran time_pct={} repeats={}, not the {pct} and {reps} asked for ({how})",
                bin.display(), run.settings.time_pct, run.settings.repeats)),
            _ => Ok(run),
        }
    };
    let (la, lb) = (args.label_a.as_str(), args.label_b.as_str());
    let mode = if args.concurrent { "A and B started together each round" } else { "A and B one at a time, in a random order each round" };
    println!("A/B: {la} = {}\n     {lb} = {}", bin_a.display(), bin_b.display());
    println!("guest image {suite_id} from {source}, run with {} ({how}; {})",
             expect.map_or("the image's own settings".to_string(), |(p, r)| format!("time_pct {p}, repeats {r}")), elf.display());
    println!("mode: {mode}\nrounds {}, threshold {:.1}%, results in {}", args.rounds, args.threshold, dir.display());

    let mut rounds: Vec<[Run; 2]> = Vec::with_capacity(args.rounds);
    for round in 1..=args.rounds {
        let (a, b) = if args.concurrent {
            let (a, b) = std::thread::scope(|s| {
                let (ha, hb) = (s.spawn(|| one(&bin_a, la)), s.spawn(|| one(&bin_b, lb)));
                (ha.join().unwrap_or_else(|_| Err("run thread panicked".into())),
                 hb.join().unwrap_or_else(|_| Err("run thread panicked".into())))
            });
            (a?, b?)
        } else if RandomState::new().hash_one(round) & 1 == 1 {
            let b = one(&bin_b, lb)?;
            (one(&bin_a, la)?, b)
        } else {
            let a = one(&bin_a, la)?;
            (a, one(&bin_b, lb)?)
        };
        let cost = |r: &Run| r.total_ns as f64 / r.total_icount.max(1) as f64;
        println!("round {round}/{}: {la} {:.1} MIPS, {lb} {:.1} MIPS, suite B/A {}, {:.0}s", args.rounds,
                 a.mips(), b.mips(), pct(cost(&b) / cost(&a)), a.wall_s.max(b.wall_s));
        super::save(&a, &dir)?;
        super::save(&b, &dir)?;
        rounds.push([a, b]);
    }

    let thr = args.threshold / 100.0;
    let lines = summarise(&rounds, thr);
    let mut text = format!("guest image {suite_id} (from {source}), {} rounds, mode: {mode}\n\n{}",
                           args.rounds, table(&lines, la, lb, thr));
    for w in warnings(&rounds, &lines, thr) { text.push_str(&format!("\nWARNING: {w}\n")); }
    let md = dir.join("ab.md");
    std::fs::write(&md, &text).map_err(|e| format!("{}: {e}", md.display()))?;
    println!("\n{text}\nwrote {}; the per-round results beside it are `iris-bench report` input", md.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use iris::bench_report::Row;

    /// Rank of the interval's ends in 1..=n, for n rounds whose ratios are 1..=n.
    fn ends(n: usize) -> (usize, usize) {
        let x: Vec<f64> = (1..=n).map(|v| v as f64).collect();
        let (lo, hi) = median_ci(&x, CONFIDENCE).unwrap();
        (lo as usize, hi as usize)
    }

    #[test]
    fn the_interval_is_the_widest_that_covers_at_the_confidence() {
        // 99.9% needs 15 rounds to trim anything: below, the range.
        assert_eq!(ends(8), (1, 8));
        assert_eq!(ends(14), (1, 14));
        assert_eq!(ends(20), (3, 18));
        assert_eq!(ends(30), (6, 25));
        assert_eq!(median_ci(&[1.0, 2.0], CONFIDENCE), None);
        assert_eq!(median_ci(&[1.0, f64::NAN, 3.0], CONFIDENCE), None);
    }

    #[test]
    fn min_rounds_is_where_the_range_first_reaches_the_confidence() {
        // Below 15 rounds the interval is the range, which misses the median
        // only when every round lands on one side of it.
        let coverage = |n: usize| 1.0 - 0.5f64.powi(n as i32 - 1);
        assert!(coverage(MIN_ROUNDS) >= CONFIDENCE);
        assert!(coverage(MIN_ROUNDS - 1) < CONFIDENCE);
        assert_eq!(ends(MIN_ROUNDS), (1, MIN_ROUNDS));
    }

    #[test]
    fn a_verdict_needs_the_interval_to_clear_the_threshold() {
        let v = |r: &[f64], thr| decide(r, median_ci(r, CONFIDENCE), thr);
        assert_eq!(v(&[1.10; 30], 0.03), "SLOWER");
        assert_eq!(v(&[1.02; 30], 0.03), "same");
        assert_eq!(v(&[1.0 / 1.10; 30], 0.03), "faster");
        // an interval from 0.9 to 1.2 can hide a change of either sign
        let wide: Vec<f64> = (0..30).map(|i| if i % 2 == 0 { 0.9 } else { 1.2 }).collect();
        assert_eq!(v(&wide, 0.03), "inconclusive");
        // one spoiled round does not move the median
        let mut r = vec![1.10; 29];
        r.push(5.0);
        assert_eq!(v(&r, 0.03), "SLOWER");
        // 10 rounds that all agree are still too few; 11 are enough
        assert_eq!(v(&[1.10; 10], 0.03), "inconclusive");
        assert_eq!(v(&[1.10; 11], 0.03), "SLOWER");
    }

    fn run_with(cs: &str, ns: u64, iters: u64, status: &str) -> Run {
        let row = Row { name: "k".into(), unit: "ops".into(), iters, work: 1, ns, icount: 10_000,
                        count: 0, exc: 0, checksum: "0x0".into(), golden: "0x0".into(), status: status.into() };
        Run { cell: cs.into(), features: vec![], machine: Default::default(), host: Default::default(),
              rows: vec![row], checked: 1, matched: 1, total_ns: ns, total_icount: 10_000,
              wall_s: 1.0, suite_id: "s".into(), settings: Default::default() }
    }

    /// `rounds` rounds in which B takes `slow` times A's time, with A's time
    /// varying 1% from round to round.
    fn rounds(n: usize, slow: f64, iters: [u64; 2], status: &str) -> Vec<[Run; 2]> {
        (0..n).map(|i| {
            let a = 1_000_000 + 10_000 * (i as u64 % 3);
            [run_with("A", a, iters[0], "OK"), run_with("B", (a as f64 * slow) as u64, iters[1], status)]
        }).collect()
    }

    #[test]
    fn a_steady_regression_is_flagged_and_an_a_a_run_is_not() {
        let slow = summarise(&rounds(30, 1.10, [10, 10], "OK"), 0.03);
        assert_eq!(slow.iter().map(|l| l.verdict).collect::<Vec<_>>(), ["SLOWER", "SLOWER"]);
        assert!(warnings(&rounds(30, 1.10, [10, 10], "OK"), &slow, 0.03).is_empty());
        let same = summarise(&rounds(30, 1.0, [10, 10], "OK"), 0.03);
        assert!(same.iter().all(|l| l.verdict == "same"));
        let few = rounds(5, 1.0, [10, 10], "OK");
        assert!(warnings(&few, &summarise(&few, 0.03), 0.03)[0].contains("only 5 round(s)"));
    }

    #[test]
    fn warnings_name_drift_and_real_checksum_failures_only() {
        let drift = rounds(MIN_ROUNDS, 1.0, [10, 20], "OK");
        let w = warnings(&drift, &summarise(&drift, 0.03), 0.03);
        assert!(w.iter().any(|x| x.contains("instructions per iteration") && x.contains("k (")), "{w:?}");
        // SKIP and UNCHECKED rows are not failures
        let skip = rounds(MIN_ROUNDS, 1.0, [10, 10], "SKIP");
        assert!(warnings(&skip, &summarise(&skip, 0.03), 0.03).is_empty());
        let bad = rounds(MIN_ROUNDS, 1.0, [10, 10], "MISMATCH");
        let w = warnings(&bad, &summarise(&bad, 0.03), 0.03);
        assert!(w.iter().any(|x| x == "checksum mismatch: k (B)"), "{w:?}");
        // half the rounds at 1.5x: an even count, so neither half is the median
        let mut noisy = rounds(12, 1.0, [10, 10], "OK");
        for (i, r) in noisy.iter_mut().enumerate() { r[1].total_ns = 1_000_000 * (1 + i as u64 % 2) * 3 / 2; }
        assert!(warnings(&noisy, &summarise(&noisy, 0.03), 0.03).iter().any(|x| x.contains("noisy")));
    }

    #[test]
    fn the_run_length_goes_by_flag_when_both_builds_take_it_and_into_the_image_otherwise() {
        let suite = iris::benchsuite::SUITE_ELF;
        let word = |img: &[u8], name| {
            let at = iris::elf::symbol_offset(img, name).unwrap();
            u32::from_be_bytes(img[at..at + 4].try_into().unwrap())
        };
        // Two builds with the flag: it is passed, and the image runs as built.
        let mut image = suite.to_vec();
        assert_eq!(run_length_args(&mut image, [true, true], 10, 3).unwrap(),
                   ["--test-device-run-config", "time_pct=10,repeats=3"]);
        assert!(image == suite);
        // A build older than the flag on either side: both run a patched image.
        for takes_flag in [[false, true], [true, false], [false, false]] {
            let mut image = suite.to_vec();
            assert!(run_length_args(&mut image, takes_flag, 10, 3).unwrap().is_empty());
            assert_eq!((word(&image, "bench_time_pct"), word(&image, "bench_repeats")), (10, 3));
        }
        assert!(set_run_length(&mut [0u8; 64], 10, 3).is_err());
    }

    #[test]
    fn a_build_takes_the_flag_when_its_help_lists_it() {
        let new = "      --test-device-dump <FILE>\n          Where the test device writes its dump\n      \
                   --test-device-run-config <FIELDS>\n          Ask a bare-metal guest for a shorter run\n";
        let old = "      --test-device-dump <FILE>\n          Where the test device writes its dump\n      \
                   --cheritest-dump-hook\n";
        assert!(help_lists_run_config(new));
        assert!(!help_lists_run_config(old));
    }

    #[test]
    fn builds_are_checked_before_they_are_compared() {
        let f = |s: &str| -> Vec<String> { s.split_whitespace().map(String::from).collect() };
        assert!(check_builds(&f("jitv2"), &f("jitv2"), false).is_ok());
        assert!(check_builds(&f("jitv2"), &f(""), false).unwrap_err().contains("different features"));
        assert!(check_builds(&f("jitv2"), &f(""), true).is_ok());
        assert!(check_builds(&f("ci_clock"), &f("ci_clock"), true).unwrap_err().contains("A is a ci_clock"));
    }
}
