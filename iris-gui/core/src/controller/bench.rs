//! The benchmark: the bare-metal suite IRIS carries inside itself, run on a
//! headless Indy of its own as a job, with progress, cancel and the report.

use std::sync::{Arc, Mutex};
use std::time::Duration;

pub use iris::bench_report::{fmt_rate, ReferenceEntry, ReferenceTable, Run, CATEGORIES};
use iris::bench_runner::{self, BenchOptions, Progress as BenchProgress};

use super::{Answer, Effect, Enabled, PickSpec};
use crate::model::job::{Job, Notify, Progress, Unit};

/// What the two modes cost on a plain interpreter build, for the tooltip and
/// the time estimate before there is evidence.
pub const QUICK_SECS: u64 = 35;
pub const FULL_SECS: u64 = 60;
const MAX_LINES: usize = 4000;
pub(crate) const SAVE_REPORT: &str = "bench.save-report";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchAction {
    Start,
    Stop,
    SetQuick(bool),
    /// An index into [`BenchController::cpus`].
    SetCpu(usize),
    SaveReport,
    Answer(Answer),
}

/// One line of the results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultRow {
    pub label: String,
    pub value: String,
    /// A comparison beside the value, or empty.
    pub note: String,
    pub tip: Option<String>,
}

impl ResultRow {
    fn plain(label: String, value: String) -> Self {
        Self { label, value, note: String::new(), tip: None }
    }

    fn tipped(label: String, value: String, tip: String) -> Self {
        Self { label, value, note: String::new(), tip: Some(tip) }
    }
}

/// A finished run, laid out for a shell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Results {
    /// The figures that stand on their own: DMIPS, guest MIPS, accuracy.
    pub headline: Vec<ResultRow>,
    /// Each category's rate, with the reference ratio as its note.
    pub categories: Vec<ResultRow>,
    /// What to read the numbers with.
    pub caveats: Vec<String>,
    /// The machine the run was made on.
    pub machine: Vec<ResultRow>,
}

/// How this run compares with a stored row, per category. The stored row
/// keeps per-kernel rates, so the category is rebuilt from them the same
/// way, weighted by this run's own times: close enough for "roughly this
/// much faster", which is all it claims.
fn category_ratio(run: &Run, reference: &ReferenceEntry, category: &str) -> Option<(f64, String)> {
    let c = CATEGORIES.iter().find(|c| c.label == category)?;
    let mine = run.category_rate(c)?;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for r in &run.rows {
        if r.status == "SKIP" || r.unit != c.unit || r.ns == 0 {
            continue;
        }
        if !c.prefixes.iter().any(|p| r.name.starts_with(p)) {
            continue;
        }
        let Some(their_rate) = reference.kernels.get(&r.name) else { continue };
        if *their_rate <= 0.0 {
            continue;
        }
        num += *their_rate * r.ns as f64;
        den += r.ns as f64;
    }
    if den <= 0.0 {
        return None;
    }
    let theirs = num / den;
    (theirs > 0.0).then(|| (mine / theirs, reference.label.clone()))
}

/// The machine a run was made on, as rows.
fn machine_rows(run: &Run) -> Vec<ResultRow> {
    let m = &run.machine;
    let mut rows = vec![ResultRow::plain(
        t!("bench-machine-cpu"),
        if m.rev.is_empty() {
            t!("bench-machine-cpu-value", cpu = m.cpu.as_str(), prid = m.prid.as_str())
        } else {
            t!("bench-machine-cpu-rev-value", cpu = m.cpu.as_str(), rev = m.rev.as_str(), prid = m.prid.as_str())
        },
    )];
    if !m.cache.is_empty() {
        let c = &m.cache;
        rows.push(ResultRow::plain(
            t!("bench-machine-l1"),
            t!(
                "bench-machine-l1-value",
                icache = fmt_bytes(c.l1i_bytes),
                iline = c.l1i_line,
                dcache = fmt_bytes(c.l1d_bytes),
                dline = c.l1d_line
            ),
        ));
        rows.push(ResultRow::plain(
            t!("bench-machine-l2"),
            if !c.l2_present {
                t!("bench-machine-l2-absent")
            } else if c.l2_bytes > 0 {
                t!("bench-machine-l2-value", size = fmt_bytes(c.l2_bytes), line = c.l2_line)
            } else {
                // Nothing but a Triton exposes the size; the PROM reads it.
                t!("bench-machine-l2-unsized", line = c.l2_line)
            },
        ));
    }
    if !m.memory.is_empty() {
        let banks: Vec<String> = m
            .memory
            .banks
            .iter()
            .map(|b| t!("bench-machine-bank", index = b.index, size = b.mb, base = format!("{:#010x}", b.base)))
            .collect();
        rows.push(ResultRow::plain(
            t!("bench-machine-memory"),
            t!("bench-machine-memory-value", total = m.memory.total_mb, banks = banks.join("  ")),
        ));
    }
    if !m.sysid.is_empty() {
        rows.push(ResultRow::plain(t!("bench-machine-board"), t!("bench-machine-board-value", sysid = m.sysid.as_str())));
    }
    rows.push(ResultRow::plain(
        t!("bench-machine-emulator"),
        if run.features.is_empty() { t!("bench-machine-no-features") } else { run.features.join(" ") },
    ));
    rows.push(ResultRow::plain(
        t!("bench-machine-host"),
        t!(
            "bench-machine-host-value",
            cpu = if run.host.cpu_model.is_empty() { t!("bench-machine-unknown-cpu") } else { run.host.cpu_model.clone() },
            os = run.host.os.as_str(),
            arch = run.host.arch.as_str(),
            cores = run.host.cores
        ),
    ));
    rows
}

/// A run in progress, for its progress bar.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchProgressView {
    pub fraction: f32,
    /// "Starting" or "dhrystone (3 of 24)".
    pub line: String,
    /// Elapsed, and time left once there is enough evidence.
    pub time: String,
}

/// Where a run is, written by the runner's callback.
#[derive(Debug, Default, Clone)]
struct Live {
    current: String,
    done: usize,
    total: usize,
}

pub struct BenchController {
    quick: bool,
    cpu: &'static iris::cpu::parts::Part,
    job: Option<Job<Box<Run>>>,
    lines: Arc<Mutex<Vec<String>>>,
    live: Arc<Mutex<Live>>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    last: Option<Result<Box<Run>, String>>,
    reference: Option<ReferenceTable>,
    notify: Arc<dyn Notify>,
}

impl BenchController {
    pub fn new(notify: Arc<dyn Notify>) -> Self {
        Self {
            // Quick by default: the same figures to a couple of percent in
            // about half the time, and every kernel still verifies.
            quick: true,
            cpu: &iris::cpu::parts::R4400_150,
            job: None,
            lines: Arc::default(),
            live: Arc::default(),
            cancel: Arc::default(),
            last: None,
            reference: None,
            notify,
        }
    }

    pub fn quick(&self) -> bool {
        self.quick
    }

    /// The CPUs the suite can run on, as labels; and which is chosen.
    pub fn cpus(&self) -> (Vec<String>, usize) {
        let all = bench_runner::BENCH_CPUS;
        (all.iter().map(|c| c.label().to_string()).collect(), all.iter().position(|c| *c == self.cpu).unwrap_or(0))
    }

    pub fn cpu_label(&self) -> String {
        self.cpu.label()
    }

    pub fn is_running(&self) -> bool {
        self.job.as_ref().is_some_and(|j| j.is_running())
    }

    /// Whether Start can run: one machine at a time measures the right thing.
    pub fn can_start(&self, machine_running: bool) -> Enabled {
        if self.is_running() {
            Enabled::No(t!("bench-running"))
        } else {
            Enabled::when(!machine_running, || t!("bench-machine-running"))
        }
    }

    pub fn progress(&self) -> Option<BenchProgressView> {
        let job = self.job.as_ref()?;
        let snap = job.snapshot();
        if !snap.is_running() {
            return None;
        }
        let live = self.live.lock().map(|l| l.clone()).unwrap_or_default();
        let (done, total) = (live.done, live.total);
        let line = if live.current.is_empty() {
            t!("bench-starting")
        } else {
            t!("bench-progress", kernel = live.current.as_str(), done = done, total = total)
        };
        let fraction = Progress::Determinate { done: done as u64, total: total as u64, unit: Unit::Items };
        Some(BenchProgressView {
            fraction: fraction.fraction().unwrap_or(0.0),
            line,
            time: remaining(done, total, snap.elapsed, self.quick),
        })
    }

    /// The last finished run, or why it failed ("stopped" when cancelled).
    pub fn last(&self) -> Option<&Result<Box<Run>, String>> {
        self.last.as_ref()
    }

    /// The console lines of the current or last run.
    pub fn lines(&self) -> Vec<String> {
        self.lines.lock().map(|l| l.clone()).unwrap_or_default()
    }

    /// The reference entry for the last run, when the bundled table has one.
    pub fn reference(&mut self) -> Option<ReferenceEntry> {
        let run = match &self.last {
            Some(Ok(r)) => r.clone(),
            _ => return None,
        };
        self.reference.get_or_insert_with(iris::bench_report::bundled_reference).matching(&run).cloned()
    }

    pub fn apply(&mut self, action: BenchAction, machine_running: bool) -> Vec<Effect> {
        let mut fx = Vec::new();
        match action {
            BenchAction::Start => {
                if let Enabled::No(why) = self.can_start(machine_running) {
                    fx.push(Effect::Toast(why));
                    return fx;
                }
                self.start();
            }
            BenchAction::Stop => self.cancel.store(true, std::sync::atomic::Ordering::Relaxed),
            BenchAction::SetQuick(q) => self.quick = q,
            BenchAction::SetCpu(i) => {
                if let Some(c) = bench_runner::BENCH_CPUS.get(i) {
                    self.cpu = c;
                }
            }
            BenchAction::SaveReport => {
                if let Some(Ok(run)) = &self.last {
                    let name = format!("iris-benchmark-{}.json", run.cell.replace(['/', ' '], "-"));
                    fx.push(Effect::PickFile(
                        PickSpec::save(SAVE_REPORT, t!("bench-save-report-picker"), name)
                            .filter(&t!("bench-save-report-filter"), &["json"]),
                    ));
                }
            }
            BenchAction::Answer(Answer::Picked { purpose: SAVE_REPORT, paths }) => {
                if let (Some(path), Some(Ok(run))) = (paths.first(), &self.last) {
                    let r = serde_json::to_string_pretty(run)
                        .map_err(|e| e.to_string())
                        .and_then(|j| std::fs::write(path, j).map_err(|e| e.to_string()));
                    if let Err(e) = r {
                        fx.push(Effect::Toast(t!("bench-save-failed", error = e)));
                    }
                }
            }
            BenchAction::Answer(_) => {}
        }
        fx
    }

    fn start(&mut self) {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let live = Arc::new(Mutex::new(Live::default()));
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.lines = lines.clone();
        self.live = live.clone();
        self.cancel = cancel.clone();
        self.last = None;
        let (quick, cpu) = (self.quick, self.cpu);
        let notify = self.notify.clone();
        self.job = Some(Job::spawn(t!("bench-heading"), false, self.notify.clone(), move |_| {
            let opts = BenchOptions {
                quick,
                cpu,
                label: iris::bench_report::cpu_model(),
                cancel: Some(cancel),
                ..Default::default()
            };
            bench_runner::run(&opts, move |p| {
                match p {
                    BenchProgress::Started { total } => {
                        if let Ok(mut l) = live.lock() {
                            l.total = total;
                        }
                    }
                    BenchProgress::Kernel { name, index } => {
                        if let Ok(mut l) = live.lock() {
                            l.done = index;
                            l.current = name;
                        }
                    }
                    BenchProgress::Line(line) => {
                        if let Ok(mut v) = lines.lock() {
                            v.push(line);
                            if v.len() > MAX_LINES {
                                let n = v.len() - MAX_LINES;
                                v.drain(..n);
                            }
                        }
                    }
                }
                notify.wake();
            })
            .map(Box::new)
        }));
    }

    /// Move a finished run into `last`.
    pub fn tick(&mut self) {
        if let Some(outcome) = self.job.as_ref().and_then(|j| j.outcome()) {
            self.job = None;
            self.last = Some(outcome);
        }
    }

    /// The results as a shell shows them: the figures that stand on their
    /// own, the per-category rates with the reference ratio beside them, the
    /// caveats, and the machine the run was made on. Built here so every
    /// shell draws the same list rather than reaching into `Run`.
    pub fn results(&mut self) -> Option<Results> {
        let reference = self.reference();
        let run = match self.last()? {
            Ok(run) => run.clone(),
            Err(_) => return None,
        };
        let headline = vec![
            ResultRow::tipped(
                t!("bench-dmips"),
                t!("bench-dmips-value", value = format!("{:.0}", run.dmips().unwrap_or(0.0))),
                t!("bench-dmips-tip"),
            ),
            ResultRow::tipped(
                t!("bench-mips"),
                t!("bench-mips-value", value = format!("{:.0}", run.mips())),
                t!("bench-mips-tip"),
            ),
            ResultRow::tipped(
                t!("bench-accuracy"),
                t!(
                    "bench-accuracy-value",
                    percent = format!("{:.0}", run.accuracy()),
                    matched = run.matched,
                    checked = run.checked
                ),
                t!("bench-accuracy-tip"),
            ),
        ];
        let mut categories = Vec::new();
        for c in CATEGORIES {
            let Some(rate) = run.category_rate(c) else { continue };
            let note = reference
                .as_ref()
                .and_then(|r| category_ratio(&run, r, c.label))
                .map(|(ratio, label)| t!("bench-ratio", ratio = format!("{ratio:.2}"), reference = label))
                .unwrap_or_default();
            categories.push(ResultRow {
                label: c.label.to_string(),
                value: format!("{}{}", fmt_rate(rate), c.suffix),
                note,
                tip: None,
            });
        }
        let mut caveats = Vec::new();
        if let Some(why) = run.settings.shortened_because() {
            caveats.push(t!("bench-caveat-quick", reason = why));
        }
        if !run.features.iter().any(|f| f == "jitv2") {
            caveats.push(t!("bench-caveat-interpreter"));
        }
        let unexpected: Vec<&str> = run
            .rows
            .iter()
            .filter(|r| r.exc > 0 && !iris::bench_report::EXPECT_EXC.contains(&r.name.as_str()))
            .map(|r| r.name.as_str())
            .collect();
        if !unexpected.is_empty() {
            caveats.push(t!("bench-caveat-exceptions", kernels = unexpected.join(", ")));
        }
        if reference.is_none() {
            caveats.push(t!("bench-no-reference"));
        }
        Some(Results { headline, categories, caveats, machine: machine_rows(&run) })
    }

    /// The report as plain text, for copying.
    pub fn report_text(run: &Run) -> String {
        report_text(run)
    }
}

/// Elapsed, plus an estimate once a few kernels have run: the first are cheap
/// integer kernels and the last expensive codecs, so an early guess is wrong.
fn remaining(done: usize, total: usize, elapsed: Duration, quick: bool) -> String {
    let e = elapsed.as_secs();
    if done < 4 || total == 0 {
        let guess = if quick { QUICK_SECS } else { FULL_SECS };
        return t!("bench-time-total", elapsed = e, total = guess);
    }
    let per = elapsed.as_secs_f64() / done as f64;
    let left = (per * (total - done) as f64).round() as u64;
    t!("bench-time-left", elapsed = e, left = left)
}

fn fmt_bytes(n: u64) -> String {
    if n >= 1 << 20 {
        format!("{} MB", n >> 20)
    } else if n >= 1 << 10 {
        format!("{} KB", n >> 10)
    } else {
        format!("{n} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_time_estimate_waits_for_evidence_before_making_one() {
        let s = remaining(2, 46, Duration::from_secs(3), true);
        assert_eq!(s, t!("bench-time-total", elapsed = 3u64, total = QUICK_SECS), "must not extrapolate from two rows");
        // Halfway through 40 s of a 46-kernel run: about 40 s more to go.
        let s = remaining(23, 46, Duration::from_secs(40), true);
        assert_eq!(s, t!("bench-time-left", elapsed = 40u64, left = 40u64));
    }
}

fn report_text(run: &Run) -> String {
    let mut s = String::new();
    s.push_str(&format!("IRIS benchmark - {}\n", run.host.cpu_model));
    s.push_str(&format!("  emulated CPU      {} ({})\n", run.machine.cpu, iris::bench_report::engine_of(run)));
    s.push_str(&format!(
        "  build features    {}\n",
        if run.features.is_empty() { "(none)".into() } else { run.features.join(" ") }
    ));
    s.push_str(&format!("  host              {} {} - {} cores\n", run.host.os, run.host.arch, run.host.cores));
    if !run.machine.rev.is_empty() {
        s.push_str(&format!("  emulated PRId     {} rev {}\n", run.machine.prid, run.machine.rev));
    }
    if !run.machine.cache.is_empty() {
        let c = &run.machine.cache;
        s.push_str(&format!(
            "  L1 cache          {} I ({} B) / {} D ({} B)\n",
            fmt_bytes(c.l1i_bytes),
            c.l1i_line,
            fmt_bytes(c.l1d_bytes),
            c.l1d_line
        ));
        s.push_str(&format!(
            "  L2 cache          {}\n",
            if !c.l2_present {
                "absent".to_string()
            } else if c.l2_bytes > 0 {
                format!("{}, {} B lines", fmt_bytes(c.l2_bytes), c.l2_line)
            } else {
                format!("present, {} B lines, size not reported", c.l2_line)
            }
        ));
    }
    if !run.machine.memory.is_empty() {
        s.push_str(&format!(
            "  memory            {} MB in {} bank(s)\n",
            run.machine.memory.total_mb,
            run.machine.memory.banks.len()
        ));
    }
    s.push('\n');
    s.push_str(&format!("  Emulated Indy     {:.1} DMIPS\n", run.dmips().unwrap_or(0.0)));
    s.push_str(&format!("  Throughput        {:.1} MIPS\n", run.mips()));
    s.push_str(&format!("  Accuracy          {:.1}%  ({}/{})\n", run.accuracy(), run.matched, run.checked));
    s.push('\n');
    for c in CATEGORIES {
        if let Some(rate) = run.category_rate(c) {
            s.push_str(&format!("  {:<16}  {}{}\n", c.label, fmt_rate(rate), c.suffix));
        }
    }
    s.push('\n');
    s.push_str(&format!("  suite             {}\n", run.suite_id));
    if let Some(why) = run.settings.shortened_because() {
        s.push_str(&format!("  quick run         {}\n", why));
    }
    s.push_str(&format!("  wall clock        {:.1} s\n", run.wall_s));
    s
}
