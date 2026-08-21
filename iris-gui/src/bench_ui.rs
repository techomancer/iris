//! Benchmark tab — measure this machine's emulated Indy, in-process.
//!
//! One button, one number. The suite is a bare-metal MIPS binary that IRIS
//! carries inside itself (`iris::benchsuite`) and runs on a headless machine of
//! its own (`iris::bench_runner`): no toolchain, no subprocess, no file written
//! anywhere, and identical on macOS, Windows and Linux. That is what makes this
//! shippable rather than a developer convenience — the older version of this
//! tab spawned `iris-bench`, which spawned `iris`, which read an ELF off disk,
//! and none of those three steps survive an application sandbox.
//!
//! It measures *the emulator*, not the host: how fast this build of IRIS runs
//! an Indy on the hardware it happens to be sitting on, and whether it still
//! computes the right answers after ten million instructions of doing it. The
//! accuracy score is the part no other emulator reports, so it is shown as
//! prominently as the speed.
//!
//! The matrix runner is still a subprocess and still developer-only: the CPU
//! model and the JIT are compile-time cargo features, so comparing them means
//! building and comparing binaries.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, RichText, ScrollArea, Ui};
use iris::bench_report::{fmt_rate, ReferenceEntry, ReferenceTable, Run, CATEGORIES};
use iris::bench_runner::{self, BenchOptions, Progress};

/// How many console lines to keep. The suite prints a few hundred; the cap is
/// there for a run that goes wrong and starts repeating itself.
const MAX_LINES: usize = 4000;

/// What the two modes cost on a plain interpreter build, for the button's
/// hover text. Measured on the reference host — treated as an order of
/// magnitude, not a promise, which is why the UI also shows real elapsed time.
const QUICK_SECS: u64 = 35;
const FULL_SECS: u64 = 60;

pub struct BenchState {
    live: Option<Live>,
    /// The last finished run, kept so the panel still says something once the
    /// worker is gone.
    last: Option<Result<Box<Run>, String>>,
    quick: bool,
    /// Parsed once. Normally empty — see `iris::bench_report::bundled_reference`.
    reference: Option<ReferenceTable>,
    dev: DevRunner,
}

impl Default for BenchState {
    fn default() -> Self {
        Self {
            live: None,
            last: None,
            // Quick by default. It reports the same figures to within a couple
            // of percent for about half the wall clock, and it gives up no
            // accuracy at all — every kernel still runs and still verifies. A
            // full run is one click away for anyone who wants the tighter
            // number, and is required before a result can go in the reference
            // table.
            quick: true,
            reference: None,
            dev: DevRunner::default(),
        }
    }
}

/// A run in flight. Everything here is written by the worker thread and read by
/// the UI thread once a frame.
struct Live {
    lines: Arc<Mutex<Vec<String>>>,
    state: Arc<Mutex<LiveState>>,
    cancel: Arc<AtomicBool>,
    done: Arc<Mutex<Option<Result<Box<Run>, String>>>>,
    started: Instant,
    quick: bool,
}

#[derive(Default, Clone)]
struct LiveState {
    /// Kernels the guest said it would run. 0 until it says.
    total: usize,
    done: usize,
    current: String,
}

impl LiveState {
    fn fraction(&self) -> f32 {
        if self.total == 0 { return 0.0; }
        (self.done as f32 / self.total as f32).clamp(0.0, 1.0)
    }
}

// ─── running ─────────────────────────────────────────────────────────────────

impl BenchState {
    pub fn is_running(&self) -> bool {
        self.live.is_some() || self.dev.is_running()
    }

    fn start(&mut self, quick: bool) {
        if self.is_running() { return; }

        let lines = Arc::new(Mutex::new(Vec::new()));
        let state = Arc::new(Mutex::new(LiveState::default()));
        let cancel = Arc::new(AtomicBool::new(false));
        let done = Arc::new(Mutex::new(None));

        let opts = BenchOptions {
            quick,
            label: iris::bench_report::cpu_model(),
            cancel: Some(cancel.clone()),
            ..Default::default()
        };

        let (l, s, d) = (lines.clone(), state.clone(), done.clone());
        std::thread::spawn(move || {
            let result = bench_runner::run(&opts, move |p| match p {
                Progress::Started { total } => s.lock().unwrap().total = total,
                Progress::Kernel { name, index } => {
                    let mut st = s.lock().unwrap();
                    st.done = index;
                    st.current = name;
                }
                Progress::Line(line) => {
                    let mut v = l.lock().unwrap();
                    v.push(line);
                    if v.len() > MAX_LINES {
                        let drop_n = v.len() - MAX_LINES;
                        v.drain(..drop_n);
                    }
                }
            });
            *d.lock().unwrap() = Some(result.map(Box::new));
        });

        self.last = None;
        self.live = Some(Live {
            lines, state, cancel, done,
            started: Instant::now(),
            quick,
        });
    }

    /// Move a finished worker's result into `last`. Called once a frame.
    fn poll(&mut self) {
        let Some(live) = &self.live else { return };
        let finished = live.done.lock().unwrap().take();
        if let Some(result) = finished {
            self.last = Some(result);
            self.live = None;
        }
    }

    fn stop(&mut self) {
        if let Some(live) = &self.live {
            live.cancel.store(true, Ordering::Relaxed);
        }
        self.dev.stop();
    }

    fn reference(&mut self) -> &ReferenceTable {
        self.reference.get_or_insert_with(iris::bench_report::bundled_reference)
    }
}

// ─── the screen ──────────────────────────────────────────────────────────────

pub fn show(ui: &mut Ui, st: &mut BenchState, machine_running: bool) {
    st.poll();

    ui.heading("Benchmark");
    ui.label(
        "Measures how fast this build of IRIS emulates an Indy on this machine, and \
         checks that it is still computing the right answers. Everything it needs is \
         built in — no disk image, no IRIX, nothing downloaded, and nothing sent \
         anywhere.",
    );
    ui.add_space(8.0);

    controls(ui, st, machine_running);

    if let Some(live) = &st.live {
        ui.add_space(8.0);
        running_view(ui, live);
    }

    match st.last.take() {
        Some(Ok(run)) => {
            ui.add_space(10.0);
            ui.separator();
            ui.add_space(6.0);
            let reference = st.reference().matching(&run).cloned();
            results_view(ui, &run, reference.as_ref());
            st.last = Some(Ok(run));
        }
        Some(Err(e)) => {
            ui.add_space(8.0);
            if e == "stopped" {
                ui.label(RichText::new("Stopped.").weak());
            } else {
                ui.label(RichText::new(format!("The benchmark did not finish: {e}"))
                    .color(Color32::from_rgb(220, 170, 90)));
            }
            st.last = Some(Err(e));
        }
        None => {}
    }

    details(ui, st);

    #[cfg(not(feature = "appstore"))]
    developer_tools(ui, st);
}

fn controls(ui: &mut Ui, st: &mut BenchState, machine_running: bool) {
    let busy = st.is_running();

    ui.horizontal(|ui| {
        // Two machines at once would measure the wrong thing: the emulator
        // would be sharing the host with itself, and every number would be a
        // reading of whatever IRIX happened to be doing. Refusing is also
        // simpler than explaining the result afterwards.
        ui.add_enabled_ui(!busy && !machine_running, |ui| {
            let secs = if st.quick { QUICK_SECS } else { FULL_SECS };
            if ui
                .button(RichText::new(primary_label()).strong())
                .on_hover_text(format!("About {secs} seconds. Runs entirely inside this app."))
                .clicked()
            {
                let quick = st.quick;
                st.start(quick);
            }
        });

        ui.add_enabled_ui(!busy, |ui| {
            ui.checkbox(&mut st.quick, "Quick")
                .on_hover_text(
                    "Shorter timed runs and one pass per kernel instead of the best of \
                     two. Every kernel still runs and still checks its answer, so the \
                     accuracy score means exactly the same thing — only the speed \
                     figures are a little noisier.",
                );
        });

        if busy && ui.button("Stop").clicked() {
            st.stop();
        }
    });

    if machine_running && !busy {
        ui.add_space(4.0);
        ui.label(
            RichText::new("Stop the emulator first — a benchmark run needs the machine to itself.")
                .weak(),
        );
    }
}

fn primary_label() -> String {
    // Name the machine the way its owner does. This measures the emulator, but
    // what varies between users is the hardware under it.
    let what = if cfg!(target_os = "macos") { "this Mac" } else { "this PC" };
    format!("Benchmark {what}")
}

fn running_view(ui: &mut Ui, live: &Live) {
    let st = live.state.lock().unwrap().clone();
    let elapsed = live.started.elapsed();

    ui.add(egui::ProgressBar::new(st.fraction()).show_percentage().animate(true));
    ui.horizontal(|ui| {
        ui.label(if st.current.is_empty() {
            "Starting the emulated machine…".to_string()
        } else {
            format!("{}  ({} of {})", st.current, st.done, st.total)
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(remaining(&st, elapsed, live.quick)).weak());
        });
    });

    // The worker writes into shared state; without this the panel only updates
    // when the pointer moves over it.
    ui.ctx().request_repaint_after(Duration::from_millis(200));
}

/// Elapsed, plus an estimate once there is enough evidence for one.
///
/// Extrapolating from kernels-completed is only honest after a few of them:
/// the first rows are the cheap integer kernels and the last are the expensive
/// codec ones, so an estimate from row two is confidently wrong. Before that,
/// say only what is known.
fn remaining(st: &LiveState, elapsed: Duration, quick: bool) -> String {
    let e = elapsed.as_secs();
    if st.done < 4 || st.total == 0 {
        let guess = if quick { QUICK_SECS } else { FULL_SECS };
        return format!("{e}s elapsed · about {guess}s in total");
    }
    let per = elapsed.as_secs_f64() / st.done as f64;
    let left = (per * (st.total - st.done) as f64).round() as u64;
    format!("{e}s elapsed · about {left}s left")
}

// ─── results ─────────────────────────────────────────────────────────────────

fn results_view(ui: &mut Ui, run: &Run, reference: Option<&ReferenceEntry>) {
    ui.heading("Results");
    ui.add_space(4.0);

    // The three figures that stand on their own. DMIPS has forty years of
    // published numbers behind it, guest MIPS is meaningful without a baseline,
    // and accuracy needs no comparison at all — which is why an empty reference
    // table costs the reader so little.
    egui::Grid::new("bench_headline").num_columns(3).spacing([18.0, 6.0]).show(ui, |ui| {
        headline(ui, "Emulated Indy", &format!("{:.0} DMIPS", run.dmips().unwrap_or(0.0)),
                 "Dhrystone 2.1, the figure every published workstation benchmark since \
                  1988 uses. A real 150 MHz Indy scored about 130.");
        ui.end_row();
        headline(ui, "Emulator throughput", &format!("{:.0} MIPS", run.mips()),
                 "Guest instructions retired per second of host time.");
        ui.end_row();
        headline(ui, "Accuracy",
                 &format!("{:.0}%  ({}/{})", run.accuracy(), run.matched, run.checked),
                 "Share of kernels whose result matched a checksum computed independently \
                  by building the same C natively. Anything below 100% is a real emulator \
                  bug and worth reporting.");
        ui.end_row();
    });

    ui.add_space(10.0);

    egui::Grid::new("bench_categories").num_columns(3).spacing([18.0, 6.0]).striped(true)
        .show(ui, |ui| {
            for c in CATEGORIES {
                let Some(rate) = run.category_rate(c) else { continue };
                ui.label(c.label);
                ui.label(RichText::new(format!("{}{}", fmt_rate(rate), c.suffix)).monospace());
                match reference.and_then(|r| ratio(run, r, c.label)) {
                    Some((ratio, label)) => {
                        ui.label(RichText::new(format!("{ratio:.2}× vs {label}")).weak());
                    }
                    None => { ui.label(""); }
                }
                ui.end_row();
            }
        });

    ui.add_space(8.0);
    if reference.is_none() {
        ui.label(
            RichText::new("Reference statistics not gathered for this platform.").weak(),
        );
    }
    caveats(ui, run);

    ui.add_space(8.0);
    machine_view(ui, run);

    ui.add_space(8.0);
    ui.horizontal(|ui| {
        if ui.button("Copy").clicked() {
            ui.ctx().copy_text(report_text(run));
        }
        if ui.button("Save report…").clicked() {
            save_report(run);
        }
    });
}

/// The machine that was measured, as the guest found it.
///
/// Read out of the emulated hardware by the suite itself — CP0 Config for the
/// caches, the memory controller for the banks — not reported from what the
/// GUI configured. When those two disagree, the guest is right and the
/// disagreement is the interesting part.
///
/// Collapsed by default: it is provenance, not a headline. But it belongs on
/// the results screen rather than only in the export, because it is what makes
/// the numbers above comparable with anything.
fn machine_view(ui: &mut Ui, run: &Run) {
    let m = &run.machine;
    egui::CollapsingHeader::new("Machine measured").default_open(false).show(ui, |ui| {
        egui::Grid::new("bench_machine").num_columns(2).spacing([18.0, 4.0]).striped(true)
            .show(ui, |ui| {
                let mut row = |k: &str, v: String| {
                    ui.label(k);
                    ui.label(RichText::new(v).monospace());
                    ui.end_row();
                };

                let rev = if m.rev.is_empty() { String::new() } else { format!(" rev {}", m.rev) };
                row("CPU", format!("{}{}  ({})", m.cpu, rev, m.prid));

                if !m.cache.is_empty() {
                    let c = &m.cache;
                    row("L1 cache", format!(
                        "{} I ({} B lines) / {} D ({} B lines)",
                        fmt_bytes(c.l1i_bytes), c.l1i_line,
                        fmt_bytes(c.l1d_bytes), c.l1d_line));
                    row("L2 cache", if !c.l2_present {
                        "absent".to_string()
                    } else if c.l2_bytes > 0 {
                        format!("{}, {} B lines", fmt_bytes(c.l2_bytes), c.l2_line)
                    } else {
                        // The architecture does not expose the size on anything
                        // but a Triton; the PROM reads it from the EEPROM.
                        format!("present, {} B lines, size not reported", c.l2_line)
                    });
                }

                if !m.memory.is_empty() {
                    let banks: Vec<String> = m.memory.banks.iter()
                        .map(|b| format!("bank{} {} MB @ {:#010x}", b.index, b.mb, b.base))
                        .collect();
                    row("Memory", format!("{} MB   {}", m.memory.total_mb, banks.join("  ")));
                }

                if !m.sysid.is_empty() { row("Board", format!("SYSID {}", m.sysid)); }
                row("Emulator", if run.features.is_empty() {
                    "no optional features".to_string()
                } else {
                    run.features.join(" ")
                });
                row("Host", format!("{} · {} {} · {} cores",
                                    m_host(run), run.host.os, run.host.arch, run.host.cores));
            });
    });
}

fn m_host(run: &Run) -> &str {
    if run.host.cpu_model.is_empty() { "unknown CPU" } else { &run.host.cpu_model }
}

/// KB under a megabyte, MB above — the way a person reads a cache size.
fn fmt_bytes(n: u64) -> String {
    if n >= 1 << 20 { format!("{} MB", n >> 20) }
    else if n >= 1 << 10 { format!("{} KB", n >> 10) }
    else { format!("{} B", n) }
}

fn headline(ui: &mut Ui, label: &str, value: &str, help: &str) {
    ui.label(label);
    ui.label(RichText::new(value).size(18.0).strong());
    ui.label(RichText::new("?").weak()).on_hover_text(help);
}

/// This run's category rate over the reference row's, when the row carries
/// enough kernels to compute one.
fn ratio(run: &Run, reference: &ReferenceEntry, category: &str) -> Option<(f64, String)> {
    let c = CATEGORIES.iter().find(|c| c.label == category)?;
    let mine = run.category_rate(c)?;

    // The stored row keeps per-kernel rates, so rebuild the category from them
    // the same way — but weighted by this run's own times, since the reference
    // row does not carry them. Close enough for a "roughly this much faster",
    // which is all the comparison claims to be.
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for r in &run.rows {
        if r.status == "SKIP" || r.unit != c.unit || r.ns == 0 { continue; }
        if !c.prefixes.iter().any(|p| r.name.starts_with(p)) { continue; }
        let Some(their_rate) = reference.kernels.get(&r.name) else { continue };
        if *their_rate <= 0.0 { continue; }
        num += *their_rate * r.ns as f64;
        den += r.ns as f64;
    }
    if den == 0.0 || num == 0.0 { return None; }
    Some((mine / (num / den), reference.label.clone()))
}

/// Everything that would make a reader draw the wrong conclusion from the
/// numbers above, said plainly rather than left for them to discover.
fn caveats(ui: &mut Ui, run: &Run) {
    if let Some(why) = run.settings.shortened_because() {
        ui.label(RichText::new(format!("Quick run — {why}. Accuracy is unaffected."))
            .weak());
    }
    if !run.features.iter().any(|f| f == "jitv2") {
        ui.label(RichText::new(
            "This build runs the interpreter. A build with the MIPS JIT scores roughly \
             four times higher, and is not comparable with these numbers.").weak());
    }
    let unexpected: Vec<&str> = run.rows.iter()
        .filter(|r| r.exc > 0 && !iris::bench_report::EXPECT_EXC.contains(&r.name.as_str()))
        .map(|r| r.name.as_str())
        .collect();
    if !unexpected.is_empty() {
        // A kernel that faults is stepped over by the harness and still reports
        // a throughput — for doing something other than what it claims.
        ui.label(RichText::new(format!(
            "Took unexpected exceptions in {} — those figures are not trustworthy.",
            unexpected.join(", ")))
            .color(Color32::from_rgb(220, 170, 90)));
    }
}

// ─── details, export ─────────────────────────────────────────────────────────

fn details(ui: &mut Ui, st: &mut BenchState) {
    let lines = st.live.as_ref().map(|l| l.lines.clone());
    let has_output = lines.is_some() || st.dev.has_output();
    if !has_output && st.last.is_none() {
        return;
    }

    ui.add_space(8.0);
    egui::CollapsingHeader::new("Details")
        .default_open(false)
        .show(ui, |ui| {
            // The guest's own console, which is where the per-kernel table and
            // any complaint from the harness appear. Secondary on purpose: a
            // wall of monospace as the primary surface reads as "something went
            // wrong" to a reader who did not ask for a log.
            let buf = lines.or_else(|| st.dev.lines());
            ScrollArea::vertical()
                .max_height(320.0)
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    match buf {
                        Some(b) => {
                            let v = b.lock().unwrap();
                            if v.is_empty() {
                                ui.label(RichText::new("(nothing yet)").weak());
                            }
                            for l in v.iter() {
                                ui.label(RichText::new(l).monospace().size(11.0));
                            }
                        }
                        None => { ui.label(RichText::new("(nothing yet)").weak()); }
                    }
                });
        });
}

fn report_text(run: &Run) -> String {
    let mut s = String::new();
    s.push_str(&format!("IRIS benchmark — {}\n", run.host.cpu_model));
    s.push_str(&format!("  emulated CPU      {} ({})\n", run.machine.cpu,
                        iris::bench_report::engine_of(run)));
    s.push_str(&format!("  build features    {}\n",
                        if run.features.is_empty() { "(none)".into() } else { run.features.join(" ") }));
    s.push_str(&format!("  host              {} {} · {} cores\n",
                        run.host.os, run.host.arch, run.host.cores));
    if !run.machine.rev.is_empty() {
        s.push_str(&format!("  emulated PRId     {} rev {}\n", run.machine.prid, run.machine.rev));
    }
    if !run.machine.cache.is_empty() {
        let c = &run.machine.cache;
        s.push_str(&format!("  L1 cache          {} I ({} B) / {} D ({} B)\n",
                            fmt_bytes(c.l1i_bytes), c.l1i_line,
                            fmt_bytes(c.l1d_bytes), c.l1d_line));
        s.push_str(&format!("  L2 cache          {}\n", if !c.l2_present {
            "absent".to_string()
        } else if c.l2_bytes > 0 {
            format!("{}, {} B lines", fmt_bytes(c.l2_bytes), c.l2_line)
        } else {
            format!("present, {} B lines, size not reported", c.l2_line)
        }));
    }
    if !run.machine.memory.is_empty() {
        s.push_str(&format!("  memory            {} MB in {} bank(s)\n",
                            run.machine.memory.total_mb, run.machine.memory.banks.len()));
    }
    s.push('\n');
    s.push_str(&format!("  Emulated Indy     {:.1} DMIPS\n", run.dmips().unwrap_or(0.0)));
    s.push_str(&format!("  Throughput        {:.1} MIPS\n", run.mips()));
    s.push_str(&format!("  Accuracy          {:.1}%  ({}/{})\n",
                        run.accuracy(), run.matched, run.checked));
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

fn save_report(run: &Run) {
    // An explicit save panel, so the only place this ever lands outside the
    // app's own container is one the user picked. Nothing is uploaded.
    let name = format!("iris-benchmark-{}.json", run.cell.replace(['/', ' '], "-"));
    let Some(path) = crate::filedialog::dialog_with(
        "Save benchmark report", &name, crate::filedialog::Anchor::Data,
        &[("JSON", &["json"])]).save_file()
    else {
        return;
    };
    match serde_json::to_string_pretty(run) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                log::warn!("saving the benchmark report to {}: {e}", path.display());
            }
        }
        Err(e) => log::warn!("serialising the benchmark report: {e}"),
    }
}

// ─── developer tools (source builds only) ────────────────────────────────────

/// The subprocess runner, for the things that genuinely need one.
///
/// `matrix` builds a separate emulator per cell, because the CPU model and the
/// JIT are cargo features rather than runtime switches — comparing them means
/// comparing binaries. `host` runs the same kernels compiled natively. Both
/// need a source checkout, a toolchain and a writable tree, so both are hidden
/// from a distributed build.
#[derive(Default)]
struct DevRunner {
    #[cfg(not(feature = "appstore"))]
    inner: dev::State,
}

impl DevRunner {
    fn is_running(&self) -> bool {
        #[cfg(not(feature = "appstore"))] { self.inner.is_running() }
        #[cfg(feature = "appstore")] { false }
    }
    fn stop(&mut self) {
        #[cfg(not(feature = "appstore"))] { self.inner.stop(); }
    }
    fn has_output(&self) -> bool {
        #[cfg(not(feature = "appstore"))] { self.inner.has_output() }
        #[cfg(feature = "appstore")] { false }
    }
    fn lines(&self) -> Option<Arc<Mutex<Vec<String>>>> {
        #[cfg(not(feature = "appstore"))] { self.inner.lines() }
        #[cfg(feature = "appstore")] { None }
    }
}

#[cfg(not(feature = "appstore"))]
fn developer_tools(ui: &mut Ui, st: &mut BenchState) {
    ui.add_space(12.0);
    ui.separator();
    egui::CollapsingHeader::new("Developer tools").default_open(false).show(ui, |ui| {
        ui.label(RichText::new(
            "These shell out to iris-bench and need a source checkout. The matrix builds \
             one emulator per cell — the CPU model and the JIT are compile-time features, \
             so comparing them means comparing binaries.").weak());
        ui.add_space(6.0);

        let busy = st.is_running();
        let have_bin = dev::iris_bench_bin();
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!busy && have_bin.is_some(), |ui| {
                if ui.button("Full matrix")
                    .on_hover_text("R4400 and R5000, interpreter and jitv2. Tens of minutes, \
                                    and it needs cargo on PATH.")
                    .clicked()
                {
                    st.dev.inner.start("matrix", &["matrix"]);
                }
                if ui.button("Measure this host")
                    .on_hover_text("The identical kernels compiled natively, for the ratio \
                                    between emulated and native.")
                    .clicked()
                {
                    st.dev.inner.start("host", &["host"]);
                }
            });
            if have_bin.is_none() {
                ui.label(RichText::new("iris-bench not built — cargo build --release --bin iris-bench")
                    .color(Color32::from_rgb(220, 170, 90)));
            }
            if let Some(dir) = dev::results_dir() {
                if !busy && ui.button("Open results").clicked() { dev::open_folder(&dir); }
            }
        });
        if let Some(o) = st.dev.inner.last_outcome() {
            ui.label(RichText::new(o).weak());
        }
    });
}

#[cfg(not(feature = "appstore"))]
mod dev {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};

    #[derive(Default)]
    pub struct State {
        lines: Arc<Mutex<Vec<String>>>,
        running: Arc<AtomicBool>,
        child: Arc<Mutex<Option<Child>>>,
        outcome: Arc<Mutex<Option<String>>>,
    }

    impl State {
        pub fn is_running(&self) -> bool { self.running.load(Ordering::Relaxed) }
        pub fn has_output(&self) -> bool { !self.lines.lock().unwrap().is_empty() }
        pub fn lines(&self) -> Option<Arc<Mutex<Vec<String>>>> {
            if self.has_output() { Some(self.lines.clone()) } else { None }
        }
        pub fn last_outcome(&self) -> Option<String> { self.outcome.lock().unwrap().clone() }

        pub fn stop(&mut self) {
            if let Some(c) = self.child.lock().unwrap().as_mut() { let _ = c.kill(); }
        }

        pub fn start(&mut self, label: &str, args: &[&str]) {
            if self.is_running() { return; }
            let Some(bin) = iris_bench_bin() else { return };

            self.lines.lock().unwrap().clear();
            *self.outcome.lock().unwrap() = None;
            self.running.store(true, Ordering::Relaxed);

            let lines = Arc::clone(&self.lines);
            let running = Arc::clone(&self.running);
            let child_slot = Arc::clone(&self.child);
            let outcome = Arc::clone(&self.outcome);
            let owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
            // The repo root, so bench/ and target/ resolve the way iris-bench
            // expects — it takes every path relative to there.
            let cwd = bin.parent().and_then(|p| p.parent()).and_then(|p| p.parent())
                .map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
            let label = label.to_string();

            std::thread::spawn(move || {
                let mut cmd = Command::new(&bin);
                cmd.current_dir(&cwd).args(&owned)
                    .stdout(Stdio::piped()).stderr(Stdio::piped());
                let mut child = match cmd.spawn() {
                    Ok(c) => c,
                    Err(e) => {
                        *outcome.lock().unwrap() = Some(format!("failed to start {}: {e}", bin.display()));
                        running.store(false, Ordering::Relaxed);
                        return;
                    }
                };

                let so = child.stdout.take();
                let se = child.stderr.take();
                *child_slot.lock().unwrap() = Some(child);

                // Both pipes on their own threads: a child that fills one while
                // we read the other would stall instead of finishing.
                let pump = |r: Option<Box<dyn std::io::Read + Send>>, sink: Arc<Mutex<Vec<String>>>| {
                    std::thread::spawn(move || {
                        if let Some(r) = r {
                            for line in BufReader::new(r).lines().map_while(Result::ok) {
                                let mut v = sink.lock().unwrap();
                                v.push(line);
                                if v.len() > MAX_LINES { let drop_n = v.len() - MAX_LINES; v.drain(..drop_n); }
                            }
                        }
                    })
                };
                let t1 = pump(so.map(|s| Box::new(s) as Box<dyn std::io::Read + Send>), Arc::clone(&lines));
                let t2 = pump(se.map(|s| Box::new(s) as Box<dyn std::io::Read + Send>), Arc::clone(&lines));

                let status = child_slot.lock().unwrap().as_mut().map(|c| c.wait());
                let _ = t1.join();
                let _ = t2.join();
                *child_slot.lock().unwrap() = None;

                let ok = matches!(status, Some(Ok(s)) if s.success());
                *outcome.lock().unwrap() =
                    Some(format!("{label} {}", if ok { "finished" } else { "failed" }));
                running.store(false, Ordering::Relaxed);
            });
        }
    }

    /// Where the pieces are, relative to wherever the GUI was launched from.
    /// The dev workflow runs it from the repo root; an installed layout puts
    /// the binaries next to the executable.
    fn locate(rel: &str) -> Option<PathBuf> {
        let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(PathBuf::from));
        let name = Path::new(rel).file_name()?.to_owned();
        let mut candidates = vec![PathBuf::from(rel), PathBuf::from("..").join(rel)];
        if let Some(d) = exe_dir {
            candidates.push(d.join(&name));
            candidates.push(d.join(rel));
        }
        candidates.into_iter().find(|p| p.exists())
    }

    pub fn iris_bench_bin() -> Option<PathBuf> {
        let exe = if cfg!(windows) { "iris-bench.exe" } else { "iris-bench" };
        locate(&format!("target/release/{exe}"))
    }

    pub fn results_dir() -> Option<PathBuf> { locate("bench/build/results") }

    pub fn open_folder(dir: &Path) {
        #[cfg(target_os = "windows")]
        let _ = Command::new("explorer").arg(dir).spawn();
        #[cfg(target_os = "macos")]
        let _ = Command::new("open").arg(dir).spawn();
        #[cfg(all(unix, not(target_os = "macos")))]
        let _ = Command::new("xdg-open").arg(dir).spawn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iris::bench_report::{HostInfo, MachineInfo, Row, RunSettings};
    use std::collections::BTreeMap;

    fn row(name: &str, unit: &str, work: u64, ns: u64) -> Row {
        Row {
            name: name.into(), unit: unit.into(), iters: 1, work, ns,
            icount: work, count: 0, exc: 0,
            checksum: "0x0".into(), golden: "0x0".into(), status: "OK".into(),
        }
    }

    fn run() -> Run {
        Run {
            cell: "test".into(),
            features: vec!["tlbvmap".into()],
            machine: MachineInfo {
                cpu: "R4400".into(), timebase: true,
                prid: "0x00000440".into(), rev: "4.0".into(), sysid: "0x00000013".into(),
                cache: iris::bench_report::CacheInfo {
                    l1i_bytes: 16384, l1i_line: 16,
                    l1d_bytes: 16384, l1d_line: 16,
                    l2_present: true, l2_line: 128, l2_bytes: 0,
                },
                memory: iris::bench_report::MemoryInfo {
                    total_mb: 256,
                    banks: vec![
                        iris::bench_report::Bank { index: 0, mb: 128, base: 0x0800_0000 },
                        iris::bench_report::Bank { index: 1, mb: 128, base: 0x1000_0000 },
                    ],
                },
                ..Default::default()
            },
            host: HostInfo { cpu_model: "Test CPU".into(), os: "linux".into(),
                             arch: "x86_64".into(), cores: 8 },
            // Two kernels of the same unit but very different durations, so a
            // test can tell a time-weighted aggregate from a plain average.
            rows: vec![
                row("int/alu", "ops", 100, 1_000_000_000),
                row("int/bitops", "ops", 300, 3_000_000_000),
            ],
            checked: 2, matched: 2,
            total_ns: 4_000_000_000, total_icount: 400,
            wall_s: 30.0,
            suite_id: "blake3:0123456789abcdef".into(),
            settings: RunSettings::default(),
        }
    }

    fn reference(alu: f64, bitops: f64) -> ReferenceEntry {
        ReferenceEntry {
            id: "ref".into(), label: "Reference machine".into(),
            cpu: "R4400".into(), engine: "interp".into(), host: "Other CPU".into(),
            measured: None, guest_mips: 1.0, dmips: 1.0, accuracy: 100.0,
            kernels: BTreeMap::from([
                ("int/alu".to_string(), alu),
                ("int/bitops".to_string(), bitops),
            ]),
        }
    }

    #[test]
    fn a_reference_that_matches_exactly_gives_a_ratio_of_one() {
        // Both kernels ran at 100 units/s here; a reference row saying the same
        // must come out as 1.00x however the two are weighted together.
        let (r, label) = ratio(&run(), &reference(100.0, 100.0), "Integer").unwrap();
        assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {r}");
        assert_eq!(label, "Reference machine");
    }

    #[test]
    fn the_comparison_weights_kernels_by_how_long_they_ran() {
        // int/bitops took 3s of the 4s, so the reference aggregate must sit at
        // 3/4 of the way to its value: (50*1 + 150*3)/4 = 125, and this run's
        // own aggregate is 400 units / 4 s = 100. A plain mean of 50 and 150
        // would be 100 and give exactly 1.0 — which is the bug this catches.
        let (r, _) = ratio(&run(), &reference(50.0, 150.0), "Integer").unwrap();
        assert!((r - 100.0 / 125.0).abs() < 1e-9, "expected 0.8, got {r}");
    }

    #[test]
    fn a_reference_row_with_no_overlapping_kernels_yields_no_comparison() {
        let mut r = reference(100.0, 100.0);
        r.kernels.clear();
        assert!(ratio(&run(), &r, "Integer").is_none());
        // And a category this run has no rows for.
        assert!(ratio(&run(), &reference(100.0, 100.0), "Codec").is_none());
    }

    #[test]
    fn the_time_estimate_waits_for_evidence_before_making_one() {
        let early = LiveState { total: 46, done: 2, current: "int/alu".into() };
        let s = remaining(&early, Duration::from_secs(3), true);
        assert!(s.contains("in total"), "must not extrapolate from two rows: {s}");
        assert!(!s.contains("left"));

        // Halfway through 40s of a 46-kernel run: about 40s more to go.
        let later = LiveState { total: 46, done: 23, current: "mem/copy".into() };
        let s = remaining(&later, Duration::from_secs(40), true);
        assert!(s.contains("40s left"), "expected an estimate near 40s, got {s}");
    }

    #[test]
    fn progress_is_a_fraction_even_before_the_guest_says_how_many() {
        assert_eq!(LiveState::default().fraction(), 0.0);
        assert_eq!(LiveState { total: 4, done: 1, ..Default::default() }.fraction(), 0.25);
        // A guest that somehow reports past its own plan must not overflow the bar.
        assert_eq!(LiveState { total: 4, done: 9, ..Default::default() }.fraction(), 1.0);
    }

    #[test]
    fn sizes_read_the_way_a_person_reads_them() {
        assert_eq!(fmt_bytes(16384), "16 KB");
        assert_eq!(fmt_bytes(1 << 20), "1 MB");
        assert_eq!(fmt_bytes(512), "512 B");
    }

    #[test]
    fn an_l2_of_unknown_size_is_reported_as_present_not_as_absent() {
        // 0 bytes means "the architecture does not say", which is the normal
        // case on everything but a Triton. Rendering that as "absent" would
        // misdescribe every R4400 result there is.
        let text = report_text(&run());
        assert!(text.contains("present, 128 B lines, size not reported"),
                "an unsized L2 must still read as present:\n{text}");
        assert!(!text.contains("L2 cache          absent"));
    }

    #[test]
    fn the_exported_report_carries_what_makes_the_numbers_meaningful() {
        let text = report_text(&run());
        for want in ["Test CPU", "R4400", "interp", "DMIPS", "MIPS", "Accuracy",
                     "Integer", "blake3:0123456789abcdef",
                     // The inventory is what makes two results comparable at
                     // all — the mem/ kernels are a readout of this hierarchy.
                     "16 KB I", "256 MB in 2 bank(s)", "0x00000440 rev 4.0"] {
            assert!(text.contains(want), "the report must mention {want}:\n{text}");
        }
        // A full run says nothing about being shortened.
        assert!(!text.contains("quick run"));

        let mut quick = run();
        quick.settings = RunSettings { groups: 0x3F, time_pct: 30, repeats: 1 };
        assert!(report_text(&quick).contains("quick run"),
                "a shortened run must say so wherever its numbers travel");
    }
}
