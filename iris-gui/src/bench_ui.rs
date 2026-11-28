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
use std::time::Duration;

use crate::i18n::{subtitle, ResponseExt};
use eframe::egui::{self, Color32, RichText, ScrollArea, Ui};
use iris_gui_core::controller::bench::{fmt_rate, BenchAction, BenchController, ReferenceEntry, Run, CATEGORIES};


/// How many console lines the developer runner keeps.
const MAX_LINES: usize = 4000;

pub struct BenchState {
    ctl: BenchController,
    dev: DevRunner,
    /// Effects for the app's effect host (the report's save panel).
    pub effects: Vec<iris_gui_core::controller::Effect>,
    machine_running: bool,
}

impl BenchState {
    pub fn new(notify: std::sync::Arc<dyn iris_gui_core::controller::Notify>) -> Self {
        Self { ctl: BenchController::new(notify), dev: DevRunner::default(), effects: Vec::new(), machine_running: false }
    }

    pub fn is_running(&self) -> bool {
        self.ctl.is_running() || self.dev.is_running()
    }

    fn act(&mut self, a: BenchAction) {
        let fx = self.ctl.apply(a, self.machine_running);
        self.effects.extend(fx);
    }

    /// An answer to a question the benchmark asked (the save panel).
    pub fn answer(&mut self, a: iris_gui_core::controller::Answer) {
        self.act(BenchAction::Answer(a));
    }

    fn stop(&mut self) {
        self.act(BenchAction::Stop);
        self.dev.stop();
    }
}

fn report_text(run: &Run) -> String {
    BenchController::report_text(run)
}

// ─── the screen ──────────────────────────────────────────────────────────────

pub fn show(ui: &mut Ui, st: &mut BenchState, machine_running: bool) {
    st.machine_running = machine_running;
    st.ctl.tick();
    if st.is_running() {
        // The run reports through shared state; follow it while it moves.
        ui.ctx().request_repaint_after(Duration::from_millis(200));
    }

    ui.heading(t!("bench-heading"));
    subtitle(ui, t!("bench-subtitle"));
    ui.add_space(8.0);

    controls(ui, st, machine_running);

    if let Some(p) = st.ctl.progress() {
        ui.add_space(8.0);
        ui.add(egui::ProgressBar::new(p.fraction).show_percentage().animate(true));
        ui.horizontal(|ui| {
            ui.label(&p.line);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(&p.time).weak());
            });
        });
    }

    let reference = st.ctl.reference();
    let mut save = false;
    match st.ctl.last() {
        Some(Ok(run)) => {
            ui.add_space(10.0);
            ui.separator();
            ui.add_space(6.0);
            save = results_view(ui, run, reference.as_ref());
        }
        Some(Err(e)) => {
            ui.add_space(8.0);
            if e == "stopped" {
                ui.label(RichText::new(t!("bench-stopped")).weak());
            } else {
                ui.label(RichText::new(t!("bench-failed", error = e.as_str()))
                    .color(Color32::from_rgb(220, 170, 90)));
            }
        }
        None => {}
    }
    if save {
        st.act(BenchAction::SaveReport);
    }

    details(ui, st);

    developer_tools(ui, st);
}

fn controls(ui: &mut Ui, st: &mut BenchState, machine_running: bool) {
    let busy = st.is_running();
    let mut actions = Vec::new();
    ui.horizontal(|ui| {
        // Two machines at once would measure the wrong thing.
        let start = st.ctl.can_start(machine_running);
        ui.add_enabled_ui(start.is_yes() && !st.dev.is_running(), |ui| {
            let secs = if st.ctl.quick() { iris_gui_core::controller::bench::QUICK_SECS } else { iris_gui_core::controller::bench::FULL_SECS };
            if ui.button(RichText::new(primary_label()).strong()).tip(t!("bench-run-tip", secs = secs)).clicked() {
                actions.push(BenchAction::Start);
            }
        });
        ui.add_enabled_ui(!busy, |ui| {
            // The CPU is a machine setting, not a build, so the suite measures
            // either from this binary.
            let (labels, mut chosen) = st.ctl.cpus();
            egui::ComboBox::from_id_salt("bench_cpu")
                .selected_text(st.ctl.cpu_label())
                .show_ui(ui, |ui| {
                    for (i, l) in labels.iter().enumerate() {
                        if ui.selectable_value(&mut chosen, i, l).clicked() {
                            actions.push(BenchAction::SetCpu(i));
                        }
                    }
                })
                .response
                .tip(t!("bench-cpu-tip"));
        });
        ui.add_enabled_ui(!busy, |ui| {
            let mut quick = st.ctl.quick();
            if ui.checkbox(&mut quick, t!("bench-quick")).tip(t!("bench-quick-tip")).changed() {
                actions.push(BenchAction::SetQuick(quick));
            }
        });
        if busy && ui.button(t!("bench-stop")).clicked() {
            actions.push(BenchAction::Stop);
        }
    });
    for a in actions {
        if a == BenchAction::Stop { st.stop(); } else { st.act(a); }
    }
    if machine_running && !busy {
        ui.add_space(4.0);
        ui.label(RichText::new(t!("bench-machine-running")).weak());
    }
}

fn primary_label() -> String {
    // Name the machine the way its owner does. This measures the emulator, but
    // what varies between users is the hardware under it.
    if cfg!(target_os = "macos") { t!("bench-run-mac") } else { t!("bench-run-pc") }
}

// ─── results ─────────────────────────────────────────────────────────────────

/// Returns whether Save report was clicked.
fn results_view(ui: &mut Ui, run: &Run, reference: Option<&ReferenceEntry>) -> bool {
    let mut save = false;
    ui.heading(t!("bench-results"));
    ui.add_space(4.0);

    // The three figures that stand on their own. DMIPS has forty years of
    // published numbers behind it, guest MIPS is meaningful without a baseline,
    // and accuracy needs no comparison at all — which is why an empty reference
    // table costs the reader so little.
    egui::Grid::new("bench_headline").num_columns(3).spacing([18.0, 6.0]).show(ui, |ui| {
        headline(ui, &t!("bench-dmips"),
                 &t!("bench-dmips-value", value = format!("{:.0}", run.dmips().unwrap_or(0.0))),
                 t!("bench-dmips-tip"));
        ui.end_row();
        headline(ui, &t!("bench-mips"), &t!("bench-mips-value", value = format!("{:.0}", run.mips())),
                 t!("bench-mips-tip"));
        ui.end_row();
        headline(ui, &t!("bench-accuracy"),
                 &t!("bench-accuracy-value", percent = format!("{:.0}", run.accuracy()),
                     matched = run.matched, checked = run.checked),
                 t!("bench-accuracy-tip"));
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
                        ui.label(RichText::new(t!("bench-ratio", ratio = format!("{ratio:.2}"), reference = label)).weak());
                    }
                    None => { ui.label(""); }
                }
                ui.end_row();
            }
        });

    ui.add_space(8.0);
    if reference.is_none() {
        ui.label(
            RichText::new(t!("bench-no-reference")).weak(),
        );
    }
    caveats(ui, run);

    ui.add_space(8.0);
    machine_view(ui, run);

    ui.add_space(8.0);
    ui.horizontal(|ui| {
        if ui.button(t!("common-copy")).clicked() {
            ui.ctx().copy_text(report_text(run));
        }
        if ui.button(t!("bench-save-report")).clicked() {
            save = true;
        }
    });
    save
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
    egui::CollapsingHeader::new(t!("bench-machine")).default_open(false).show(ui, |ui| {
        egui::Grid::new("bench_machine").num_columns(2).spacing([18.0, 4.0]).striped(true)
            .show(ui, |ui| {
                let mut row = |k: &str, v: String| {
                    ui.label(k);
                    ui.label(RichText::new(v).monospace());
                    ui.end_row();
                };

                row(&t!("bench-machine-cpu"), if m.rev.is_empty() {
                    t!("bench-machine-cpu-value", cpu = &m.cpu, prid = &m.prid)
                } else {
                    t!("bench-machine-cpu-rev-value", cpu = &m.cpu, rev = &m.rev, prid = &m.prid)
                });

                if !m.cache.is_empty() {
                    let c = &m.cache;
                    row(&t!("bench-machine-l1"), t!("bench-machine-l1-value",
                        icache = fmt_bytes(c.l1i_bytes), iline = c.l1i_line,
                        dcache = fmt_bytes(c.l1d_bytes), dline = c.l1d_line));
                    row(&t!("bench-machine-l2"), if !c.l2_present {
                        t!("bench-machine-l2-absent")
                    } else if c.l2_bytes > 0 {
                        t!("bench-machine-l2-value", size = fmt_bytes(c.l2_bytes), line = c.l2_line)
                    } else {
                        // The architecture does not expose the size on anything
                        // but a Triton; the PROM reads it from the EEPROM.
                        t!("bench-machine-l2-unsized", line = c.l2_line)
                    });
                }

                if !m.memory.is_empty() {
                    let banks: Vec<String> = m.memory.banks.iter()
                        .map(|b| t!("bench-machine-bank", index = b.index, size = b.mb,
                                    base = format!("{:#010x}", b.base)))
                        .collect();
                    row(&t!("bench-machine-memory"), t!("bench-machine-memory-value",
                        total = m.memory.total_mb, banks = banks.join("  ")));
                }

                if !m.sysid.is_empty() {
                    row(&t!("bench-machine-board"), t!("bench-machine-board-value", sysid = &m.sysid));
                }
                row(&t!("bench-machine-emulator"), if run.features.is_empty() {
                    t!("bench-machine-no-features")
                } else {
                    run.features.join(" ")
                });
                row(&t!("bench-machine-host"), t!("bench-machine-host-value",
                    cpu = m_host(run), os = &run.host.os, arch = &run.host.arch,
                    cores = run.host.cores));
            });
    });
}

fn m_host(run: &Run) -> String {
    if run.host.cpu_model.is_empty() { t!("bench-machine-unknown-cpu") } else { run.host.cpu_model.clone() }
}

/// KB under a megabyte, MB above — the way a person reads a cache size.
fn fmt_bytes(n: u64) -> String {
    if n >= 1 << 20 { format!("{} MB", n >> 20) }
    else if n >= 1 << 10 { format!("{} KB", n >> 10) }
    else { format!("{} B", n) }
}

fn headline(ui: &mut Ui, label: &str, value: &str, help: String) {
    ui.label(label);
    ui.label(RichText::new(value).size(18.0).strong());
    ui.label(RichText::new("?").weak()).tip(help);
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
        ui.label(RichText::new(t!("bench-caveat-quick", reason = why))
            .weak());
    }
    if !run.features.iter().any(|f| f == "jitv2") {
        ui.label(RichText::new(t!("bench-caveat-interpreter")).weak());
    }
    let unexpected: Vec<&str> = run.rows.iter()
        .filter(|r| r.exc > 0 && !iris::bench_report::EXPECT_EXC.contains(&r.name.as_str()))
        .map(|r| r.name.as_str())
        .collect();
    if !unexpected.is_empty() {
        // A kernel that faults is stepped over by the harness and still reports
        // a throughput — for doing something other than what it claims.
        ui.label(RichText::new(t!("bench-caveat-exceptions", kernels = unexpected.join(", ")))
            .color(Color32::from_rgb(220, 170, 90)));
    }
}

// ─── details, export ─────────────────────────────────────────────────────────

fn details(ui: &mut Ui, st: &mut BenchState) {
    let lines = st.ctl.lines();
    let dev = st.dev.lines();
    if lines.is_empty() && dev.is_none() && st.ctl.last().is_none() {
        return;
    }
    ui.add_space(8.0);
    egui::CollapsingHeader::new(t!("bench-details"))
        .default_open(false)
        .show(ui, |ui| {
            // The guest's own console, where the per-kernel table and any
            // complaint from the harness appear.
            let buf: Vec<String> = if !lines.is_empty() {
                lines
            } else {
                dev.map(|b| b.lock().unwrap().clone()).unwrap_or_default()
            };
            ScrollArea::vertical()
                .max_height(320.0)
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if buf.is_empty() {
                        ui.label(RichText::new(t!("bench-details-empty")).weak());
                    }
                    for l in &buf {
                        ui.label(RichText::new(l).monospace().size(11.0));
                    }
                });
        });
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
    inner: dev::State,
}

impl DevRunner {
    fn is_running(&self) -> bool {
        self.inner.is_running()
    }
    fn stop(&mut self) {
        self.inner.stop();
    }
    fn has_output(&self) -> bool {
        self.inner.has_output()
    }
    fn lines(&self) -> Option<Arc<Mutex<Vec<String>>>> {
        self.inner.lines()
    }
}

fn developer_tools(ui: &mut Ui, st: &mut BenchState) {
    ui.add_space(12.0);
    ui.separator();
    egui::CollapsingHeader::new(t!("bench-dev")).default_open(false).show(ui, |ui| {
        subtitle(ui, t!("bench-dev-subtitle"));
        ui.add_space(6.0);

        let busy = st.is_running();
        let have_bin = dev::iris_bench_bin();
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!busy && have_bin.is_some(), |ui| {
                if ui.button(t!("bench-dev-matrix"))
                    .tip(t!("bench-dev-matrix-tip"))
                    .clicked()
                {
                    st.dev.inner.start("matrix", &["matrix"]);
                }
                if ui.button(t!("bench-dev-host"))
                    .tip(t!("bench-dev-host-tip"))
                    .clicked()
                {
                    st.dev.inner.start("host", &["host"]);
                }
            });
            if have_bin.is_none() {
                ui.label(RichText::new(t!("bench-dev-not-built"))
                    .color(Color32::from_rgb(220, 170, 90)));
            }
            if let Some(dir) = dev::results_dir() {
                if !busy && ui.button(t!("bench-dev-open-results")).clicked() { dev::open_folder(&dir); }
            }
        });
        if let Some(o) = st.dev.inner.last_outcome() {
            ui.label(RichText::new(o).weak());
        }
    });
}

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
                        *outcome.lock().unwrap() = Some(t!("bench-dev-start-failed", path = bin.display(), error = e.to_string()));
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
                *outcome.lock().unwrap() = Some(match (label.as_str(), ok) {
                    ("matrix", true) => t!("bench-dev-matrix-finished"),
                    ("matrix", false) => t!("bench-dev-matrix-failed"),
                    (_, true) => t!("bench-dev-host-finished"),
                    (_, false) => t!("bench-dev-host-failed"),
                });
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
        let mut candidates = vec![crate::launch_dir().join(rel), crate::launch_dir().join("..").join(rel)];
        if let Some(d) = exe_dir {
            candidates.push(d.join(&name));
            candidates.push(d.join(rel));
            if let Some(root) = d.parent().and_then(Path::parent) { candidates.push(root.join(rel)); }
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
