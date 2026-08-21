//! Benchmark tab — run `iris-bench` from the GUI and watch it.
//!
//! The suite is a developer tool and the command line is its natural home; this
//! exists so that "how fast is this build, and is it still right" is one click
//! away rather than a remembered incantation, and so the answer is legible
//! while it is still running. A full matrix takes tens of minutes and rebuilds
//! the emulator once per cell — a progress-free spinner would be useless, so
//! the child's output is streamed line by line into the panel.
//!
//! Nothing here talks to a running machine. `iris-bench` spawns its own
//! headless emulator with its own bare-metal config, so this is safe to use
//! while a normal IRIX session is up.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui::{self, Color32, RichText, ScrollArea, Ui};

/// How many output lines to keep. A matrix run prints a cargo build per cell;
/// the interesting part is always the tail.
const MAX_LINES: usize = 4000;

#[derive(Default)]
pub struct BenchState {
    lines: Arc<Mutex<Vec<String>>>,
    running: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
    /// What finished last, and how — kept after the run so the panel still says
    /// something once the thread is gone.
    last: Option<Outcome>,
    what: String,
}

struct Outcome {
    label: String,
    ok: bool,
    detail: String,
}

/// Where the pieces are, relative to wherever the GUI was launched from. The
/// dev workflow runs it from the repo root; an installed layout puts the
/// binaries next to the executable.
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

fn iris_bench_bin() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "iris-bench.exe" } else { "iris-bench" };
    locate(&format!("target/release/{}", exe))
}

fn suite_elf() -> Option<PathBuf> { locate("bench/build/irisbench.elf") }
fn host_bin() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "irisbench-host.exe" } else { "irisbench-host" };
    locate(&format!("bench/build/{}", exe))
}
fn results_dir() -> Option<PathBuf> { locate("bench/build/results") }

impl BenchState {
    pub fn is_running(&self) -> bool { self.running.load(Ordering::Relaxed) }

    fn start(&mut self, label: &str, args: &[&str]) {
        if self.is_running() { return; }
        let Some(bin) = iris_bench_bin() else {
            self.last = Some(Outcome {
                label: label.to_string(),
                ok: false,
                detail: "iris-bench not built — run `cargo build --release --bin iris-bench`"
                    .to_string(),
            });
            return;
        };

        self.lines.lock().unwrap().clear();
        self.what = label.to_string();
        self.last = None;
        self.running.store(true, Ordering::Relaxed);

        let lines = Arc::clone(&self.lines);
        let running = Arc::clone(&self.running);
        let child_slot = Arc::clone(&self.child);
        let owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        // The repo root, so bench/ and target/ resolve the way iris-bench
        // expects — it takes every path relative to there.
        let cwd = bin.parent().and_then(|p| p.parent()).and_then(|p| p.parent())
            .map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
        let label = label.to_string();

        std::thread::spawn(move || {
            let mut cmd = Command::new(&bin);
            cmd.current_dir(&cwd)
                .args(&owned)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => {
                    lines.lock().unwrap().push(format!("failed to start {}: {}", bin.display(), e));
                    running.store(false, Ordering::Relaxed);
                    return;
                }
            };

            let so = child.stdout.take();
            let se = child.stderr.take();
            *child_slot.lock().unwrap() = Some(child);

            // Both pipes on their own threads: a child that fills one while we
            // read the other would stall instead of finishing.
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
            lines.lock().unwrap().push(if ok {
                format!("--- {} finished ---", label)
            } else {
                format!("--- {} failed ---", label)
            });
            running.store(false, Ordering::Relaxed);
        });
    }

    fn stop(&mut self) {
        if let Some(c) = self.child.lock().unwrap().as_mut() { let _ = c.kill(); }
    }

    /// Pull the headline numbers back out of the streamed output. iris-bench
    /// prints one summary line per cell in a fixed shape, so this is a read of
    /// what already scrolled past rather than a second source of truth.
    fn summarize(&self) -> Vec<String> {
        self.lines
            .lock()
            .unwrap()
            .iter()
            .filter(|l| l.contains("accuracy") || l.starts_with("report:") || l.starts_with("matrix:"))
            .cloned()
            .collect()
    }
}

pub fn show(ui: &mut Ui, st: &mut BenchState) {
    ui.heading("Benchmark");
    ui.label(
        "Runs bench/ — a bare-metal MIPS suite that measures this build of IRIS and \
         checks that it is still computing the right answers. No IRIX and no disk \
         image needed; it starts its own headless emulator, so it is safe to use \
         while a machine is running.",
    );
    ui.add_space(6.0);

    // ── prerequisites ───────────────────────────────────────────────────────
    let bench_bin = iris_bench_bin();
    let elf = suite_elf();
    let hostb = host_bin();

    egui::Grid::new("bench_paths").num_columns(2).striped(true).show(ui, |ui| {
        let row = |ui: &mut Ui, name: &str, p: &Option<PathBuf>, hint: &str| {
            ui.label(name);
            match p {
                Some(p) => { ui.label(RichText::new(p.display().to_string()).monospace()); }
                None => { ui.label(RichText::new(hint).color(Color32::from_rgb(220, 170, 90))); }
            }
            ui.end_row();
        };
        row(ui, "iris-bench", &bench_bin, "not built — cargo build --release --bin iris-bench");
        row(ui, "suite binary", &elf, "not built — make -C bench  (needs a MIPS cross toolchain)");
        row(ui, "host baseline", &hostb, "not built — make -C bench hostbench");
    });

    ui.add_space(8.0);
    let busy = st.is_running();

    ui.horizontal(|ui| {
        ui.add_enabled_ui(!busy && bench_bin.is_some() && elf.is_some(), |ui| {
            if ui
                .button("Run once")
                .on_hover_text(
                    "Run the suite against target/release/iris as it is built right now. \
                     A couple of minutes on the interpreter.",
                )
                .clicked()
            {
                st.start("run", &["run", "--label", "gui"]);
            }
        });

        ui.add_enabled_ui(!busy && bench_bin.is_some() && hostb.is_some(), |ui| {
            if ui
                .button("Measure this host")
                .on_hover_text(
                    "Run the identical kernels natively, for the ratio between \
                     emulated and native. About ten seconds.",
                )
                .clicked()
            {
                st.start("host", &["host"]);
            }
        });

        ui.add_enabled_ui(!busy && bench_bin.is_some() && elf.is_some(), |ui| {
            if ui
                .button("Full matrix")
                .on_hover_text(
                    "R4400 and R5000, interpreter and jitv2. Builds a separate emulator \
                     for each — the CPU model and the JIT are compile-time cargo features. \
                     Tens of minutes, and it needs cargo on PATH.",
                )
                .clicked()
            {
                st.start("matrix", &["matrix"]);
            }
        });

        if busy && ui.button("Stop").clicked() { st.stop(); }

        if let Some(dir) = results_dir() {
            if !busy && ui.button("Open results").clicked() { open_folder(&dir); }
        }
    });

    if busy {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(format!("{} running…", st.what));
        });
        // A background thread is writing lines; without this the panel only
        // updates when the pointer moves over it.
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
    }

    let summary = st.summarize();
    if !summary.is_empty() {
        ui.add_space(6.0);
        ui.separator();
        for line in &summary {
            ui.label(RichText::new(line).monospace().strong());
        }
    }
    if let Some(out) = &st.last {
        ui.label(
            RichText::new(format!("{}: {}", out.label, out.detail))
                .color(if out.ok { Color32::LIGHT_GREEN } else { Color32::from_rgb(220, 170, 90) }),
        );
    }

    ui.add_space(6.0);
    ui.separator();
    ui.label("Output");
    ScrollArea::vertical()
        .max_height(320.0)
        .stick_to_bottom(true)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let lines = st.lines.lock().unwrap();
            if lines.is_empty() {
                ui.label(RichText::new("(nothing yet)").weak());
            }
            for l in lines.iter() {
                ui.label(RichText::new(l).monospace().size(11.0));
            }
        });
}

fn open_folder(dir: &Path) {
    #[cfg(target_os = "windows")]
    let _ = Command::new("explorer").arg(dir).spawn();
    #[cfg(target_os = "macos")]
    let _ = Command::new("open").arg(dir).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = Command::new("xdg-open").arg(dir).spawn();
}
