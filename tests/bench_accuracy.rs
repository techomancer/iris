//! The benchmark suite's accuracy (`bench/`) under the emulator this build
//! produced: one case per CPU, for the engine the build was compiled with.
//!
//!     cargo test --release --test bench_accuracy -- --ignored
//!     cargo test --release --features jitv2 --test bench_accuracy -- --ignored
//!
//! Every kernel checksums its result against a golden value compiled into the
//! guest image, so a wrong answer anywhere in the CPU model shows up here as a
//! mismatch. Only accuracy is judged, never speed: a shared machine's
//! throughput varies by more than most real regressions.
//!
//! `iris-bench run --iris` runs the guest on this build's `iris` and saves the
//! parsed result; set `IRIS_BENCH_RESULTS` to collect those for
//! `iris-bench report`. The guest is the checked-in
//! `bench/prebuilt/irisbench.elf`, or `IRIS_BENCH_ELF` to judge a rebuilt one.

mod common;

use std::process::Command;

use common::*;

/// Seconds before a hung guest is killed. A full run takes about a minute on
/// the interpreter; the margin is for a loaded machine.
const TIMEOUT_S: u64 = 2400;

/// Kernels that take exceptions on purpose.
const EXPECT_EXC: &[&str] = &["sys/exception", "sys/tlb_miss"];

/// Every way the result could be wrong.
fn judge(run: &serde_json::Value, cpu: &str) -> Vec<Finding> {
    let mut errs = Vec::new();
    // A build whose features did not take runs the wrong CPU and every result
    // is mislabelled. The guest's banner is authoritative because it reads PRId.
    let want = cpu.split('_').next().unwrap().to_uppercase();
    let got = run["machine"]["cpu"].as_str().unwrap_or("");
    if got != want {
        errs.push(Finding::run(format!("guest reports {got}, expected {want}")));
    }
    if !run["machine"]["timebase"].as_bool().unwrap_or(false) {
        errs.push(Finding::run("no host time base: every timing is a guess"));
    }
    // A kernel that faults is stepped over by the exception dispatcher and
    // still reports a throughput, so unexpected exceptions are checked as well
    // as the checksums themselves.
    for r in run["rows"].as_array().map(Vec::as_slice).unwrap_or_default() {
        let name = r["name"].as_str().unwrap_or("?");
        if r["status"] == "MISMATCH" {
            let s = |k: &str| r[k].as_str().unwrap_or("?").to_string();
            errs.push(Finding::test(name, format!("checksum {} != {}", s("checksum"), s("golden"))));
        }
        let exc = r["exc"].as_u64().unwrap_or(0);
        if exc != 0 && !EXPECT_EXC.contains(&name) {
            errs.push(Finding::test(name, format!("took {exc} unexpected exceptions")));
        }
    }
    errs
}

fn guest_elf() -> std::path::PathBuf {
    std::env::var_os("IRIS_BENCH_ELF")
        .map(Into::into)
        .unwrap_or_else(|| repo("bench/prebuilt/irisbench.elf"))
}

fn cell(cpu: &str) {
    require_optimised_build();
    let family = cpu.split('_').next().unwrap();
    let cell = format!("{}-{}", family, engine());
    let dir = run_dir(&format!("bench-{cpu}"));
    // iris-bench runs the emulator from the config's grandparent, `dir`.
    let config = machine_config("bench/run/bare.toml", &dir);
    let results = std::env::var_os("IRIS_BENCH_RESULTS")
        .map(Into::into)
        .unwrap_or_else(|| dir.join("results"));

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_iris-bench"));
    cmd.current_dir(&dir)
        .arg("run")
        .arg("--iris").arg(env!("CARGO_BIN_EXE_iris"))
        .arg("--elf").arg(guest_elf())
        .arg("--config").arg(&config)
        .arg("--label").arg(&cell)
        .arg("--out").arg(&results)
        // iris-bench has its own timeout; this one is the backstop.
        .arg("--timeout").arg(timeout(TIMEOUT_S).as_secs().to_string());
    if let Some(path) = junit_path(&format!("bench-{cell}")) {
        cmd.arg("--junit").arg(path);
    }
    cmd.args(["--", "--cpu", cpu]);
    let limit = timeout(TIMEOUT_S) + std::time::Duration::from_secs(60);
    let out = run(cmd, limit, &dir.join("bench.log")).unwrap_or_else(|e| panic!("{cell}: {e}"));
    if !out.status.success() {
        panic!(
            "{cell}: iris-bench failed ({}{}): {}\n--- last 20 lines ---\n{}",
            out.status,
            out.crash().map(|c| format!(", {c}")).unwrap_or_default(),
            out.stderr.trim(),
            tail(&out.stdout, 20)
        );
    }

    // The file iris-bench says it wrote, rather than a guess at its name.
    // A relative path is relative to iris-bench's working directory, `dir`.
    let wrote = out
        .stdout
        .lines()
        .rev()
        .find_map(|l| l.trim_start().strip_prefix("wrote "))
        .unwrap_or_else(|| panic!("{cell}: iris-bench saved no result\n{}", tail(&out.stdout, 20)));
    let path = dir.join(wrote);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{cell}: no result at {}: {e}", path.display()));
    let run: serde_json::Value = serde_json::from_str(&text).expect("parse the saved result");
    let errs = judge(&run, cpu);
    let (matched, checked) = (&run["matched"], &run["checked"]);
    if !errs.is_empty() {
        panic!(
            "{}\n{matched}/{checked} matched\n{}",
            headline(&cell, &errs),
            errs.iter().map(|e| format!("check: {e}")).collect::<Vec<_>>().join("\n")
        );
    }
    println!("check: {cell}: {matched}/{checked} matched");
}

#[test]
#[ignore = "bare-metal suite: needs an optimised build; cargo test --release --test bench_accuracy -- --ignored"]
fn r4400() {
    cell("r4400");
}

#[test]
#[ignore = "bare-metal suite: needs an optimised build; cargo test --release --test bench_accuracy -- --ignored"]
fn r5000() {
    cell("r5000");
}

/// The R10000 always runs the shadow caches, so the default build covers it.
#[test]
#[ignore = "bare-metal suite: needs an optimised build; cargo test --release --test bench_accuracy -- --ignored"]
fn r10000_195() {
    cell("r10000_195");
}
