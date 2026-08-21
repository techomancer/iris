//! The benchmark suite's report: its data model, its parser, and the reference
//! table a result is compared against.
//!
//! The suite (`bench/`) is a bare-metal MIPS binary that prints a human table
//! and then a machine-readable block between `IRIS-BENCH-BEGIN` and
//! `IRIS-BENCH-END`. Everything here is about turning that block into numbers.
//!
//! This lives in the library rather than in `iris-bench` because three callers
//! need it and only one of them is that binary: the CLI, the in-process runner
//! (`crate::bench_runner`), and the GUI reading a saved result or the reference
//! table. One parser, one schema, one definition of what "accuracy" means.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const BEGIN: &str = "IRIS-BENCH-BEGIN";
pub const END: &str = "IRIS-BENCH-END";


/// Kernels whose whole point is to take exceptions. Everywhere else a nonzero
/// count is a defect — see BF_TAKES_EXC in bench/harness/benchlib.h.
pub const EXPECT_EXC: &[&str] = &["sys/exception", "sys/tlb_miss"];

/// Dhrystones per second per DMIPS, by the VAX 11/780 convention every
/// published Dhrystone figure since 1988 uses.
pub const DHRY_PER_DMIPS: f64 = 1757.0;

// ─── data model ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub name: String,
    pub unit: String,
    pub iters: u64,
    pub work: u64,
    pub ns: u64,
    pub icount: u64,
    pub count: u64,
    /// Exceptions taken during the timed run. Nonzero for a kernel that is not
    /// meant to take any means it measured something other than what it
    /// claims — the harness flags those on the line, and the report repeats it.
    pub exc: u64,
    pub checksum: String,
    pub golden: String,
    pub status: String,
}

impl Row {
    /// Work units per second. The unit is the kernel's own, so this is only
    /// comparable across cells for the same kernel — which is exactly how the
    /// report uses it.
    pub fn rate(&self) -> f64 {
        if self.ns == 0 { 0.0 } else { self.work as f64 * 1e9 / self.ns as f64 }
    }
    /// Guest instructions retired per host second. Zero when there is no
    /// instruction counter (host runs, or an emulator without the test
    /// device's timebase registers).
    pub fn mips(&self) -> f64 {
        if self.ns == 0 || self.icount == 0 { 0.0 } else { self.icount as f64 * 1e3 / self.ns as f64 }
    }
}

/// The cache hierarchy, read out of CP0 Config by the guest.
///
/// Worth recording on every result rather than inferring from the CPU name:
/// the `mem/` kernels are a direct readout of this hierarchy, so two results
/// with different L1 sizes are not measuring the same thing however similar
/// their rates look.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheInfo {
    pub l1i_bytes: u64,
    pub l1i_line: u64,
    pub l1d_bytes: u64,
    pub l1d_line: u64,
    pub l2_present: bool,
    pub l2_line: u64,
    /// 0 when the architecture does not report it — which is every CPU but a
    /// Triton R5000. The PROM knows the real figure from the EEPROM; CP0 does
    /// not expose it, so this is left unknown rather than guessed.
    pub l2_bytes: u64,
}

impl CacheInfo {
    /// True when the guest reported nothing — a host run, or a result recorded
    /// before the inventory existed.
    pub fn is_empty(&self) -> bool {
        self.l1i_bytes == 0 && self.l1d_bytes == 0
    }
}

/// One RAM bank, as the memory controller has it programmed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bank {
    pub index: u8,
    pub mb: u64,
    pub base: u64,
}

/// Installed memory, from the MC's MEMCFG registers — valid whether or not the
/// PROM ran, since `--load-elf` programs them exactly as POST would.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_mb: u64,
    pub banks: Vec<Bank>,
}

impl MemoryInfo {
    pub fn is_empty(&self) -> bool {
        self.total_mb == 0
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MachineInfo {
    pub cpu: String,
    pub prid: String,
    pub fir: String,
    pub config: String,
    pub l2: bool,
    pub testdev: bool,
    pub timebase: bool,
    pub count_hz: u64,
    pub count_hz_measured: bool,
    pub work_bytes: u64,
    /// CPU revision as `major.minor`, from PRId. Empty on an older result.
    #[serde(default)]
    pub rev: String,
    /// Board/system revision, from the MC's SYSID register.
    #[serde(default)]
    pub sysid: String,
    /// `default` on all three so results recorded before the guest reported an
    /// inventory still load — as an empty one, which is what they had.
    #[serde(default)]
    pub cache: CacheInfo,
    #[serde(default)]
    pub memory: MemoryInfo,
}

/// The run configuration the guest actually used, from the `#run` line.
///
/// A shortened run is still *accurate* — every kernel that ran verified against
/// its golden checksum — but its rates are noisier, so a result has to carry
/// this rather than let a reader assume a full one. `Default` is what the suite
/// does when nobody asks for anything, which is also what a result recorded
/// before the `#run` line existed must be read as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunSettings {
    /// The suite's own `BG_*` group mask. `BG_ALL` is every group.
    pub groups: u32,
    /// Per-kernel target time, as a percentage of the suite's default.
    pub time_pct: u32,
    /// Timed passes per kernel.
    pub repeats: u32,
}

/// `BG_ALL` from `bench/harness/benchlib.h`: integer, fpu, memory, imaging,
/// codec, sys.
pub const BG_ALL: u32 = 0x3F;

impl Default for RunSettings {
    fn default() -> Self {
        Self { groups: BG_ALL, time_pct: 100, repeats: 2 }
    }
}

impl RunSettings {
    /// Every group, full-length timed runs, best-of-two. Only a full run's
    /// numbers belong in the reference table.
    pub fn is_full(&self) -> bool {
        *self == Self::default()
    }
    /// Why this run is not comparable with a full one, for a human.
    pub fn shortened_because(&self) -> Option<String> {
        if self.is_full() { return None; }
        let mut why = Vec::new();
        if self.groups != BG_ALL {
            why.push(format!("only groups {:#04x} ran", self.groups));
        }
        if self.time_pct != 100 {
            why.push(format!("timed runs were {}% of full length", self.time_pct));
        }
        if self.repeats != 2 {
            why.push(format!("{} timed pass(es) per kernel instead of 2", self.repeats));
        }
        Some(why.join("; "))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HostInfo {
    pub os: String,
    pub arch: String,
    pub cpu_model: String,
    pub cores: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    /// Cell label: "r4400-interp", "r5000-jitv2", "host", …
    pub cell: String,
    /// Cargo features the emulator was built with, from its own banner.
    pub features: Vec<String>,
    pub machine: MachineInfo,
    pub host: HostInfo,
    pub rows: Vec<Row>,
    pub checked: usize,
    pub matched: usize,
    pub total_ns: u64,
    pub total_icount: u64,
    /// Wall clock for the whole process, including emulator startup — always
    /// larger than total_ns, which counts only timed regions.
    pub wall_s: f64,
    /// Hash of the guest binary this result came from. Reference numbers only
    /// mean anything against the exact suite that produced them: add or change
    /// a kernel and every stored figure silently becomes a comparison between
    /// two different workloads. Empty for a host run (no guest binary).
    /// `default` so results recorded before this field existed still load.
    #[serde(default)]
    pub suite_id: String,
    /// How the suite was configured for this run. `default` so results
    /// recorded before the `#run` line existed still load, as full runs —
    /// which is what they were.
    #[serde(default)]
    pub settings: RunSettings,
}

impl Run {
    pub fn accuracy(&self) -> f64 {
        if self.checked == 0 { 0.0 } else { self.matched as f64 * 100.0 / self.checked as f64 }
    }
    pub fn mips(&self) -> f64 {
        if self.total_ns == 0 || self.total_icount == 0 { 0.0 }
        else { self.total_icount as f64 * 1e3 / self.total_ns as f64 }
    }
    pub fn row(&self, name: &str) -> Option<&Row> {
        self.rows.iter().find(|r| r.name == name)
    }
    /// Dhrystone 2.1 DMIPS.
    pub fn dmips(&self) -> Option<f64> {
        self.row("int/dhrystone").map(|r| r.rate() / DHRY_PER_DMIPS)
    }
    /// Whetstone passes per second. Deliberately not converted to MWIPS: that
    /// needs a "Whetstone instructions per loop" constant taken from a
    /// reference implementation, and a figure resting on an unverified factor
    /// of a thousand would look authoritative without being so. Passes per
    /// second is exact and is what cell-to-cell comparison uses.
    pub fn whet_loops(&self) -> Option<f64> {
        self.row("fpu/whetstone").map(|r| r.rate())
    }
    /// LINPACK 100x100 MFLOPS.
    pub fn linpack_mflops(&self) -> Option<f64> {
        self.row("fpu/linpack").map(|r| r.rate() / 1e6)
    }
}

// ─── parsing the suite's machine block ───────────────────────────────────────

/// What one parse of the suite's machine block yielded.
#[derive(Debug, Clone)]
pub struct Parsed {
    pub machine: MachineInfo,
    pub rows: Vec<Row>,
    pub checked: usize,
    pub matched: usize,
    pub total_ns: u64,
    pub total_icount: u64,
    pub settings: RunSettings,
}

pub fn parse_block(text: &str) -> Result<Parsed, String> {
    let begin = text.find(BEGIN).ok_or_else(|| {
        "no IRIS-BENCH-BEGIN in the output — the suite did not reach its report".to_string()
    })?;
    let end = text[begin..].find(END).ok_or_else(|| {
        "output ends before IRIS-BENCH-END — the suite died partway through".to_string()
    })? + begin;

    let mut machine = MachineInfo::default();
    let mut rows = Vec::new();
    let mut settings = RunSettings::default();
    let (mut checked, mut matched, mut total_ns, mut total_ic) = (0usize, 0usize, 0u64, 0u64);

    for line in text[begin..end].lines().skip(1) {
        let line = line.trim();
        if line.is_empty() { continue; }
        if let Some(rest) = line.strip_prefix('#') {
            let mut it = rest.split_whitespace();
            let kind = it.next().unwrap_or("");
            let kv: BTreeMap<&str, &str> = it
                .filter_map(|tok| tok.split_once('='))
                .collect();
            let num = |k: &str| -> u64 {
                kv.get(k).and_then(|v| parse_u64(v)).unwrap_or(0)
            };
            match kind {
                "machine" => {
                    machine.cpu = kv.get("cpu").unwrap_or(&"unknown").to_string();
                    machine.prid = kv.get("prid").unwrap_or(&"").to_string();
                    machine.fir = kv.get("fir").unwrap_or(&"").to_string();
                    machine.config = kv.get("config").unwrap_or(&"").to_string();
                    machine.l2 = num("l2") != 0;
                    machine.testdev = num("testdev") != 0;
                    machine.timebase = num("timebase") != 0;
                    machine.rev = kv.get("rev").unwrap_or(&"").to_string();
                    machine.sysid = kv.get("sysid").unwrap_or(&"").to_string();
                }
                "timebase" => {
                    machine.count_hz = num("count_hz");
                    machine.count_hz_measured = num("measured") != 0;
                }
                "work" => machine.work_bytes = num("bytes"),
                "cache" => {
                    machine.cache = CacheInfo {
                        l1i_bytes: num("l1i"),
                        l1i_line: num("l1i_line"),
                        l1d_bytes: num("l1d"),
                        l1d_line: num("l1d_line"),
                        l2_present: num("l2") != 0,
                        l2_line: num("l2_line"),
                        l2_bytes: num("l2_bytes"),
                    };
                }
                "memory" => {
                    machine.memory.total_mb = num("total_mb");
                    machine.memory.banks = (0..4)
                        .filter_map(|i| {
                            let mb = num(&format!("bank{}_mb", i));
                            (mb > 0).then(|| Bank {
                                index: i as u8,
                                mb,
                                base: num(&format!("bank{}_base", i)),
                            })
                        })
                        .collect();
                }
                "run" => {
                    settings = RunSettings {
                        groups: num("groups") as u32,
                        time_pct: num("time_pct") as u32,
                        repeats: num("repeats") as u32,
                    };
                }
                "totals" => {
                    checked = num("checked") as usize;
                    matched = num("matched") as usize;
                    total_ns = num("ns");
                    total_ic = num("icount");
                }
                _ => {}
            }
            continue;
        }

        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 11 { continue; }
        rows.push(Row {
            name: f[0].to_string(),
            unit: f[1].to_string(),
            iters: parse_u64(f[2]).unwrap_or(0),
            work: parse_u64(f[3]).unwrap_or(0),
            ns: parse_u64(f[4]).unwrap_or(0),
            icount: parse_u64(f[5]).unwrap_or(0),
            count: parse_u64(f[6]).unwrap_or(0),
            exc: parse_u64(f[7]).unwrap_or(0),
            checksum: f[8].to_string(),
            golden: f[9].to_string(),
            status: f[10].to_string(),
        });
    }

    if rows.is_empty() {
        return Err("the report block held no benchmark rows".to_string());
    }
    Ok(Parsed { machine, rows, checked, matched, total_ns, total_icount: total_ic, settings })
}

pub fn parse_u64(s: &str) -> Option<u64> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse().ok()
    }
}

/// Pull the feature list out of the emulator's own startup banner, so a saved
/// result records what produced it rather than what the caller believed.
pub fn parse_features(stderr: &str) -> Vec<String> {
    for line in stderr.lines() {
        if let Some(rest) = line.strip_prefix("iris: build features: ") {
            let rest = rest.trim();
            if rest == "(none)" { return Vec::new(); }
            return rest.split_whitespace().map(str::to_string).collect();
        }
    }
    Vec::new()
}

// ─── headline categories ─────────────────────────────────────────────────────

/// One line of the summary a person actually reads: a benchmark family reduced
/// to a single throughput.
///
/// A group's kernels do not all measure the same thing — `int/` reports both
/// `ops` and `dhry`, `mem/` reports `B`, `acc` and `upd` — so a category takes
/// only the kernels carrying the group's dominant unit and aggregates those.
/// Summing work and time separately (rather than averaging per-kernel rates)
/// weights each kernel by how long it ran, which is what makes the result a
/// throughput rather than an average of incomparable numbers.
pub struct Category {
    pub label: &'static str,
    /// Kernel-name prefixes that belong to this family.
    pub prefixes: &'static [&'static str],
    /// The work unit to aggregate. Kernels reporting anything else are left
    /// out — they are measuring something this figure does not claim to cover.
    pub unit: &'static str,
    /// How to render the aggregate, after the SI prefix.
    pub suffix: &'static str,
}

pub const CATEGORIES: &[Category] = &[
    Category { label: "Integer",  prefixes: &["int/"],          unit: "ops", suffix: "ops/s" },
    Category { label: "Floating", prefixes: &["fpu/"],          unit: "ops", suffix: "ops/s" },
    Category { label: "Memory",   prefixes: &["mem/"],          unit: "B",   suffix: "B/s" },
    Category { label: "Imaging",  prefixes: &["img/", "vid/"],  unit: "px",  suffix: "px/s" },
    Category { label: "Codec",    prefixes: &["codec/"],        unit: "B",   suffix: "B/s" },
];

impl Run {
    /// This run's throughput for one category, or `None` when no kernel in it
    /// produced a usable measurement (skipped on this CPU, or the group was not
    /// selected).
    pub fn category_rate(&self, c: &Category) -> Option<f64> {
        let (mut work, mut ns) = (0u128, 0u128);
        for r in &self.rows {
            if r.status == "SKIP" || r.unit != c.unit || r.ns == 0 { continue; }
            if !c.prefixes.iter().any(|p| r.name.starts_with(p)) { continue; }
            work += r.work as u128;
            ns += r.ns as u128;
        }
        if ns == 0 || work == 0 { return None; }
        Some(work as f64 * 1e9 / ns as f64)
    }
}

/// A rate with an SI prefix: `38.5 M`, `7.32 k`. The caller appends the unit.
pub fn fmt_rate(v: f64) -> String {
    if v <= 0.0 { return "-".into(); }
    if v >= 1e9 { format!("{:.2} G", v / 1e9) }
    else if v >= 1e6 { format!("{:.2} M", v / 1e6) }
    else if v >= 1e3 { format!("{:.2} k", v / 1e3) }
    else { format!("{:.1}", v) }
}

// ─── host identification ─────────────────────────────────────────────────────

pub fn host_info() -> HostInfo {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    HostInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_model: cpu_model(),
        cores,
    }
}

/// A NUL-terminated `sysctl` name's value as a string, or `None` if the key is
/// absent or does not hold one.
#[cfg(target_os = "macos")]
fn sysctl_string(name: &[u8]) -> Option<String> {
    let mut len: libc::size_t = 0;
    let name = name.as_ptr() as *const libc::c_char;

    // SAFETY: `name` is NUL-terminated by the caller's byte-string literal. A
    // null `oldp` asks only for the length, which is what `len` receives.
    let rc = unsafe {
        libc::sysctlbyname(name, std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0)
    };
    if rc != 0 || len <= 1 {
        return None;
    }

    let mut buf = vec![0u8; len];
    // SAFETY: `buf` holds exactly the `len` bytes the call above asked for, and
    // `len` is passed by pointer so the kernel can shorten it.
    let rc = unsafe {
        libc::sysctlbyname(name, buf.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0)
    };
    if rc != 0 {
        return None;
    }

    buf.truncate(len);
    if let Some(nul) = buf.iter().position(|b| *b == 0) {
        buf.truncate(nul);
    }
    let s = String::from_utf8(buf).ok()?;
    let s = s.trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

pub fn cpu_model() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in s.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    if k.trim() == "model name" || k.trim() == "Model" {
                        return v.trim().to_string();
                    }
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        // `sysctlbyname`, not a spawned `sysctl`. This would otherwise be the
        // only subprocess left anywhere in the benchmark path, and the whole
        // point of running the suite in-process is that a sandboxed
        // application should need none. Returns "Apple M1"-style names on
        // Apple silicon and the Intel brand string on Intel.
        if let Some(s) = sysctl_string(b"machdep.cpu.brand_string\0") {
            return s;
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(s) = std::env::var("PROCESSOR_IDENTIFIER") { return s; }
    }
    "unknown".to_string()
}


// ─── reference rows ──────────────────────────────────────────────────────────
//
// `data/bench_reference.json` is the table the GUI compares a user's result
// against. It ships checked in and **starts empty** — a machine with no row is
// the normal case, not an error, and the GUI says "reference statistics not
// gathered for this platform" rather than inventing one. Rows are added by
// running the suite on a machine and pasting what this subcommand prints.
//
// Deliberately a static file updated by hand: the alternative (a user-writable
// override, an import/export pair, a fetch) is a lot of machinery for a table
// that changes when someone gets a new Mac.

/// One machine's numbers, as they appear in `data/bench_reference.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceEntry {
    pub id: String,
    pub label: String,
    /// Emulated CPU (`R4400`/`R5000`) and execution engine (`interp`/`jitv2`).
    /// Both matter: the two engines differ by about 4x, so a row without them
    /// cannot be compared with anything.
    pub cpu: String,
    pub engine: String,
    pub host: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measured: Option<String>,
    pub guest_mips: f64,
    pub dmips: f64,
    pub accuracy: f64,
    /// Work units per second, per kernel.
    pub kernels: BTreeMap<String, f64>,
}

/// The reference table that ships with the application.
///
/// Checked in, compiled in, and **normally empty** — a machine with no row gets
/// "reference statistics not gathered for this platform" rather than an
/// invented comparison. There is no upload, no download and no user-writable
/// override: it is a static file and a pull request. See
/// `data/bench_reference.README.md`.
///
/// Note this is *not* the correctness oracle. The golden checksums accuracy is
/// scored against are compiled into the guest image, where nobody can edit them
/// to "fix" a failure. Externalising the performance table therefore carries no
/// correctness risk at all.
pub fn bundled_reference() -> ReferenceTable {
    const JSON: &str = include_str!("../data/bench_reference.json");
    serde_json::from_str(JSON).unwrap_or_else(|e| {
        // A malformed table must not take the application down over a
        // comparison it can perfectly well do without.
        log::warn!("bench_reference.json is not readable ({e}); treating it as empty");
        ReferenceTable { schema: 1, suite_id: String::new(), entries: Vec::new() }
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceTable {
    pub schema: u32,
    /// The guest binary these numbers were measured against. A result whose own
    /// `suite_id` differs is not comparable — treat the table as empty rather
    /// than comparing across two different workloads.
    pub suite_id: String,
    pub entries: Vec<ReferenceEntry>,
}

impl ReferenceTable {
    /// The row to compare `run` against, if there is one.
    ///
    /// Three things have to agree before a comparison means anything: the
    /// **suite**, because two different workloads under one name is not a
    /// comparison; the **emulated CPU**, because the R4400 and R5000 cache
    /// models differ deeply; and the **engine**, because the interpreter and
    /// jitv2 are about 4x apart. A mismatch on any of them is treated exactly
    /// like an empty table, so callers need one fallback path rather than four.
    pub fn matching(&self, run: &Run) -> Option<&ReferenceEntry> {
        if self.entries.is_empty() || self.suite_id != run.suite_id {
            return None;
        }
        let engine = engine_of(run);
        let candidates = || {
            self.entries.iter()
                .filter(|e| e.cpu == run.machine.cpu && e.engine == engine)
        };
        // Prefer the same host CPU — that is a like-for-like number rather than
        // a cross-machine one — and otherwise take any row for this cell.
        candidates()
            .find(|e| e.host == run.host.cpu_model)
            .or_else(|| candidates().next())
    }
}

/// `interp` unless the emulator's own feature banner says otherwise. Read from
/// the banner rather than inferred, so a mislabelled row is impossible.
pub fn engine_of(run: &Run) -> &'static str {
    if run.features.iter().any(|f| f == "jitv2") { "jitv2" } else { "interp" }
}

/// Today as `YYYY-MM-DD`, from the system clock. Hinnant's civil-from-days.
pub fn today() -> String {
    let days = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => (d.as_secs() / 86_400) as i64,
        Err(_) => return "unknown".into(),
    };
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{:02}-{:02}", if m <= 2 { y + 1 } else { y }, m, d)
}

pub fn reference_entry(run: &Run, id: &str, label: Option<&str>, measured: Option<&str>) -> ReferenceEntry {
    ReferenceEntry {
        id: id.to_string(),
        label: label.map(str::to_string).unwrap_or_else(|| {
            format!("{} — {}", run.host.cpu_model, engine_of(run))
        }),
        cpu: run.machine.cpu.clone(),
        engine: engine_of(run).to_string(),
        host: run.host.cpu_model.clone(),
        measured: Some(measured.map(str::to_string).unwrap_or_else(today)),
        guest_mips: (run.mips() * 10.0).round() / 10.0,
        dmips: run.dmips().map(|v| (v * 10.0).round() / 10.0).unwrap_or(0.0),
        accuracy: (run.accuracy() * 10.0).round() / 10.0,
        kernels: run.rows.iter()
            .filter(|r| r.status != "SKIP" && r.work > 0)
            .map(|r| (r.name.clone(), (r.rate() * 100.0).round() / 100.0))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real report block, trimmed to four rows. Kept verbatim rather than
    /// generated: the point of the test is that the parser still agrees with
    /// what `bench/harness/main.c` actually prints.
    const BLOCK: &str = "\
benchmark                 unit       rate/s    guest-MIPS   time%  acc
------------------------------------------------------------------
int/alu                   ops      61247926      103.36          ok
------------------------------------------------------------------

IRIS-BENCH-BEGIN v1
#machine cpu=R4400 prid=0x00000440 fir=0x00000500 config=0x00c08483 l2=1 testdev=1 timebase=1 sysid=0x00000013 rev=4.0
#cache l1i=16384 l1i_line=16 l1d=16384 l1d_line=16 l2=1 l2_line=128 l2_bytes=0
#memory total_mb=256 banks=2 bank0_mb=128 bank0_base=0x08000000 bank1_mb=128 bank1_base=0x10000000
#timebase count_hz=32999459 measured=1
#work base=0x88300000 bytes=25165824
#run groups=0x0000003f time_pct=30 repeats=1
#cols name unit iters work ns icount count exc checksum golden status
int/alu ops 975094 15601504 254727060 26327595 8405943 0 0x152c986014248fb5 0x152c986014248fb5 OK
int/dhrystone dhry 51941 51941 246911934 22031467 8148070 0 0xe5c6550cf608d8ca 0xe5c6550cf608d8ca OK
mem/copy B 1 4194304 213900417 2621565 7058653 0 0xb603341656905fe2 0xb603341656905fe2 OK
sys/tlb_miss miss 256 524288 232062918 6710007 7658053 3 0x0000000000000000 0x0000000000000000 UNCHECKED
#totals benches=4 checked=3 matched=3 ns=947602329 icount=57690634
IRIS-BENCH-END
";

    fn parsed() -> Parsed {
        parse_block(BLOCK).expect("the block the guest prints must parse")
    }

    #[test]
    fn the_machine_block_parses() {
        let p = parsed();
        assert_eq!(p.machine.cpu, "R4400");
        assert!(p.machine.l2 && p.machine.testdev && p.machine.timebase);
        assert_eq!(p.machine.count_hz, 32_999_459);
        assert_eq!(p.machine.work_bytes, 25_165_824);
        assert_eq!(p.rows.len(), 4, "the human table above the block is not a row");
        assert_eq!((p.checked, p.matched), (3, 3));
        assert_eq!(p.total_ns, 947_602_329);

        let dhry = p.rows.iter().find(|r| r.name == "int/dhrystone").unwrap();
        assert_eq!(dhry.work, 51_941);
        assert_eq!(dhry.status, "OK");
        // Exceptions are parsed from their own column, not inferred.
        assert_eq!(p.rows.iter().find(|r| r.name == "sys/tlb_miss").unwrap().exc, 3);
    }

    #[test]
    fn the_machine_inventory_parses() {
        let m = parsed().machine;
        assert_eq!(m.rev, "4.0");
        assert_eq!(m.sysid, "0x00000013");

        assert_eq!(m.cache, CacheInfo {
            l1i_bytes: 16384, l1i_line: 16,
            l1d_bytes: 16384, l1d_line: 16,
            l2_present: true, l2_line: 128,
            // Not architecturally reported on anything but a Triton — 0 means
            // "unknown", and must never be read as "no L2".
            l2_bytes: 0,
        });
        assert!(!m.cache.is_empty());
        assert!(m.cache.l2_present && m.cache.l2_bytes == 0,
                "an L2 of unknown size is still an L2");

        assert_eq!(m.memory.total_mb, 256);
        assert_eq!(m.memory.banks, vec![
            Bank { index: 0, mb: 128, base: 0x0800_0000 },
            Bank { index: 1, mb: 128, base: 0x1000_0000 },
        ]);
        assert_eq!(m.memory.banks.iter().map(|b| b.mb).sum::<u64>(), m.memory.total_mb,
                   "the banks must add up to the total the guest reported");
    }

    /// The guest names any MIPS CPU from PRId, including ones this emulator
    /// does not model. Every name it can produce has to be a single token, or
    /// the whitespace-then-`=` parse of the machine block truncates it —
    /// "MIPS imp 0xab" once parsed as cpu="MIPS".
    #[test]
    fn a_cpu_the_emulator_does_not_model_survives_the_round_trip() {
        let exotic = BLOCK
            .replace("cpu=R4400", "cpu=MIPS-imp-0xab")
            .replace("prid=0x00000440", "prid=0x0000ab37")
            .replace("rev=4.0", "rev=3.7");
        let m = parse_block(&exotic).unwrap().machine;
        assert_eq!(m.cpu, "MIPS-imp-0xab");
        assert_eq!(m.prid, "0x0000ab37");
        assert_eq!(m.rev, "3.7");
        assert!(!m.cpu.contains(char::is_whitespace),
                "a CPU name with a space in it truncates on parse");
        // The keys *after* cpu= must still be found, which is what a spaced
        // name would have broken.
        assert_eq!(m.fir, "0x00000500");
        assert!(m.timebase);
    }

    /// A host run reports no inventory, and a result recorded before the
    /// inventory existed has none either. Both must load as "empty", never as
    /// a machine with no cache and no RAM.
    #[test]
    fn a_result_with_no_inventory_loads_as_empty_not_as_zero() {
        let older = BLOCK
            .lines()
            .filter(|l| !l.starts_with("#cache") && !l.starts_with("#memory"))
            .collect::<Vec<_>>()
            .join("\n");
        let m = parse_block(&older).unwrap().machine;
        assert!(m.cache.is_empty());
        assert!(m.memory.is_empty());
        assert!(m.memory.banks.is_empty());
    }

    #[test]
    fn the_run_configuration_comes_from_the_block_not_from_the_caller() {
        let p = parsed();
        assert_eq!(p.settings, RunSettings { groups: BG_ALL, time_pct: 30, repeats: 1 });
        assert!(!p.settings.is_full());
        assert!(p.settings.shortened_because().is_some());
    }

    /// A result recorded before `#run` existed was a full run, and has to load
    /// as one — otherwise every stored result would suddenly be "shortened"
    /// and refused by the reference merge.
    #[test]
    fn a_block_without_a_run_line_is_a_full_run() {
        let older = BLOCK.replace("#run groups=0x0000003f time_pct=30 repeats=1\n", "");
        let p = parse_block(&older).unwrap();
        assert!(p.settings.is_full());
        assert!(p.settings.shortened_because().is_none());
    }

    #[test]
    fn a_truncated_report_is_an_error_rather_than_an_empty_result() {
        let cut = BLOCK.split("#totals").next().unwrap();
        assert!(parse_block(cut).is_err(), "a run that died mid-report must not parse");
        assert!(parse_block("nothing here").is_err());
    }

    fn a_run() -> Run {
        let p = parsed();
        Run {
            cell: "test".into(),
            features: vec!["tlbvmap".into()],
            machine: p.machine,
            host: HostInfo { cpu_model: "Test CPU".into(), ..Default::default() },
            rows: p.rows,
            checked: p.checked,
            matched: p.matched,
            total_ns: p.total_ns,
            total_icount: p.total_icount,
            wall_s: 30.0,
            suite_id: "blake3:0123456789abcdef".into(),
            settings: p.settings,
        }
    }

    #[test]
    fn derived_figures_use_the_conventions_they_claim_to() {
        let run = a_run();
        assert_eq!(run.accuracy(), 100.0);
        // DMIPS is dhrystones/s over the VAX 11/780 constant, not a raw rate.
        let dhry = run.row("int/dhrystone").unwrap();
        let want = dhry.rate() / DHRY_PER_DMIPS;
        assert!((run.dmips().unwrap() - want).abs() < 1e-9);
        assert!(run.dmips().unwrap() > 0.0);
    }

    #[test]
    fn a_category_aggregates_only_its_own_kernels_and_unit() {
        let run = a_run();
        let int = CATEGORIES.iter().find(|c| c.label == "Integer").unwrap();
        // int/alu is "ops"; int/dhrystone is "dhry" and must not be folded in,
        // since adding dhrystones to ALU operations is not a throughput.
        let alu = run.row("int/alu").unwrap();
        let want = alu.work as f64 * 1e9 / alu.ns as f64;
        assert!((run.category_rate(int).unwrap() - want).abs() < 1e-6);

        // A category with no kernels present is absent, not zero.
        let img = CATEGORIES.iter().find(|c| c.label == "Imaging").unwrap();
        assert!(run.category_rate(img).is_none());
    }

    #[test]
    fn the_reference_table_refuses_every_mismatch() {
        let run = a_run();
        let row = ReferenceEntry {
            id: "ref".into(), label: "Reference".into(),
            cpu: "R4400".into(), engine: "interp".into(), host: "Test CPU".into(),
            measured: None, guest_mips: 50.0, dmips: 70.0, accuracy: 100.0,
            kernels: BTreeMap::new(),
        };
        let table = |suite: &str, e: ReferenceEntry| ReferenceTable {
            schema: 1, suite_id: suite.into(), entries: vec![e],
        };

        assert!(table(&run.suite_id, row.clone()).matching(&run).is_some());

        // A different suite is two different workloads under one name.
        assert!(table("blake3:ffffffffffffffff", row.clone()).matching(&run).is_none());
        // A different emulated CPU: the cache models differ deeply.
        assert!(table(&run.suite_id, ReferenceEntry { cpu: "R5000".into(), ..row.clone() })
            .matching(&run).is_none());
        // A different engine: the two are about 4x apart.
        assert!(table(&run.suite_id, ReferenceEntry { engine: "jitv2".into(), ..row.clone() })
            .matching(&run).is_none());
        // An empty table is the normal shipping state, not an error.
        assert!(ReferenceTable { schema: 1, suite_id: String::new(), entries: Vec::new() }
            .matching(&run).is_none());
    }

    #[test]
    fn the_same_host_cpu_wins_over_a_merely_compatible_row() {
        let run = a_run();
        let base = ReferenceEntry {
            id: "other".into(), label: "Other".into(),
            cpu: "R4400".into(), engine: "interp".into(), host: "Some Other CPU".into(),
            measured: None, guest_mips: 1.0, dmips: 1.0, accuracy: 100.0,
            kernels: BTreeMap::new(),
        };
        let mine = ReferenceEntry { id: "mine".into(), host: "Test CPU".into(), ..base.clone() };
        let table = ReferenceTable {
            schema: 1, suite_id: run.suite_id.clone(), entries: vec![base, mine],
        };
        assert_eq!(table.matching(&run).unwrap().id, "mine");
    }

    /// The table that actually ships has to parse, whatever is in it.
    #[test]
    fn the_bundled_reference_table_loads() {
        let t = bundled_reference();
        assert_eq!(t.schema, 1);
        if !t.entries.is_empty() {
            assert!(!t.suite_id.is_empty(),
                    "a populated table must name the suite its rows were measured against");
            for e in &t.entries {
                assert!(!e.cpu.is_empty() && !e.engine.is_empty(),
                        "row {} has no cpu/engine, so nothing can be compared to it", e.id);
            }
        }
    }

    #[test]
    fn rates_are_formatted_with_an_si_prefix() {
        assert_eq!(fmt_rate(38_500_000.0), "38.50 M");
        assert_eq!(fmt_rate(7_320.0), "7.32 k");
        assert_eq!(fmt_rate(0.0), "-");
        assert_eq!(fmt_rate(-1.0), "-");
    }
}
