//! How much would a content-hash cache of compiled pages save? (R10000 JIT only.)
//!
//! A compiled page function does not depend on which physical page it came
//! from: branch targets are page-relative, `j`/`jal` resolve at run time, and
//! the `PhysicalCodePage` pointer codegen receives is bookkeeping, never
//! emitted. So code compiled for one page could serve any page with the same
//! bytes, as long as its dispatch switch covers the entries that page needs.
//! This measures how often that situation arises, before anything is built.
//!
//! `IRIS_JIT_HASHSTATS=1` turns it on. At every compile that really happens
//! (past churn avoidance and every early-out), it records two keys:
//!
//! - **page**: all 1024 words plus FR mode. Equal keys compile identically for
//!   any entry set, so a hit whose earlier entry set covers this one could
//!   reuse the earlier code outright.
//! - **exact**: the words the walk visited, the visited mask, the entry set and
//!   FR mode: what the compile actually consumed.
//!
//! Each hit is split by same/other physical page, and by whether a mega-flush
//! happened in between (a flush resets the code arena, so an in-session cache
//! cannot reuse anything across one). A line goes to stderr every
//! `REPORT_EVERY` compiles, since a killed emulator never reaches a final
//! report.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::cpu::jitv2::{BITMAP_WORDS, ENTRIES_PER_PAGE};

/// Bumped by `Jitv2::mega_flush`.
pub static FLUSH_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Time spent in `compile_region_uncommitted` (Cranelift codegen for one page),
/// summed over the deferred-path compiles: what a cache hit would save.
static COMPILE_NS: AtomicU64 = AtomicU64::new(0);
static COMPILE_N: AtomicU64 = AtomicU64::new(0);

pub fn note_compile_time(d: std::time::Duration) {
    if enabled() {
        COMPILE_NS.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
        COMPILE_N.fetch_add(1, Ordering::Relaxed);
    }
}

const REPORT_EVERY: u64 = 500;

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("IRIS_JIT_HASHSTATS").as_deref(), Ok("1")))
}

struct Seen {
    pfn: u32,
    epoch: u64,
    entries: [u64; BITMAP_WORDS],
}

#[derive(Default)]
struct Stats {
    compiles: u64,
    page: HashMap<u64, Seen>,
    exact: HashMap<u64, Seen>,
    // page-key hits: [same pfn, other pfn] x [same epoch, across a flush]
    page_hit: [[u64; 2]; 2],
    // ... of those, the earlier entry set covered this compile's
    page_covered: [[u64; 2]; 2],
    exact_hit: [[u64; 2]; 2],
    instrs: u64,
    instrs_page_covered_same_epoch: u64,
    // same pfn + same epoch + covered, by why churn avoidance let it through:
    // [page had no compile snapshot, snapshot present but the skip rejected]
    covered_why: [u64; 2],
    // ... and when the skip rejected, which check refused (index = the
    // `PhysicalCodePage::last_skip_reject` code)
    skip_reject: [u64; 8],
}

/// `IRIS_JIT_HASHSTATS_LOG=<file>`: also append one line per compile, so two
/// runs can be compared for cross-run reuse (what a persistent cache would
/// hit). Fields: page-content key, pfn, flush epoch, FR mode, instruction
/// count, entry bitmap (16 hex words).
fn log_line(kp: u64, pfn: u32, epoch: u64, entries: &[u64; BITMAP_WORDS], fr1: bool, instrs: usize) {
    static LOG: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let log = LOG.get_or_init(|| {
        std::env::var_os("IRIS_JIT_HASHSTATS_LOG").and_then(|p| {
            std::fs::OpenOptions::new().create(true).append(true).open(p).ok().map(Mutex::new)
        })
    });
    if let Some(f) = log {
        use std::io::Write;
        let mut line = format!("{kp:016x} {pfn:x} {epoch} {} {instrs} ", fr1 as u8);
        for w in entries { line.push_str(&format!("{w:016x}")); }
        line.push('\n');
        let _ = f.lock().unwrap().write_all(line.as_bytes());
    }
}

fn stats() -> &'static Mutex<Stats> {
    static S: OnceLock<Mutex<Stats>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Stats::default()))
}

fn key_page(words: &[u32; ENTRIES_PER_PAGE], fr1: bool) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    words.hash(&mut h);
    fr1.hash(&mut h);
    h.finish()
}

fn key_exact(words: &[u32; ENTRIES_PER_PAGE], used: &[u64; BITMAP_WORDS], entries: &[u64; BITMAP_WORDS], fr1: bool) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    used.hash(&mut h);
    entries.hash(&mut h);
    fr1.hash(&mut h);
    for (i, w) in words.iter().enumerate() {
        if used[i >> 6] & (1u64 << (i & 63)) != 0 {
            w.hash(&mut h);
        }
    }
    h.finish()
}

/// Record one compile that is about to happen.
pub fn record(pfn: u32, words: &[u32; ENTRIES_PER_PAGE], used: &[u64; BITMAP_WORDS],
              entries: &[u64; BITMAP_WORDS], fr1: bool, instr_count: usize, had_snapshot: bool,
              reject_code: u8) {
    let epoch = FLUSH_EPOCH.load(Ordering::Relaxed);
    let kp = key_page(words, fr1);
    log_line(kp, pfn, epoch, entries, fr1, instr_count);
    let ke = key_exact(words, used, entries, fr1);
    let mut s = stats().lock().unwrap();
    s.compiles += 1;
    s.instrs += instr_count as u64;

    if let Some(prev) = s.page.get(&kp) {
        let where_ = (prev.pfn != pfn) as usize;
        let when = (prev.epoch != epoch) as usize;
        let covered = (0..BITMAP_WORDS).all(|i| entries[i] & !prev.entries[i] == 0);
        s.page_hit[where_][when] += 1;
        if covered {
            s.page_covered[where_][when] += 1;
            if when == 0 {
                s.instrs_page_covered_same_epoch += instr_count as u64;
                if where_ == 0 {
                    s.covered_why[had_snapshot as usize] += 1;
                    if had_snapshot {
                        s.skip_reject[(reject_code as usize).min(7)] += 1;
                    }
                }
            }
        }
    }
    // Keep the widest entry set seen for this content in this epoch: that is
    // what a cache would hold after recompiling to add coverage.
    let merged = match s.page.get(&kp) {
        Some(prev) if prev.epoch == epoch => {
            let mut m = prev.entries;
            for i in 0..BITMAP_WORDS { m[i] |= entries[i]; }
            m
        }
        _ => *entries,
    };
    s.page.insert(kp, Seen { pfn, epoch, entries: merged });

    if let Some(prev) = s.exact.get(&ke) {
        let where_ = (prev.pfn != pfn) as usize;
        let when = (prev.epoch != epoch) as usize;
        s.exact_hit[where_][when] += 1;
    }
    s.exact.insert(ke, Seen { pfn, epoch, entries: *entries });

    if s.compiles % REPORT_EVERY == 0 {
        eprintln!(
            "jitv2 hashstats: compiles={} flushes={} distinct_page={} distinct_exact={} \
             page_hit[same_pfn,other_pfn]x[same_epoch,across_flush]={:?} \
             page_covered={:?} exact_hit={:?} instrs={} instrs_reusable_in_session={} \
             covered_why[no_snapshot,skip_rejected]={:?} skip_reject_by_code={:?} \
             codegen_ms_total={} codegen_us_avg={}",
            s.compiles, epoch, s.page.len(), s.exact.len(),
            s.page_hit, s.page_covered, s.exact_hit, s.instrs, s.instrs_page_covered_same_epoch,
            s.covered_why, s.skip_reject,
            COMPILE_NS.load(Ordering::Relaxed) / 1_000_000,
            COMPILE_NS.load(Ordering::Relaxed) / 1000 / COMPILE_N.load(Ordering::Relaxed).max(1));
    }
}
