//! JIT v2 corpus runner: walks every snapshot in a `jitv2_corpus/`-style
//! directory (see `jitv2/comp.rs`) through the reachability walker
//! (`jitv2/analyzer.rs`) and prints aggregate stats — the Phase 0 sizing
//! measurements the design doc calls for (§9): region-size histogram,
//! exclusion-reason counts, 0xFFC hit rate.
//!
//! Usage: jitv2-analyze [corpus_dir]   (default: jitv2_corpus)

use std::collections::HashMap;
use std::path::Path;

use iris::cpu::jitv2::analyzer::{instrs_linear, Analyzer, StopReason};
use iris::cpu::jitv2::ENTRIES_PER_PAGE;

/// Parsed from a corpus filename `pfn_<pfn:08x>_off_<offset:04x>.bin`
/// (`jitv2/comp.rs::corpus_path`).
struct Entry {
    pfn: u32,
    offset: u16,
    path: std::path::PathBuf,
}

fn parse_filename(name: &str) -> Option<(u32, u16)> {
    let name = name.strip_suffix(".bin")?;
    let rest = name.strip_prefix("pfn_")?;
    let (pfn_hex, rest) = rest.split_once("_off_")?;
    let pfn = u32::from_str_radix(pfn_hex, 16).ok()?;
    let offset = u16::from_str_radix(rest, 16).ok()?;
    Some((pfn, offset))
}

fn load_page(path: &Path) -> std::io::Result<[u32; ENTRIES_PER_PAGE]> {
    let bytes = std::fs::read(path)?;
    let mut words = [0u32; ENTRIES_PER_PAGE];
    // Raw memory, word for word — the same representation comp.rs wrote
    // (slice::from_raw_parts over a [u32; N], no byte-order conversion).
    let src: &[u32] = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4)
    };
    let n = src.len().min(words.len());
    words[..n].copy_from_slice(&src[..n]);
    Ok(words)
}

#[derive(Default)]
struct Stats {
    files_walked: usize,
    files_failed: usize,
    /// Entries whose first instruction is itself excluded (§4.4) — an empty
    /// region, the "excluded-first-instruction" sticky-rejection reason
    /// (§6.4). Not counted in `region_sizes`/`stop_reasons` since nothing
    /// was walked.
    entries_rejected_excluded_first: usize,
    region_sizes: Vec<usize>,
    stop_reasons: HashMap<&'static str, usize>,
    foreign_page_slot_hits: usize,
    entries_at_word_1: usize, // offset 4 bytes == word index 1, the total-entry predicate's always-checkable offset (§6.1.4)
}

fn stop_reason_name(r: StopReason) -> &'static str {
    match r {
        StopReason::PageLeaving => "page_leaving",
        StopReason::RegJump => "reg_jump",
        StopReason::Excluded => "excluded",
        StopReason::ForeignPageSlot => "foreign_page_slot",
        StopReason::Truncated => "truncated",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus_dir = args.get(1).map(String::as_str).unwrap_or("jitv2_corpus");

    let read_dir = match std::fs::read_dir(corpus_dir) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("jitv2-analyze: cannot read corpus dir '{}': {}", corpus_dir, e);
            std::process::exit(1);
        }
    };

    let mut entries: Vec<Entry> = Vec::new();
    for dirent in read_dir {
        let dirent = match dirent { Ok(d) => d, Err(_) => continue };
        let file_name = dirent.file_name();
        let name = match file_name.to_str() { Some(n) => n, None => continue };
        if let Some((pfn, offset)) = parse_filename(name) {
            entries.push(Entry { pfn, offset, path: dirent.path() });
        }
    }

    eprintln!("jitv2-analyze: {} corpus entries found in '{}'", entries.len(), corpus_dir);

    let mut stats = Stats::default();
    // One Analyzer, reused across every corpus entry — its scratch buffer is
    // reset in place per walk() call rather than heap-allocated per job,
    // matching how the real compile thread will use it.
    let mut analyzer = Analyzer::new();

    for entry in &entries {
        let page = match load_page(&entry.path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("jitv2-analyze: failed to read {}: {}", entry.path.display(), e);
                stats.files_failed += 1;
                continue;
            }
        };

        let entry_word = entry.offset / 4;
        let page_base = entry.pfn.wrapping_mul(iris::cpu::jitv2::PAGE_SIZE);
        let (result, non_empty) = analyzer.walk(&page, entry_word, page_base);

        stats.files_walked += 1;
        if entry_word == 1 {
            stats.entries_at_word_1 += 1;
        }
        if !non_empty {
            stats.entries_rejected_excluded_first += 1;
            continue;
        }
        let mut region_size = 0usize;
        for instr in instrs_linear(result) {
            region_size += 1;
            for reason in [instr.fallthrough_exit, instr.taken_exit].into_iter().flatten() {
                *stats.stop_reasons.entry(stop_reason_name(reason)).or_insert(0) += 1;
                if reason == StopReason::ForeignPageSlot {
                    stats.foreign_page_slot_hits += 1;
                }
            }
        }
        stats.region_sizes.push(region_size);
    }

    print_report(&stats);
}

fn print_report(stats: &Stats) {
    println!("=== JIT v2 corpus analysis ===");
    println!("files walked: {}   failed to read: {}", stats.files_walked, stats.files_failed);
    println!("entries rejected (excluded first instruction, empty region): {} ({:.2}%)",
        stats.entries_rejected_excluded_first,
        stats.entries_rejected_excluded_first as f64 * 100.0 / stats.files_walked.max(1) as f64);
    println!("entries at word offset 1 (byte 4, the 0xFFC-safe total-entry offset): {}", stats.entries_at_word_1);
    println!();

    if !stats.region_sizes.is_empty() {
        let mut sizes = stats.region_sizes.clone();
        sizes.sort_unstable();
        let n = sizes.len();
        let sum: usize = sizes.iter().sum();
        let mean = sum as f64 / n as f64;
        let median = sizes[n / 2];
        let p90 = sizes[(n * 9 / 10).min(n - 1)];
        let max = *sizes.last().unwrap();
        let min = sizes[0];
        println!("region size (instructions reached per entry):");
        println!("  min={} median={} mean={:.1} p90={} max={}", min, median, mean, p90, max);

        // Simple bucketed histogram.
        let buckets = [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, usize::MAX];
        let mut counts = vec![0usize; buckets.len()];
        for &s in &sizes {
            let idx = buckets.iter().position(|&b| s <= b).unwrap();
            counts[idx] += 1;
        }
        println!("  histogram (region size <= bucket):");
        for (b, c) in buckets.iter().zip(counts.iter()) {
            if *c == 0 { continue; }
            let label = if *b == usize::MAX { "inf".to_string() } else { b.to_string() };
            let pct = *c as f64 * 100.0 / n as f64;
            println!("    <= {:>5}: {:>7} ({:5.1}%)", label, c, pct);
        }
    }
    println!();

    println!("stop reasons (per terminal instruction across all walked regions):");
    let mut reasons: Vec<(&str, usize)> = stats.stop_reasons.iter().map(|(&k, &v)| (k, v)).collect();
    reasons.sort_by(|a, b| b.1.cmp(&a.1));
    let total_stops: usize = reasons.iter().map(|(_, v)| v).sum();
    for (name, count) in &reasons {
        let pct = *count as f64 * 100.0 / total_stops.max(1) as f64;
        println!("  {:<14} {:>7} ({:5.1}%)", name, count, pct);
    }
    println!();
    println!("0xFFC foreign-page-slot hits: {} ({:.2}% of walked files)",
        stats.foreign_page_slot_hits,
        stats.foreign_page_slot_hits as f64 * 100.0 / stats.files_walked.max(1) as f64);
}
