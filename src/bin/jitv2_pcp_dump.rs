//! Standalone offline analyzer for a `j2 dump-pcp` capture file
//! (`src/jitv2/pcp_dump.rs`'s format): loads the dumped `PhysicalCodePage`
//! state + raw 4KB memory, re-runs the real reachability walker over every
//! offset that was requested/compiled/denylisted at capture time, and
//! prints disassembly + classification for each — the offline counterpart
//! to the `j2 analyze`/`j2 pcp` monitor commands, for a page pulled from a
//! machine that already crashed/stopped and isn't available to query live
//! anymore.
//!
//! Usage: jitv2_pcp_dump <dump_file> [--offset <hex>] [--compile]
//!   --offset <hex>  also walk+disassemble starting from this specific word
//!                    offset (page-relative, e.g. 0x60), regardless of
//!                    whether it was requested/compiled/denylisted at dump
//!                    time — for probing "what would the walk from here
//!                    even look like" ad hoc, same as `j2 analyze`.
//!   --compile        also run real codegen (Codegen::compile_region_uncommitted)
//!                    against the merged multi-entry region and report
//!                    whether it would compile cleanly today (useful when
//!                    the dump predates a codegen fix and you want to know
//!                    if this exact page would compile now).

use iris::jitv2::analyzer::{instrs_linear, Analyzer};
use iris::jitv2::pcp_dump::PcpDump;
use iris::jitv2::{ENTRIES_PER_PAGE, PAGE_SIZE};
use iris::mips_dis::disassemble;

fn print_usage_and_exit() -> ! {
    eprintln!("Usage: jitv2_pcp_dump <dump_file> [--offset <hex>] [--compile]");
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage_and_exit();
    }
    let path = &args[1];

    let mut extra_offset: Option<u16> = None;
    let mut do_compile = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--offset" => {
                let v = args.get(i + 1).unwrap_or_else(|| print_usage_and_exit());
                let v = v.strip_prefix("0x").unwrap_or(v);
                extra_offset = Some(u16::from_str_radix(v, 16).unwrap_or_else(|_| {
                    eprintln!("jitv2_pcp_dump: bad --offset value '{}'", args[i + 1]);
                    std::process::exit(1);
                }));
                i += 2;
            }
            "--compile" => {
                do_compile = true;
                i += 1;
            }
            other => {
                eprintln!("jitv2_pcp_dump: unknown argument '{}'", other);
                print_usage_and_exit();
            }
        }
    }

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("jitv2_pcp_dump: cannot read '{}': {}", path, e);
            std::process::exit(1);
        }
    };
    let dump = match PcpDump::from_bytes(&bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("jitv2_pcp_dump: {}: {}", path, e);
            std::process::exit(1);
        }
    };

    let page_base_phys = dump.pfn * PAGE_SIZE;
    println!("=== {} ===", path);
    println!("pfn={:#010x}  phys_base={:#010x}", dump.pfn, page_base_phys);
    println!("current_gen={}  entry_gen={}  {}",
        dump.current_gen, dump.entry_gen,
        if dump.current_gen == dump.entry_gen { "(func/compiled ARE fresh vs this snapshot's gen)" }
        else { "(func/compiled are STALE vs this snapshot's current_gen — page mutated after last publish)" });
    println!("fr1={}", dump.fr1);
    println!();

    // §13: `is_requested`/`is_compiled`/`is_denylisted` are WORD-indexed
    // (0..ENTRIES_PER_PAGE), matching `PhysicalCodePage`'s own bitmap
    // indexing — `requested`/`compiled`/`denylisted`/`entry_words` below are
    // all word indices, not byte offsets. `fmt_offsets` (and every other
    // "page_off" display in this tool) prints BYTE offsets, so every one of
    // these lists gets multiplied by 4 before formatting — passing raw word
    // indices to fmt_offsets was a real bug in an earlier version of this
    // tool: it silently mislabeled word index 0x90 as "page_off 0x90"
    // (which looks byte-offset-shaped but is actually byte offset 0x240),
    // and the range-collapsing logic's `+4` step never matched anything
    // since adjacent word indices differ by 1, not 4 — so nothing ever
    // collapsed and every printed value was 4x too small to be the byte
    // offset it looked like.
    let requested_words: Vec<u16> = (0..ENTRIES_PER_PAGE as u16).filter(|&o| dump.is_requested(o as usize)).collect();
    let compiled_words: Vec<u16> = (0..ENTRIES_PER_PAGE as u16).filter(|&o| dump.is_compiled(o as usize)).collect();
    let denylisted_words: Vec<u16> = (0..ENTRIES_PER_PAGE as u16).filter(|&o| dump.is_denylisted(o as usize)).collect();

    println!("requested offsets ({}): {}", requested_words.len(), fmt_offsets(&to_byte_offsets(&requested_words)));
    println!("compiled offsets  ({}): {}", compiled_words.len(), fmt_offsets(&to_byte_offsets(&compiled_words)));
    println!("denylisted offsets ({}): {}", denylisted_words.len(), fmt_offsets(&to_byte_offsets(&denylisted_words)));
    println!();

    // Entry set for the real walk: every offset that was either compiled or
    // requested at capture time (the union is what a real compile would
    // have covered, or was about to cover) — denylisted offsets are
    // reported above but deliberately excluded from the walk (that's the
    // whole point of denylisting: don't re-walk them). Word indices, same
    // as `walk_multi_entry` itself expects — `--offset` is given in bytes
    // on the command line (matching every other "offset" this tool takes
    // input as), so it's converted to a word index here before joining.
    let mut entry_words: Vec<u16> = requested_words.iter().chain(compiled_words.iter()).copied().collect();
    entry_words.sort_unstable();
    entry_words.dedup();
    if let Some(extra_byte_offset) = extra_offset {
        let extra_word = extra_byte_offset / 4;
        if !entry_words.contains(&extra_word) {
            entry_words.push(extra_word);
            entry_words.sort_unstable();
        }
    }

    if entry_words.is_empty() {
        println!("(nothing requested or compiled at capture time, and no --offset given — nothing to walk)");
        return;
    }

    let mut analyzer = Analyzer::new();
    // Scope the walk's borrow of `analyzer` to just what's needed to build
    // `visited` — `covered()`/`has_fpu()` (called right after) are separate
    // `&analyzer` borrows that can't coexist with `instrs`, which is really
    // `&analyzer.instrs` under the hood.
    let visited: Vec<_> = {
        let instrs = analyzer.walk_multi_entry(&dump.words, &entry_words, page_base_phys, usize::MAX);
        instrs_linear(instrs).cloned().collect()
    };

    println!("=== multi-entry walk from {} entry point(s): {} ===", entry_words.len(), fmt_offsets(&to_byte_offsets(&entry_words)));
    let covered = analyzer.covered().to_vec();
    let declined: Vec<u16> = entry_words.iter().copied().filter(|w| !covered.contains(w)).collect();
    if !declined.is_empty() {
        println!("entries DECLINED by the walk (excluded at entry, per current analyzer rules): {}", fmt_offsets(&to_byte_offsets(&declined)));
    }
    println!("has_fpu={}", analyzer.has_fpu());
    println!();

    println!("{} instructions visited:", visited.len());
    for instr in &visited {
        let paddr = page_base_phys + (instr.word as u32) * 4;
        let dis = disassemble(instr.raw, paddr as u64, None);
        let mut tags = Vec::new();
        if instr.is_entry_point { tags.push("ENTRY"); }
        if instr.is_fallback { tags.push("fallback"); }
        if instr.is_branch_fallback_successor { tags.push("branch-fallback-successor"); }
        if instr.is_slot_only { tags.push("slot-only"); }
        if instr.is_branch_target { tags.push("branch-target"); }
        let tag_str = if tags.is_empty() { String::new() } else { format!(" [{}]", tags.join(",")) };
        let exits = match (instr.fallthrough_exit, instr.taken_exit) {
            (None, None) => String::new(),
            (ft, tk) => format!(" (fallthrough_exit={:?} taken_exit={:?})", ft, tk),
        };
        println!("  page_off={:#05x} phys={:#010x}: {:08x} {}{}{}",
            instr.word * 4, paddr, instr.raw, dis, tag_str, exits);
    }

    if do_compile {
        println!();
        println!("=== attempting real codegen (compile_region_uncommitted) ===");
        run_compile_attempt(&mut analyzer, dump.fr1);
    }
}

/// Word indices (0..ENTRIES_PER_PAGE) -> byte offsets (page-relative),
/// matching every "page_off" value this tool otherwise displays.
fn to_byte_offsets(words: &[u16]) -> Vec<u16> {
    words.iter().map(|&w| w * 4).collect()
}

/// `offsets` must already be BYTE offsets (page-relative, word-aligned —
/// see `to_byte_offsets`), not word indices, or both the collapsing logic
/// below and the printed values will be wrong.
fn fmt_offsets(offsets: &[u16]) -> String {
    if offsets.is_empty() {
        return "(none)".to_string();
    }
    // Collapse consecutive runs (adjacent words, +4 bytes apart) into
    // ranges — a page with hundreds of set bits in one contiguous block
    // (common: a compiled region's every internal word technically counts)
    // would otherwise flood the terminal with one hex number per line.
    let mut ranges: Vec<(u16, u16)> = Vec::new();
    for &o in offsets {
        match ranges.last_mut() {
            Some((_, end)) if *end + 4 == o => *end = o,
            _ => ranges.push((o, o)),
        }
    }
    ranges.iter()
        .map(|&(s, e)| if s == e { format!("{:#05x}", s) } else { format!("{:#05x}..{:#05x}", s, e) })
        .collect::<Vec<_>>()
        .join(", ")
}

fn run_compile_attempt(analyzer: &mut Analyzer, fr1: bool) {
    use iris::jitv2::codegen::Codegen;
    let mut codegen = Codegen::new();
    // `analyzer`'s scratch buffer already holds the merged walk from
    // `main`'s own `walk_multi_entry` call — codegen needs an owned,
    // mutable copy (it writes `block_id` back into it).
    let mut instrs_owned = analyzer.instrs_snapshot();
    let has_fpu = analyzer.has_fpu();
    match codegen.compile_region_uncommitted(&mut instrs_owned, fr1, true, has_fpu, std::ptr::null_mut()) {
        Some(_func_id) => println!("compile_region_uncommitted: OK (produced a FuncId — this region compiles cleanly against the current codegen)"),
        None if codegen.last_compile_ran_out_of_memory() => {
            println!("compile_region_uncommitted: arena ran out of memory (not a real decline — retry with a fresh arena)");
        }
        None => {
            #[cfg(feature = "developer")]
            let reason = if codegen.last_decline_was_verifier_error() {
                "Cranelift verifier rejected the generated IR (real codegen bug)"
            } else {
                "no emitter for some visited instruction (analyzer/codegen table gap)"
            };
            #[cfg(not(feature = "developer"))]
            let reason = "no emitter for some visited instruction, or a Cranelift verifier rejection \
                (build with --features jitv2,developer for the specific breakdown)";
            println!("compile_region_uncommitted: DECLINED — {}", reason);
        }
    }
}
