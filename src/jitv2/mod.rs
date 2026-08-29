//! JIT v2: physical-page PIC region compiler. See `rules/jitv2/jit-v2-design.md`.

pub mod jitv2;
pub mod comp;
pub mod opcode_support;
pub mod analyzer;
pub mod codegen;
pub mod paged_memory;
#[cfg(feature = "j2wp")]
pub mod pcp_dump;
pub mod equiv_test;

#[cfg(not(feature = "j2wp"))]
pub use jitv2::JitEntry;
pub use jitv2::{
    CompileQueue, CompileRequest, JitFn, JitStats, Jitv2, PageSlot, Pfn, PhysicalCodePage,
    ARENA_RESERVE_SIZE, BITMAP_WORDS, CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES, COMPILE_QUEUE_CAPACITY,
    ENTRIES_PER_PAGE, JITV2_INITIAL_PAGE_CAPACITY, PAGE_SIZE,
    min_calls_before_compile, set_min_calls_before_compile,
};
pub use paged_memory::{PagedArenaMemoryProvider, PagedArenaState};
#[cfg(feature = "developer")]
pub use jitv2::{BatchFlushReason, CodeSizeBucket, RejectReason, REJECT_REASON_COUNT};

/// The jitv2 dirty-page probe — see `rules/jitv2/dirty-cache-page-probe.md`.
/// Absent under `tcache`, which closes that blind spot by construction.
#[cfg(not(feature = "tcache"))]
pub use jitv2::{install_jit_page_probe, clear_jit_page_probe, jit_page_has_dirty_lines};

#[cfg(test)]
mod zz_corpus {
    use crate::jitv2::analyzer::Analyzer;
    use crate::jitv2::codegen::Codegen;
    use crate::jitv2::JitFn;
    use crate::jitv2::jitv2::ENTRIES_PER_PAGE;

    /// Compile every corpus page named in IRIS_CORPUS_LIST and report the
    /// total emitted bytes. Same input set for both builds = an apples-to-
    /// apples measure of emitted code volume on real guest code.
    #[test]
    fn zz_corpus_sizes() {
        let list = match std::env::var("IRIS_CORPUS_LIST") { Ok(v)=>v, Err(_)=>return };
        // Match the real emulator: opt_level is a process-wide static that
        // defaults to `none` under `developer`. Production runs `speed`.
        let speed = std::env::var("IRIS_OPT_SPEED").is_ok();
        Codegen::set_opt_level_speed(speed);
        println!("OPTLEVEL speed={}", speed);
        let names = std::fs::read_to_string(&list).expect("list file");
        let mut total: u64 = 0;
        let mut n_ok = 0u64;
        let mut n_decl = 0u64;
        for name in names.lines() {
            let name = name.trim();
            if name.is_empty() { continue; }
            let bytes = match std::fs::read(name) { Ok(b)=>b, Err(_)=>continue };
            if bytes.len() < ENTRIES_PER_PAGE*4 { continue; }
            let mut pw = [0u32; ENTRIES_PER_PAGE];
            for i in 0..ENTRIES_PER_PAGE {
                pw[i] = u32::from_ne_bytes([bytes[i*4],bytes[i*4+1],bytes[i*4+2],bytes[i*4+3]]);
            }
            // offset encoded in filename: ..._off_XXXX.bin
            let off = name.rsplit("_off_").next()
                .and_then(|t| t.strip_suffix(".bin"))
                .and_then(|t| u16::from_str_radix(t,16).ok())
                .unwrap_or(0);
            if off as usize >= ENTRIES_PER_PAGE { continue; }
            let mut an = Analyzer::new();
            let (walked, ok) = an.walk_bounded(&pw, off, 0x8000_0000u32, usize::MAX);
            if !ok { continue; }
            let mut ins = *walked;
            let mut cg = Codegen::new();
            let f: Option<JitFn> = cg.compile_region(&mut ins, off, true, false);
            if f.is_some() { total += cg.last_code_size() as u64; n_ok += 1; }
            else { n_decl += 1; }
            std::mem::forget(cg);
        }
        println!("CORPUS ok={} declined={} total_bytes={}", n_ok, n_decl, total);
    }
}
