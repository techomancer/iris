//! The interpreter's hot paths must not regress in a release binary.
//!
//! Reads the symbols and machine code of a built `iris` (lightning, fat LTO:
//! without LTO, calls into std go through the GOT and the control test reports
//! the binary unreadable) and checks that nothing in the interpreter's hot code
//! calls a formatting function. x86-64 ELF only. The tests are ignored so a
//! plain `cargo test` skips them visibly; CI runs them after the release build:
//!
//!     cargo test -p iris-checks -- --ignored
//!
//! The binary is IRIS_HOT_PATH_BINARY, else target/release/iris. See
//! clanker/perf/hot-path-inlining-must-be-explicit.md.

use iced_x86::{Decoder, DecoderOptions, Mnemonic, OpKind};
use object::{Architecture, Object, ObjectSection, ObjectSymbol, SymbolKind};
use regex::Regex;
use std::sync::OnceLock;

// Methods that print by design. Formatting calls in them are expected, and
// the first two double as the control: if none shows a formatting call, the
// disassembly cannot show them and a clean result means nothing.
const CONTROL: &str = r"MipsExecutor.*::(report_[a-z_]*error|handle_exception)";
const COLD: &str = r"::(report_[a-z_]*error|fault_note|post_refused_store|idle_profile_report|backtrace_from|jitv2_monitor_pfn|handle_exception|debug_|monitor_)";
const FMT_CALL: &str = r"<(std::io::stdio::_e?print|.*::write_fmt|core::fmt::|alloc::fmt::format|<[ui](8|16|32|64|size)>::_fmt)";
// A hit-path fetch must inline, so it has no symbol of its own.
const OUT_OF_LINE_FETCH: &str = r"(CpuCache|ShadowCache|DecodeCache).*>::fetch$";
// #[inline(never)] in the source, so they exist whatever the inliner decides.
// If one is missing a type was renamed and the fetch check passes vacuously.
const ANCHORS: [&str; 3] = ["fill_l1i_line", "index_store_data", "fetch_slow"];

struct Call {
    /// `ADDR <name>:`, the form COLD and CONTROL match.
    caller: String,
    site: u64,
    /// `<symbol+0xoff>`, the form FMT_CALL matches.
    target: String,
}

struct Binary {
    /// Every symbol, demangled.
    names: Vec<String>,
    /// Direct calls and jumps made by MipsExecutor code.
    calls: Vec<Call>,
}

fn demangle(raw: &str) -> String {
    format!("{:#}", rustc_demangle::demangle(raw))
}

fn load() -> Binary {
    let path = std::env::var("IRIS_HOT_PATH_BINARY").unwrap_or_else(|_| {
        let target = std::env::var("CARGO_TARGET_DIR")
            .unwrap_or_else(|_| format!("{}/../target", env!("CARGO_MANIFEST_DIR")));
        format!("{target}/release/iris")
    });
    let data = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("{path}: {e}\nbuild it first, or set IRIS_HOT_PATH_BINARY")
    });
    let elf = object::File::parse(&*data).expect("parse the ELF binary");
    assert_eq!(elf.architecture(), Architecture::X86_64, "only x86-64 is supported");

    let names = elf.symbols().filter_map(|s| s.name().ok()).map(demangle).collect();

    // Functions by address, to resolve each branch target to its symbol.
    let mut funcs: Vec<_> = elf
        .symbols()
        .filter(|s| s.kind() == SymbolKind::Text && s.size() > 0)
        .filter_map(|s| Some((s.address(), s.size(), demangle(s.name().ok()?), s.section_index()?)))
        .collect();
    funcs.sort_by_key(|f| f.0);
    funcs.dedup_by_key(|f| f.0);

    let mut calls = Vec::new();
    for (addr, size, name, sec) in &funcs {
        if !name.contains("MipsExecutor") {
            continue;
        }
        let sec = elf.section_by_index(*sec).unwrap();
        let off = (addr - sec.address()) as usize;
        let code = &sec.data().unwrap()[off..off + *size as usize];
        for ins in Decoder::with_ip(64, code, *addr, DecoderOptions::NONE) {
            if !matches!(ins.mnemonic(), Mnemonic::Call | Mnemonic::Jmp)
                || ins.op0_kind() != OpKind::NearBranch64
            {
                continue;
            }
            let t = ins.near_branch_target();
            let i = funcs.partition_point(|f| f.0 <= t);
            let Some(callee) = i.checked_sub(1).map(|i| &funcs[i]) else { continue };
            let plus = if callee.0 == t { String::new() } else { format!("+{:#x}", t - callee.0) };
            calls.push(Call {
                caller: format!("{addr:016x} <{name}>:"),
                site: ins.ip(),
                target: format!("<{}{plus}>", callee.2),
            });
        }
    }
    Binary { names, calls }
}

fn binary() -> &'static Binary {
    static B: OnceLock<Binary> = OnceLock::new();
    B.get_or_init(load)
}

#[test]
#[ignore = "needs the release binary"]
fn hit_path_fetch_is_inlined() {
    let re = Regex::new(OUT_OF_LINE_FETCH).unwrap();
    let bad: Vec<_> = binary().names.iter().filter(|n| re.is_match(n)).collect();
    assert!(bad.is_empty(), "a hit-path fetch is out of line: {bad:#?}");
}

#[test]
#[ignore = "needs the release binary"]
fn anchor_functions_exist() {
    for a in ANCHORS {
        let n = format!("::{a}");
        assert!(binary().names.iter().any(|s| s.contains(&n)), "anchor {a} not found");
    }
}

#[test]
#[ignore = "needs the release binary"]
fn disassembly_shows_formatting_calls_in_control_methods() {
    let (ctl, fmt) = (Regex::new(CONTROL).unwrap(), Regex::new(FMT_CALL).unwrap());
    let seen = binary().calls.iter().filter(|c| ctl.is_match(&c.caller) && fmt.is_match(&c.target));
    assert!(seen.count() > 0, "unreadable binary: no formatting call in the control methods");
}

#[test]
#[ignore = "needs the release binary"]
fn hot_code_has_no_formatting_calls() {
    let (cold, fmt) = (Regex::new(COLD).unwrap(), Regex::new(FMT_CALL).unwrap());
    let bad: Vec<_> = binary()
        .calls
        .iter()
        .filter(|c| !cold.is_match(&c.caller) && fmt.is_match(&c.target))
        .map(|c| format!("{}\n   {:x}: call {}", c.caller, c.site, c.target))
        .collect();
    assert!(bad.is_empty(), "{} formatting calls in hot code:\n{}", bad.len(), bad.join("\n"));
}
