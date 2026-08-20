//! Single source of truth for "does jitv2 have a real emitter for this raw
//! instruction word" — shared by `analyzer.rs` (deciding `Excluded` vs
//! `Sequential` at classify time) and `codegen.rs` (the actual emitter
//! lookup). Kept Cranelift-free like `analyzer.rs` itself: this module only
//! answers `bool`, it never touches `cranelift_*` types or emits IR.
//!
//! Before this module existed, `analyzer::classify()` had its own
//! independent opinion of what's compilable (anything not explicitly
//! excluded fell through to `Sequential`), separate from `codegen.rs`'s
//! `lookup_semantics`/`lookup_cp1_semantics`/`lookup_branch_or_jump`/
//! `lookup_regjump` tables — the two could (and did) drift: several
//! opcodes classify() called `Sequential` had no codegen emitter at all,
//! silently poisoning (declining, not just excluding-in-place) every
//! region that happened to walk through them (`compile_region`'s upfront
//! rejection loop declines the *whole* region if any visited instruction
//! lacks an emitter — see `rules/jitv2/unsupported-instructions.md` for the
//! inventory that motivated this fix). Routing `classify()` through
//! `has_emitter()` instead makes that drift structurally impossible: an
//! opcode without an emitter is `Excluded` (the analyzer never walks past
//! it as a head, so it's a clean region boundary instead of poisoning
//! everything downstream of it) until an emitter actually exists, at which
//! point it becomes `Sequential` automatically — no `analyzer.rs` edit
//! needed when a new emitter lands.
//!
//! The static coverage table lives on `InstrKind::has_jitv2_emitter`
//! (`mips_instr_stats.rs`) — the same flat per-instruction enum `decode_into`
//! classifies through, so jitv2's opcode coverage and the interpreter's own
//! instruction identity map can't drift into two independently-maintained
//! opcode/funct match trees.
//!
//! On top of that static table sits a **runtime per-instruction enable
//! bitmask** (`ENABLED`, one bool per `InstrKind`, `j2 <name> [on|off]`) —
//! this is what actually gates `has_emitter` at runtime, not
//! `has_jitv2_emitter` directly. `has_jitv2_emitter` says "codegen has an
//! emitter for this at all, ever"; `ENABLED` says "...and it's currently
//! allowed to be used," which is what makes the toggle useful for bisecting
//! a live-boot divergence: flip individual opcodes (or a whole `j2 alu
//! off`-style category, which bulk-sets/-clears every `InstrKind` in that
//! category) off one at a time and see which one the divergence tracks.
//! `j2 <category> [on|off]` (`set_category_enabled`) is a bulk setter over
//! the exact same per-instruction table a future `j2 instr <name> [on|off]`
//! will flip one bit at a time — there is only ever one table, categories
//! are just a convenient way to touch many bits of it at once.
//!
//! `InstrKind::has_jitv2_emitter`'s match arms must mirror `codegen.rs`'s
//! `lookup_semantics`/`lookup_cp1_semantics` coverage exactly (same
//! opcode/funct/rs partitioning) — that table is the *coverage* question
//! only ("is there an emitter"), not the emitter itself, and not the runtime
//! toggle.

use crate::mips_instr_stats::{classify_instr, InstrCategory, InstrKind, NUM_INSTR_KINDS};
use std::sync::atomic::{AtomicBool, Ordering};

/// Per-`InstrKind` runtime enable bit, initialized `true` for every kind
/// `InstrKind::has_jitv2_support` covers (emitter-table coverage plus
/// branch/jump/regjump, which have real emitters but live outside
/// `has_jitv2_emitter`'s table — see that method's doc comment) and `false`
/// for everything else (matches pre-toggle behavior exactly — see
/// [`has_emitter`]). A process
/// global, same pattern as `analyzer::FALLBACK_ENABLED`: `classify()`/
/// `has_emitter()` have no other per-call config, and this only ever changes
/// from the monitor console followed by a `j2 flush` so already-compiled
/// regions are rebuilt under the new policy. `OnceLock`, not a `const`
/// array, because the initial `true`/`false` pattern depends on
/// `InstrKind::has_jitv2_emitter()`, which isn't `const fn`.
static ENABLED: std::sync::OnceLock<Box<[AtomicBool]>> = std::sync::OnceLock::new();

fn enabled_table() -> &'static [AtomicBool] {
    ENABLED.get_or_init(|| {
        (0..NUM_INSTR_KINDS)
            .map(|k| {
                // SAFETY: k is in range 0..NUM_INSTR_KINDS, matching the enum's contiguous discriminants.
                let kind: InstrKind = unsafe { std::mem::transmute(k as u16) };
                AtomicBool::new(kind.has_jitv2_support())
            })
            .collect()
    })
}

/// Whether `kind` is currently enabled for jitv2 compilation. This is the
/// runtime toggle, not the static "does an emitter exist" question — see the
/// module doc comment.
pub fn instr_enabled(kind: InstrKind) -> bool {
    enabled_table()[kind as usize].load(Ordering::Relaxed)
}

/// Enable/disable one instruction. Caller is responsible for a `j2 flush`
/// afterward (already-compiled regions are unaffected until recompiled).
/// Enabling a kind `has_jitv2_emitter()` doesn't cover is a no-op in
/// practice (there's no emitter to run), but is not rejected here — the
/// table only tracks "may this be used," not "can this ever work."
pub fn set_instr_enabled(kind: InstrKind, on: bool) {
    enabled_table()[kind as usize].store(on, Ordering::Relaxed);
}

/// Bulk `set_instr_enabled` over every `InstrKind` whose `category()`
/// intersects `category` — `j2 alu|fpu|branch|loadstore|cop0 [on|off]`'s
/// implementation. Touches the exact same per-instruction table a future
/// single-instruction toggle will use, so a category flip and a follow-up
/// single-instruction override compose naturally (flip the category off,
/// then re-enable just the one instruction under test).
pub fn set_category_enabled(category: InstrCategory, on: bool) {
    for k in 0..NUM_INSTR_KINDS {
        // SAFETY: k is in range 0..NUM_INSTR_KINDS, matching the enum's contiguous discriminants.
        let kind: InstrKind = unsafe { std::mem::transmute(k as u16) };
        if kind.category().intersects(category) {
            set_instr_enabled(kind, on);
        }
    }
}

/// True if every `InstrKind` in `category` is currently enabled — used by
/// `j2 <category>` (no on/off argument) to report the category's current
/// state. Categories with mixed state (e.g. after a single-instruction
/// override) report as "off" (conservative: "on" should mean "fully on").
pub fn category_enabled(category: InstrCategory) -> bool {
    (0..NUM_INSTR_KINDS).all(|k| {
        // SAFETY: k is in range 0..NUM_INSTR_KINDS, matching the enum's contiguous discriminants.
        let kind: InstrKind = unsafe { std::mem::transmute(k as u16) };
        !kind.category().intersects(category) || instr_enabled(kind)
    })
}

/// (supported count, enabled count) among `InstrKind`s with jitv2 support whose
/// category intersects `category`. `supported == enabled` means "fully on",
/// `enabled == 0` means "fully off", anything else is a mixed/partial state —
/// `write_status` uses this three-way split to decide whether a category needs
/// per-instruction detail or a one-line summary suffices.
fn category_counts(category: InstrCategory) -> (usize, usize) {
    let mut supported = 0;
    let mut enabled = 0;
    for k in 0..NUM_INSTR_KINDS {
        // SAFETY: k is in range 0..NUM_INSTR_KINDS, matching the enum's contiguous discriminants.
        let kind: InstrKind = unsafe { std::mem::transmute(k as u16) };
        if !kind.has_jitv2_support() || !kind.category().intersects(category) {
            continue;
        }
        supported += 1;
        if instr_enabled(kind) {
            enabled += 1;
        }
    }
    (supported, enabled)
}

const ALL_CATEGORIES: &[(&str, InstrCategory)] = &[
    ("alu", InstrCategory::ALU),
    ("fpu", InstrCategory::FPU),
    ("branch", InstrCategory::BRANCH),
    ("loadstore", InstrCategory::LOADSTORE),
    ("cop0", InstrCategory::COP0),
];

/// Inspection dump for `j2 instrs [category]` (`mips_exec.rs`). Reports one
/// summary line per category ("N/M enabled") by default; a category only gets
/// expanded to a per-instruction breakdown when it's in a mixed state (not
/// uniformly on or off) — a fully-on or fully-off category is already fully
/// described by its summary line, so listing all its instructions individually
/// would just be noise. `filter`, if given, restricts the report to one
/// category and always expands it to per-instruction detail regardless of
/// whether it's mixed (an explicit request for one category is a request to
/// see it in full). `Reserved` and other kinds with no jitv2 support at all
/// are omitted everywhere (they're permanently `Excluded`; toggling them is
/// meaningless).
pub fn write_status(w: &mut dyn std::io::Write, filter: Option<InstrCategory>) -> std::io::Result<()> {
    for &(name, cat) in ALL_CATEGORIES {
        if let Some(f) = filter {
            if cat != f {
                continue;
            }
        }
        let (supported, enabled) = category_counts(cat);
        if supported == 0 {
            continue;
        }
        writeln!(w, "{:<10} {}/{} enabled", name, enabled, supported)?;
        if filter.is_some() || (enabled != 0 && enabled != supported) {
            let mut rows: Vec<(&'static str, bool)> = (0..NUM_INSTR_KINDS)
                .filter_map(|k| {
                    // SAFETY: k is in range 0..NUM_INSTR_KINDS, matching the enum's contiguous discriminants.
                    let kind: InstrKind = unsafe { std::mem::transmute(k as u16) };
                    if !kind.has_jitv2_support() || !kind.category().intersects(cat) {
                        return None;
                    }
                    Some((kind.name(), instr_enabled(kind)))
                })
                .collect();
            rows.sort_unstable_by_key(|&(name, _)| name);
            for (iname, on) in rows {
                writeln!(w, "    {:<14} {}", iname, if on { "on" } else { "off" })?;
            }
        }
    }
    Ok(())
}

/// True if `codegen.rs` has a real emitter for `raw` *and* it's currently
/// enabled (see the module doc comment) — a plain data-flow instruction
/// (`lookup_semantics`/`lookup_cp1_semantics`), a branch/jump
/// (`lookup_branch_or_jump`), or a register-indirect jump
/// (`lookup_regjump`). Callers that already know `raw` is a branch/jump/
/// regjump via `analyzer::classify`'s own dispatch (`Classify::Branch`/
/// `Jump`/`RegJump`) don't call this — those are handled by construction and
/// consult [`instr_enabled`] directly instead (there's no per-instruction
/// emitter-coverage question for them, only the runtime toggle).
pub fn has_emitter(raw: u32) -> bool {
    let op = ((raw >> 26) & 0x3F) as u8;
    let rs = ((raw >> 21) & 0x1F) as u8;
    let rt = ((raw >> 16) & 0x1F) as u8;
    let funct = (raw & 0x3F) as u8;
    let kind = classify_instr(op, rs, rt, funct);
    kind.has_jitv2_emitter() && instr_enabled(kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mips_isa::*;

    fn r_type(op: u32, rs: u32, rt: u32, rd: u32, sa: u32, funct: u32) -> u32 {
        (op << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct
    }
    fn i_type(op: u32, rs: u32, rt: u32, imm: u16) -> u32 {
        (op << 26) | (rs << 21) | (rt << 16) | imm as u32
    }

    #[test]
    fn plain_addu_has_emitter() {
        assert!(has_emitter(r_type(OP_SPECIAL, 1, 2, 3, 0, FUNCT_ADDU)));
    }

    #[test]
    fn addiu_has_emitter() {
        assert!(has_emitter(i_type(OP_ADDIU, 1, 2, 5)));
    }

    #[test]
    #[cfg(feature = "mips4")]
    fn movz_has_emitter() {
        assert!(has_emitter(r_type(OP_SPECIAL, 1, 2, 3, 0, FUNCT_MOVZ)));
    }

    #[test]
    #[cfg(not(feature = "mips4"))]
    fn movz_has_no_emitter_without_mips4() {
        // MOVZ is MIPS IV; without the feature it must fall back to the
        // interpreter so Reserved Instruction can be raised (see
        // has_jitv2_emitter's mips4 gate in mips_instr_stats.rs).
        assert!(!has_emitter(r_type(OP_SPECIAL, 1, 2, 3, 0, FUNCT_MOVZ)));
    }

    #[test]
    fn teq_has_emitter() {
        assert!(has_emitter(r_type(OP_SPECIAL, 1, 2, 0, 0, FUNCT_TEQ)));
    }

    #[test]
    fn tgei_has_emitter() {
        assert!(has_emitter(i_type(OP_REGIMM, 1, RT_TGEI, 5)));
    }

    #[test]
    fn prefx_has_no_emitter_yet() {
        // PREFX is COP1X-encoded and, unlike plain PREF, exec_prefx checks
        // STATUS_CU1 and raises cpu_unusable if it's clear — a genuine
        // COP0-adjacent side effect this codebase's hard-no on
        // privilege/COP0-touching instructions excludes from jitv2 for now.
        assert!(!has_emitter(r_type(OP_COP1X, 1, 2, 3, 4, FUNCT_PREFX)));
    }

    #[test]
    fn daddi_has_emitter() {
        assert!(has_emitter(i_type(OP_DADDI, 1, 2, 5)));
    }

    #[test]
    fn daddiu_has_emitter() {
        assert!(has_emitter(i_type(OP_DADDIU, 1, 2, 5)));
    }

    #[test]
    fn lwl_has_emitter() {
        assert!(has_emitter(i_type(OP_LWL, 1, 2, 0)));
    }

    #[test]
    fn swl_has_emitter() {
        assert!(has_emitter(i_type(OP_SWL, 1, 2, 0)));
    }

    #[test]
    fn cop1x_has_no_emitter_yet() {
        assert!(!has_emitter(r_type(OP_COP1X, 1, 2, 3, 4, FUNCT_MADD_S)));
    }

    #[test]
    fn cp1_add_s_has_emitter() {
        assert!(has_emitter(r_type(OP_COP1, RS_S, 2, 3, 0, FUNCT_FADD)));
    }

    #[test]
    fn cp1_movz_fmt_has_no_emitter_yet() {
        assert!(!has_emitter(r_type(OP_COP1, RS_S, 2, 3, 0, FUNCT_FMOVZ)));
    }

    #[test]
    fn mfc1_has_emitter() {
        assert!(has_emitter(r_type(OP_COP1, RS_MFC1, 2, 3, 0, 0)));
    }

    #[test]
    fn cop0_has_no_emitter() {
        // OP_COP0 is excluded at the analyzer level entirely (not routed
        // through has_emitter at all in practice), but has_emitter must
        // still correctly say "no" for it — nothing in either lookup table
        // matches OP_COP0.
        assert!(!has_emitter(r_type(OP_COP0, RS_MFC0, 2, 3, 0, 0)));
    }

    #[test]
    fn branch_and_regjump_report_false_here_by_design() {
        // BEQ/JR are handled by analyzer::classify's own Branch/RegJump
        // dispatch before has_emitter would ever be consulted for them —
        // has_emitter only answers the Sequential-vs-Excluded question for
        // everything else, so it correctly says "no" for these (codegen's
        // lookup_branch_or_jump/lookup_regjump are the real answer for
        // that different question).
        assert!(!has_emitter(i_type(OP_BEQ, 1, 2, 0)));
        assert!(!has_emitter(r_type(OP_SPECIAL, 1, 0, 0, 0, FUNCT_JR)));
    }

    // The tests below mutate the process-global ENABLED table, so they must not run
    // concurrently with each other (or with anything else touching it) — same
    // one-mutex-per-module discipline as analyzer::FALLBACK_TEST_LOCK.
    static TOGGLE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn category_toggle_bulk_sets_and_clears() {
        let _lock = TOGGLE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Note: category_enabled(ALU) is *not* true by default — Nop/Syscall/Break
        // are ALU-category but exception-only (never dispatched through codegen at
        // all), so "fully on" was never the starting state for this category. Check
        // individual supported kinds instead.
        assert!(instr_enabled(InstrKind::Addu));
        assert!(instr_enabled(InstrKind::Sll));

        set_category_enabled(InstrCategory::ALU, false);
        assert!(!instr_enabled(InstrKind::Addu));
        assert!(!instr_enabled(InstrKind::Sll));
        assert!(!category_enabled(InstrCategory::ALU));
        // FPU untouched by an ALU toggle.
        assert!(instr_enabled(InstrKind::Fadd_s));

        set_category_enabled(InstrCategory::ALU, true);
        assert!(instr_enabled(InstrKind::Addu));
        assert!(instr_enabled(InstrKind::Sll));
    }

    #[test]
    fn single_instruction_override_composes_with_category_toggle() {
        let _lock = TOGGLE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_category_enabled(InstrCategory::ALU, false);
        assert!(!instr_enabled(InstrKind::Addu));

        // Re-enable just one instruction under an otherwise-disabled category.
        set_instr_enabled(InstrKind::Addu, true);
        assert!(instr_enabled(InstrKind::Addu));
        assert!(!instr_enabled(InstrKind::Sll));
        // Mixed state within the category reports as "off" (conservative).
        assert!(!category_enabled(InstrCategory::ALU));

        set_category_enabled(InstrCategory::ALU, true);
        assert!(instr_enabled(InstrKind::Sll));
    }

    #[test]
    fn disabled_instruction_makes_has_emitter_false() {
        let _lock = TOGGLE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let raw = r_type(OP_SPECIAL, 1, 2, 3, 0, FUNCT_ADDU);
        assert!(has_emitter(raw));
        set_instr_enabled(InstrKind::Addu, false);
        assert!(!has_emitter(raw));
        set_instr_enabled(InstrKind::Addu, true);
        assert!(has_emitter(raw));
    }

    #[test]
    fn disabling_branch_category_makes_analyzer_exclude_branches() {
        // Branch/Jump/RegJump never call has_emitter (analyzer::classify resolves
        // them by construction) — this is the toggle's other half, exercised through
        // analyzer::classify directly, confirming the branch_category_gate wiring.
        let _lock = TOGGLE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let beq = i_type(OP_BEQ, 1, 2, 0);
        let jr = r_type(OP_SPECIAL, 1, 0, 0, 0, FUNCT_JR);
        let j = i_type(OP_J, 0, 0, 0);

        assert_ne!(crate::jitv2::analyzer::classify(beq, 5, 0), crate::jitv2::analyzer::Classify::Excluded);
        assert_ne!(crate::jitv2::analyzer::classify(jr, 5, 0), crate::jitv2::analyzer::Classify::Excluded);
        assert_ne!(crate::jitv2::analyzer::classify(j, 5, 0), crate::jitv2::analyzer::Classify::Excluded);

        set_category_enabled(InstrCategory::BRANCH, false);
        assert_eq!(crate::jitv2::analyzer::classify(beq, 5, 0), crate::jitv2::analyzer::Classify::Excluded);
        assert_eq!(crate::jitv2::analyzer::classify(jr, 5, 0), crate::jitv2::analyzer::Classify::Excluded);
        assert_eq!(crate::jitv2::analyzer::classify(j, 5, 0), crate::jitv2::analyzer::Classify::Excluded);

        set_category_enabled(InstrCategory::BRANCH, true);
        assert_ne!(crate::jitv2::analyzer::classify(beq, 5, 0), crate::jitv2::analyzer::Classify::Excluded);
    }

    #[test]
    fn write_status_summarizes_fully_on_and_expands_mixed() {
        let _lock = TOGGLE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_category_enabled(InstrCategory::ALU, true);
        set_category_enabled(InstrCategory::BRANCH, true);

        // Branch is fully on: summary line only, no per-instruction breakdown.
        let mut out = Vec::new();
        write_status(&mut out, None).unwrap();
        let text = String::from_utf8(out).unwrap();
        let branch_line = text.lines().find(|l| l.starts_with("branch")).unwrap();
        assert!(branch_line.contains("enabled"));
        // No indented "jr"/"beq" rows should immediately follow a fully-on category —
        // check there's no indented line naming a branch instruction anywhere.
        assert!(!text.contains("    jr "));

        // Disable exactly one ALU instruction: ALU becomes mixed, so its section
        // must expand to per-instruction detail, including the disabled one.
        set_instr_enabled(InstrKind::Addu, false);
        let mut out = Vec::new();
        write_status(&mut out, None).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("    addu"));
        assert!(text.contains("off"));

        set_instr_enabled(InstrKind::Addu, true);
    }

    #[test]
    fn write_status_filter_always_expands() {
        let _lock = TOGGLE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_category_enabled(InstrCategory::FPU, true);
        // FPU is fully on, but an explicit filter should still expand it.
        let mut out = Vec::new();
        write_status(&mut out, Some(InstrCategory::FPU)).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("    fadd.s"));
        assert!(!text.contains("alu"));
    }
}
