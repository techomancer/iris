//! Which of the kernel's CP0 instructions may stay inside a compiled region.
//!
//! `OP_COP0` classifies as [`Classify::Excluded`](super::analyzer::Classify),
//! and before this an excluded word was a hard region boundary: every
//! `mfc0`/`mtc0`/`eret` cut a region in two. That is a large part of why the
//! kernel interpreted so much — a whole-kernel census found **677 COP0 sites**
//! in `unix.B`, and the exception, timer and interrupt paths are dense with
//! them.
//!
//! Admitted words stay in the region as **interpreter-fallback heads** that
//! call the real `exec_cop0`, privilege gate and side effects included. There
//! is deliberately no native CP0 emitter and there must not be one: a second
//! implementation of CP0 would drift from the first. Nothing here changes what
//! a CP0 instruction *does*; it changes only whether the region has to end on
//! it.
//!
//! # What a region bakes in at compile time
//!
//! Exactly two things, and they are the whole safety argument:
//!
//! 1. **FR mode.** `codegen::emit_fr_mode_guard` checks `STATUS_FR` once per
//!    region entry on behalf of every FPR access in the region, which is only
//!    legitimate while nothing in the region can change FR mid-flight. Status
//!    (CP0 register 12) carries FR, so it is absent from [`MTC0_SAFE_REGS`]
//!    and handled separately: `compile_region_uncommitted` declines any region
//!    that combines `has_fpu` with a Status-writing fallback head. A region
//!    with no CP1 instruction has no FPR-access emitter and no guard at all,
//!    so a Status write there has nothing to invalidate — and that is exactly
//!    the shape of the kernel exception and timer paths this exists for.
//!
//! 2. **The pending-interrupt sample.** `emit_pending_interrupt_preamble`
//!    samples `core.hot.interrupts != 0`, where the interpreter's
//!    `step_preamble!` tests `(pending | core.cp0_cause) != 0` and then
//!    `(Cause.IP & Status.IM)`. So the interpreter can deliver on Cause IP
//!    bits that never appear in `hot.interrupts` — the two **software**
//!    interrupts IP0/IP1, writable only by `MTC0 Cause`. That is why Cause
//!    (13) is not in [`MTC0_SAFE_REGS`] despite looking like an obvious safe
//!    candidate: compiling through it would let a region run arbitrarily far
//!    past a software interrupt the interpreter would have delivered at once.
//!
//! Nothing else. Translation is a live nutlb probe on every access, the code
//! cache is keyed on *physical* page, and cache geometry is a machine property
//! no CP0 write changes. Privilege and ASID transitions flush the nutlb
//! through the paths the interpreter already runs
//! (`status_changed_cb` -> `resync_privilege_state`,
//! `handle_cp0_side_effects`' EntryHi/Config flushes) — and those run here too,
//! because the real handler runs.
//!
//! First tried as an experiment, which measured it worth **about 6-7% of
//! boot time**. The Cargo feature and `IRIS_JIT_ATOMIZED` env policy it
//! carried there are deliberately left behind:
//! they existed to A/B the policy while it was being established, and it has
//! been. The `j2 cop0 off` monitor toggle still works for bisecting a live
//! divergence, through the ordinary per-`InstrKind` enable table.

use crate::cpu::mips_isa::*;

/// CP0 registers an `MTC0`/`DMTC0` may write from inside a region.
///
/// Omitted on purpose: **12 (Status)**, which carries `STATUS_FR` and is
/// handled by the `has_fpu` gate in `compile_region_uncommitted` instead;
/// **13 (Cause)**, whose IP0/IP1 software-interrupt bits the compiled
/// preamble cannot see (see the module docs); and the TLB-shape registers
/// 10 (EntryHi) and 16 (Config), which are safe but worthless — no measured
/// site writes them in a hot path.
pub const MTC0_SAFE_REGS: &[u32] = &[0, 2, 3, 4, 5, 6, 9, 11, 14, 17, 18, 19, 20, 28, 29, 30];

/// CP0 register 12. Named because three places have an opinion about it: this
/// module excludes it from [`MTC0_SAFE_REGS`], `compile_region_uncommitted`
/// gates it on `!has_fpu`, and `emit_fr_mode_guard` is the check that gate
/// protects.
pub const CP0_STATUS: u32 = 12;

/// Whether `raw` writes CP0 Status — the one admitted class whose safety
/// rests on a second condition rather than on the write being inert.
/// `compile_region_uncommitted` uses this to decline a region that would
/// combine it with a CP1 instruction.
///
/// Deliberately **not** gated on anything: the blanket `j2 fallback on`
/// toggle admits `MTC0 Status` as a fallback head too, and has done since
/// interpreter fallback landed, so without this check the same stale-FR
/// hazard already existed on that path. One comparison closes it for both.
pub fn writes_cp0_status(raw: u32) -> bool {
    if (raw >> 26) & 0x3F != OP_COP0 {
        return false;
    }
    let rs = (raw >> 21) & 0x1F;
    (rs == RS_MTC0 || rs == RS_DMTC0) && ((raw >> 11) & 0x1F) == CP0_STATUS
}

/// Whether this `OP_COP0` word may stay inside a compiled region as an
/// interpreter-fallback head rather than ending it.
///
/// A *reachability* predicate, not an emitter-coverage one: it deliberately
/// does not go through `opcode_support::has_emitter`, because there is no
/// native emitter for any of these and there must not be one.
/// `compile_region_uncommitted` exempts `is_fallback` words from its
/// must-have-an-emitter rejection loop precisely because they run through the
/// interpreter.
pub fn stays_in_region(raw: u32) -> bool {
    if (raw >> 26) & 0x3F != OP_COP0 {
        return false;
    }
    // The monitor half of the switch: `j2 cop0 off`, or a single InstrKind
    // flipped off, must reach here too, so a divergence found on a live boot
    // can be bisected without relaunching.
    let kind = crate::cpu::mips_instr_stats::classify_instr(
        (OP_COP0 as u8) & 0x3F,
        ((raw >> 21) & 0x1F) as u8,
        ((raw >> 16) & 0x1F) as u8,
        (raw & 0x3F) as u8,
    );
    if !crate::cpu::jitv2::opcode_support::instr_enabled(kind) {
        return false;
    }

    let rs = (raw >> 21) & 0x1F;
    let rd = (raw >> 11) & 0x1F;
    let funct = raw & 0x3F;
    match rs {
        // Reads. `MipsCore::read_cp0` mutates only CP0-internal bookkeeping
        // (`update_random` for Random, the Count memoization) and returns a
        // value; no register's read has an effect outside CP0, so every `rd`
        // is admissible. The Status.CU0 privilege gate is `exec_cop0`'s and
        // still runs — this is a fallback head, not a native emitter.
        RS_MFC0 | RS_DMFC0 => true,
        RS_MTC0 | RS_DMTC0 => {
            if rd == CP0_STATUS {
                // Admitted, but only for a region with no CP1 instruction —
                // a condition `compile_region_uncommitted` enforces, since the
                // walk has not finished computing `has_fpu` when this runs.
                return true;
            }
            MTC0_SAFE_REGS.contains(&rd)
        }
        // Everything else under this `rs` is a TLB op (TLBR/TLBWI/TLBWR/TLBP
        // — out of scope: complex semantics, rare, and a subtle bug there
        // corrupts memory in an unrelated address space minutes later) or
        // WAIT (the idle loop, owned elsewhere).
        RS_TLB => funct == FUNCT_ERET,
        // CFC0/CTC0/BC0 and every unassigned `rs` raise Reserved Instruction.
        // A fallback head would deliver that correctly, but an instruction
        // whose only outcome is an exception is not worth compiling through:
        // it never has a successor to reach.
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cop0(rs: u32, rd: u32, funct: u32) -> u32 {
        (OP_COP0 << 26) | (rs << 21) | (rd << 11) | funct
    }

    #[test]
    fn reads_are_all_admitted() {
        for rd in 0..32u32 {
            assert!(stays_in_region(cop0(RS_MFC0, rd, 0)), "mfc0 rd={rd}");
            assert!(stays_in_region(cop0(RS_DMFC0, rd, 0)), "dmfc0 rd={rd}");
        }
    }

    #[test]
    fn cause_is_never_admitted() {
        // CP0 13. Its IP0/IP1 software-interrupt bits are invisible to the
        // compiled preamble's `hot.interrupts` sample, so a region could run
        // far past an interrupt the interpreter delivers immediately.
        assert!(!stays_in_region(cop0(RS_MTC0, 13, 0)));
        assert!(!stays_in_region(cop0(RS_DMTC0, 13, 0)));
    }

    #[test]
    fn tlb_ops_are_never_admitted_but_eret_is() {
        for funct in [FUNCT_TLBR, FUNCT_TLBWI, FUNCT_TLBWR, FUNCT_TLBP] {
            assert!(!stays_in_region(cop0(RS_TLB, 0, funct)), "funct={funct:#x}");
        }
        assert!(stays_in_region(cop0(RS_TLB, 0, FUNCT_ERET)));
    }

    #[test]
    fn status_is_admitted_here_and_gated_in_codegen() {
        let w = cop0(RS_MTC0, CP0_STATUS, 0);
        assert!(stays_in_region(w), "admitted by this predicate...");
        assert!(writes_cp0_status(w), "...and flagged for compile_region's has_fpu gate");
        // Nothing else claims to write Status.
        assert!(!writes_cp0_status(cop0(RS_MTC0, 9, 0)));
        assert!(!writes_cp0_status(cop0(RS_MFC0, CP0_STATUS, 0)));
    }

    #[test]
    fn non_cop0_words_are_rejected_outright() {
        assert!(!stays_in_region(0));
        assert!(!writes_cp0_status(0));
        // A COP1 word whose rs happens to equal RS_MTC0 is not a CP0 write.
        assert!(!writes_cp0_status((OP_COP1 << 26) | (RS_MTC0 << 21) | (CP0_STATUS << 11)));
    }
}
