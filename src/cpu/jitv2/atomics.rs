//! LL/SC may stay inside a compiled region.
//!
//! `ll`, `lld`, `sc` and `scd` classify as `Excluded`, and an excluded word
//! ended the region: every one cost an exit to the dispatcher and a re-entry.
//! Measured with cpu-tests/jitcov on IP28 / R10000, that made them
//! *slower* compiled than interpreted (0.6-0.8x),
//! and that kernel's idle loop is an
//! `lld`/`scd` compare-and-swap behind a cache barrier, so it paid that on
//! every pass.
//!
//! They now stay in the region as **interpreter-fallback heads**, exactly as
//! `jitv2::cop0` keeps the safe CP0 subset: the head calls the interpreter's
//! own `exec_ll`/`exec_sc`, so the link bit, LLAddr, address and TLB
//! exceptions, and the rule that any exception between the pair breaks the
//! link all behave exactly as interpreted. There is no native emitter and there
//! should not be one: a second implementation of the link state would drift.
//!
//! Against the two things a region bakes in (see `jitv2::cop0`): neither
//! instruction writes Status (so the FR-mode guard is unaffected) or Cause (so
//! the pending-interrupt sample is unaffected).
//!
//! One consequence to know: an `sc` that stores into the very page it is
//! executing from goes through the interpreter's store path, which bumps the
//! page generation, but the rest of the running region is not re-checked until
//! it exits. Self-modifying code through `sc` on its own page is not something
//! any guest here does.

use crate::cpu::mips_isa::*;

/// Whether this word is an `ll`/`lld`/`sc`/`scd` that may stay in a region as
/// an interpreter-fallback head. Honours the per-instruction enable table, so
/// `j2 loadstore off` (or one kind switched off) puts them back to ending the
/// region, for bisecting a divergence on a live boot.
pub fn stays_in_region(raw: u32) -> bool {
    let op = (raw >> 26) & 0x3F;
    if !matches!(op, OP_LL | OP_LLD | OP_SC | OP_SCD) {
        return false;
    }
    let kind = crate::cpu::mips_instr_stats::classify_instr(
        op as u8,
        ((raw >> 21) & 0x1F) as u8,
        ((raw >> 16) & 0x1F) as u8,
        (raw & 0x3F) as u8,
    );
    crate::cpu::jitv2::opcode_support::instr_enabled(kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i_type(op: u32, rs: u32, rt: u32, imm: u32) -> u32 {
        (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)
    }

    #[test]
    fn the_four_atomics_stay_in_the_region() {
        for op in [OP_LL, OP_LLD, OP_SC, OP_SCD] {
            assert!(stays_in_region(i_type(op, 5, 8, 16)), "op {op:#x}");
        }
    }

    #[test]
    fn nothing_else_is_admitted_here() {
        for op in [OP_LW, OP_SW, OP_LD, OP_SD, OP_CACHE, OP_COP0] {
            assert!(!stays_in_region(i_type(op, 5, 8, 16)), "op {op:#x}");
        }
    }
}
