# MOVF/MOVT are CP1 accesses with an integer encoding

`MOVCI` (MIPS IV `MOVF`/`MOVT`, `SPECIAL` funct `0x01`) conditionally moves a GPR
based on an **FP condition code read out of FCSR**. That makes it a coprocessor-1
access: MIPS64 lists Coprocessor Unusable for it, and it must fault when
`Status.CU1` is clear.

Neither engine checked. `exec_movci` read `get_fpu_cc(cc)` directly, and
`emit_movci` loaded `fpu_fcsr` directly. Fixed in both.

## Why it was easy to miss

Everything else that touches CP1 state is `OP_COP1`-encoded and therefore routed
through a gate that already checks CU1:

- Interpreter: each CP1 handler opens with
  `if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }`
  — ~40 copies, including `exec_fmovcf_s`, the COP1-*encoded* `MOVF.fmt`.
- JIT: `emit_cp1_cu1_guard` is emitted on the `lookup_cp1_semantics` dispatch
  arm, unconditionally, for every entry in that table.

MOVCI is dispatched from the **integer** `SPECIAL` funct table instead, so it
reaches neither. `emit_movci`'s own doc comment even noted the routing
("registered in `lookup_semantics`, not `lookup_cp1_semantics`") without drawing
the consequence.

The guard belongs **inside `emit_movci`**, not at the `lookup_semantics` dispatch
site: MOVCI is the only entry in that table touching CP1 state, and guarding the
whole table would be wrong for the ~150 genuinely-integer emitters it shares with.
It must be the **first** IR emitted — `emit_cp1_cu1_guard` terminates the current
block and switches to a fresh continue block, so anything emitted before it lands
in the wrong block.

## Why no test caught it

**Both engines omitted the check identically.** The equivalence and lockstep
harnesses compare the JIT against the interpreter, so a spec deviation the two
*share* is structurally invisible to them. No amount of equivalence testing finds
this class of bug; only a spec/manual comparison does.

## The test trap this creates

`movci_matches_interpreter_across_all_cc_and_tf_combinations` seeded
`fpu_fcsr = 0` and never set CU1. After the fix, **both engines take the CpU path
identically** — so the test still passed while its entire cc/tf/cc_actual sweep
became dead code. Verified empirically: it went green with every MOVCI in it
faulting before executing.

A "still passes" signal after adding a fault path is not reassurance; it is the
symptom to check for. The test now sweeps `cu1 ∈ {false, true}` and asserts
`took_exception == !cu1` on every iteration, so it can never silently go vacuous
again.

## MOVZ/MOVN need no such check

Verified: they read only `core.gpr`, never FCSR or any FPR, and MIPS64 lists no
Coprocessor Unusable for them. Same for plain `PREF` (unlike COP1X-encoded
`PREFX`, which is a genuine CP1 access and already checks).

Related: `analyzer::is_fpu_instruction` matches only `OP_COP1`/`OP_COP1X`/the CP1
load-stores, so a region containing only MOVCI gets `has_fpu == false` and no
`emit_fr_mode_guard`. That is **correct** — MOVCI reads FCSR, whose layout is
fixed, not FR-packed FPRs — but it is a non-obvious asymmetry now that MOVCI does
get the CU1 guard.
