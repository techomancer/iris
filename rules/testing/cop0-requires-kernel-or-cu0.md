# CP0 instructions must be gated on Kernel-or-CU0

`exec_cop0` (`src/mips_exec.rs`) is the single dispatch point for MFC0/DMFC0/MTC0/DMTC0/
TLBR/TLBWI/TLBWR/TLBP/ERET/WAIT. It had **no privilege check at all**, so a user-mode
process could execute `MTC0 $x, Status` and write itself directly into kernel mode.

This was a strictly larger hole than the stale-`translate_fn` bug that
`rules/testing/cp0-status-writes-must-resync-privilege-state.md` describes: that one needed
stale state and was permissive-only for KUSEG; this one needed nothing at all. An
adversarial review of this subsystem rated the `translate_fn` issue CRITICAL and **missed
this entirely**.

## The rule

CP0 is usable iff `Status.CU0 == 1` **or** the CPU is in Kernel mode. Supervisor gets **no**
implicit access on R4000/R4400 — it is a segment privilege level, not a coprocessor-0 one.

```rust
let cp0_usable = match self.core.get_privilege_mode() {
    PrivilegeMode::Kernel => true,
    _ => (self.core.cp0_status & STATUS_CU0) != 0,
};
if !cp0_usable { return self.cpu_unusable(0); }
```

`cpu_unusable(0)` is the right helper — CE=0 names coprocessor 0 in `Cause`.

`exec_cache` already had this idiom and is the model. Its comment used to read "must be
kernel or supervisor, or CU0 set", which contradicted its own `match` (Supervisor falls in
the `_` arm and needs CU0). Code was right, comment was wrong; both now say the same thing.

## Why one edit covers every engine

jitv2 classifies the whole `OP_COP0` major opcode `Excluded` (`analyzer.rs`), so compiled
code never emits CP0 semantics of its own — it retires these words through
`interp_fallback_fn` into this same `exec_cop0`. There is no second copy to keep in sync.

## Blast radius: nothing broke

- **cpu-tests / bench**: `cpu-tests/harness/start.S` sets `ST_CU0` explicitly *and* leaves
  KSU=0. Doubly safe. Verified: 2101 passed / 61 failed both with and without the gate —
  the 61 are pre-existing FPU/cache failures, unrelated.
- **Rust unit tests**: every COP0-executing test is in Kernel at execution time, because
  `reset` leaves `Status.ERL` set and EXL|ERL forces Kernel.
  `test_nutlb_kernel_entry_unreachable_from_user` looks like a counterexample but is not:
  its `MTC0 Status` runs while KSU is still 0, and it executes no COP0 afterward.
- **gdb / monitor / MCP**: use `write_cp0` directly, never `exec_cop0`. Unaffected.

## Latent fragility

`equiv_test::benign_excluded_mtc0` and friends pass *only* because the seeded executor
inherits reset's `ERL`. A future helper that clears ERL and sets KSU=USER without CU0 would
turn every use of it into EXC_CPU and produce a confusing cascade of failures in tests that
have nothing to do with privilege. Its doc comment now says so.

## Test

`test_cop0_requires_kernel_or_cu0` — from genuine user mode with CU0 clear, `MTC0 Status`
must raise EXC_CPU with `Cause.CE == 0` and must **not** change KSU; with CU0 set the same
instruction is permitted. Compare only KSU, not the whole Status word: exception delivery
sets EXL, so a whole-word comparison would fail for the wrong reason. Verified to fail with
the gate removed.
