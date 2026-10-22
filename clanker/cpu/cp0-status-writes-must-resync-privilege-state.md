# Every `cp0_status` write must call `resync_privilege_state()`

`MipsExecutor` caches state derived from CP0 Status:

- `translate_fn` — one of six monomorphised translators (privilege × 32/64-bit)
- `fpr_read_*` / `fpr_write_*` — six accessors selected by `STATUS_FR`
- the untagged translation caches (nanotlb, nutlb, jitv2 `pcp`), which carry no
  permission bits of their own

Before this was unified, these were re-derived **only** from `MipsCore::write_cp0` reg 12
via `status_changed_cb`. Every site that wrote the field directly left them stale.

**The API.** `MipsExecutor::resync_privilege_state()` does all three;
`set_cp0_status(new)` stores the word and then resyncs. `power_on` is the reference shape.
Both are `pub` because some callers (`restore_state_digest`, `CoreSnapshot::restore_into`)
sit outside the main `impl` block.

**Flushing is not a substitute for re-deriving.** This is the trap. The old `exec_eret`
comment claimed the nanotlb/nutlb flush was "what makes ERET safe here". It is not: the
flush *guarantees* the next access misses and therefore calls `translate_fn`. A stale
`translate_fn` is not bypassed by a flush, it is guaranteed to be consulted.

## Why the bug was invisible, and why it still mattered

`get_privilege_mode()` returns `Kernel` whenever `EXL|ERL` is set, *before* reading KSU.
Kernels use the mandatory MIPS idiom: write Status with the target KSU while EXL is still
1, then `ERET` clears EXL. So the callback fires while privilege still reads Kernel and
installs `translate_32_kernel`; ERET then clears EXL without re-deriving.

**Consequence: `translate_32_user` was never installed during a real IRIX boot.** The only
place it was ever live is `test_nutlb_kernel_entry_unreachable_from_user`, which clears
EXL *and* ERL in the same MTC0 that sets KSU=USER — a state IRIX never writes.

And a stale kernel translator is *byte-identical* to the user one for all legitimate user
memory: `translate_32bit_impl` consults `PRIV` only for segments 4/5/6/7
(KSEG0/KSEG1/KSSEG/KSEG3). Segments 0..=3 (KUSEG — the whole 2 GB user space) ignore
`PRIV` and go straight to `tlb_translate_impl`, which takes no privilege parameter at all;
MIPS TLB entries carry no user/kernel bit (only V/D/G/ASID). So the exposure was
permissive-only: a user access to ≥0x80000000 succeeding where it should raise ADEL/ADES.
No correct guest can observe it, and it cannot break a boot.

**The real hazard was the inverse direction.** Had `translate_32_user` ever been live when
an exception fired, `deliver_exception_at` sets EXL and points PC at the handler vector
(KSEG0/KSEG1) by direct field write. The handler's own first fetch would take a spurious
ADEL — and since EXL is already set, `was_exl` suppresses the EPC update, so the guest
never recovers: an unrecoverable double-fault hang. The code was correct only by
coincidence. That is the reason to fix it.

Severity: **Medium** (latent hazard + fidelity gap), not the "CRITICAL privilege bypass"
an adversarial review claimed.

**One genuinely live bug fixed:** soft reset. `core.reset(true)` sets
`cp0_status = BEV|ERL` and `pc = 0xBFC00000` (KSEG1). Without a resync, a soft reset taken
in user mode leaves the user translator installed and faults on the reset vector fetch
itself. Survived only because IRIX takes soft resets from kernel mode.

## Site table

| Site | Status |
|---|---|
| `write_cp0` reg 12 (MTC0/DMTC0) | via `status_changed_cb` → resync |
| `exec_eret` (clears ERL/EXL) | **fixed** — was flush-only |
| `handle_exception` / `_at` / `_syscall` | **fixed** — were flush-only |
| step-preamble soft reset | **fixed** — had nothing (live bug) |
| snapshot TOML `load_state` | **fixed** — had nothing |
| gdb `write_regs` / `write_reg` | **fixed** — had nothing |
| lockstep `CoreSnapshot::restore_into` | **fixed** — had nothing |
| `power_on` | collapsed onto the API (also gained the nanotlb/pcp flush it omitted) |
| `restore_state`, `restore_state_digest` | collapsed onto the API |
| `MipsExecutor::new` | explicit, unchanged |

## Constraints to preserve

- **`deliver_exception_at` / `deliver_exception` stay free functions on `&mut MipsCore`.**
  `bin/jitv2_verify.rs` calls them with no executor at all, so the resync cannot move
  inside them — it belongs in the three executor wrappers.
- **Do not add a `MipsCore::set_cp0_status()` that fires the callback.**
  `install_status_cb` stores `self as *mut Self`; `MipsExecutor` is unpinned and is moved
  at least twice between `new()` and `Arc::new(Mutex::new(..))` in `MipsCpu::new`, which is
  why the install sits there. Installing earlier records a dead stack frame. It would also
  fail silently on the ~35 directly-constructed test executors, where the callback is
  `None`.
- **jitv2 never compiles COP0** (`analyzer.rs`: all of `OP_COP0` → `Classify::Excluded`),
  so ERET/MTC0 always run in the interpreter and `jit-v2-design.md`'s "Status is a region
  constant" invariant — load-bearing for the region-wide CU1/FR guard — is preserved.
  Do not make Status mutable inside a compiled region.
- The default-off interp-fallback path calls the real `exec_eret`, so it is covered.

## Tests

Each was verified to fail when *only* its own fix is reverted — none is a tautology:

- `test_eret_to_user_mode_refuses_kseg0` — probes KSEG0 *before* the ERET too, so it
  cannot pass vacuously.
- `test_exception_from_user_accepts_kseg0` — the mirror; this is the one that disarms the
  double-fault landmine.
- `test_soft_reset_from_user_fetches_reset_vector` — drives the real `step_preamble` by
  setting bit 63 of `core.hot.interrupts`.

`nanotlb-asid-mutation-flush.md`'s list of flush sites reads as a completed safety
argument but was only ever complete for *flushing*, never for `translate_fn`/`fpr_mode`
re-derivation. It now points at `resync_privilege_state`.
