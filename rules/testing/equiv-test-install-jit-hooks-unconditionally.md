# `run_multipage` must install JIT hooks on the interpreter arm too

**Symptom.** Under `--features jitv2,jitv2_lockstep`, the whole test *process* aborted
(SIGABRT, not a normal test failure) on
`full_mutex_lock_cas_call_chain_matches_interpreter`:

```
panicked at src/mips_core.rs:1029:
  jitv2: write hook called before MipsExecutor::install_jit_hooks
```

Because the hook is an `unsafe extern "C"` fn, its panic aborts rather than unwinding, so
one test took down every other test in the binary — which is why this looked like a mass
failure rather than a single bad test.

**Cause.** `equiv_test::run_multipage` installed hooks only on the JIT arm:

```rust
if jit {
    exec.jitv2_inline_compile = true;
    exec.install_jit_hooks();       // <-- interpreter arm never got them
}
for _ in 0..steps { exec.step_jit(); }
```

But **both** arms drive `step_jit()`, and under `jitv2_lockstep` the interpreter arm still
executes compiled code (that is the whole point — the JIT runs alongside the interpreter
for the per-instruction comparison). So a store instruction on the "interpreter" arm
reached `core.write64_fn`, still pointing at `jit_hooks_not_installed_write64`.

The test only tripped it because it is one of the few multipage fixtures containing real
stores (`sd a0, 0(sp)` in its `mutex_lock`).

**Fix.** Install unconditionally; `install_jit_hooks` only writes fn pointers and is
correct for either arm. Keep `jitv2_inline_compile` gated on `jit` — that flag, not the
hooks, is what distinguishes the two arms.

```rust
exec.install_jit_hooks();
if jit { exec.jitv2_inline_compile = true; }
```

**Rule.** Any equiv-test helper that calls `step_jit()` needs the hooks installed,
regardless of whether that arm is nominally "the interpreter". `jitv2_lockstep` erases the
distinction. Gate the *compile* flag, never the hooks.

## Still-failing lockstep tests (pre-existing, unrelated)

After the above, `--features jitv2,jitv2_lockstep -- --test-threads=1` is 831 passed / 2
failed. Both reproduce on older commits with only this hook fix applied, so neither is
caused by it:

- `lockstep_fpu_trunc_w_s_non_integer_value_sets_inexact` — the known conversion-under-
  lockstep bug (JIT writes 0 for CVT/TRUNC and drops FCSR Inexact). Same family as the
  ~12 CVT equiv failures that appear only in the `jitv2 + jitv2_lockstep` combination.
- `full_mutex_lock_cas_call_chain_matches_interpreter` — **FIXED**, see
  `equiv-test-caller-must-preserve-ra.md`. It was the test, not the engine: the synthetic
  caller never saved `ra` around its own `jal`, so it looped forever and never returned.
  Both engines were sampled mid-flight.
