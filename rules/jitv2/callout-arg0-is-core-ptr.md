# JIT→Rust callouts: arg 0 is always the `MipsCore` pointer

## The rule

Every JIT→Rust hook (`MipsCore`'s `read*_fn` / `write*_fn` /
`handle_exception_fn` / `kill_entry_fn` / `interp_fallback_fn` /
`dev_trace_bp_fn` / `lockstep_*_fn` / `fpu_cvt_*_fn` / `fpu_set_mode_fn`)
takes as arg 0 the **`*mut MipsCore` that compiled code was itself entered
with** — `JitFn`'s own arg 0, i.e. `ctx.core_ptr` in codegen.

Hooks that need the surrounding `MipsExecutor<T, C>` recover it themselves
with `mips_exec::exec_from_core`, the Rust `container_of`:

```rust
core.cast::<u8>()
    .sub(std::mem::offset_of!(MipsExecutor<T, C>, core))
    .cast::<MipsExecutor<T, C>>()
```

`offset_of!` asks the compiler for the layout it actually chose, so this is
correct under `repr(Rust)` field reordering — it does *not* assume `core`
sits at offset 0 (it does today, so it compiles to nothing).

## Why — the bug this prevents

There used to be a second, separately-stashed pointer: `MipsCore::jit_ctx`, a
type-erased `*mut MipsExecutor<T,C>` set by `install_jit_hooks`. Each call
site loaded it and passed it as arg 0. Two differently-typed pointers behind
one `*mut c_void` parameter meant every new callout had to pick the right
one, and **picking wrong was silent**.

It happened: `emit_fpu_cvt_to_int_call` passed `core_ptr` where the helper
expected the executor. `jit_cvt_to_int` cast it to `*mut MipsExecutor<T,C>`
and took `&mut exec.core` — so it wrote the converted result into an FPR
array at `self + 2*offsetof(core)`. No fault, no crash; the real `core.fpr`
was simply never updated, surfacing much later as a `jitv2_lockstep`
divergence where CVT/ROUND/TRUNC/CEIL/FLOOR results never landed.

One pointer for every callout removes the choice, and therefore the bug
class.

## Secondary effects

- **One fewer load per callout.** The `jit_ctx` load is gone from ~17 emit
  sites in `codegen.rs`; arg 0 is now a reg-reg move (often elided).
- **`install_jit_hooks` no longer captures an address.** It installs only
  function pointers, so — unlike `interrupts_ptr` — it does *not* require the
  executor to be at its final, stable address. An executor that moves after
  installation stays correct instead of dangling. The boxing in
  `equiv_test`'s `run_jit`/`run_multipage` is now belt-and-braces.
- **Code size:** ~2.6% *less* emitted code than before the change, measured
  over 500 real corpus pages (4,229,600 → 4,120,593 bytes) — but only with
  the `callout_core_arg` copy below. Getting this wrong cost 3% in the other
  direction; see the next section.

## Arg 0 is biased — always unbias through `core_from_arg`

Compiled code does **not** pass a usable pointer. `codegen::callout_core_arg`
adds `CALLOUT_CORE_BIAS` (4) first, and `mips_exec::core_from_arg` subtracts
it back. That function is the single place the contract is undone; every hook
funnels through it, directly or via `exec_from_core`.

Dereferencing a callout's `ctx` raw reads `MipsCore` at a 4-byte offset —
every field lands on the wrong bytes, silently, with no fault. A
`debug_assert!` in `core_from_arg` catches null/garbage but cannot catch an
unbiased-but-valid pointer, so the rule is procedural: **one entry point, no
exceptions.** `jitv2_verify`'s shim uses it too, even though it only wants the
`MipsCore` and never the executor.

### Why the bias exists

Arg 0 carries a `FixedReg(%rdi)` constraint at the call. The same core pointer
is also the region's base register (`0x50(%r14)`, …) and is live across that
call. Handing regalloc one SSA value for both roles puts a single long live
range under a fixed-register conflict at *every* callout — and regalloc2
splits a value only `MAX_SPLITS_PER_SPILLSET` (**2**) times before giving up
and calling `split_into_minimal_bundles`, which spills and reloads at every
use. Real regions have hundreds of calls, so two splits is nowhere near
enough.

A biased pointer is a distinct value born at the call and dead immediately
after, so it never joins the base pointer's spillset and never reaches that
cliff. Emitted arg 0 is a single `leaq 4(%r14), %rdi`.

**A `+0` copy does not work.** The `opt_level=speed` optimizer folds it back
into the same value — verified byte-identical output. Only a nonzero bias
survives. The magnitude is irrelevant (1 and 4 measured within noise); 4 keeps
the biased pointer inside `MipsCore` so it never points outside its own
allocation.

### Measurements (500 real corpus pages, `opt_level=speed`)

| | total bytes | vs baseline |
|---|---|---|
| before the change | 4,022,594 | — |
| passing `core_ptr` directly | 4,117,029 | +2.35% |
| **with the bias** | **3,945,940** | **−1.91%** |

Worst single page:

| | `movq` | total instrs | spill stores | reloads | size |
|---|---|---|---|---|---|
| before | 9162 | 21955 | 697 | 175 | 101,545 |
| direct | 10662 | 23455 | 2851 | 2387 | 106,022 |
| **bias** | **8540** | 21951 | **697** | **175** | **99,678** |

Spills and reloads return to baseline exactly; `movq` drops 622 *below*
baseline, which is the removed `jit_ctx` load finally showing up as a win.

## Measure codegen changes at `opt_level=speed`, against the corpus

Two traps cost a lot of time here:

1. **`opt_level` flips with the `developer` feature.**
   `CODEGEN_OPT_LEVEL_SPEED` defaults to `!cfg!(feature = "developer")`, so any
   test run with `--features developer` compiles at `opt_level=none` — a
   different register allocator path than production. Results there can invert:
   the `+0` copy looked like a 2.6% *win* at `none` and was a byte-for-byte
   no-op at `speed`.

2. **Synthetic loops do not show it.** Straight-line LW/ADDU chains have too
   few calls to reach the 2-split cap, and showed the broken version as
   *smaller*. Only real guest pages (many blocks, many calls, values live
   across them) expose the spilling. Use `jitv2_corpus/`.

Cranelift has no user-facing "make a distinct copy" primitive — no `copy`,
no opaque/blackbox barrier in CLIF — which is why the bias trick is what it
is. `enable_pinned_reg` is also not usable: it makes `%r15` non-callee-saved
for compiled code, so a JIT function stops preserving its caller's `%r15` and
corrupts the interpreter loop (observed: `free(): invalid size`).
