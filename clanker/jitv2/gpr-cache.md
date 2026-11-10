# Guest registers are cached in host registers only inside loops, and written through

Status: implemented 2026-11-10, on by default (`j2 gprcache`, 6 on x86-64,
10 on AArch64 by callee-saved count, not yet by measurement). Plan and the
two downstream implementations it started from:
[../plans/jitv2-gpr-forwarding.md](../plans/jitv2-gpr-forwarding.md).

## The rule

- A page function **with a loop** keeps the `N` registers its loops use most
  in Cranelift `Variable`s. A function without a loop has none and compiles
  exactly as before.
- **Write-through.** `emit_write_gpr` stores to `MipsCore` and `def_var`s.
  Memory is the architectural state at every instant, so exits, bails,
  exceptions, the monitor, snapshots and the other engine need nothing new.
  There is no dirty state and no spill.
- **Variables are defined per external entry**, in a stub the dispatch
  `Switch` reaches first, not once in the entry block. A definition there
  would be live into every entry, and every call would load every register
  any path of the page reads. In a stub, SSA construction keeps only the
  loads some path from that entry reads before writing.
- **A callout that writes a GPR itself must redefine the variable.** There
  are two: `emit_interp_fallback_head` (the interpreter ran an arbitrary
  instruction: reload all, `emit_reload_gpr_cache`), and the shared load
  helper in `emit_load` (wrote `gpr[rt]`: reload that one, after the status
  check so a fault keeps the old value). The read callout never targets a
  GPR slot any more; it writes `jit_read_scratch` and the value comes back
  as an SSA value.
- Off under `jitv2_lockstep`: its reference interpreter writes registers
  between any two emitted instructions.

**When adding a callout**: if the Rust side can write `core.gpr[..]` and
control returns into the region, redefine the affected variables after it.
The callouts that return into a region today, and what they write:

| Callout | Writes GPRs |
|---|---|
| `read*_fn` (slow memory read) | no, writes through `dst` = `jit_read_scratch` |
| shared load helper (`j2 memhelpers on`) | `gpr[rt]`, reloaded in `emit_load` |
| `write*_fn`, `write64_masked_fn`, shared store helper | no |
| `fpu_binop_commit_fn`, `fpu_cvt_*_fn`, `fpu_set_mode_fn` | no (FPRs, FCSR) |
| `interp_fallback_fn` | any, all reloaded |
| `dev_trace_bp_fn`, `fetch_verify_fn`, `kill_entry_fn` | no |
| `lockstep_*_fn` | any; cache is off |
| `handle_exception*_fn` | does not return into the region |

## The load/store join (landed with it)

Inline loads and stores used to join their fast and slow paths on a status
block param and check it after the join, and a load's value went through
`jit_read_scratch` on both paths. Now the fast path jumps to the join with
the loaded value as the block param and nothing else; the callout, its
status check and the scratch reload are all on the cold path
(`emit_mem_read_ext`, `emit_mem_write`). The shared helpers keep the old
status-returning shape (`emit_mem_read_raw`, `emit_mem_write_split`).

## Measurements

Host: Threadripper 2990WX, `lightning,jitv2`, baseline `7b068a1c`, three
alternating runs per binary and CPU, geometric mean of the 46 per-kernel
median throughput ratios. All 40 checked kernels matched on every run.

| CPU | join only (`gprcache 0`) | join + cache (6) |
|---|---:|---:|
| R4400 | 1.079 | 1.276 |
| R5000 | 1.061 | 1.242 |
| R10000 | 1.069 | 1.271 |

Loops gain most: `int/alu` 3.8x, `int/alu64` 3.3x, `codec/crc32` 2.2x,
`mem/latency_l1` 1.8x, `img/resize` 1.6x, `int/branch` 1.6x. Dhrystone gains
13-16%, nearly all of it from the join. Slightly down, on more than one CPU:
`img/dct8x8` 0.96-0.98 (already with the join alone), `codec/rle` 0.86-0.99,
`sys/exception` 0.88-0.99, `sys/llsc` 0.91-0.96. The suite's run-to-run
spread on `mem/latency_dram`, `mem/random` and the `sys/` group is larger
than those differences.

**IRIX 6.5.22 on the Indy shows no change**, R4400, cold boot from a fresh
overlay, twice each: time to the login prompt 25.4-25.8 s both; a cc build
plus `find`/`tar`/`compress` over NFS and XFS 75 s vs 72-79 s; a call-heavy
C program (sprintf, libm) warm 1.85-1.9 s vs 1.75-1.88 s. Every output and
checksum identical. That program takes 6 s on its first run and 2.1 s on its
second: the guest is waiting for compiles, see below. Mean compile time in
those boots was 52-54 ms, against 49-52 ms with the cache off.

## What was tried and dropped

**All 31 registers, every function** (what xiri's `66f5a59b` does, minus its
FPU/fallback exclusion, plus per-entry stubs): 1.27 on the suite, but
`fpu/whetstone` 0.76-0.84 and `mem/stream_triad` 0.81, and the IRIX program
no faster (1.90-1.93 s against 1.85-1.91 s) with compiles a fifth longer
(mean 60 ms against 52). Straight-line code enters, runs each
instruction once and leaves through a call or return, so a variable saves
nothing there and costs its entry load, block params and allocator work.

**Budget.** 3, 4, 6, 8, 10, 12, 16 all land at 1.22-1.29 on the suite, inside
the noise of one another; 31 loses the FP kernels. Cached registers are live
across the cold callouts in a loop body, so those beyond the callee-saved
registers are saved and restored around each one.

**Reloading every variable after every callout**, so that none is live across
a call. Legal because of write-through, and it does remove the pressure, but
over a 584-page IRIX corpus (`zz_corpus_sizes`, 26,151 regions) it emitted
+11% code at 8 registers and +34% at 31, with compiles 22% longer at 31, for
the same suite throughput. Keeping them live: +2% and +4% code, compile time
unchanged. Removed.

**Counting uses everywhere instead of in loops**, at the same budget. Same
suite result, since the suite's hot code is loops; every other function
then carries variables that buy it nothing.

## Compile time is the bigger lever in IRIX

Found while measuring, not fixed here. In the 2.5-minute boot-and-build run
above, four compile workers spent 270-290 s compiling: 5,200-5,450 page
functions at a mean of 50 ms each, the slowest 330-380 ms. The workers are
saturated for the whole run and the guest interprets whatever is queued. A
codegen change that adds compile time is paid for there; a throughput
benchmark on a handful of pages cannot see it. One did show it: with reload
after every callout at 31 registers, `fpu/whetstone` and `fpu/transcend` ran
at a tenth of baseline, because the kernel finished before its page's
half-second compile did.

Iris now prints the totals when it stops (`jitv2 compiles: N in T ms (mean,
max), R rejected`) and reports the first region Cranelift rejects in every
build. Before `lightning` a reject was silent, and looked like slow code.

## Verifying a change here

- `cargo test --features jitv2 --lib jitv2`, then again with
  `IRIS_GPRCACHE=31 IRIS_GPRCACHE_EVERYWHERE=1`, which puts every
  instruction of every test region behind the cache (the default caches only
  in loops, which few equivalence tests have).
- `gpr_cache_loop_callouts_and_faulting_load` and
  `gpr_cache_reloads_after_interpreter_fallback` re-run themselves in child
  processes per budget and with the shared helpers, because the budget is
  process-wide. The second fails if the reload after a fallback is removed.
- `cpu-tests` under jitv2: the same tests fail as before (R4400 17 lines,
  R5000 12), also in the stress configuration.
- A boot with the two stress variables set: R4400 to the desktop, R5000 to
  a shell, with the cc build and checksum run on each.
