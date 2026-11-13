# Plan: keep guest registers in SSA values across a jitv2 region

Status: **done 2026-11-10.** What landed, and what the measurements changed,
is in [../jitv2/gpr-cache.md](../jitv2/gpr-cache.md). In short: stage 1 (the
join) and stage 3 (the region cache) are in, on by default; stage 2 (local
forwarding) was not needed, since per-entry stubs give the region cache the
same "no load nobody reads" property; the budget turned out to matter less
than *where* registers are cached (loops only); and the reload-after-callout
idea in stage 3 was measured and dropped. The "related work" list at the end
is still open, and compile time is now the first item on it.

Written from xiri's `perf/sgiemu-port` (`66f5a59b`, read in full) and atom's
summary of revenir's JIT patches (0025 gpr-forward, 0026 mem-join; the patches
themselves are in a private repository and were not read). Main at `95a3af0b`.

## Why

`emit_read_gpr` / `emit_write_gpr` are a load and a store against `core_ptr`.
[interrupt-check-frequency-gates-gpr-forwarding.md](../jitv2/interrupt-check-frequency-gates-gpr-forwarding.md)
measured that Cranelift's own store-to-load forwarding recovers almost nothing:
42% of GPR loads in the sample region read registers the region never stores,
and callouts clobber the rest. It named a per-region register cache as the
only path. Two downstream trees have since built one.

## What the others measured

**xiri `66f5a59b`.** 31 Cranelift `Variable`s, all loaded at function entry.
Reads `use_var`; writes store to `MipsCore` *and* `def_var` (write-through),
so every exit, exception and helper sees precise registers with no spill
protocol. After the shared load helper (which writes its destination itself)
the variable is reloaded, only on success. Disabled for `developer`,
`jitv2_lockstep`, any region with an interpreter fallback, and any region with
an FPU instruction.

`iris-bench`, Ryzen 8945HS, `lightning,jitv2`, 3 alternating runs: geomean
+7.7% R4400, +7.2% R5000. `int/alu` 2.25x, `int/alu64` 2.36x, `codec/lz`
1.4-1.7x, Dhrystone flat. Losses: `mem/unaligned` -11%, `mem/latency_dram`
-6%, `fpu/whetstone` -7% on R4400. With FPU regions included it lost 20-25% on
the FP kernels, hence the exclusion. No IRIX workload was timed.

**revenir (reported only).** 0026 mem-join: the loaded value rides the
fast/slow join as an SSA value instead of a scratch slot, status checked only
on the slow path; 6-10% on x86 and ARM. 0025 gpr-forward: "a pass that removes
reloads of guest registers", x86-64 only, tight loops +14%.

## Reading of the regressions (hypothesis, to be measured in stage 0)

1. **Entry cost.** 31 loads on every entry into compiled code, used or not.
   Hurts short regions and anything that leaves and re-enters often.
2. **Register pressure across calls.** A variable that is live across a
   callout must sit in a callee-saved register or be spilled to the stack.
   x86-64 SysV has 5 usable callee-saved registers (Win64 7), aarch64 has 10.
   With 31 live values and a callout every few instructions (FPU, slow memory
   paths), the cache turns L1 loads from `MipsCore` into stack spills and
   reloads plus moves. This is the per-architecture budget.

## Stages

Each stage lands separately, behind a runtime knob (`j2 gprfwd off|local|region`,
`j2 gprbudget <n>`), and is timed before the next starts.

### 0. Baseline and callout audit

- `iris-bench matrix` (R4400/R5000 x interp/jitv2), baseline and candidate
  side by side, alternating order, sha and features recorded. Add one live
  IRIX figure (boot to login, and a Dhrystone or compile in the guest):
  emitted-code wins have failed to show up in wall-clock before.
- Table of every callout the codegen emits, classified: writes no GPR /
  writes one known GPR (through a `dst` pointer) / may write any GPR
  (interpreter fallback, lockstep) / never returns into the region. The table
  goes in `clanker/jitv2/` and is the contract stages 2 and 3 rely on. A new
  helper that writes a GPR must be added to it.

### 1. Finish mem-join

Main already carries the status as a block param (`emit_mem_read_split`). The
value does not: the fast path stores to `dst` (the GPR slot or
`jit_read_scratch`) and `emit_mem_read` reloads it. Make the value a second
block param: fast path passes the extended load, slow path passes a reload of
`dst` after the callout; one GPR write after the join. Independent win, and
the register cache needs loads to produce an SSA value anyway. Check how the
shared `mem_helpers` path interacts (it writes the destination inside the
helper; either return the value or keep xiri's reload-on-success).

### 2. Local forwarding

A known-value map `[Option<Value>; 32]` in `EmitCtx`, valid within one
straight-line run. Read: return the known value or load and remember. Write:
store and remember. Reset at every word that is an entry point, branch target,
fallback successor, or otherwise has more than one predecessor. Invalidate by
callout class from the stage-0 table. No entry loads, no phis.

Open choice, decided by measurement: drop everything at each callout (zero
cross-call pressure) or keep values alive across no-GPR-write callouts.

### 3. Region cache with a budget

xiri's write-through variables, with three changes:

- **Budget.** Only the top N registers by static use count in the function
  (weight uses inside loops) get a variable; the rest stay on the stage-2
  path. N is per host architecture and a runtime knob; start around 4 on
  x86-64 and 8 on aarch64 and sweep.
- **Entry loads only for cached registers.**
- **Reload instead of exclude.** After a may-write-any-GPR callout, re-`def`
  the cached variables from memory. FPU and fallback regions then need no
  blanket exclusion; whether they profit is a measurement.

Things that are safe by construction and should stay so: write-through keeps
`MipsCore` precise at every exit; the monitor, GDB stub and snapshot loader
only touch registers while the CPU is outside compiled code, and every entry
reloads; GPRs are per core under SMP.

### 4. Not planned: lazy write-back

Dropping the store and spilling dirty registers at exits is where the real
correctness risk is (every exception and bail path becomes a spill site).
Revisit only if stores show up in a profile after stage 3.

## Verification

- Port xiri's equivalence test: store/load loop against the interpreter with
  loop-carried values, a delay-slot register that changes, and a final
  faulting load that must keep the old destination and report the right EPC.
- A debug knob that, at each pending-interrupt check, compares every cached
  variable with its `MipsCore` slot and traps on mismatch. Lockstep must run
  with the cache off (its inline interpreter writes GPRs), so this is the
  only in-situ check of the cache itself.
- `cargo test --workspace --features jitv2`, cpu-tests against the known
  baseline, `iris-bench` accuracy, IRIX 6.5 boot to desktop on R4400, R5000
  and R10000.

## Related work to pick up separately

- **Compile time first.** A live IRIX boot keeps four compile workers
  saturated at ~50 ms per page function; see "Compile time is the bigger
  lever" in the note above. revenir's 0008 (incremental compile) and 0004
  (idle workers) are aimed at exactly this.

- From xiri: PS/2 Set 2 -> Set 1 translation (`484452d9`), TLB tags keep
  Region and VPN[39:32] in 32-bit mode (`9df1df14`, `43fdfaaf`), MC restore
  republishes the bank map (`b6d6f5c2`, check the machine loader first).
- From revenir, once the patches can be read: idle compile workers (0004),
  same-page fast dispatch (0005), incremental compile (0008), persistent
  cache cap (0021, overlaps `atom/jitcache-bounds`).
- Chaining (0011-0015, with 0023 frame pointers on x86-64) needs its own plan:
  code lifetime in the shared SMP pool is the open problem, and it interacts
  with stage 3 (every chained entry reloads its cached registers).
