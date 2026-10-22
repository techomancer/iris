# `WAIT` is not a NOP — and `InstrKind::Nop` never meant "nop"

Two related fixes, both prompted by an r5k-vs-r4k `instr_used.txt` diff whose
**only** difference was a line reading `nop`.

## The stats bug: `FUNCT_WAIT => Nop`

`mips_instr_stats.rs` classified `WAIT` as `InstrKind::Nop`. That was misleading
in both directions:

- A **real** `nop` is `sll $0,$0,0`, which classifies as `InstrKind::Sll`. So
  `Nop` never counted a single actual nop.
- The only encoding reaching `Nop` was `WAIT`. So `nop` appearing in a report
  meant *"a WAIT executed"* and nothing else.

The diff therefore read as "r4k doesn't count nops" when it actually meant
**"only r5k executes WAIT"** — and that misreading sent an investigation down
the wrong path for a while.

`WAIT` now has its own `InstrKind::Wait` (`IsaClass::Cop0`,
`InstrCategory::COP0`, name `"wait"`). `InstrKind::Nop` is now unreachable and
is kept only so the discriminants of later variants stay stable —
`NUM_INSTR_KINDS` and the report path's `transmute` both index by discriminant.

## Why only R5000 executes it

IRIX uses `WAIT` **only** on the R4600/R5000 idle path. `kern/ml/R4Kasm.s`:

```
#ifdef R4600
LEAF(wait_for_interrupt)
	...
	mfc0	a1, C0_SR
	mtc0	zero, C0_SR		/* IE=0: test p_nextthread atomically */
	lbu	t1,0(a0)		/* check idle flag */
	beqz	t1,1f
	mtc0	a1, C0_SR		/* (slot) restore, return */
	j	ra
1:
EXPORT(wait_for_interrupt_fix_loc)
	mtc0	a1,C0_SR		/* must be adjacent to avoid	*/
	c0	C0_WAIT			/* a race with an interrupt	*/
```

The R4400 build takes the `#else` arm in `kern/os/machdep.c` — a plain C spin,
`while (local_idle() && !idler());` — with no `WAIT` at all. **This is the only
ISA-level difference between the two profiles on an IRIX 6.5 boot**: an
`instr_used.txt` diff between them shows `wait` and nothing else.

## The semantics fix: WAIT completes, then stalls

`WAIT` stalls until an interrupt is pending. It was implemented as a plain NOP
that retired and advanced PC. Interrupt *delivery* was still correct (the
interpreter samples before every instruction, and MAME's `mips3.cpp` does the
same), but it retired a fictitious instruction on every pass through the idle
loop.

### The instruction COMPLETES — `EPC = WAIT + 4`

Per the MIPS spec the processor stalls *after* WAIT graduates, so an interrupt
taken during the stall reports `EPC = WAIT + 4` and `ERET` resumes at the
*following* instruction. **An implementation that leaves PC on the WAIT is
wrong** and was tried first: IRIX's idle path is

```
	mtc0	a1,C0_SR
	c0	C0_WAIT
	NOP_0_4
	j	ra
```

— the function is meant to *return* once an interrupt has been taken. Leaving PC
on the WAIT re-executes it forever and never reaches the `j ra`.

So `exec_wait` calls `handle_exec_complete()` first, then stalls.

### The stall must advance `hot.cycles`, and must pace it

Both halves matter and they pull in opposite directions:

- **Must advance.** `hot.cycles` is the clock other threads wait on. The
  WD33C93A's deferred-interrupt path (`wd33c93a.rs`, *"Required for
  OpenBSD/NetBSD"*) spins until `cpu_cycles` has moved 10000. Holding still
  during the stall **deadlocks** it against a CPU waiting for the very interrupt
  that spin is about to deliver. So a host sleep/park is not an option either.
- **Must be paced.** It is also the *virtual time base* (10ns/cycle), and under
  `ci_clock` CP0 Count derives straight from it (`count_now`). Bumping once per
  host loop iteration runs guest time at hundreds of millions of cycles per real
  second inside one instruction — Count leaps and timers fire early.

The stall therefore advances cycles **paced off the host clock** at 10ns/cycle,
sampled from `Instant`, not counted per iteration. An idle guest's clock then
tracks wall time the same way a running one does.

### An unreleasable WAIT returns `EXEC_RETRY` *before* the PC update

A WAIT executed with `Status.IE` clear (or `EXL`/`ERL` set) can never be
released by an interrupt. Handled by returning `EXEC_RETRY` **before**
`handle_exec_complete()`, so PC stays on the instruction:

- the guest correctly never progresses — matching hardware, which hangs;
- the CPU thread still returns to its run loop every iteration, so the monitor,
  the debugger and the soft-reset check stay responsive.

Falling *through* instead (letting the guest walk past a WAIT that should have
blocked) is wrong, and spinning inside the instruction freezes the monitor and
debugger with it. Proven: replacing the early return with `if false` makes
`test_wait_with_interrupts_disabled_retries_without_advancing` **hang** rather
than fail.

IRIX never reaches this case — `wait_for_interrupt_fix_loc` requires the
`mtc0 a1,C0_SR` that re-enables interrupts to be *adjacent* to the WAIT, which
is what its "must be adjacent to avoid a race with an interrupt" comment is
about.

## JIT

No change needed: `analyzer.rs` classifies all of `OP_COP0` as
`Classify::Excluded` (the comment names WAIT), and codegen has no `WAIT`
emitter — so it has always retired through the interpreter. Checked that
`emit_interp_fallback_head` returns any non-`EXEC_COMPLETE` status directly, so
`EXEC_RETRY` propagates correctly if in-region fallback is ever enabled.
