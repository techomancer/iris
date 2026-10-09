# A retry bail out of an inlined delay slot must arm the branch

## Symptom

IRIX 6.5.22, Indy R5000 + XZ (GR2), jitv2 `lightning` build:
`/usr/demos/General_Demos/solidview/solidview -umats body.mat body.fea`
died with SIGSEGV (sometimes SIGBUS) inside libgl, at a different spot from
run to run. The interpreter ran it fine.

The core showed `gl_c_bgnpolygon` running with `t9` = `gl_i_bgnpolygon`.
libgl lays those two out back to back:

```
gl_i_bgnpolygon: lui at; ori at; lui v0; ori v0; sw zero,0(v0)
                 jr ra
                 sw zero,0(at)        <- delay slot
gl_c_bgnpolygon: ...                  <- execution fell through to here
```

`gl_c_*` derives `gp` from `t9`, so every GOT load in it was shifted by
`gl_i_*`'s size and it called into the wrong function. The `jr ra` had not
jumped.

## Cause

`emit_check_mem_status` turns a non-exception status (`EXEC_RETRY`, i.e. the
device returned `BUS_BUSY`) into `emit_bail(ctx.word)`. That sets `core.pc`
to the instruction's own address and leaves the retry to the interpreter.

For an inlined delay slot, `ctx.word` is the slot. Outside lockstep/developer
builds, nothing has armed `core.in_delay_slot`/`core.delay_slot_target`: the
branch's transfer lives only in compiled code. So the interpreter re-runs the
slot as a plain instruction and goes on to slot + 4. The branch is lost.

GR2 is where this happens for real: its HQ FIFO write returns `BUS_BUSY` when
the FIFO is full, and GL code stores to the FIFO from delay slots all the time.
FIFO fill depends on timing, which is why the crash site moved between runs.

## Fix

`EmitCtx::slot_target` holds the branch's pending target while a slot is
being emitted (`emit_slot_semantics` sets it). The retry block stores it to
`delay_slot_target` and sets `in_delay_slot` before bailing, as
`branch_delay` would have. An entry word reached as a foreign slot does not
need this: both fields are already live from the dispatch that reached it.

Tests: `jr_with_busy_delay_slot_still_jumps`,
`beq_with_busy_delay_slot_still_branches`,
`bne_not_taken_with_busy_delay_slot_skips_to_fallthrough` (equiv_test.rs).

## Gotchas from the hunt

- **`jitv2_lockstep` can't catch this bug, for two reasons.** First, it is too
  slow to fill the GR2 FIFO, so `BUS_BUSY` never happens and the retry path
  never runs. Second, its slot bracket stores both `in_delay_slot` and
  `delay_slot_target`, so even a retry would resume correctly. Lockstep can't
  see bugs that need device backpressure (FIFO full, bus busy). Look for
  those at full speed, with the `j2 <category> off` bisect.
- **Bisect with several runs per setting.** The crash came about 1 run in 1
  with everything on, but one "all on" run survived, and `j2 inline_mem off`
  still crashed 2 times in 4. A single surviving run proves nothing.
- `j2 <alu|branch|loadstore|fpu|cop0> off` + `j2 flush` is a live bisect
  that needs no rebuild. Here only ALU + load/store + branch together
  crashed, because the slot is only inlined when its branch is compiled.
- PC breakpoints are compiled out of `lightning` builds (`bp add` is
  accepted and never fires).
- dbx's backtrace from the core was misleading. The useful part was
  `printregs` plus disassembling the GOT slots the faulting code used.
