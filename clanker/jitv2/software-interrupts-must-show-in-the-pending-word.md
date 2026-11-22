# Software interrupts must show in the pending word

Status: fixed 2026-11-22 (`MipsCore::sync_soft_interrupts`).

**Symptom.** On the Tezro (4 x R16000, jitv2), quake's `timedemo demo1`
froze for 0.3 to 1.3 s several times a run, at random; scores varied
16 to 29 fps. The guest's clock was fine (UST never went back, no system
gap). Quake was asleep in the kernel outside any system call, at a
libGLcore PC, on a wait address inside `rrm_pane` (the Rendering Resource
Manager). The VPro board was idle for the whole stall: swap armed, latched
at the next retrace, retrace interrupt delivered.

**What the trace showed.** Perfetto's per-CPU `interrupt taken` events:
for the whole stall one CPU took `ip 0x81` (IP7 clock plus software
interrupt IP0) every 10 ms, and the stall ended the moment that CPU took
`ip 0x1` alone. IRIX had posted deferred work (the wake-up) as a software
interrupt; its clock handler services IP7 and returns.

**Cause.** Compiled code polls only `core.hot.interrupts` for a pending
interrupt (`emit_pending_interrupt_preamble`, `emit_entry_interrupt_bail`).
Software interrupts lived only in `cp0_cause`, which the interpreter's
`step_preamble!` reads but compiled code does not. A software interrupt
left pending across an `eret` into compiled code was taken only when a
hardware interrupt next sent the CPU through the interpreter's check, and
when that was the clock, the handler left IP0 pending again.

**Rule.** Every write of Cause IP0/IP1 (guest `mtc0`, snapshot and state
restores) mirrors them into `hot.interrupts` through
`MipsCore::sync_soft_interrupts`. The interpreter merges only the external
bits (IP2-IP7) back into Cause, so the mirror only makes the word nonzero;
delivery still reads Cause. Anything new that sets Cause's software bits
must call it.

**Tools that found it.** A small guest watcher sampling the process from `/proc` every 50 ms
(state, wait channel, `pr_syscall`, user PC; `pr_syscall` reads one above
`sys.s`), kernel symbols from `/unix` for the wait address, and a perfetto
trace (`all,-prim,-jit`) with frame gaps taken from `swap arm` slices.
