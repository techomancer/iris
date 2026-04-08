# JIT Dispatch Architecture Rules

## Interpreter-first, never JIT-first

The dispatch loop must run the interpreter in sustained bursts (hundreds of steps)
between JIT block executions. The interpreter's step() does critical per-instruction
bookkeeping: cp0_count advancement, interrupt checking, cp0_compare crossover,
delay slot state machine.

**NEVER** check the JIT cache every iteration (JIT-first). Even one exec.step()
between JIT blocks is insufficient. Tested at 58% JIT ratio — kernel panicked.

**Minimum probe interval: 100.** Below this, the system approaches JIT-first
behavior and crashes. The adaptive ProbeController enforces this via IRIS_JIT_PROBE_MIN.

## No block chaining

**NEVER** execute multiple JIT blocks consecutively without returning to the
interpreter. Manual interrupt checks between chained blocks are insufficient —
they miss CP0 timing, soft reset, software interrupts. Tested with up to 16
chained blocks — kernel panic at 0x880097ac.

## Post-block bookkeeping is mandatory

After every JIT block execution (normal exit path):
1. Advance cp0_count by `block_len * count_step`
2. Check cp0_compare crossover for timer interrupt (CAUSE_IP7)
3. Credit `local_cycles += block_len`
4. Check for pending interrupts via atomic load
5. Merge external IP bits into cp0_cause
6. If unmasked interrupt pending, call exec.step() to service it

On the exception path:
1. Advance cp0_count for instructions executed BEFORE the fault:
   `instrs_before_fault = (ctx.pc - block_start_pc) / 4`

Omitting post-block cp0_count advancement causes timer drift and kernel panics.
This was present in the original initial JIT but accidentally dropped in the
rewrite. The bug was masked with short straight-line ALU blocks but manifested
immediately with branch compilation (longer, more frequent blocks).
