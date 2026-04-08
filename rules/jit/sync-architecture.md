# JIT Sync Architecture Rules

## sync_to_executor: minimal writeback only

sync_to_executor must ONLY write back:
- GPRs (core.gpr)
- PC (core.pc)
- hi, lo

It must NOT write back:
- cp0_status, cp0_cause, cp0_epc, cp0_badvaddr
- cp0_count, cp0_compare, count_step
- nanotlb (all 3 entries)
- fpr, fpu_fcsr
- local_cycles, cached_pending

**Why:** JIT memory helpers (read/write) call exec methods directly, which
modify these fields on the executor in-place. The JitContext copy is stale
for these fields after helpers run. Writing them back would clobber changes
made by exception handlers and TLB fill operations.

## sync_to_executor must clear delay slot state

Always set:
- exec.in_delay_slot = false
- exec.delay_slot_target = 0

JIT blocks handle delay slots internally. Clearing prevents the interpreter
from jumping to a stale target on the next step().
