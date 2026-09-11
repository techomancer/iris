# `deliver_exception_at` clears `in_delay_slot` itself

Addresses `cpucritique.md` finding BD-2.

## The change

`core.in_delay_slot = false;` now lives at the end of `deliver_exception_at`
(`src/mips_core.rs`), and the three executor wrappers (`handle_exception`,
`handle_exception_at`, `handle_exception_syscall`) no longer each do it themselves.

Safe because `bd` is fully consumed *before* this point — folded into `Cause.BD` and
`cp0_epc` (the `- 4` for the BD case) at the top of the function. Clearing the live flag
afterwards cannot lose information. The vector's first instruction is never in a delay
slot, so this is unconditionally correct.

## Why it matters, and why it was not a live bug

All three wrappers already cleared it, so no interpreter or jitv2 path was affected. But
`deliver_exception` / `deliver_exception_at` are free functions on `&mut MipsCore`
precisely so they can be called *without* an executor — `src/bin/jitv2_verify.rs` does
exactly that, and had no wrapper to do the clearing. That caller could enter a handler with
`in_delay_slot` still set.

This is the same shape as the CP0-Status resync bug
(`rules/testing/cp0-status-writes-must-resync-privilege-state.md`): housekeeping duplicated
across every call site, correct at all of them today, and one new call site away from being
wrong. Hanging it on the single function that performs the transition makes it
unforgettable.

The critique rated BD-2 HIGH. It was defensive-only at the time — **LOW-MEDIUM** is honest
— but the fix is two lines net and removes a real footgun, so it is worth doing regardless
of the label.

## Test

`test_deliver_exception_clears_in_delay_slot` asserts directly on a bare `MipsCore` (the
path that had no coverage): after `deliver_exception_at(.., bd = true)`,

- `in_delay_slot` is clear,
- `Cause.BD` is still set,
- `EPC == fault_pc - 4`.

The last two are the ones that matter: they prove the clear happens *after* the BD
information has been captured, not instead of it.
