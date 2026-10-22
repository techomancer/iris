# idle-pause: a masked interrupt must end the park too

The guard in `idle-pause-must-not-park-with-interrupts-masked.md` covers IM
*zero*. IM *partly* clear is just as unsatisfiable, in the one situation
IRIX spends a lot of time in: a raised spl.

An IRIX 6.5.22 Indy (R4400, `lightning,jitv2,idle-pause`) hung after some
hours of light use: keyboard, serial console and telnet all dead, the
`MIPS-CPU` thread at 0%. The monitor showed it parked:

```
stopped pc=ffffffff880e3a98
CP0 Status: 0x00004001 (IM:_6______ K:K IE)
```

IE set and IM6 alone: `splhi`. On IP22 IP6 is the bus error (`err_stat`, see
`ioc.rs`), the one line IRIX leaves unmasked at high spl. The code around the
PC calls the spl routine, a function, then `splx`, in a loop on a flag in a
kernel structure - a flag an interrupt handler clears. Most of each pass is
at high spl, so the detector's batch-end samples land there, the state
repeats, and `park` waits for `ip & im`. The timer's interrupt (IP4) then
arrives masked; the guest would take it a few instructions on, at `splx`,
but parked it never gets there. Nothing raises IP6, so the park never ends.
`stop`/`step`/`cont` on the monitor unstuck it, which is what gave it away:
the first step went straight to the exception vector.

The mask `update` samples is the guest's mask *of that moment*, not of the
loop. So:

- `update` declines to park while any line is pending, masked or not (the
  guest is about to take it);
- `park` returns when any line becomes pending, masked or not.

"Nothing pending, wait for anything" is the condition that is always
satisfiable by the guest's own timer, whatever spl the sample caught. A line
held pending and masked for long would keep the CPU from parking (spinning,
as without idle-pause), never from running.

Tests: `update_declines_to_park_while_a_masked_interrupt_is_pending`,
`a_masked_interrupt_ends_the_park` (`idle_park.rs`), and
`idle_park_wakes_for_an_interrupt_masked_by_a_raised_spl`
(`mips_exec_test.rs`), which runs the spl loop through `step_int`, the
1000-step batches, `update` and `park` as `MipsCpu::start` does and raises
IP4 from another thread once the CPU has parked. All three hang (end at
their deadline) without the fix.

Tests that park take `PARK_TEST_LOCK`: there is one `PARKER` slot, so two
parking at once wake each other's threads. They wait for `parked()`, not a
guessed sleep, to keep the lock briefly - `an_interrupt_ends_the_park...`
measures latency under the whole suite's load and was already occasionally
over its limit (about 1 run in 25 with `--features idle-pause,jitv2`, on
`main` as with this change).
