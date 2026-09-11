# A synthetic call chain must preserve `ra`, or it never returns

`full_mutex_lock_cas_call_chain_matches_interpreter` (`src/jitv2/equiv_test.rs`) failed for
months under `--features jitv2,jitv2_lockstep`. Every GPR matched between the JIT and the
interpreter; only `pc` differed, by 8 bytes. It looked like a subtle JIT divergence. It was
a bug in the test fixture.

## What was actually wrong

The synthetic caller did:

```
caller+0x00  addiu a1, zero, 20
caller+0x04  or    a0, s6, zero
caller+0x08  jal   mutex_lock      <-- clobbers ra with caller+0x10
caller+0x0c  addiu a1, zero, 24    (slot)
caller+0x10  lw    t6, 84(s3)
caller+0x14  jr    ra              <-- ra is caller+0x10, not the sentinel
caller+0x18  nop                   (slot)
```

`jal` writes the return address into `ra`. The caller had no frame, so it never saved the
sentinel value it was seeded with. By the time its own `jr ra` executed, `ra` pointed at
`caller+0x10` — and the chain entered a three-instruction infinite loop:

```
+0x10 -> +0x14 -> +0x18 -> +0x10 -> ...
```

A trace made it obvious:

```
step 3  pc=...8023735c ra=...80237360   <- jal overwrote ra
step 7  pc=...80237360 ra=...80237360
step 10 pc=...80237360 ra=...80237360   <- looping
step 13 pc=...80237360 ra=...80237360
```

So **the call chain never returned**, and the fixed `steps = 60` sampled two engines
mid-flight. The JIT and interpreter retire different numbers of instructions per `step()`
(the JIT runs a whole region), so they sat at different points in that loop. Nothing was
diverging; they were simply photographed at different moments.

## The fix

Give the caller a real frame, exactly as compiled MIPS code has one:

```
caller+0x00  addiu sp, sp, -32
caller+0x04  sd    ra, 16(sp)     <-- save the sentinel
caller+0x08  addiu a1, zero, 20
caller+0x0c  or    a0, s6, zero
caller+0x10  jal   mutex_lock
caller+0x14  addiu a1, zero, 24   (slot)
caller+0x18  lw    t6, 84(s3)
caller+0x1c  ld    ra, 16(sp)     <-- restore it
caller+0x20  jr    ra             --> sentinel
caller+0x24  addiu sp, sp, 32     (slot)
```

Plus a `jr $ra`-to-self sentinel at the landing address (the
`assert_bc1_fallback_matches` idiom) so the PC is *pinned* once reached: a RegJump is always
a region boundary, so the JIT exits and `step()` returns every dispatch. The previous
landing page was all zeroes, and NOPs *advance* PC — they do not quiesce.

## Why the old comment was misleading

> *"Generous step budget; both engines quiesce at the ra sentinel (an all-zero page —
> they'll just spin on NOPs identically past that...)"*

Two errors in one sentence: they never reached the sentinel, and an all-zero page would not
have quiesced them if they had. It read as a completed safety argument and was neither.

## Guard added

```rust
assert_eq!(interp.pc, ra_sentinel,
    "setup: the interpreter must reach the ra sentinel within {steps} steps ...");
```

This is what caught the real bug: the first attempt fixed only the sentinel, and the guard
immediately reported the chain still stopping at `caller+0x18`. Without it, a future
regression that stops the chain early would silently go back to comparing mid-flight states.

## Do not tune the step count

The result is now step-count independent — verified passing at 45, 60 and 137 steps. That
is the property worth keeping. A count tuned until the two engines happen to line up would
hide exactly the divergence this test exists to find.

Also verified non-vacuous: injecting a deliberate one-register difference into the JIT arm's
seed makes it fail as expected.
