# Gotchas found while building this suite

Each of these first showed up as a plausible-looking emulator bug. None of them
was one.

## The assembler inserts traps into `div`

GAS expands `div rs, rt` into a **macro**, not an instruction:

```
        bnez    s1, 1f
        div     zero, s0, s1        # the real instruction, in the delay slot
        break   0x7                 # divisor was zero
1:      li      at, -1
        bne     s1, at, 2f
        lui     at, 0x8000
        bne     s0, at, 2f
        nop
        break   0x6                 # 0x80000000 / -1
2:
```

So `muldiv/div_by_zero_no_trap` and the overflow tests "failed" against traps
the CPU never raised — the assembler had bolted them on. Two fixes, both in
place:

- **`.set nomacro`** in the standard prologue, so any such expansion is now an
  assembly error rather than a silent one. (`.set noreorder` must come first;
  GAS rejects the other order.)
- The divides are written with an explicit `$0` destination —
  `div $0, %1, %2` — which is the plain instruction.

Consequence: pseudo-instructions are unavailable inside test asm. Write
`daddu %0, $zero, $zero` rather than `move`, and bring large constants in
through an `"r"` constraint rather than `li`.

## A `u32` address given to `cache` is zero-extended, and misses silently

In 64-bit mode KSEG0/KSEG1 are the *sign-extended* ranges
`0xffffffff80000000..0xffffffffbfffffff`. A `u32` such as `0x8821c000` sitting
in a 64-bit register is `0x000000008821c000` — xkuseg, TLB-mapped, nothing like
the address intended.

Feeding that to `cache` is doubly quiet: a `Hit_*` operation that misses is
architecturally a **no-op**, so every flush in the harness was doing nothing at
all, and the only visible symptom was cache tests failing for unrelated-looking
reasons. `SEXT_PTR` / `K1_PTR` in `iris.h` force the sign extension, and the
range helpers now work on `char *` rather than `u32`.

## A primary-cache writeback does not reach memory

This one is real R4400 behaviour and the most valuable of the three.

`Hit_Writeback_Invalidate_D` writes the line back to the **secondary cache**,
not to memory. An uncached KSEG1 read goes to memory and sees the old value —
which looks precisely like a broken writeback in the emulator.

The probe that settled it:

```
k0 reg = 0xffffffff8821c000        pointers are fine
k1 reg = 0xffffffffa821c000
after k1 write, k1 reads 0x11112222    uncached path works
k0 reads 0xc0ffee00 (no flush)         cached line is stale, as expected
after k0 write + wb, k1 reads 0x11112222   <-- writeback did not reach memory
```

Reaching memory takes a second operation against the **SD** (secondary data)
target. `Config.SC == 0` says a secondary cache is present, which
`cache_detect()` reads at startup, and the range helpers in `testlib.c` cascade
`PD → SD` when it is. With that, the same probe ends `k1 reads 0x33334444`.

This is why IRIX flushes both levels rather than just the primary, and it is a
trap any cache test on this family will fall into exactly once.

## o32 splits a 64-bit value across two registers

See [toolchain.md](toolchain.md). `"=r"(u64)` under o32 binds only the first
register of an even/odd pair, so a 64-bit result comes back half-read — and on
a big-endian target the half you get is the high word, which made `addu`
producing `0x80000000` read back as `0x8000000000000000` and look like a
sign-extension bug in the CPU. The suite builds n32.
