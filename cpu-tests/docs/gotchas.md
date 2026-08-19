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

## `0xFC000000` is not a reserved opcode — it is `SD`

The first reserved-instruction test used primary opcode 0x3F on the assumption
that the top of the map was unassigned. It is `SD`: `.word 0xFC000000` is
`sd $zero, 0($zero)`, which stores to virtual address 0, misses in the TLB, and
reports **TLBS through the XTLB refill vector** rather than RI. A perfectly
good TLB test, and a useless RI test.

The genuine holes on R4400/R5000 are primary opcodes **0x1C..0x1F**. (Later
MIPS32 revisions claimed 0x1C and 0x1F as SPECIAL2/SPECIAL3; neither of these
parts implements them.) The suite uses 0x1C, 0x1D and 0x1F, and deliberately
skips **0x1E** — IRIS's own jitv2 uses that as a region-boundary sentinel
(`src/mips_isa.rs:64`), so testing it would measure the JIT's tooling rather
than the CPU.

## An exception taken with EXL set has nowhere to return to

`t_exception_with_exl_set_preserves_epc` hung the whole suite on first run.
The rule under test is that an exception taken while `Status.EXL` is already
set does **not** update EPC — the first exception's EPC has to survive. But the
harness's default handler resumes *through* EPC, so with EXL pre-set it ERETs
to whatever EPC happened to hold (in that test, a deliberate sentinel) and
faults again forever.

Fixed by `exl_resume_handler` in `start.S`: a handler a test can install via
`exc_user_handler` that resumes at a caller-supplied `exl_resume_pc` instead of
at EPC. Any future test that deliberately corrupts the normal resume path needs
the same treatment.

## `-msoft-float` makes the assembler refuse FP mnemonics

The suite is built `-msoft-float` so GCC never emits FP code of its own — the
FPU tests change `FR` and `FCSR` underneath it, and compiler-generated FP would
silently depend on the state being changed. The side effect is that GAS rejects
`mfc1` and friends outright. Any asm block containing an FP instruction needs
an explicit `.set hardfloat`; the test files define an `AF` prologue for it.

## `(u64)(unsigned long)ptr` zero-extends under n32 — twice bitten

The same mistake as the `cache` one, in a different disguise. `unsigned long`
is **32 bits** under n32, so `(u64)(unsigned long)p` truncates the pointer and
then *zero*-extends it. `0xffffffff88228000` becomes `0x0000000088228000` —
xkuseg, not KSEG0.

In `mips4/cop1x` that produced a TLB-miss-on-store at
`BadVAddr=0x000000008822800c`, which reads exactly like a broken `LWXC1` and is
really a broken cast. The fix is to pass the **pointer** as the asm operand:
pointers are sign-extended by the ABI, so they are correct by construction.

The rule for this suite: never route an address through an integer type on the
way into asm. Use the pointer, or `SEXT_PTR` / `K1_PTR` when a numeric address
is genuinely needed.

## o32 splits a 64-bit value across two registers

See [toolchain.md](toolchain.md). `"=r"(u64)` under o32 binds only the first
register of an even/odd pair, so a 64-bit result comes back half-read — and on
a big-endian target the half you get is the high word, which made `addu`
producing `0x80000000` read back as `0x8000000000000000` and look like a
sign-extension bug in the CPU. The suite builds n32.

## A generated expectation must use the value the register can hold

`gen/fpvectors.py` writes each test vector's operands as exact rationals and
computes the answer from them. `2^30 - 1` needs thirty significand bits; a
single-precision register has twenty-four. So the table said

    cvt.w.s(1073741823.0) = 1073741823

while the machine was necessarily converting `1073741824.0` — the operand as
stored — and answering `1073741824`. Two tests failed against a table that no
correct implementation could have matched.

The fix is one function, `roundtrip()`, applied to every operand before it
reaches either the table or the expectation: encode it to the target format,
decode it back, and compute from *that*. The same trap is waiting in any
generated FP suite, and it fails in the most misleading possible way — the
emulator looks wrong by exactly one ulp.

The generator's `--check` mode exists for the same reason. It recomputes every
vector with the host's own IEEE arithmetic — a completely separate
implementation from the exact-rational one — and refuses to write the tables if
the two disagree. It caught nothing on the first run and would have caught
this, had the conversion tables been part of what it checks.

## `make run` needs the bare machine config

`run/run-local.sh` invoked IRIS without `--config run/bare.toml`, so the default
configuration applied: `default_scsi()` attaches `scsi1.raw` and `cdrom4.iso`,
neither of which exists anywhere in this tree, and startup is fatal when a
configured disk is missing. `make run` failed before the guest ran a single
instruction, while `run/matrix.sh` — which passes the config — worked fine.

## Under `--load-elf` the SCC produces nothing, and `--serial-log` is empty

The same script also asked for `--serial-log FILE`, and the file was always
zero bytes. The reason is not the logging: it is that **the SCC has never been
programmed**. `con_init()` does nothing because "the PROM leaves the console
configured" — true when booting through the PROM, and false under `--load-elf`,
which is precisely the path that skips it. An unprogrammed channel has
`WR5.TX_ENABLE` clear, so the byte is queued and the TX thread never latches
it, and nothing reaches the backend to be teed
(`rules/testing/scc-serial-output-from-bare-metal-code.md` §2).

Confirmed by running with `--serial-log` and *without* `--test-device`: the log
is still empty and the machine sits there, so the SCC really is the silent one
rather than the tee.

So under `--load-elf` the test device is the only working output path, and
headless IRIS prints it to its own stdout. `run/run-local.sh` now tees that,
which is what `matrix.sh` and the CI workflow always did. Making the serial
path work in this mode means programming WR5 (and probably WR4/WR9/WR11-14) in
`con_init()`; the hook is already there, with a comment saying so.

## The last line printed is not the last line to arrive

`IRIS-CPUTEST-DONE rc=100` reached the PROM-boot log as `IRIS-CPUTEST-DONE rc=`,
which matters more than it looks: `run/run-prom.sh` decides pass or fail by
matching `rc=0` on that line, so a completely green run would have been reported
as a failure.

Two independent races, both of which had to be fixed:

1. **The guest halts the machine mid-transmission.** `scc_putc` waits for the
   transmit *buffer* before handing over each byte, which says nothing about the
   byte already in the shift register — and `testdev_exit` stops the machine
   immediately after. `con_flush()` in `harness/console.c`, called before the
   exit, drains it.

2. **`iris-ci serial-wait` returns on the token, not on the line.** The rc
   digits are still in flight when it returns, so a `grep` at that instant reads
   an incomplete line. The script now waits for `rc=[0-9]+` before reading
   anything out of the log.

The first one loses the bytes for good; the second only reads too early. Both
produce the same symptom, which is why fixing one of them was not enough.
