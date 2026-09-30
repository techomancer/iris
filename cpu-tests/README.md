# cpu-tests

MIPS III / MIPS IV tests that run **on the emulated CPU** with no operating
system. They print PASS/FAIL over the serial console and report the failure
count as the emulator's exit code.

One binary covers every CPU it knows: it reads `PRId` at startup and picks
expectations for R4400 (MIPS III) or R5000 (MIPS IV) — both validated on real
Indys — or R4600 (MIPS III), whose expectations are inferred rather than
measured; [docs/r4600.md](docs/r4600.md) says from what.

**The suite is tests only** — emulator fixes land separately. Findings are
reported, not fixed here, so the suite keeps showing them until `src/` changes.

## Build

Needs a MIPS cross toolchain:

```sh
sudo apt-get install gcc-mips-linux-gnu binutils-mips-linux-gnu
make                        # -> build/cputest.elf
```

No root? `make toolchain-local` unpacks the same packages into
`~/.local/opt`; the Makefile finds them automatically.

`make ONLY=group_umode` (any `group_*` from `harness/tests.c`) builds a suite of
`identity` plus that one group - for iterating on a slow target such as an HDL
simulator. Do a clean build when switching between it and the full suite.

## Run

```sh
cargo build --release       # in the repo root, first
make run                    # loads the ELF straight into RAM and runs it
```

`make run` takes a few minutes — 240 tests, some of which take real exceptions
or sweep the caches. You get a line per test and a summary:

```
alu/addu_sign_extends ...................... PASS
...
 RESULT: 2041 checks passed, 121 failed  (240 tests)
IRIS-CPUTEST-DONE rc=100
```

The exit code **is** the failure count, saturated at 100 so that it can never
collide with the harness's own 127 ("unknown CPU, refusing to run"). The R4400
cell exceeds that today, so read the `RESULT:` line for the real number.
`IRIS-CPUTEST-DONE` is the token to match on if you are scripting it.

### Other CPUs and engines

The CPU is a runtime choice (`--cpu r4400|r5000`); the JIT is a compile-time
cargo feature, so each engine needs its own IRIS build:

```sh
cargo build --release                                # interpreter
cargo build --release --features jitv2               # jitv2 (no env var needed)
# then add --cpu r5000 to the run command above for an R5000
```

`run/matrix.sh` does all four CPU × engine combinations and builds what it
needs; `CELLS="r4400-jitv2" run/matrix.sh` runs just one. The *Bare-metal
suites* CI workflow (`.github/workflows/suites.yml`) runs every cell on each
push that touches the CPU, and gates on the known failing count per CPU.

### Booting it like real hardware

```sh
make image                  # volume-header image via mkvh
run/run-prom.sh             # PROM: boot -f dksc(0,2,8)cputest
```

Slower, but it exercises the real path: the PROM reads the volume header, loads
the ELF, and jumps to it. This is what the bootable CD will use.

### Which machines this runs on

**An SGI Indy or Indigo2 (IP22/IP24), and nothing else** — on real hardware or
emulated. The harness is written for that machine: it links and self-relocates
to a fixed KSEG0 address chosen because IP22/IP24 RAM begins at `0x08000000`
(`harness/link.ld` explains why at length), it drives the console through the
Z85C30 SCC via IOC2 at `0x1FBD9800`, and the exception vectors and TLB work
assume that map. Under IRIS none of that costs anything, because IP22/IP24 is
what IRIS emulates.

Unlike `bench/`, this suite also **legitimately refuses an unrecognised CPU**
(exit 127). That is not the same limitation and should not be "fixed" the same
way: these tests check R4400-versus-R5000 behaviour by construction — cache
geometry, MIPS IV availability, FPU quirks — so a CPU the suite has no case
for genuinely has no expected answer here. The R4600 case was added the honest
way: each of its answers is either documented, shared by both measured parts,
or accepts both measured parts' answers — [docs/r4600.md](docs/r4600.md).
`bench/` has no CPU-specific anything and does run on any MIPS III-or-later
part.

The assumptions, which of them were tested, and what a port to another SGI
family would actually involve, are written up once in
[`rules/testing/bare-metal-harness-platform-assumptions.md`](../rules/testing/bare-metal-harness-platform-assumptions.md)
— the harness is shared with `bench/`, so the limitations are shared too.

## Expected results

The expectations are **validated against real hardware**: an SGI Indy, R4400
rev 6.0, passes 240/240 (`oracle/r4400-rev6.0-run4.log`). Measured against that,
the emulator reports:

| | pass | fail | failing tests |
|---|---:|---:|---:|
| Indy R4400 rev 6.0 | 2164 | **0** | **0** |
| Indy R5000 rev 1.0 | 2135 | **0** | **0** |
| IRIS `--cpu r4400` | 2040 | 124 | 27 |
| IRIS `--cpu r5000` | 2027 | 108 | 22 |

Every emulator failure is a known finding, listed in
[docs/findings.md](docs/findings.md) — and all 15 of the original FP failures are
now hardware-confirmed as real IRIS bugs rather than bad tests. Anything else is
new: start with [docs/gotchas.md](docs/gotchas.md), which collects the times the
*test* was wrong, including the sixteen the Indy caught that IRIS agreed with.

`run/diff-hw.py` classifies a hardware log against an emulator one; the archived
hardware logs live in `oracle/`.

### Added since the oracle runs, not yet run on silicon

These went in on branch `claude/r4600-cputests` after the Indys ran the suite,
so the silicon counts above do not include them and no Indy has passed them
yet. Each is plain architecture rather than a part-specific quirk:

| test | checks |
|---|---|
| `excep/cp0_unusable_user` | CP0 instructions from User and Supervisor mode raise Coprocessor Unusable, CE = 0 (R4000 manual ch. 5) |
| `excep/cp0_usable_cu0` | ... unless `Status.CU0` is set |
| `identity/config_k0` | re-enabled: Config.K0 is writable, with the sweep kept in registers so no dirty D-cache line can be lost across a change of KSEG0's attribute |
| `mem/load_then_use` | the instructions right behind a load read its value correctly, cached and not |
| `mem/load_then_trap` | a syscall, break, trap, overflow or reserved instruction right behind a load: exactly one exception, EPC on it, the load complete |
| `mem/load_then_more` | a CP0 read, a divide, a jump, and nearby loads and stores right behind a load |
| `fpu/trap_behind_a_load` | an FP trap right behind an integer load, while the load may still be waiting on its fill |
| `mem/load_then_load` | a load right behind a load that does not name its register - same line, two lines, the same word at two widths, three in a row, the first load's value as the third's base, the second's used at once, after a store, before a store, an LWL merge, KSEG1 against KSEG0 - with both lines cold, both warm, and each one warm alone |
| `mem/load_then_load_evict` | the second of two loads evicts the first's D-cache line, or finds its own line just written back to memory |
| `mem/load_then_store` | a store right behind a load, in the load's line and another, read back, in the same four cache states |
| `tlb/load_then_mapped_load` | two loads where one or both walk the TLB, and the second's value is used at once |
| `umode/*` (12 tests) | User mode the way IRIX 6 runs it: n32 processes with `Status.UX` = 1 under a 32-bit kernel (KX = 0), so every exception and ERET switches addressing mode. Each test runs a few words at kuseg `0x00400000` under three Status settings - KX = SX = UX = 1, all clear (IRIX 5), and UX alone (IRIX 6) - and records every exception: vector, Cause, EPC, BadVAddr, Context, XContext, EntryHi. Syscalls as the first and second instruction after ERET (libc's `_getuid`), a break, a pending interrupt, load and fetch TLB refills (XTLB vector when UX = 1; fixed up and retried), KSEG0 and misaligned loads (AdEL), and 64-bit operations with UX set |

The first results for them are from the `sgiindy_MiSTer` FPGA core presenting as
an R4600: all pass (2409 checks over 250 tests in its simulator, both with every
load stalling execute and with loads that stall only when they must, including
behind another load or store). `mem/load_then_load` found that core releasing a
stalled load's execute hold on the completion of the load ahead of it.

The `umode` group came out of IRIX 6.5's installer dying on that core with `init
died (why = 3, what = 0xb)`: init's saved frame showed a TLB miss at the general
exception vector's own address, reported on the `syscall` in `_getuid`. The core
built the vector address with the kernel's 32-bit mode and fetched it under the
process's UX = 1. Before its fix `umode/syscall_second` reproduced init's frame
exactly; after it, 2752 checks pass over 267 tests on the FPGA, all but
`umode/fetch_miss_entry` (the core still takes an instruction-TLB miss on an ERET
target as a nested exception). IRIS passes all 12.

## Writing a test

Add a function to the right file in `tests/`, register it in that file's table,
and rebuild:

```c
static void t_addu_sign_extends(void)
{
    u64 r;
    u32 a = OPAQUE(0x7FFFFFFFu), b = OPAQUE(1u);
    __asm__ __volatile__(A "addu %0, %1, %2" Z : "=r"(r) : "r"(a), "r"(b));
    CHECK_EQ(r, 0xFFFFFFFF80000000ull);
}

static const struct test tests[] = {
    TEST("alu/addu_sign_extends", t_addu_sign_extends, CPU_ALL),
};
```

`CPU_ALL`, or any combination of `CPU_R4400`, `CPU_R5000` and `CPU_R4600`,
says which parts it applies to; the others are skipped rather than failed.
Inside a test, `is_r4400()` / `is_r5000()` / `is_r4600()` name one part and
`has_mips4()` asks the ISA question — use that one for MIPS IV availability, so
the R4600 is not mistaken for an R5000. `A`/`Z` are the strict asm prologue —
use them, or the assembler will quietly rewrite your instructions (see
[docs/gotchas.md](docs/gotchas.md)).

A whole new area also needs a `struct test_group` and one line in
`harness/tests.c`.

FP tests have their own conventions — `AF` rather than `A` for any block with
an FP instruction, never a `$f*` clobber, values crossing through memory — all
of them collected in `tests/fpu/fpu_common.h`, which also provides the
`observe_s`/`observe_d` helpers that run one operation and report what the FPU
and the CPU each did about it.

## Layout

```
harness/   startup, exception vectors, CHECK macros, console
tests/     identity alu muldiv mem branch excep cp0 tlb umode fpu cache mips4
gen/       fpvectors.py — computes the FP expectation tables
run/       run-local.sh  matrix.sh  run-prom.sh  bare.toml  boot.toml
docs/      findings gotchas status oracle memory-map toolchain
```

`tests/fpu/` is eight files rather than one — arithmetic, traps, denormals,
comparisons, generated vectors, double precision, the FR=0 register file, and
the odd corners — sharing `fpu_common.h` for the conventions that make FP tests
work at all under `-msoft-float`. [docs/status.md](docs/status.md) has the
breakdown.

### Generated expectations

`make vectors` regenerates `tests/fpu/fpvectors.{c,h}` from `gen/fpvectors.py`,
which computes IEEE-754 results with exact rational arithmetic and cross-checks
them against the host FPU before writing anything. The generated files are
checked in, so building the suite needs only the cross toolchain; python3 is
needed only to change them.

## More

- [docs/findings.md](docs/findings.md) — what the suite found, and what each
  finding actually costs.
- [docs/gotchas.md](docs/gotchas.md) — read this before believing a new failure.
- [docs/status.md](docs/status.md) — coverage, current numbers, what is missing.
- [docs/oracle.md](docs/oracle.md) — where expected values come from, and what
  the suite deliberately refuses to assert.
- [docs/memory-map.md](docs/memory-map.md) — why it links at `0x88200000`.
- [docs/toolchain.md](docs/toolchain.md) — why n32, and why no libgcc.
- [PLAN.md](PLAN.md) — the original roadmap and the verified PROM boot path.
