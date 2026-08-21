# What the bare-metal suites assume about the machine

`cpu-tests/` and `bench/` share a harness — `cpu-tests/harness/{start.S,
console.c,iris.h,link.ld}` — and that harness is written for an **SGI Indy or
Indigo2 (IP22/IP24)**. Under IRIS that is free, because IP22/IP24 is what IRIS
emulates. It only costs anything on real hardware, or if IRIS ever grows a
second platform.

This note records where the assumptions are, so nobody has to re-derive them,
and marks clearly which claims were tested and which are reasoning.

## The split: the CPU half is portable, the machine half is not

| | portable? |
|---|---|
| the kernels / tests themselves | **yes** — `-march=mips3`, so any MIPS III-or-later CPU executes them |
| `bench/`'s golden checksums | **yes** — `golden/golden.h` is one flat table computed natively, *not* per-CPU |
| CPU identification | **yes** — named from PRId alone (`bench/harness/main.c: cpu_name`) |
| load address | **no** — IP22/IP24 |
| console | **no** — IP22/IP24 |
| memory inventory | **no** — IP22/IP24 |
| host time base | **n/a** — an IRIS device; absent on all real hardware, and handled |

So the *measurement* is already platform-independent and the *bring-up* is not.
A port is a bring-up problem, not a benchmark problem.

## The three assumptions, in the order they bite

**1. The load address — this is the one that stops it.** Both suites link at
KSEG0 `0x88200000`, i.e. physical `0x08200000`, and `start.S` self-relocates
there from wherever the PROM dropped the image (it must: `sys/tlb_miss` and the
cpu-tests TLB tests rewrite all 48 entries, which would unmap code executing
from a mapped region). `cpu-tests/harness/link.ld` says why that address:

> physical RAM on IP22/IP24 begins at LOMEM_BASE = `0x08000000`, not at 0 […]
> everything from `0x00080000` up to `0x08000000` is unmapped.

On a machine whose RAM is based at 0, physical `0x08200000` is 130 MB up — not
RAM at all on a smaller machine, and the wrong place regardless. The relocating
copy goes into the void and the jump after it lands in nothing. `bench`'s work
area (`_work_start` at `0x88300000`, probing upward) inherits the same
assumption.

**2. The console.** `IOC_BASE 0xBFBD9800` — Z85C30 SCC channel B via IOC2.
Note that `scc_putc` *reads* RR0 before writing, so on a machine where nothing
decodes that address it is a bus error per character, not a quiet no-op.

**3. The memory inventory** (`bench` only). Bank layout comes from the memory
controller at `MC_BASE 0xBFA00000`. Cache geometry does *not* — that is read
from CP0 Config and is portable.

Also `TESTDEV_BASE 0xBF400000`, but that is an IRIS device by definition. Its
absence is expected, detected, and reported: without it the suites fall back to
CP0 Count at a measured-if-possible frequency and say so in the header.

## What was actually tested

Verified by experiment in the emulator:

- **CPU identification is genuinely generic.** The emulator's PRId was
  temporarily patched to `0x0926` and to `0xab37`; the guest named them
  `R10000 rev 2.6` and `MIPS-imp-0xab rev 3.7` and ran all 46 kernels at 40/40
  both times. (It used to *refuse* anything but an R4400 or R5000, on the
  stated grounds that "the golden checksums are selected by PRId" — which was
  false; see the note under **Gotchas** below.)
- **No `bench` kernel is CPU-gated** — every one is `BENCH(...)`, i.e.
  `BCPU_ALL`. `grep -c BENCH_CPU bench/kernels/*.c` is 0 across the board.
- **cpu-tests is the opposite** and deliberately so: its tests check
  R4400-versus-R5000 behaviour by construction, so it keeps its two-value
  `CPU_*` and genuinely has no expected answer for a third CPU.

Reasoning, **not** tested — treat as a starting point, not as fact:

- That IP32 (O2) bases RAM at physical 0 and uses CRIME/MACE rather than
  HPC3/IOC2. This is the premise the whole "it will not run on an O2"
  conclusion rests on.
- Anything about Octane or Origin.
- The exact ARCS vector layout (below).

No run on real SGI hardware is recorded anywhere in this repo. `run-prom.sh`
exists and is built for it, but every number we have is the emulator's opinion
of itself.

## If someone ports it: use ARCS, not per-machine drivers

The obvious approach — write a MACE UART driver for the O2, another for
Octane — is the wrong one, because it is per-machine and it has a chicken-and-egg
problem (you cannot debug a console with no console).

Every SGI machine from the Indigo onwards boots **ARCS** firmware, which
defines a standard callable console interface *and* a memory-map query. That is
one path for the whole family, and it happens to solve both of the hard
assumptions above: console, and where RAM actually is (so the relocation target
can be discovered rather than hardcoded). The board SYSID line would stay
per-machine, and it is cosmetic.

Two things to know before starting:

- ARCS is a **PROM-booted** path only. IRIS's `--load-elf` never runs the PROM,
  so the direct SCC console has to stay as the emulator's fast path — two
  console paths selected at runtime on whether a firmware vector exists, not
  one.
- **It can be developed without real hardware.** IRIS executes a real 512 KB
  SGI PROM image as guest code (`src/prombin.rs`, `Prom::from_file_or_embedded`),
  so the PROM's own ARCS implementation runs inside the emulator, and
  `bench/run/run-prom.sh` already boots the suite through it. Real hardware then
  confirms it generalises rather than being the development loop.
  `src/debug.md` has notes on the PROM's `arcs_printf`.

## Gotchas already paid for

- **A CPU name must be a single token.** It is emitted as the value of `cpu=`
  in the machine block, which is parsed by splitting on whitespace and then on
  `=`. `MIPS imp 0xab` silently truncated to `cpu="MIPS"`; it is now
  `MIPS-imp-0xab`, with a test asserting no whitespace.
- **A wrong reason in a comment outlives the code it explains.** The refusal to
  run on an unknown CPU claimed the goldens were selected by PRId. They never
  were — `golden.h` is one flat table. The real mechanism was that
  `cpu_kind == 0` matched no kernel's CPU mask, which is a very different
  problem with a very different fix. Check the mechanism, not the comment.
