# bench — the IRIS benchmark suite

Two questions, one suite:

* **How fast is this build of IRIS, and where does the time go?** Per kernel,
  in guest instructions per host second, so a change to the emulator can be
  measured rather than felt.
* **Is it still computing the right answers at speed?** Every kernel produces a
  deterministic checksum compared against a value computed independently on the
  host. The share that match is the accuracy score.

It runs bare metal — no IRIX, no PROM required — as a static MIPS III binary
loaded straight into RAM. The same C is compiled a second time for the host, so
the machine you are running the emulator on is measured by *the same kernels*
and the comparison between them is real rather than rhetorical.

```
make -C bench                 # build the guest binary
make -C bench golden          # (re)compute the expected checksums natively
make -C bench hostbench       # build the host baseline
make -C bench bench           # run once against ../target/release/iris
make -C bench matrix          # build and run every CPU x engine cell
make -C bench report          # turn the saved results into markdown
```

---

## What it measures

46 kernels in six groups. Every one is deterministic, self-checking where a
checksum is meaningful, and autoscaled by the harness to about 250 ms per timed
run so the whole suite finishes in a couple of minutes on the interpreter and
less on a JIT.

| group | kernels | what it is really testing |
|---|---|---|
| `int/` | alu, alu_ilp, alu64, muldiv, branch, bitops, **dhrystone** | dispatch cost, 64-bit paths, unpredictable branches |
| `fpu/` | scalar_s, scalar_d, divsqrt, transcend, **whetstone**, **linpack**, matmul | the FPU under sustained load, not one instruction at a time |
| `mem/` | latency L1/L2/DRAM, copy, fill, stream copy/scale/triad, unaligned, random | the cache hierarchy, as a curve rather than a number |
| `img/` | rgb2ycbcr, convolve3x3, sharpen5x5, dct8x8, resize, rotate90, composite, dither, histogram | what the machine was bought to do |
| `vid/` | motion_est, yuv2rgb | the MPEG inner loop, and playback |
| `codec/` | crc32, adler32, rle, lz, huffman | table lookups, hash chains, bit packing |
| `sys/` | tlb_hit, tlb_miss, exception, cache_flush, uncached, llsc | paths that exist only because this is an emulator |

**The imaging group is the point of the whole thing.** An Indy shipped with a
camera on the monitor and Photoshop in the catalogue; the workloads people ran
were images and video. A 3x3 convolution is three strided reads and a
multiply-accumulate per pixel and lives or dies on the cache model. A DCT is a
register-pressure problem. Motion estimation is a branch-free absolute
difference storm. Floyd-Steinberg is a strictly serial dependency across a
whole frame that nothing can reorder. Between them they exercise translation,
memory and dispatch in the proportions real software uses, which no ALU chain
does.

**The `sys` group is the other point.** TLB refills, exception round trips,
cache maintenance and uncached device reads are where an emulator can be
catastrophically slower than the hardware it stands in for, and they are
invisible to every conventional benchmark. `rules/perf/` and `rules/jitv2/` are
full of work whose payoff shows up here and nowhere else.

### Industry-standard figures

Three kernels report in units with four decades of published numbers behind
them, so an emulated Indy can be put next to a real one:

| kernel | unit | derived as |
|---|---|---|
| `int/dhrystone` | DMIPS | Dhrystone 2.1, rate / 1757 |
| `fpu/linpack` | MFLOPS | LINPACK 100x100 (dgefa + dgesl), rate / 1e6 |

`iris-bench report` computes both. One caveat and one deliberate omission:

* **LINPACK** generates the system once and restores it with a copy between
  solves, since the harness times whatever `run()` does while the reference
  implementation times only the factor and solve. That is ~80 KB of copy
  against ~690 k flops, and it is identical on every cell being compared.
* **Whetstone** is here for its instruction mix — the classic module structure
  and weights — but is reported in **passes per second, not MWIPS**. Converting
  needs the "Whetstone instructions per loop" constant from a reference
  implementation, and that is not something this suite can verify; a derived
  MWIPS resting on an unchecked factor of a thousand would look authoritative
  without being so. It also uses the suite's own transcendentals
  (`harness/bmath.h`) rather than a libm, because a libm difference between host
  and guest would be scored as an emulator fault.

The `mem/stream_*` kernels are STREAM-shaped and deliberately not STREAM: same
four loops, arrays sized to miss every cache, but an independent
implementation. Do not quote them as STREAM results.

---

## How a benchmark is put together

```c
static u64 v_dct(void);        /* one fixed reference run -> checksum */
static u64 r_dct(u32 iters);   /* iters passes -> work units performed */

BENCH("img/dct8x8", "blk", v_dct, r_dct, 1, BG_IMG),
```

`verify()` and `run()` are separate on purpose. The harness picks the iteration
count for `run()` by measurement, so it differs between a fast host and a slow
one — a checksum taken from the timed loop would depend on how fast the machine
is and could never be compared to anything.

The rules a kernel has to follow:

* **Call `work_alloc()` once per `run()`, outside the iteration loop.** It is a
  bump allocator with no free. A kernel that allocates per iteration exhausts
  24 MB in a handful of passes — and only at whatever iteration count the
  autoscaler happened to pick, which makes it look like a data-dependent
  failure rather than the plain bug it is.
* **Never checksum anything wider than a byte as raw bytes.** The golden values
  come from a little-endian host and the guest is big-endian; `cksum_bytes` over
  a `short[]` compares byte order, not results, and reports the difference as an
  emulator fault. Use `cksum_u64`/`cksum_f64`, which are defined on values.
* **Initialise everything you checksum.** A buffer whose tail is never written
  folds in whatever the allocator last held, which is not the same on the host
  as on the guest and not even the same between two host builds.
* **Say so if you mean to take exceptions.** `BENCH_EXC` marks a kernel that
  faults on purpose. Everywhere else a nonzero count means the shared dispatcher
  stepped over a faulting instruction and the kernel produced a number for doing
  something other than what it claims.

That last one is not hypothetical. `mem/unaligned` scored a plausible 871 k
accesses/s while taking an address error on three loads in four, because
`*(const u32 *)p` promises an alignment the pointer does not have and GCC
emitted a plain `lw`. The checksum caught it; the exception counter now catches
the whole class.

---

## The time base

Timing anything with CP0 Count would be measuring the timer model as much as the
workload — under IRIS, Count is virtual, materialised from a wall-clock anchor
at a `count_hz` inferred from the guest's own Compare writes. So the test device
(`--test-device`) carries two extra registers:

| register | offset | what it is |
|---|---|---|
| `TESTDEV_HOST_NS_LO/HI` | `0x10` / `0x14` | host monotonic nanoseconds |
| `TESTDEV_ICOUNT_LO/HI` | `0x18` / `0x1C` | guest instructions retired |
| `TESTDEV_CAPS` | `0x20` | capability bits |

Reading the LO half latches the whole 64-bit value; HI returns the high word of
that same latch, so LO-then-HI cannot tear.

The instruction counter is `MipsCore::hot.cycles`, advanced once per retired
instruction by **both** the interpreter and jitv2 (`emit_increment_cycles`),
which is what makes "guest MIPS" comparable across engines and the single most
useful number the suite produces.

The suite still measures the CP0 Count rate against the host clock at startup
and prints it. That is not decoration — it is a report on the emulator's timer
model, and it is what makes the fallback path honest when there is no host clock
at all.

**Probing matters.** An emulator built before these registers existed decodes
only 16 bytes and repeats, so `0x20` aliases back onto `SIGNATURE` — whose low
bit is set, so a naive `caps & CAP_TIMEBASE` says yes and every timing comes
back frozen. That happened on the first run. The suite now rejects a CAPS word
equal to the signature and then requires the clock to actually advance. And it
never *writes* an unprobed offset: on an old device `0x1C` aliases onto `EXIT`.

Without a host clock — real hardware, or an older build — everything falls back
to CP0 Count at an assumed 100 MHz (a 200 MHz R4x00, Count = clock/2), and the
header says so in as many words.

---

## Output

A human table, streamed a line at a time because the run takes minutes and
something that prints nothing until it is over is indistinguishable from a hang:

```
benchmark                 unit       rate/s    guest-MIPS   time%  acc
------------------------------------------------------------------
int/alu                   ops      38479208       64.93          ok
fpu/linpack               flop      5874487       34.25          ok
img/dct8x8                blk           3766       49.21          ok
sys/tlb_miss              miss       1948786       24.94          -
------------------------------------------------------------------
  wall clock         15.48 s
  guest work        756.28 M instructions
  emulator speed     48.85 MIPS (guest instructions per host second)
  accuracy            97.5 %  (39 of 40 checksums matched)
```

then the two rankings that answer "what is taking a while" — which are
different lists, because a kernel can dominate the wall clock simply by being
long, and a kernel can be terrible per instruction while barely registering in
the total:

```
  Where the time went (largest share of wall clock)
  Where the emulator works hardest (fewest guest MIPS)
```

then a machine-readable block between `IRIS-BENCH-BEGIN` and `IRIS-BENCH-END`
that `iris-bench` parses. Everything in it is an integer — a freestanding `%f`
would need its own float formatter, and nanoseconds, instructions, work units
and checksums are all exact as integers. Rates are derived by the host, which
has a real printf.

---

## iris-bench

```
iris-bench run     [--iris PATH] [--elf PATH] [--label NAME]
iris-bench host    # measure this machine with the same kernels
iris-bench matrix  [--cells r4400-interp,r5000-jitv2] [--force-build]
iris-bench report  [--baseline CELL] [--format md|json|text]
iris-bench cells   # what matrix knows how to build
```

`matrix` builds a **separate emulator per cell**, because the CPU model and the
JIT are compile-time cargo features and there is no runtime switch to flip:

| cell | features |
|---|---|
| `r4400-interp` | (default) |
| `r5000-interp` | `r5k` |
| `r4400-jitv2` | `jitv2` |
| `r5000-jitv2` | `r5k,jitv2` |
| `r4400-lightning` | `lightning` |
| `r4400-jitv2-lightning` | `jitv2,lightning` |

Each build is copied to `bench/build/iris-<cell>` before the next one starts —
the next `cargo build` overwrites `target/release/iris`, and a matrix that races
its own artefacts produces results labelled with the wrong build. After the run,
the guest's own `#machine cpu=` line (read from PRId, so it is authoritative) is
checked against what the cell claims. cpu-tests has the same guard for the same
reason: an `--features r5k` build once overwrote the binary between the copy and
the run, and an "R4400" cell silently exercised an R5000.

The report gives per-cell summaries with DMIPS/MWIPS/MFLOPS, per-kernel
throughput with speedups against a baseline cell and against native, any
checksum mismatches, any unexpected exceptions, and per-cell time-share and
efficiency rankings.

---

## The other half: `bench/irix/`

Everything above runs with no operating system, which is the right way to
measure a CPU and the wrong way to answer "is this usable". `bench/irix/`
measures the machine as a user meets it — a filesystem on an emulated SCSI
disk, IRIX's buffer cache and syscall path, the tools that shipped in the box,
and the X server driving REX3:

```sh
# with an emulator already running --ci and IRIX sitting at a shell prompt
iris-bench irix --socket /tmp/iris.sock
```

Steps live in `bench/irix/steps.toml` and are ordinary shell one-liners: `dd`
through the filesystem cold and warm, 500 small files created/stat'd/removed,
`sum`/`compress`/`gzip`/`tar`, 256 MB through the read/write syscall pair with
no disk in the path, loopback ping, and `xwd` reading the root window back out
of the framebuffer. A step naming a `requires` program is **skipped** when that
program is not installed rather than failed — IRIX installs vary enormously and
a missing `x11perf` is not a benchmark result.

Timing is done on the host around one `iris-ci run`, with the measured no-op
round trip subtracted, so nothing depends on the guest having a usable clock.
Commands run as `sh -c '<cmd>'` with an IRIX-shaped PATH whatever the login
shell is — so a step command may not contain a single quote.

Nothing here is checksummed. These are IRIX's own tools against IRIX's own
filesystem and their output is not ours to predict; the accuracy score belongs
to the bare-metal half.

> **Status: written, not yet run.** There is no IRIX disk image in this working
> tree, so the step list has been reasoned about rather than executed. The
> per-step `requires` gate is what makes that safe to ship — an install that
> lacks a tool skips it — but expect to adjust paths on first contact.

## Running it elsewhere

**Through the PROM**, which is also how you would run it on real hardware:

```sh
make -C bench image        # volume-header image via mkvh
bench/run/run-prom.sh      # boot -f dksc(0,2,8)irisbench
```

Slower than `--load-elf`, but it exercises the real path: the PROM reads the
volume header, loads the ELF and jumps to it. An image built this way can be
burned to a CD and booted on an actual Indy — which is the only way to get a
reference number that is not an emulator's opinion of itself. There is no test
device on real hardware, so results come back over serial with CP0 Count as the
time base; the header says so.

**Filtering.** There is no runtime selector — a bare-metal binary loaded with
`--load-elf` has nowhere to take arguments from. Comment out a group in
`harness/groups.c` and rebuild, or filter in the report.

---

## Layout

```
harness/     benchlib (time base, work area, checksums), main (the runner),
             bmath (a libm that is the same everywhere), link.ld, string.c,
             tlbasm.S (the TLB refill handler for sys/tlb_miss)
kernels/     integer fpu memory imaging codec sys
gen/         golden.c (the oracle) + hostplat.c (the host platform layer)
golden/      golden.h — generated, checked in
run/         bare.toml, run-local.sh, run-prom.sh
```

The toolchain probe, the SCC console and the startup/exception code are shared
with `cpu-tests/` rather than duplicated. What is *not* shared is the float ABI
(this suite wants the FPU; cpu-tests drives it by hand under `-msoft-float`),
the link layout (this one needs megabytes of working set above `_end`), and the
runner.

The two suites answer different questions and neither replaces the other.
cpu-tests asks "is this instruction correct", one instruction at a time with
clean state. This asks "is it still correct after ten million of them, and how
long did that take".
