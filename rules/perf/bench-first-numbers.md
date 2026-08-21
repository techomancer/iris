# First benchmark matrix — 2026-08-21

The first complete run of `bench/`, kept as a baseline and for the four
findings it produced. Reproduce with `iris-bench matrix`; compare with
`iris-bench report --baseline r4400-interp`.

Host: Intel Core i5-9500T @ 2.20 GHz, 6 cores, Linux x86_64. A modest desktop
CPU — the ratios below are what a laptop-class machine gives, not a workstation.

| cell | accuracy | guest MIPS | DMIPS | LINPACK MFLOPS |
|---|---:|---:|---:|---:|
| r4400-interp | 100% (40/40) | 50.8 | 71.1 | 5.86 |
| r5000-interp | 100% (40/40) | 44.2 | 58.9 | 5.04 |
| r4400-jitv2 | 100% (40/40) | **201.8** | **214.1** | 10.08 |
| r5000-jitv2 | 100% (40/40) | 209.6 | 197.4 | 9.33 |

**214 DMIPS with jitv2** puts an emulated Indy in the same range as the real
200 MHz R4400 it stands in for, on this host. The interpreter is about a third
of that.

**Run-to-run variation is under 1%.** Two independent full matrix runs on an
otherwise idle machine agreed to within 1.1% on every headline figure (50.8 vs
51.0 MIPS, 201.8 vs 202.9, 214.1 vs 213.0 DMIPS). So a 5% change between
commits is real and worth chasing; anything under 2% is noise. That is what the
best-of-two-at-250 ms design buys, and it is why the CI job gates on accuracy
rather than on throughput — a shared runner has nothing like this floor.

---

## 1. jitv2 buys nothing on floating point

Guest MIPS, interpreter → jitv2 on R4400:

| kernel | interp | jitv2 | |
|---|---:|---:|---|
| `int/alu` | 65.0 | 631.6 | **9.7x** |
| `int/muldiv` | 60.6 | 671.9 | 11.1x |
| `fpu/scalar_d` | 23.6 | 23.5 | **1.00x** |
| `fpu/scalar_s` | 24.0 | 24.3 | 1.01x |
| `fpu/divsqrt` | 18.3 | 18.6 | 1.02x |
| `fpu/linpack` | 34.2 | 58.8 | 1.7x |
| `fpu/matmul` | 43.2 | 95.4 | 2.2x |

The three kernels that are *nothing but* FP arithmetic land within 2% of the
interpreter. The mixed ones (linpack, matmul) gain, and gain roughly in
proportion to how much integer and address arithmetic they carry. So the
compiled FP path costs the same as the interpreted one: dispatch is not what an
FP instruction spends its time on here, the FP semantics are, and jitv2 emits
the same semantics.

FP emitters exist (`opcode_support.rs` asserts `Fadd_s` has one and the
per-kind toggle defaults on), so this is not a coverage gap. It is where the
next real FP speedup has to come from, and it is worth roughly 4x on
`fpu/scalar_*` before it would stop being the limiter.

## 2. The emulated L2 costs 3-4x on memory-bound code

Plain `r5k` reports `Config.SC = 1` (no secondary cache), so the L2 model is
bypassed entirely. That makes the R5000 cells a natural controlled experiment,
and the result is large:

| kernel | r4400-interp | r5000-interp | |
|---|---:|---:|---|
| `mem/copy` | 10.8 MIPS | 41.9 MIPS | **3.9x** |
| `mem/stream_copy` | 9.6 | 34.3 | 3.6x |
| `mem/stream_scale` | 11.0 | 34.7 | 3.2x |
| `mem/latency_dram` | 10.0 | 24.7 | 2.5x |

Everything else goes the other way — R5000 is ~20% *slower* on pure ALU work
(65.0 → 51.8 MIPS on `int/alu`), which is the price of the 2-way L1 probe on
every fetch. So the R5000 cells are not "faster"; they are the same machine
with the L2 model switched off, and switching it off is worth 3-4x on anything
that streams memory.

HACKING.md already says "fully emulated L2 was a mistake in hindsight". This is
that sentence with a number on it.

## 3. jitv2 makes memory-bound loops *slower* on R4400

On the L2-enabled configuration only:

| kernel | interp | jitv2 |
|---|---:|---:|
| `mem/copy` | 10.8 | **9.1** |
| `mem/stream_copy` | 9.6 | **7.9** |
| `mem/stream_scale` | 11.0 | **9.2** |
| `mem/stream_triad` | 12.8 | **11.7** |

A 10-15% regression, and it does not appear on R5000 (where the same kernels go
41.9 → 128.3), so it is an interaction with the L2 path rather than with the
loops themselves. Worth a look: these are the only kernels in the whole suite
where turning the JIT on costs throughput.

## 4. Region boundaries cost what the design says they cost

`CACHE` and `LL`/`SC` are architecturally excluded from jitv2 — deliberate
region boundaries, per `rules/jitv2/unsupported-instructions.md`. The suite
prices them:

| kernel | interp | jitv2 |
|---|---:|---:|
| `sys/cache_flush` | 55.3 | 38.7 |
| `sys/llsc` | 57.5 | 42.8 |
| `sys/tlb_miss` | 24.8 | 21.8 |

A 25-30% loss on code dominated by a boundary instruction. That is the expected
shape of the tradeoff rather than a bug, but it is the first time it has had a
number, and it says what a lock-heavy or cache-management-heavy guest workload
actually pays.

---

## Two more things worth knowing

**`sys/uncached` is 4x faster under jitv2** (69.6 → 289.2 MB/s) — uncached
KSEG1 reads go straight down the MC bus and the win there is all dispatch.

**`img/rgb2ycbcr` and `img/composite` are ~2.5x slower on R5000-jitv2 than on
R4400-jitv2** (309 → 130 MIPS, 214 → 100 MIPS) while every other imaging kernel
is within 20%. Both are the byte-store-heaviest kernels in the group. Not
chased down.

**Native ratio.** With jitv2 the emulator runs integer ALU work at about 1/6 of
this host's native rate, imaging kernels at 1/30 to 1/100, and FP at 1/150 or
worse. `vid/motion_est` reads as 1/499, but the host build almost certainly
vectorises that SAD loop into `psadbw`; treat the extreme end of that column as
a compiler comparison, not an emulator one.
