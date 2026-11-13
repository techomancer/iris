# The guest compiler changes the benchmark

Kind: measure. Measured 2026-11; re-measure before reusing the numbers.

The bench kernels are C compiled for MIPS, so the compiler decides how many
guest instructions a kernel runs. A different compiler on unchanged sources is a
different workload, and every rate moves with it while the emulator stays the
same.

## What happened

`bench/prebuilt/irisbench.elf` was rebuilt with Debian's GCC 14.2.0 (6adc7927)
after being built with GCC 16.2.1. Dhrystone and rotate90 then ran about 37%
more guest instructions per iteration, so their rates fell about 0.73-0.76x with
no change in emulator cost, and the drop was read as an interpreter regression.
`check-prebuilt` hashed the sources, not the compiler, so CI passed the image.

## Guest instructions per iteration, GCC 14.2.0 against GCC 16.2.1

Plain interpreter, R4400, full cache model, deterministic mode, all 46 kernels
of the current sources built both ways. Seven kernels differ by more than 0.5%
(GCC 14 to GCC 16):

| kernel | change |
|---|---:|
| dhrystone | 428.15 to 313.10 (-26.9%) |
| rotate90 | 1,440,274 to 1,049,282 (-27.1%) |
| lz | -9.0% |
| rle | +4.4% |
| composite | +3.1% |
| whetstone | -0.6% |
| huffman | +0.6% |

The other 39 differ by less than 0.5%. Rates measured on different images
are not comparable, kernel by kernel.

## How to see it

- Instructions per iteration (`icount / iters` per row of a result JSON) shows a
  compiler change directly: it is the same on every run of one image.
- The DWARF producer string names the compiler: `grep -a 'GNU C' irisbench.elf`.
- To compare two emulator builds, hold the guest fixed:
  `iris-bench run --iris BIN --elf FIXED`. Never use the image embedded in each
  commit, or a bisect lands on the commit that rebuilt the image.
  `iris-bench report` warns when the runs it compares used different images.

## What the pin does

`cpu-tests/toolchain.mk` names one compiler, GCC 16.2.1 for `mips64-linux-gnu`
from Fedora 45, and `cpu-tests/toolchain.Containerfile` installs exactly it.
Every guest binary is built with it. A Fedora GCC 16.2.1 build of the sources
matches the earlier GCC 16.2.1 image on all 46 kernels.

The builds are reproducible: both suites compile with
`-ffile-prefix-map=<repository root>=.`, so debug information does not record
the checkout's path. Two builds of `bench` in different directories gave the
same sha256 (0c92ebe4b0e8...).

The compiler also moves cpu-tests: GCC 16.2.1 shrank the code and moved the
stack, which exposed two tests that depended on where it landed
([cpu-tests-must-not-depend-on-code-size](../testing/cpu-tests-must-not-depend-on-code-size.md)).
The same tests fail under GCC 12.4 and GCC 16.2.1 in all four CPU and engine
cells.
