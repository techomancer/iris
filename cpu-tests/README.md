# cpu-tests — bare-metal MIPS CPU test suite for IRIS

Self-checking MIPS III / MIPS IV tests that run **on the emulated CPU** with no
operating system, print PASS/FAIL over the serial console, and report the
failure count as the emulator's exit code.

One binary covers both CPUs: it reads `PRId` at startup and selects
expectations for **R4400** (`0x00000440`, MIPS III) or **R5000**
(`0x00002321`, MIPS IV). IRIS picks the CPU at build time — `cargo build` for
R4400, `cargo build --features r5k,r5ksc_triton` for R5000.

## Running it

```sh
make                     # build cputest.elf (needs a MIPS cross toolchain)
make run                 # run it under IRIS via --load-elf
run/matrix.sh            # R4400/R5000 x interp/JIT
run/run-prom.sh          # boot it through the PROM from a disk image
```

Without root, `make toolchain-local` unpacks a cross toolchain into
`~/.local/opt`. See [docs/toolchain.md](docs/toolchain.md).

## Layout

```
harness/   start.S (relocation, vectors, exception dispatch), testlib, console
tests/     identity alu muldiv mem branch excep cp0 tlb fpu cache mips4
run/       run-local.sh  matrix.sh  run-prom.sh  bare.toml  boot.toml
docs/      toolchain  memory-map  oracle  findings  gotchas
```

## Documentation

- **[docs/findings.md](docs/findings.md)** — architectural deviations the suite
  has surfaced in IRIS, and what each one costs in practice.
- **[docs/gotchas.md](docs/gotchas.md)** — cases where the suite was wrong and
  IRIS was right. Read this before believing a new failure.
- **[docs/oracle.md](docs/oracle.md)** — where expected values come from, and
  what the suite deliberately declines to assert.
- **[docs/memory-map.md](docs/memory-map.md)** — why the suite links at
  `0x88200000` and relocates itself there.
- **[docs/toolchain.md](docs/toolchain.md)** — why n32, and why no libgcc.
- **[PLAN.md](PLAN.md)** — the original roadmap and the verified PROM boot path.

## Relationship to the Rust tests

`src/mips_exec_test.rs` (101 tests) and `src/mips_tlb_test.rs` exercise the
interpreter's opcode handlers directly under `cargo test`. This suite exercises
the whole machine the way IRIX does — decode, JIT, caches, TLB, exception
vectors, CP0 timers — and is the only thing that can run the *same binary*
against the interpreter, the JIT, and (eventually) real SGI hardware for
comparison.

The two are complementary. The FP condition-code bug in
[docs/findings.md](docs/findings.md) is an example of something only this side
could find: IRIS's own unit test set the condition code through a helper, so it
never exercised the instruction field that was being decoded wrongly.

## Scope of this branch

**Tests only — no emulator changes.** Findings are reported, not fixed, so the
suite keeps showing them. The one fix written so far lives on its own branch,
`fix/fp-condition-code`; see [docs/findings.md](docs/findings.md) §1.
