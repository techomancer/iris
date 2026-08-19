# cpu-tests — bare-metal MIPS CPU test suite for IRIS

Self-checking MIPS III / MIPS IV tests that run **on the emulated CPU** with no
operating system, printing PASS/FAIL over the serial console. Built as a static
ELF32-BE binary loaded by the SGI PROM, and packaged as a bootable SGI CD image
(volume header + EFS partition 7).

Two CPUs, one binary: the suite reads `PRId` at startup and selects
expectations for **R4400** (`0x00000440`, MIPS III) or **R5000**
(`0x00002321`, MIPS IV). IRIS picks the CPU at build time —
`cargo build` for R4400, `cargo build --features r5k` for R5000.

**Status: planning, blocked.** Nothing is built yet.

- [PLAN.md](PLAN.md) — design, verified PROM boot path, test inventory,
  oracle strategy, phased roadmap, decisions taken.
- [EMULATOR-SUPPORT-PROMPT.md](EMULATOR-SUPPORT-PROMPT.md) — four additive
  IRIS features this suite needs (volume-directory writer, direct ELF load,
  TFTP server, test device). Written to be handed to a separate IRIS session.
  **Suite work starts once those land.**

## Relationship to the Rust tests

`src/mips_exec_test.rs` (101 tests) and `src/mips_tlb_test.rs` exercise the
interpreter's opcode handlers directly under `cargo test`. This suite exercises
the whole machine the way IRIX does — decode, JIT, caches, TLB, exception
vectors, CP0 timers — and is the only thing that can run the same binary
against the interpreter, the JIT, and real SGI hardware for comparison.
