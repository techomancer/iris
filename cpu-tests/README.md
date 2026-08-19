# cpu-tests

MIPS III / MIPS IV tests that run **on the emulated CPU** with no operating
system. They print PASS/FAIL over the serial console and report the failure
count as the emulator's exit code.

One binary covers both CPUs: it reads `PRId` at startup and picks expectations
for R4400 (MIPS III) or R5000 (MIPS IV).

**This branch is tests only** — no emulator changes. Findings are reported, not
fixed, so the suite keeps showing them.

## Build

Needs a MIPS cross toolchain:

```sh
sudo apt-get install gcc-mips-linux-gnu binutils-mips-linux-gnu
make                        # -> build/cputest.elf
```

No root? `make toolchain-local` unpacks the same packages into
`~/.local/opt`; the Makefile finds them automatically.

## Run

```sh
cargo build --release       # in the repo root, first
make run                    # loads the ELF straight into RAM and runs it
```

`make run` takes about a minute. You get a line per test and a summary:

```
alu/addu_sign_extends ...................... PASS
...
 RESULT: 783 checks passed, 19 failed  (172 tests)
IRIS-CPUTEST-DONE rc=19
```

The exit code **is** the failure count. `IRIS-CPUTEST-DONE` is the token to
match on if you are scripting it.

### Other CPUs and engines

The CPU and the JIT are compile-time cargo features, so each combination needs
its own IRIS build:

```sh
cargo build --release                                # R4400, interpreter
cargo build --release --features r5k,r5ksc_triton    # R5000
cargo build --release --features jitv2               # jitv2 (no env var needed)
```

Then run as above. `run/matrix.sh` does all four combinations and builds each
one for you; `CELLS="r4400-jitv2" run/matrix.sh` runs just one.

> The jitv2 cells have never been run. They are written but unproven — the
> first green run is still owed.

### Booting it like real hardware

```sh
make image                  # volume-header image via mkvh
run/run-prom.sh             # PROM: boot -f dksc(0,2,8)cputest
```

Slower, but it exercises the real path: the PROM reads the volume header, loads
the ELF, and jumps to it. This is what the bootable CD will use.

## Expected results

| | pass | fail | failing tests |
|---|---:|---:|---|
| R4400 | 783 | 19 | 7 |
| R5000 | 802 | 3 | 3 |

Every failure is a known finding, listed in [docs/findings.md](docs/findings.md).
Anything else is new — start with [docs/gotchas.md](docs/gotchas.md), which
collects the times the *test* was wrong and the emulator was right.

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

`CPU_ALL`, `CPU_R4400` or `CPU_R5000` says which parts it applies to; the
others are skipped rather than failed. `A`/`Z` are the strict asm prologue —
use them, or the assembler will quietly rewrite your instructions (see
[docs/gotchas.md](docs/gotchas.md)).

A whole new area also needs a `struct test_group` and one line in
`harness/tests.c`.

## Layout

```
harness/   startup, exception vectors, CHECK macros, console
tests/     identity alu muldiv mem branch excep cp0 tlb fpu cache mips4
run/       run-local.sh  matrix.sh  run-prom.sh  bare.toml  boot.toml
docs/      findings gotchas status oracle memory-map toolchain
```

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
