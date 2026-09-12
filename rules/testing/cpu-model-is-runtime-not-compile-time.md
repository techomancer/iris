# The emulated CPU is a runtime flag now, not a cargo feature

`--cpu r4400|r5000` selects the model at construction. Both are compiled into
every build as separate monomorphisations, and `src/config.rs:1334` says so
outright: the `r5k` cargo feature "no longer selects the model and is vestigial
for that purpose".

`src/machine.rs:752` dispatches `build_cpu!(R4400Cache)` / `build_cpu!(R5000Cache)`
from `cfg.machine.cpu`, and MIPS IV availability rides along with it — `C::MIPS4`
is an associated const on that type parameter (`R4400Cache::MIPS4 == false`,
`R5000Cache::MIPS4 == true`, asserted in `src/mips_exec_test.rs:212`). So
`--cpu r5000` really does get MIPS IV, and `--cpu r4400` really does raise
Reserved Instruction on it. No rebuild, no feature flag.

## Two places still say otherwise

- `cpu-tests/README.md` — "The CPU and the JIT are compile-time cargo features,
  so each combination needs its own IRIS build". Half right: **jitv2 still is**
  a cargo feature, the CPU is not.
- `cpu-tests/run/matrix.sh` — its header comment repeats the claim, and
  `build_iris()` still builds a separate IRIS per CPU cell and caches it in
  `build/iris-<cpu>-<engine>/`.

The matrix still produces correct results; it is just doing two builds where one
would do. For a plain R4400-vs-R5000 comparison, one binary and two runs is
enough:

```sh
run/run-local.sh build/cputest.elf --cpu r4400
run/run-local.sh build/cputest.elf --cpu r5000
```

That also matters for a hardware diff, where both cells must come from the same
guest ELF — see
[running-cpu-tests-on-real-hardware.md](running-cpu-tests-on-real-hardware.md).

## Unrelated but adjacent: IRIS_DEBUG_LOG does nothing at startup

`src/main.rs:89` reads `IRIS_DEBUG_LOG` and enables the named devlog modules,
but it runs while `devlog::DEVLOG.get()` is still `None`, so the whole block —
including its own "DIAG: enabled …" message — is silently skipped. Setting
`IRIS_DEBUG_LOG=scsi,pdma` produces no trace at all.

Use the monitor instead (`Monitor listening on 127.0.0.1:8888`): `scsi debug on`,
`scsi regs`, `scsi wdt`. `scsi regs` in particular is how the WD33C93A constants
in `cpu-tests/harness/scsilog.c` were derived from the PROM's own driver.
