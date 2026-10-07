# The emulated CPU is a runtime flag now, not a cargo feature

`--cpu r4400|r5000|r10000` selects the model at construction. All three are
compiled into every build as separate monomorphisations. The `r5k` and
`mips4` Cargo features were removed on 2026-10-01.

`src/machine.rs:752` dispatches `build_cpu!(R4400Cache)` / `build_cpu!(R5000Cache)`
from `cfg.machine.cpu`, and MIPS IV availability rides along with it — `C::MIPS4`
is an associated const on that type parameter (`R4400Cache::MIPS4 == false`,
`R5000Cache::MIPS4 == true`, asserted in `src/cpu/mips_exec_test.rs:212`). So
`--cpu r5000` really does get MIPS IV, and `--cpu r4400` really does raise
Reserved Instruction on it. No rebuild, no feature flag.

## Matrix builds

`cpu-tests/README.md` and `run/matrix.sh` describe CPU selection at runtime.
The runner retains separate cached files per cell (`build/iris-<cpu>-<engine>`)
but passes no CPU Cargo feature; only the JIT engine needs `jitv2`.

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

## Startup logging

`src/main.rs` applies `IRIS_DEBUG_LOG` after `Machine::new` initializes the
devlog registry. It adds a stderr sink and enables the named modules. This
startup path no longer has the initialization-order problem recorded here.
Devlog output requires a `developer` build; ordinary `log` warnings use the
CLI's `env_logger` backend.

Use the monitor instead (`Monitor listening on 127.0.0.1:8888`): `scsi debug on`,
`scsi regs`, `scsi wdt`. `scsi regs` in particular is how the WD33C93A constants
in `cpu-tests/harness/scsilog.c` were derived from the PROM's own driver.
