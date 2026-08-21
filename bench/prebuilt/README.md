# bench/prebuilt — the guest binary, checked in

`iris` links `irisbench.elf` in with `include_bytes!` (`src/benchsuite.rs`) so
the benchmark runs on a machine with no MIPS cross toolchain and no build step:
a released app, a sandboxed one, or anyone who just wants the number.
`make -C bench` still builds `build/irisbench.elf` for development; this is a
copy of a known-good one.

A checked-in build product that can drift is worse than no build product, and
this one drifts dangerously: accuracy is scored against golden checksums
compiled *into* the image, so a stale image against fresh goldens reports
failures that are not real. `.github/workflows/bench.yml` rebuilds it and fails
on any difference.

Refresh it with `make -C bench prebuilt` after changing anything the guest is
built from — the kernels, the harness, `cpu-tests/harness/`, the link script,
the compiler flags — and commit the result alongside the source change.

`PROVENANCE` records what the checked-in bytes hash to. It is written by the
`prebuilt` target; nothing reads it, but a reviewer can check it by hand and a
`git log` on it shows every time the image moved.
