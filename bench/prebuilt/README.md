# bench/prebuilt — the guest binary, checked in

`iris` links `irisbench.elf` in with `include_bytes!` (`src/benchsuite.rs`) so
the benchmark runs on a machine with no MIPS cross toolchain and no build step:
a released app, a sandboxed one, or anyone who just wants the number.
`make -C bench` still builds `build/irisbench.elf` for development; this is a
copy of a known-good one.

A checked-in build product that can drift is worse than no build product, and
this one drifts dangerously: accuracy is scored against golden checksums
compiled *into* the image, so a stale image against fresh goldens reports
failures that are not real. `.github/workflows/suites.yml` rebuilds it and fails
on any difference.

Refresh it with `make -C bench prebuilt` after changing anything the guest is
built from — the kernels, the harness, `cpu-tests/harness/`, the link script,
the compiler flags — and commit the result alongside the source change.

`PROVENANCE` records a digest of every source the image is built from, written
by the `prebuilt` target and verified by `make -C bench check-prebuilt` (which
CI runs). It is a *source* digest, not an image one, because two correct builds
of identical source do not produce identical bytes — the image is compiled with
`-g`, so DWARF records the build directory, and toolchain versions differ. The
image's own hash is recorded alongside for humans, and nothing compares it.

The digest cannot tell you the image *works*. The embedded-runner test does, by
running the suite out of the binary the image is linked into and requiring 100%
accuracy against the golden checksums compiled into it.
