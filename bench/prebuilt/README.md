# bench/prebuilt — the guest binary, checked in

`iris` links `irisbench.elf` in with `include_bytes!` (`src/benchsuite.rs`) so
the benchmark runs on a machine with no MIPS cross toolchain and no build step:
a released app, a sandboxed one, or anyone who just wants the number.
`make -C bench` still builds `build/irisbench.elf` for development; this is a
copy of a known-good one.

A checked-in build product that can drift is worse than no build product, and
this one drifts dangerously: accuracy is scored against golden checksums
compiled *into* the image, so a stale image against fresh goldens reports
failures that are not real. And the compiler decides how much guest code each
kernel runs, so an image from another compiler is a different benchmark.

## After you change a kernel

The image depends on `bench/kernels`, `bench/harness`, `bench/golden`,
`cpu-tests/harness`, the link script and the compiler flags. After changing any
of them, rebuild it in the toolchain container
([cpu-tests/docs/toolchain.md](../../cpu-tests/docs/toolchain.md) has the
Podman form):

```sh
docker build -t iris-guest-toolchain -f cpu-tests/toolchain.Containerfile cpu-tests
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/src -w /src \
    iris-guest-toolchain make -C bench prebuilt
```

then commit `bench/prebuilt/irisbench.elf` with the source change. Moving a line
in a guest source changes the bytes too (debug line tables), so a comment edit
needs a refresh as well. `make -C cpu-tests all` rebuilds the cpu-tests ELF; it
is not checked in.

## How it is checked

`make -C bench check-prebuilt` reads the compiler out of the image's debug
information and fails, naming both versions, if it is not the pinned one. It
needs no toolchain. CI also rebuilds the image in the container and fails on any
byte difference; the builds are reproducible, so a difference means the image
is stale, hand-copied, or from another compiler.

This replaces an earlier digest of the guest sources (`PROVENANCE`). The digest
could not see the compiler or the flags, which is how an image from another
compiler got in. The cost: the rebuild needs the toolchain image, so it depends
on Docker Hub and Fedora's mirrors, and an outage there turns the check red.

Neither check can tell you the image *works*. The embedded-runner test does, by
running the suite out of the binary the image is linked into and requiring 100%
accuracy against the golden checksums compiled into it.
