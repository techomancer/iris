# libchdman-rs builds offline only with LIBCHDMAN_FORCE_SOURCE

Status: current (2026-11-19, libchdman-rs 0.289.6). Issue techomancer/iris#213.

Symptom: a `cargo vendor` + `--offline` build (distro packaging) fails in
libchdman-rs's build script, which tries to download a prebuilt static
archive.

Cause: iris enables libchdman-rs's `prebuilt` feature, so the build script
downloads `libchdman` for the host target. Features are additive; a packager
cannot turn `prebuilt` off from outside, and iris keeps it on so normal builds
stay fast and need no C++20 toolchain.

Rule:

- Since 0.289.1 the crate ships the MAME CHD subset in `mame-chd-src/`, so
  `cargo vendor` output contains everything. `LIBCHDMAN_FORCE_SOURCE=1`
  makes the build script compile it and skip the download. Keep iris's
  version floor at 0.289.1 or later.
- On x86_64 the source build passes `-mavx2 -mfma` to the FLAC sub-build, so
  the binary can SIGILL on pre-Haswell CPUs. Generic-baseline packages need
  `LIBCHDMAN_CPU=portable`.
- No other dependency downloads at build time. Checked by building
  `--release --bin iris` and checking `iris-gui --features iris/lightning,bundled`
  under `sandbox-exec` with outbound IP denied and a fresh `CARGO_HOME`
  pointed at the vendor directory.
- Do not bring back a `chd` Cargo feature to solve this; CHD stays
  unconditional (see `chd-disks-need-features-chd-at-build-time.md`).

The user-facing steps are in FEATURES.md, "Offline and distribution builds".
