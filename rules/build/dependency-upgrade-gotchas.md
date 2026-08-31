# Dependency upgrade gotchas (2026-08 sweep)

Notes from the crate-update pass that moved iris off `bitfield 0.14 / cpal 0.15
/ env_logger 0.10 / png 0.17 / rfd 0.15 / socket2 0.5 / spin 0.10 /
windows-sys 0.52 / libchdman-rs 0.288` and iris-gui off `dirs 5 / if-addrs 0.13
/ toml 0.8`. Only the non-obvious parts are recorded here.

## windows-sys: `Win32_Foundation` was arriving by accident

`src/thread_affinity.rs` calls `SetThreadAffinityMask`/`GetCurrentThread`, and
both signatures name `Foundation::HANDLE`. windows-sys gates that module behind
the **`Win32_Foundation`** feature, which `Win32_System_Threading` does *not*
imply — in 0.52 the generated binding even carried a
`#[doc = "Required features: \"Win32_Foundation\""]` marker.

iris only ever declared `Win32_System_Threading`. The Windows build worked
purely because `rfd`, `socket2`, and `anstyle-wincon` resolve to the *same*
windows-sys version and enable `Win32_Foundation` themselves, so Cargo's feature
unification filled the gap. That is a build that depends on who else happens to
be in the graph — drop or re-version any of those crates and the Windows build
breaks in a way that reproduces on no other platform.

`Cargo.toml` now declares both features explicitly. **Don't "simplify" that back
down to one feature.**

macOS/Linux dev boxes cannot catch this: the module only compiles on Windows, so
Cargo accepting the manifest proves nothing about the Windows build. Until
2026-08-31 `windows-sys` was a hard (non target-gated) dependency, which at
least made Cargo *validate the feature names* on every host; it is now gated to
`cfg(target_os = "windows")` (see the riscv64 section below), so off-Windows not
even the feature names are checked.

## cpal 0.18 API churn (`src/hal2.rs`)

Four unrelated breaks in the audio-output path:

- `SampleRate` is now `pub type SampleRate = u32` — a plain alias, not a tuple
  struct. `cpal::SampleRate(rate)` → `rate`.
- All error types collapsed into one `cpal::Error` + `ErrorKind`. The stream
  error callback takes `cpal::Error`; `cpal::StreamError` is gone.
- `build_output_stream` takes `config: StreamConfig` **by value**. `StreamConfig`
  is `Copy`, so the f32-then-i16 fallback can still pass the same config twice.
- `DeviceTrait::name()` is gone. `DeviceTrait: Display`, so the device name is
  `device.to_string()` / `{}`. (`description()` returns the structured
  `DeviceDescription` if more than the name is ever needed.)

`HostTrait` lost `id()`, but the concrete `cpal::Host` keeps an inherent `id()`,
so `host.id()` still compiles.

## rfd ≥ 0.16 dropped the async-runtime features

`rfd 0.15` took `features = ["xdg-portal", "async-std"]`. 0.16+ removed the
`async-std`/`tokio` executor features outright; `xdg-portal` now implies its own
`pollster` executor. The manifest is just `features = ["xdg-portal"]` — listing
`pollster` alongside it is redundant.

rfd also gained a `wayland` feature (for parenting dialogs to a Wayland surface).
iris calls no `set_parent`, so it stays off, and this is not a regression: the
old `default-features = false` config never had an equivalent either.

## Smaller renames

- **socket2 0.6**: `Socket::set_ttl` → `set_ttl_v4` (disambiguated from the IPv6
  hop limit). `src/net.rs` sets it on an `IPV4` ICMP socket, so the rename is a
  straight substitution.
- **png 0.18**: `Reader::output_buffer_size()` returns `Option<usize>` (`None`
  when the size would overflow `usize`).

## Verified drop-ins — don't re-derive these

- **`dirs` 5 → 6 does not move any user data.** `config_dir()` and `data_dir()`
  are byte-identical in both versions on Linux (XDG), macOS (Application
  Support), and Windows (Roaming AppData). iris-gui's `gui.json` machine store
  and `iris-gui.pid` stay exactly where they were; no migration needed.
- **`bitfield` 0.14 → 0.19 is a clean drop-in** despite five major versions.
  All eight `bitfield! { ... }` blocks (rex3, vc2, mips_cache_v2, hal2,
  saa7191, mips_exec) compile untouched.
- **`libchdman-rs` 0.288 → 0.289** needs no code change. The `chd_disk` tests
  exercise it for real — they create compressed CHDs, write COW diffs, and
  flatten them — so a green `cargo test --features chd chd_disk` is meaningful
  coverage of the upgrade, not just a compile check.

## What this sweep deliberately left alone

`egui`/`eframe` (0.35, though 0.36.1 exists) and the whole windowing stack that
shares crates with them: `winit`, `glutin`, `glutin-winit`, `glow`,
`raw-window-handle`. `winit` in particular is `[patch.crates-io]`-ed to
`third_party/winit-0.30.13` for the App Store private-API fix — see
`rules/macos/appstore-private-api.md` before touching it.

Side effect worth knowing: iris's own `glow` (0.13) and egui's (0.17) are still
two versions in the lockfile. `png` used to be split the same way and is now
unified on 0.18.

## windows-sys must stay under `cfg(target_os = "windows")` (riscv64, 2026-08-31)

Symptom: the **riscv64 cross leg alone** fails while all nine other release legs
are green —

```
error[E0425]: cannot find type `MEMORY_BASIC_INFORMATION` in this scope
  --> .../windows-sys-0.61.2/src/Windows/Win32/System/Memory/mod.rs:109
error: could not compile `windows-sys` (lib) due to 2 previous errors
```

Cause: `windows-sys` sat in the plain `[dependencies]` table, so Cargo built it
for **every** target. Its `windows_link::link!` declarations expand on any
target, but the arch-dependent structs are defined only for the architectures
Windows runs on (x86, x86_64, aarch64, arm). `riscv64gc` matches none of them,
so `VirtualQuery`/`VirtualQueryEx` end up naming a type that was never defined.
x64/arm64 Linux and macOS never noticed: their arch *is* one Windows supports,
so the structs exist and the crate compiles as dead weight.

Trigger: commit 73a10e2 added the `Win32_System_Memory` feature (for ppmem's
`map_windows.rs`). The previously enabled features happened to declare nothing
that referenced an arch-gated struct, so the latent breakage only surfaced then.

Fix: the dependency now lives in
`[target.'cfg(target_os = "windows")'.dependencies]`. Both consumers were
already gated — `src/thread_affinity.rs` behind `#[cfg(windows)]`, and
`src/ppmem/map_windows.rs` reached only via `#[cfg_attr(windows, path = ...)]`
in `src/ppmem/map.rs` — so nothing off-Windows ever referenced the crate.

**Before adding another `Win32_*` feature**, confirm the crate stays out of the
non-Windows graphs:

```
cargo tree -e normal --target riscv64gc-unknown-linux-gnu -i windows-sys
# must print "nothing to print"
```

Related: `Cargo.lock` is gitignored, so CI re-resolves the graph on every run.
A build can go red with no commit touching the code that broke — check the
resolved crate versions in the log before hunting through the diff.
