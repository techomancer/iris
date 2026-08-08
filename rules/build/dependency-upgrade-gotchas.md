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

macOS/Linux dev boxes cannot catch this: `windows-sys` is a hard (non
target-gated) dependency, so Cargo *validates the feature names* everywhere, but
the module itself only compiles on Windows. Cargo accepting the manifest proves
nothing about the Windows build.

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
