# Mac App Store rejects winit's private SkyLight blur API (`CGSSetWindowBackgroundBlurRadius`)

**Symptom.** App Store review rejects the `iris-gui` binary under **Guideline
2.5.1 (Performance — Software Requirements)**:

> The app uses or references the following non-public or deprecated APIs:
> Contents/MacOS/iris-gui — Symbols: `_CGSSetWindowBackgroundBlurRadius`

**Root cause.** `eframe 0.29` pulls in `winit 0.30` for window creation. winit's
macOS backend (`platform_impl/macos/window_delegate.rs::set_blur`) calls the
private SkyLight APIs `CGSSetWindowBackgroundBlurRadius` /
`CGSMainConnectionID`, declared in `platform_impl/macos/ffi.rs`. The call site
is reached unconditionally during window init (`set_blur(attrs.blur)`), so the
import lands in the linked binary **even though iris-gui never requests blur**
(`egui::ViewportBuilder` leaves `blur = false`). Apple's static binary scan
flags the imported symbol regardless of whether it's called at runtime.

Confirm with:

```
nm -u target/release/iris-gui | grep -i CGSSetWindowBackgroundBlurRadius
```

`U _CGSSetWindowBackgroundBlurRadius` = present (rejected). No output = clean.
(`_CGShieldingWindowLevel` also shows up but is a **public** CoreGraphics API —
Apple does not flag it.)

**Fix.** Vendor a patched winit and override it via `[patch.crates-io]`:

- `third_party/winit-0.30.13/` — copy of the crate with:
  - `set_blur` stubbed to a no-op (no `ffi::CGS…` calls),
  - the two private `extern` declarations removed from `ffi.rs` (and the
    now-unused `NSInteger` / `AnyObject` imports dropped).
- Root `Cargo.toml`: `[patch.crates-io] winit = { path = "third_party/winit-0.30.13" }`.

Only the `0.30.x` requirement (eframe → egui-winit → glutin-winit) matches the
patch. `iris`'s own `winit 0.29` dependency is the keyboard `KeyCode` type only
and creates no window inside `iris-gui`, so its `set_blur` is dead-stripped —
patching just the 0.30 copy removes the symbol entirely (verified with `nm -u`).

**Two-version gotcha.** Cargo allows only one `[patch.crates-io]` entry per
crate name, so you cannot patch both 0.29.15 and 0.30.13. That's fine here —
only the eframe (0.30) window code reaches `set_blur`. If a future change makes
`iris` create a winit-0.29 window inside the GUI process, re-check `nm -u`;
you'd then have to unify on a single winit version before patching.

**When bumping eframe/winit:** re-vendor the matching winit version, re-apply
the two-edit patch, and re-run the `nm -u` check before submitting.

## ⚠️ CORRECTION (2026-08-03): the symbol is STILL PRESENT — winit **0.29** also has it

The claim above that iris's own `winit 0.29` copy "creates no window inside
iris-gui, so its `set_blur` is dead-stripped" **does not hold**. Measured on a
plain `cargo build --release -p iris-gui` (profile `lto = "fat"`):

```
$ nm -u target/release/iris-gui | grep -i CGS
_CGSMainConnectionID
_CGSSetWindowBackgroundBlurRadius
```

**This is pre-existing, not caused by the egui 0.29 → 0.35 bump** — the
`IRIS.app` bundle built 2026-07-02 (long before) has both symbols too. Patching
only the 0.30 copy is not sufficient.

**Source: CONFIRMED — the unpatched `winit 0.29.15`.** The vendored
`third_party/winit-0.30.13/` stub is intact and doing its job; the import comes
from iris's *own* winit 0.29 dependency, which `[patch.crates-io]` never touched
(it only matches the `0.30.x` requirement). Only two crates in the whole registry
declare the symbol — `winit-0.29.15` and `winit-0.30.13` — and the binary embeds
source paths for **both**:

```
$ strings -a target/aarch64-apple-darwin/release/iris-gui \
    | grep -oE "winit-0\.29\.[0-9]+|third_party/winit-0\.30\.13" | sort | uniq -c
  12 third_party/winit-0.30.13      <- patched, clean
  24 winit-0.29.15                  <- UNPATCHED registry copy
```

…including `…/winit-0.29.15/src/platform_impl/macos/window.rs`, which is exactly
where the call lives (`window.rs:597`, extern at `ffi.rs:120`).

**Merely depending on winit 0.29 links its macOS backend — reachability is not
enough to strip it.** Commenting out `pub mod ui;` in `src/lib.rs` and rebuilding
does **not** remove the symbol (measured). winit's macOS backend registers
Objective-C classes via `declare_class!`, which emits `#[used]` statics that
survive dead-stripping regardless of whether any Rust code calls into them. The
Jun-16 commit's assumption — "creates no window in iris-gui, so its blur code is
dead-stripped" — is therefore wrong.

⚠️ **Do not attribute this by `nm`-ing rlibs in `target/release/deps/`.** The
release profile is `lto = "fat"`, so rlibs hold LLVM bitcode, not machine code —
`nm` reports "no symbols" or errors with `Unknown attribute kind`, which reads
like a clean result and proves nothing. Use `strings -a` on the linked binary
(the profile sets `debug = 1`, so source paths are embedded), or bisect by
removing a dependency and re-linking.

### FIXED (2026-08-03): unify the graph on ONE winit

The fix was not a new patch — it was removing the *second* winit so the existing
patch covers everything. iris's own deps moved 0.29 → 0.30.13:

| dep | was | now |
|---|---|---|
| `winit` (root + iris-gui) | 0.29 | 0.30 |
| `glutin` | 0.31 | 0.32 |
| `glutin-winit` | 0.4 | 0.5 |
| `raw-window-handle` | 0.5 | 0.6 |

Cheaper than it looks, because two things did **not** break:
- **`KeyCode` is byte-identical between winit 0.29 and 0.30** — `ps2.rs`,
  `push_kb` and all keyboard code needed zero changes.
- **`EventLoop::run(closure)` still exists in 0.30** (deprecated, not removed) —
  `src/ui.rs` did *not* need an `ApplicationHandler` rewrite.

Only 8 mechanical edits in `src/ui.rs` + `src/headless_gl.rs`:
`WindowBuilder` → `WindowAttributes::default()`, `.with_window_builder()` →
`.with_window_attributes()`, and rwh 0.6 made `raw_window_handle()` /
`build_surface_attributes()` return `Result`.

Verified clean on the real pipeline build **and** the signed bundle:
```
$ ./scripts/build-macos.sh appstore
$ nm -u IRIS.app/Contents/MacOS/iris-gui | grep CGS   -> (nothing)
$ strings -a IRIS.app/Contents/MacOS/iris-gui | grep -oE "winit-0\.29\.[0-9]+|third_party/winit-0\.30\.13" | sort | uniq -c
  12 third_party/winit-0.30.13    # and NO winit-0.29
```
Control: `_CGShieldingWindowLevel` (public) is still present, so `nm` is working.

## Can we drop the vendored copy and take winit from git? NOT YET — both doors shut

The upstream fix **is** in winit **master** — `winit-appkit/src/window_delegate.rs`
gates the call behind `#[cfg(feature = "private-apple-apis")]`, off by default.
But it is unusable here, and both alternatives were measured, not assumed:

| source | version | has fix? | usable? |
|---|---|---|---|
| crates.io `0.30.13` | 0.30.13 | ✗ | ✓ (what we vendor + stub) |
| crates.io `0.31.0-beta.1/2` | 0.31.0-beta | ✗ (no `private-apple-apis` in published features) | — |
| git `master` | **0.31.0-beta.2** | ✓ | ✗ **version mismatch** |
| git `v0.30.x` branch | 0.30.13 | ✗ **not backported** (still calls it unconditionally) | — |

egui-winit 0.35 requires `winit = "0.30.13"` (`^0.30.13`), which `0.31.0-beta.2`
does not satisfy. `[patch.crates-io] winit = { git = "…", branch = "master" }`
fails **silently and dangerously**:

```
warning: patch `winit v0.31.0-beta.2 (git master)` was not used in the crate graph
     Adding winit v0.30.13      <- falls back to the UNPATCHED registry copy
```

i.e. the git patch is ignored and the private symbol comes straight back. Do not
use it. Our vendored stub already produces exactly what master's default does
(no `CGS*` import), so there is nothing to gain until **egui/eframe bumps to a
winit 0.31 release** — at that point switch to upstream, leave
`private-apple-apis` off, delete `third_party/winit-0.30.13/` and the `[patch]`,
and re-run the `nm -u` check.

**Reproduce the check:** `./scripts/build-macos.sh appstore` then
`nm -u target/aarch64-apple-darwin/release/iris-gui | grep CGS`. A plain
`cargo build --release -p iris-gui` reproduces it too — both were measured.
There is still **no CI gate**; add one to the appstore workflow.

**No CI gate exists for this** — there is no `nm -u` check in `.github/workflows/`
or `scripts/`. Add one to the appstore workflow so a regression can't ship.

## Upstream status (don't file a new bug — already tracked)

- winit issue **#4205** "_CGSSetWindowBackgroundBlurRadius non-public or
  deprecated API" — open, milestone **winit 0.31.0**.
- winit PR **#4541** "macOS: Feature-gate `CGSSetWindowBackgroundBlurRadius`" —
  open/in-progress. Puts the call behind a `private-apple-apis` Cargo feature
  (off by default → symbol absent unless opted in). Resolves #4205.
- #4574 (dup App Store rejection report) closed as duplicate; #4538 (remove it
  outright) abandoned.

**Migration:** once IRIS moves to a winit (≥0.31) that ships the feature gate —
which only happens after eframe bumps to a winit-0.31 release and we bump eframe
— delete `third_party/winit-0.30.13/` and the `[patch.crates-io]`, and just make
sure the `private-apple-apis` feature stays disabled (and that eframe doesn't
enable it). Re-run the `nm -u` check to confirm.
