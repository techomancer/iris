# PCAP release + elevation plan

Status: **in progress** (updated 2026-06-20). The PCAP feature itself is **done**
and lives on branch `add-pcap-builds` (GenSayer's "Add PCap support" + danifunker
edition, merged). This document now tracks the remaining elevation / installer /
release-wiring work to completion.

> Tracking convention: every actionable item below is a `- [ ]` checkbox. Tick it
> (`- [x]`) as it lands. Items are tagged **[branch]** (lives on `add-pcap-builds`,
> the source/installer files) or **[main]** (lives on the fork's `main`, the CI
> pipeline). See the ownership rule immediately below — it is load-bearing.

---

## ⚠️ Branch / file-ownership rule (LOCKED 2026-06-20)

The build-pipeline files exist **only** on the fork's `main` branch:
`release.yml`, `appstore.yml`, `sync-upstream.yml`, and vendored build-only files
(e.g. the generated `COMBINED-LICENSE.txt`). They are **not** on `add-pcap-builds` and we must
**not** merge `origin/main` to pull them in.

Work therefore splits by file type:

| Lives on **`add-pcap-builds`** (this branch) | Lives on **`main`** (pipeline, separate) |
|---|---|
| Elevation/capture code in `iris-gui` + `iris` | `release.yml` build-variant jobs (Phase 1) |
| `installer/iris-gui.iss` Inno `[Code]` block | `make_dmg` Info.plist + ChmodBPF-copy steps |
| `installer/macos/chmod-bpf/*` resources | Pcap-installer build step (Windows job) |
| `installer/*.entitlements` edits | Native build-deps install (apt/pacman/WinPcap) |
| `Cargo.toml` packaging metadata (deb/rpm deps, setcap) | `DOWNLOADS.md` generator rows/footnotes |

So: a Phase that says "alter `release.yml`" is **main work** even when the
resources it consumes are authored here. The two halves meet only at release time.

---

## Current state (2026-06-20)

- [x] **Phase 0 — feature merged.** `add-pcap-builds` carries the pcap feature:
  - root `Cargo.toml`: `pcap = ["dep:pcap"]`, `pcap = { version = "2", optional = true }`
  - `iris-gui/Cargo.toml`: `pcap = ["iris/pcap"]`
  - `src/net_pcap.rs` (`PcapEngine`, `open_capture`, `is_pcap_mode`) present.
- [x] **Build sanity** on this branch (verified 2026-06-20, all green):
  - [x] `cargo build --features pcap`
  - [x] `cargo build -p iris-gui --features pcap`
  - [x] `cargo build -p iris-gui` (no-feature / App-Store path still green)
  - [x] `cargo build` (plain CLI) + 278 lib tests green

---

## Locked decisions (2026-06-19)

| Decision | Choice |
|---|---|
| Sequencing | Feature first (done), then elevation/installer, then release wiring on `main` |
| Build variants | PCAP implies lightning — 3 total: `standard`, `lightning`, `pcap` (= lightning+pcap, + CHD like every build). No 4-way cross-product. |
| In-app elevation | In scope for v1. Linux: rusty-backup `pkexec` whole-process re-exec. macOS: **ChmodBPF** (unprivileged app + one-time admin install). |
| Windows Npcap | Installer detects existing wpcap; if missing, offers to download+launch Npcap from npcap.com. Plus a matrix link. |
| App Store | **Never** ships PCAP (sandbox can't open a raw capture). Leave `appstore.yml` untouched. |

---

# Work tracker

## A. On-branch work (`add-pcap-builds`) — source / installer files

### A1 — Capture-permission detection + GUI surfacing  *(cross-cutting, DONE 2026-06-20)*
- [x] **[branch]** Surface `PcapEngine::open_capture` failure to the GUI: added
  `PcapStatus` enum (`src/net.rs`) + `NatControl.pcap_status` field/methods;
  `PcapEngine` sets Active/PermissionDenied/DeviceError via `classify_open_error`
  (`src/net_pcap.rs`, string-classifies EPERM/EACCES — pcap 2.x has no perm
  variant); `Machine::net_pcap_status()` accessor; `handle.rs` samples it into
  `Status` + exposes `EmulatorHandle::pcap_status()`. Unit-tested.
- [x] **[branch]** UX entry points (both, per decision): explicit "Enable packet
  capture…" button in the Network tab (`ConfigAction::EnablePacketCapture`) **and**
  an auto-prompt modal in `main.rs` when a running pcap-mode machine reports
  `PermissionDenied` (one-shot per run). Both call the new cross-platform
  `iris-gui/src/capture_access.rs` dispatch (per-OS `enable()` + `permission_hint()`;
  Linux/macOS/Windows stubs return the manual steps until A2/A4/A7 fill them in).

### A2 — Linux elevation  *(DONE 2026-06-20)*
- [x] **[branch]** Ported `relaunch_with_elevation()` into the Linux `imp` of
  `iris-gui/src/capture_access.rs`. pkexec `env …` re-exec, re-injecting
  `DISPLAY/WAYLAND_DISPLAY/WAYLAND_SOCKET/XAUTHORITY/XDG_RUNTIME_DIR/HOME/APPIMAGE/
  ARGV0` + `SUDO_USER` and `SUDO_UID/GID` read from `/proc/self` (no `libc` dep).
  Uses generic polkit (`pkexec`) — no custom `.policy`.
- [x] **[branch]** AppImage wrinkle kept: re-exec `$APPIMAGE` when set.
- [x] **[branch]** Trigger wired via A1 (auto on `PermissionDenied`, or the
  explicit button) → `capture_access::enable_packet_capture()`.
- [ ] **[branch]** ⚠️ *Local type-check pending* — no `rustup`/Linux target on the
  dev box, so the Linux `imp` was reviewed (edition-2021-correct) but compiles for
  the first time on Dani's Linux build. (Edge: a cancelled pkexec dialog ends the
  app since `exec` already replaced it — documented; setcap avoids the prompt.)
- [x] **[branch]** setcap is the package default → see A6.

### A3 — macOS ChmodBPF resources  *(DONE 2026-06-20)*
- [x] **[branch]** `installer/macos/chmod-bpf/io.github.danifunker.iris.ChmodBPF.plist`
  — LaunchDaemon, `RunAtLoad` (label matches bundle id `io.github.danifunker.iris`).
  `plutil -lint` OK.
- [x] **[branch]** `installer/macos/chmod-bpf/ChmodBPF` — script: idempotent
  `dseditgroup -o create access_bpf`, `chgrp access_bpf /dev/bpf*`, `chmod g+rw`,
  reapply on boot. `sh -n` clean, executable.

### A4 — macOS in-app ChmodBPF install flow (iris-gui)  *(DONE 2026-06-20)*
- [x] **[branch]** Implemented in `iris-gui/src/capture_access.rs` (macOS `imp`).
  Resources are **embedded via `include_str!`** (no .app-bundle dependency → works
  in a plain `cargo run` dev build, and the **`make_dmg` copy step is no longer
  needed** — see B2). Staged to a temp dir; one privileged `/bin/sh` script run via
  a single `osascript … with administrator privileges` prompt.
- [x] **[branch]** Privileged step: copy plist → `/Library/LaunchDaemons/`, script
  → `/Library/Application Support/IRIS/ChmodBPF/`; `dseditgroup -o create` + add the
  real `$USER` (captured pre-elevation); chmod current `/dev/bpf*`;
  `launchctl bootout||true` then `bootstrap system …`.
- [x] **[branch]** `bpf_accessible()` probe (`/dev/bpf0..15`, EACCES→false,
  Ok/EBUSY→true) short-circuits to `Enabled`; otherwise install → `NeedsRelaunch`
  ("quit & reopen"); cancel detected via osascript `-128`.
- [ ] **[branch]** *(deferred)* Fallback one-shot whole-process sudo re-exec if the
  user declines the daemon — low priority; the daemon path is the primary flow.

### A5 — macOS entitlements (on-branch half of packaging)  *(DONE 2026-06-20)*
- [x] **[branch]** Added `com.apple.security.automation.apple-events` to
  `installer/iris-gui-notarized.entitlements` (osascript drives Apple Events).
  `plutil -lint` OK. *(The matching `NSAppleEventsUsageDescription` Info.plist key
  is still **[main]** — the plist is generated by `make_dmg`; see B2.)*

### A6 — Linux package metadata (on-branch half of packaging)  *(DONE 2026-06-20)*
- [x] **[branch]** `iris-gui/Cargo.toml`: deb keeps `depends = "$auto"` (shlibdeps
  resolves libpcap per-variant + handles the Ubuntu `t64` rename — hardcoding
  would break noble and over-constrain non-pcap debs); rpm gets explicit
  `requires = { libpcap = "*" }` (cargo-generate-rpm doesn't auto-detect).
- [x] **[branch]** setcap postinst (the no-root default): `installer/linux/postinst`
  (deb, via `maintainer-scripts`) + `post_install_script` (rpm) run
  `setcap cap_net_raw,cap_net_admin+eip /usr/bin/iris-gui`, best-effort/guarded so
  install never fails. ⚠️ *Packaging only verifiable on a Linux build (cargo-deb /
  cargo-generate-rpm) — flagged for Dani.* Manifest parses (`cargo metadata` OK).

### A7 — Windows Npcap installer logic  *(DONE 2026-06-20, revised: no auto-download)*
Decision (2026-06-20): **never silently download or bundle Npcap.** The flow is
detect → **open the npcap.com page in the browser** → user installs it → re-check.
- [x] **[branch]** `installer/iris-gui.iss` `[Code]` block, **gated behind ISPP
  `#ifdef Pcap`** so the standard installer is byte-for-byte unchanged; only the
  pcap installer (compiled `iscc /DPcap=1`, a **[main]** step) gets it.
  1. `NpcapInstalled()` — `{sys}\Npcap\wpcap.dll`, legacy `{sys}\wpcap.dll`, and
     the `npcap` service registry key.
  2. When missing, a `CreateCustomPage` (after Select Tasks, skipped when present
     via `ShouldSkipPage`) explains Npcap is needed and has an **"Open the Npcap
     download page"** button → `ShellExec` to `https://npcap.com/#download`. No
     `DownloadTemporaryFile`, no `Exec` of any downloaded file.
  3. **Try again:** `NextButtonClick` re-checks; if still missing, a Yes/No prompt
     lets the user stay on the page to install + re-check, or continue without it.
  4. Removed the old `[Tasks]` npcap entry and the `NpcapVersion` define. Installer
     stays per-user/no-admin (Npcap's own installer raises its own UAC).
- [x] **[branch]** Runtime Windows path (`capture_access.rs`) aligned to the same
  philosophy: `enable()` checks for Npcap; if present, advises Administrator +
  relaunch; if missing, opens `npcap.com` in the browser (never downloads) and
  asks the user to install + relaunch.
- [ ] **[branch]** ⚠️ *Not compile-tested* — `ISCC` is Windows-only; Pascal
  reviewed against the Inno API. Flagged for Dani's Windows test.

## B. On-main work (`main`) — CI pipeline, do NOT commit on this branch

> **Status (2026-06-20):** B1–B3 are **implemented on branch `pcap-release-pipeline`**
> (cut off `main`; `release.yml` only, +218 lines). They live there, not on
> `add-pcap-builds`, per the file-ownership rule. The boxes below stay unchecked in
> this doc because the doc travels with `add-pcap-builds`; both branches merge to
> `main` for a full pcap release. Note: the WinPcap link uses `LIBPCAP_LIBDIR`
> (the crate's `build.rs` honors it), not `LIB`/`rustc-link-search`.

### B1 — release.yml pcap build variant (Phase 1)
- [ ] **[main]** Mirror the `lightning` step across every job, adding a 3rd `pcap`
  build. GUI features `iris/lightning,iris/pcap,bundled`; CLI features
  `chd,camera,rex-jit,lightning,pcap` (keep `chd` explicit on CLI).
- [ ] **[main]** Native build deps per job: Linux packages `libpcap-dev` (apt);
  Linux AppImage `libpcap` (pacman) + quick-sharun bundles `libpcap.so`; Windows
  **WinPcap Developer Pack** (BSD) on the runner + `LIB`/`rustc-link-search` so
  `#[link(name="wpcap")]` resolves; macOS none (in SDK).
- [ ] **[main]** Artifact naming: `IRIS-<component>[-pcap]-<os>-<arch>-<VER>.<ext>`
  (insert `-pcap` where `-lightning` goes).
- [ ] **[main]** Windows job: matrix `cli_features_pcap`, GUI pcap portable zip,
  **pcap installer build** (where A7's `.iss` `[Code]` runs), CLI pcap zip + uploads.
- [ ] **[main]** Linux AppImage + packages jobs: pcap GUI + CLI builds, packages, uploads.

### B2 — release.yml make_dmg (Phase 3 main half)
- [ ] **[main]** Add `NSAppleEventsUsageDescription` to the `make_dmg` Info.plist
  heredoc (release.yml ~`:348`) for the **pcap variant** (needed for the osascript
  admin prompt under the hardened runtime).
- [x] ~~copy `installer/macos/chmod-bpf/*` into the bundle~~ **DROPPED** — A4 embeds
  the resources via `include_str!`, so no bundle copy is needed. (The `make_dmg
  "pcap"` variant + `sign_notarize_package …iris-pcap…` + uploads are still part of
  B1's variant work.)

### B3 — DOWNLOADS.md generator (Phase 1 docs)
- [ ] **[main]** Add pcap filenames to the missing-asset checklist; add a 📡 PCAP
  column/rows + explainer to the three tables; add the Npcap note/link to the
  Windows footnote.

---

## Testing matrix (Dani — needs real hardware/OS)

These cannot be validated from this dev box alone; flagged for you to run. ✅ = I
can verify locally (build/logic); 🧪 = needs your hands-on test on that OS.

| Area | What to test | Where |
|---|---|---|
| Build (all features) | `cargo build --features pcap` + `-p iris-gui --features pcap` + plain `-p iris-gui` | ✅ this box (macOS) |
| **macOS ChmodBPF** | Run the in-app "Enable packet capture" → one admin prompt → daemon installs → quit/reopen → pcap mode bridges onto wired iface | 🧪 macOS |
| macOS daemon persistence | Reboot → `/dev/bpf*` still group-readable (LaunchDaemon `RunAtLoad`) | 🧪 macOS |
| macOS Gatekeeper | `launchctl bootstrap` works on current macOS; runtime-installed daemon not blocked | 🧪 macOS |
| **Linux pkexec** | pcap mode → capture EPERM → elevation modal → pkexec prompt → re-exec as root → capture works (X11 **and** Wayland) | 🧪 Linux |
| Linux AppImage | Same, re-execing `$APPIMAGE` (not the FUSE `current_exe()`) | 🧪 Linux (AppImage build) |
| Linux setcap | deb/rpm install → `setcap` postinst → runs unprivileged with capture | 🧪 Linux (after B1 packages) |
| **Windows Npcap** | Installer with no driver → opt-in task → downloads + launches Npcap → IRIS pcap mode captures (Administrator) | 🧪 Windows |
| Windows detect | Installer with Npcap already present → skips the download page | 🧪 Windows |
| Windows build link | pcap crate finds the WinPcap Dev Pack import lib on the runner | 🧪 Windows (CI, after B1) |

---

## Licensing summary (clean except Windows-Npcap)

- IRIS core: **BSD-3-Clause** (`LICENSE`). The CHD path (`libchdman-rs` >= 0.288.8 and the MAME CHD core it vendors) is also **BSD-3-Clause** — see `LICENSE-libchdman-rs.txt`. (Earlier `0.287.0-l7` builds were GPL-3.0 via `LICENSE-GPL3.txt`; no longer required.)
- `pcap` crate: **MIT OR Apache-2.0**.
- libpcap (Linux/macOS): **BSD-3-Clause** — bundle in AppImage / depend in deb/rpm; ships in macOS.
- Windows **Npcap**: proprietary, **redistribution forbidden** — never bundle; user installs it.
  Build-link against the **BSD WinPcap Dev Pack**, not the Npcap SDK.

## Runtime requirements (document regardless of in-app elevation)

- Linux: root or `setcap cap_net_raw,cap_net_admin+eip`. AppImage → run via pkexec/sudo.
- macOS: one-time admin install of **ChmodBPF** (app then runs unprivileged); quit &
  reopen IRIS after install. Wired-only — many Wi-Fi APs reject the bridged MAC.
- Windows: Administrator + a WinPcap-compatible driver (Npcap) installed.
- Default backend stays **NAT** everywhere; PCAP is opt-in via `[network] mode = "pcap"`.

## Resolved decisions (2026-06-20)

- **Implementation order**: **A1 (cross-cutting) first**, then pick an OS. All
  three elevation paths depend on the capture-failure surfacing + Networking-tab hook.
- **Elevation UX trigger**: **both** — an explicit "Enable packet capture" button
  in the Networking tab AND an automatic modal when a pcap-mode machine hits a
  permission error on start. (Affects A1/A2/A4.)
- **Linux package path**: **setcap-in-postinst is the default** for deb/rpm (no
  root GUI); pkexec re-exec is the AppImage/portable fallback. (Affects A2/A6.)

## Open risks

- Windows link step: confirm the pcap `build.rs` finds the WinPcap Dev Pack import
  lib on the runner (`LIB`/search dir). Highest-uncertainty Phase-1 item. **[main]**
- macOS ChmodBPF: `launchctl bootstrap` vs legacy `load`; group-membership-needs-
  relaunch UX; runtime-installed daemon is outside notarization (confirm Gatekeeper
  doesn't block it).
- Artifact/upload count grows ~50% (3rd variant). Accepted.
</content>
</invoke>
