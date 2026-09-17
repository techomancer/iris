# App Store review response — IRIS

> **Note:** `docs/appstore-review-notes.txt` was never committed to this repository;
> the text below is what remains here.
>
> **Paste-ready version → `docs/appstore-review-notes.txt`** (3,915 chars, under
> the field's 4,000 limit). Copy it verbatim into **App Review Information →
> Notes** in App Store Connect; leave the **App Sandbox Information** section
> blank. That file is the *superset* actually submitted: it adds the
> IRIX-media / IP disclaimer, "the app contacts no external services," the
> inbound port-mapping / FTP use case, and the full entitlement list (all seven
> keys, incl. the honest `allow-jit` note — see below). This markdown file is the
> working/source document (rationale, history, verification commands).

Originally written for the 1.0 (20260610.2118) review (June 16, 2026), which
raised two issues:

1. **Guideline 2.5.1** — private API `_CGSSetWindowBackgroundBlurRadius`.
2. **Guideline 2.4.5(i)** — entitlements without obvious matching functionality
   (`com.apple.security.device.camera`, `com.apple.security.network.server`).

## App Sandbox Information screen — N/A

The App Store Connect **App Sandbox Information** screen is *only* for
temporary-exception entitlements (`com.apple.security.temporary-exception.*`).
IRIS uses none, so that screen stays blank. The entitlement justifications below
go in **App Review Information → Notes**, not there.

No entitlement has been added since the original submission — the later features
(per-disk CHD copy-on-write + exit-time fold, the in-core pure-Rust NFS server,
the Networking-tab redesign / FTP ALG / in-app file bridge) all run inside the
existing grants. The in-core NFS server in particular opens **zero host sockets**
(it lives entirely in the user-mode NAT), so it does not even rely on
`network.server`. PCAP capture is sandbox-incompatible and ships only in the
non-App-Store release builds, never under the `appstore` feature.

### `com.apple.security.cs.allow-jit` — kept, described honestly

The `appstore` feature force-sets `IRIS_NO_JIT=1` (`iris-gui/src/main.rs:111`), so
the App Store build runs the MIPS CPU interpreter-only — JIT is never allocated
(the binary would otherwise `SIGKILL` on the first JITed page under MAS signing,
since `allow-unsigned-executable-memory` is rejected). The entitlement is left in
place for parity with the Developer-ID builds, and the notes describe it
truthfully as present-but-disabled rather than claiming the build uses JIT.
(Decision 2026-06-20: keep + describe honestly, over removing it outright.)

---

## 1. Guideline 2.5.1 — private API (fixed in binary)

The symbol came from the `winit` windowing crate (pulled in by `eframe`), whose
macOS backend calls `CGSSetWindowBackgroundBlurRadius` in `WindowDelegate::set_blur`.
IRIS never requests window blur, but the symbol is linked in regardless and
Apple's static scan flags it.

**Fix:** vendored a patched `winit` (`third_party/winit-0.30.13/`, wired via
`[patch.crates-io]` in the root `Cargo.toml`) that removes the private extern
declarations and makes `set_blur` a no-op. Verified the symbol is gone from the
linked binary:

```
nm -u target/release/iris-gui | grep CGSSetWindowBackgroundBlurRadius   # → no output
```

(`_CGShieldingWindowLevel` remains; it is a public CoreGraphics API and was not
flagged.) See `rules/macos/appstore-private-api.md`.

A new binary is required for this fix, so we also strengthened the two
entitlements below with visible, testable functionality rather than removing
them.

---

## 2. Guideline 2.4.5(i) — entitlement justifications

Both entitlements back real functionality. To make them easy to verify we added
in-app features that exercise each one directly, without needing to boot IRIX.

### `com.apple.security.device.camera`

**What it's for:** IRIS emulates the SGI Indy's **IndyCam** / VINO video-input
hardware. When the user selects the host camera as the video source, IRIS
captures live frames from the Mac's camera (AVFoundation) and feeds them to the
emulated VINO device. The matching `NSCameraUsageDescription` is in `Info.plist`.

**How to test (reviewer steps):** *(no boot or login required)*
1. Launch IRIS. In the left column click **Edit config…**, then click the
   **Video-In** button that appears below it.
2. Click **📷 Test Camera**. (The same test is also under **Help ▶ →
   Diagnostics → Test Camera…**, which requires a running machine.)
3. macOS shows the camera-permission prompt; allow it.
4. A live preview from the Mac camera appears, with a status line showing the
   capture resolution and a rising frame count. Closing the window releases the
   camera (indicator light turns off).

> Paste-ready reply:
>
> IRIS emulates the SGI Indy IndyCam (VINO video-input) hardware. The camera
> entitlement lets the app capture live video from the Mac's camera and feed it
> to the emulated video-input device. You can verify this directly: open the
> **Video-In** tab and click **Test Camera** — macOS will prompt for camera
> access and the app then shows a live preview from the camera. The matching
> NSCameraUsageDescription is included in Info.plist.

### `com.apple.security.network.server`

**What it's for:** two things —
- The emulator exposes the emulated machine's **serial console** (IRIX ttyd1,
  `127.0.0.1:8881`) and **PROM monitor** on loopback TCP, so a terminal can
  attach to the guest console. The app's own **Serial console…** window connects
  to this server (loopback), which is the visible end-to-end demonstration: the
  emulator *listens* (network.server) and the viewer *connects* (network.client).
- It binds **inbound port-forwards** into the emulated SGI Ethernet (SEEQ 8003)
  when the user configures them on the Networking tab.

(The clean-shutdown "Send IRIX halt" action no longer uses a socket — it now
types at the console in-process — so the server entitlement is only used for the
genuine server features above.)

**How to test (reviewer steps):**
1. Launch IRIS and **Start** a machine (the bundled config boots to the PROM).
2. Open **Machine ▶ → Serial console…** (also under **Help ▶ → Diagnostics →
   Serial console…**).
3. The window shows "● connected to 127.0.0.1:8881" and streams the live guest
   serial console. Typing a line and pressing Enter sends it to the guest.
   This confirms the app's loopback serial **server** is live and accepting a
   connection.

> Paste-ready reply:
>
> IRIS exposes the emulated workstation's serial console and PROM monitor as
> loopback TCP servers (e.g. 127.0.0.1:8881) so a terminal can attach to the
> guest console, and it binds user-configured inbound port-forwards into the
> emulated Ethernet. You can verify this without external tools: Start a machine
> and open **Machine → Serial console…** — the app connects to its own loopback
> serial server and streams the live guest console, and you can type into it.

---

## Summary of binary changes in this resubmission

- Vendored/patched `winit` to drop the private blur API (2.5.1). No window blur
  was ever used.
- Added **Video-In → Test Camera** (live host-camera preview) so the camera
  entitlement is user-visible.
- Added **Machine → Serial console…** (in-app loopback serial viewer) so the
  network.server entitlement is user-visible.
- Moved "Send IRIX halt" to an in-process console path (no longer opens a
  loopback socket).
