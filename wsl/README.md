# IRIS on Windows / WSL

This folder contains Windows 11 and WSL launch/build guidance. The examples
use a WSL checkout at `~/iris-wsl-build`; adjust paths for your own checkout.

## One-click launch (Windows)

| Script | What it does |
|--------|----------------|
| `run-iris-premiere.bat` | **Recording:** full-feature `iris.exe` CLI (lightning+rex-jit+idle-pause; best 3D/X11 perf) |
| `capture-app-crash.ps1` | Tee stderr to `premiere-debug.log` + on-screen capture checklist |
| `run-iris-windows.bat` | IRIS CLI with full performance features (lightning+rex-jit+idle-pause) |
| `run-iris-gui-windows.bat` | iris-gui launcher (configure machines in the UI) |
| `run-iris-gui-premiere.bat` | **Premiere GUI:** `premiere` feature (lightning+idle-pause) + `IRIS_GUI_GL=1` |
| `run-iris-ci.bat` | Headless CI with `iris-windows.toml` (TCP `127.0.0.1:19851`) |
| `run-iris.bat` | IRIS CLI via WSL (iris-ci, Linux tools) |
| `run-iris-gui.bat` | iris-gui via WSL |

**For YouTube / max graphics:** tune the machine in **iris-gui**, **export** to `irix-install/iris-windows.toml`, **stop** the GUI VM, then run **`run-iris-premiere.bat`** for the 3D segment (native CLI OpenGL — faster than the GUI CPU compositor).

---

## Runtime and build settings

These settings are part of this repository and share the core configuration schema:

### Performance stack

| Area | What |
|------|------|
| **Launch scripts** | `run-iris-premiere.bat` / `run-iris-windows.bat` build with `lightning,rex-jit,idle-pause`; **feature stamp** rebuild via `ensure-build.bat` |
| **Premiere GUI** | `cargo build -p iris-gui --features premiere` — embedded `lightning` + `idle-pause`; `run-iris-gui-premiere.bat` |
| **GUI prefs** | Debug tab settings (`gui_gl_capture`, `no_idle`, `debug_log`) persisted in `gui.json`; **File → Prepare for premiere…** exports TOML |
| **Idle refresh** | Status-bar-only heartbeat skips full compositor + partial egui upload — [rules/perf/gui-idle-refresh.md](../rules/perf/gui-idle-refresh.md) |
| **Audio** | HAL2 wall-clock pacing/resampling; Display tab `[audio]` prebuf / cpal buffer |
| **jitv2 (experimental)** | `--features jitv2` — physical-page region compiler, auto-enabled at runtime once compiled in (no env-var toggle); use `jitv2_lockstep` for diagnostics or `jitv2_opcodefusion` separately (they conflict); `[jitv2] threads` configures the pool |

See [HELP.md](../HELP.md) for monitor commands, serial ports, NVRAM, etc.

### VM hardware / RAM (iris-gui + core)

| Area | What |
|------|------|
| **RAM presets** | Memory menu + Memory tab: 32–512 MB on IP24/IP22; IP28 also offers 768/1024 MB |
| **RAM workflow** | Edits disabled while VM is running; **“Applied at next Start”** when stopped; shows config vs last-started total |
| **Extended RAM fix** | If PROM only POSTs lomem, core **synthesizes MEMCFG** for himem banks 2–3 when configured (`src/dev/mc.rs`) — see [rules/irix/extended-ram-memcfg.md](../rules/irix/extended-ram-memcfg.md) |
| **MHz vs MIPS** | Status-bar **MIPS** = real host speed; IRIX System Manager **MHz** = `hinv` inventory (cosmetic). Debug tab explains build features |

**Important:** `banks` in config is applied only when the VM **Starts**. Changing RAM in the GUI while IRIX is running updates the saved config, not the live guest — **Stop → change → Start**.

### MHz vs MIPS (don’t chase the wrong number)

| Display | Meaning |
|---------|---------|
| System Manager **~166 MHz** | Guest inventory from PROM/kernel — **not** PC emulation speed |
| Status bar **MIPS** | Instructions per wall-clock second on your PC |
| Status bar **Hz** | Guest kernel tick rate (CP0 Compare or PIT) — **not** CPU MHz |

Enabling `rex-jit`/`jitv2` raises MIPS; hinv MHz stays the same. That is expected.

---

## Native Windows build (recommended for daily use)

```powershell
cargo +nightly-x86_64-pc-windows-msvc build --release --bin iris --features lightning,rex-jit,idle-pause
cargo +nightly-x86_64-pc-windows-msvc build -p iris-gui --release
```

Runtime (CLI):

```powershell
.\target\release\iris.exe --config irix-install\iris-windows.toml
```

Or use `wsl\run-iris-premiere.bat` (builds with the right features automatically).

Close `iris-gui.exe` before rebuilding if the linker reports “Access is denied”.

### REX3 JIT warm-up (do this before recording)

The REX3 draw-mode JIT learns across sessions. **First boot after install is always the slowest.**

| Profile | Path | Purpose |
|---------|------|---------|
| REX3 JIT | `%USERPROFILE%\.iris\rex-jit-profile.bin` | Draw-mode shaders |

**Before recording:** boot IRIX once, open a GL app (`glxgears`, 4Dwm), interact for 5–10 minutes, quit cleanly. The second boot replays the saved profile and is smoother.

Monitor (telnet `127.0.0.1:8888`): `perf snapshot`, `rex jit status`, `hal2 status` (cpal underrun counter; codec A dedicated pump).

**Profile script:** `wsl\profile.ps1` — Task Manager + monitor scrape.

**CI smoke:** `wsl\smoke-premiere.ps1` — headless iris + `iris-ci ping`. Indigo2 profile: `irix-install\iris-indigo2-smoke-ci.toml`.

### Phase 3 unified config

Export from iris-gui includes `[perf]` and `[machine]` sections (plus `[jitv2]` if built with the `jitv2` feature). See [rules/perf/phase3-platform.md](../rules/perf/phase3-platform.md).

Windows CI default socket: `127.0.0.1:19851` (TCP). Unix: `/tmp/iris.sock`.

**A/B recording script:** `wsl\premiere-ab-checklist.ps1` — Take A (slow) vs Take B (premiere CLI) commands.

### Premiere GUI build

```powershell
cargo +nightly-x86_64-pc-windows-msvc build -p iris-gui --release --features premiere
wsl\run-iris-gui-premiere.bat
```

The `premiere` feature enables `lightning` + `idle-pause` on the embedded iris core (same stack as CLI premiere, minus native vsync window).

### Dual-path performance (recording)

| Path | Best for |
|------|----------|
| `run-iris-premiere.bat` (CLI) | **Max 3D/X11** — native OpenGL + vsync |
| `run-iris-gui-premiere.bat` | In-process demo with GL capture |
| `run-iris-gui-windows.bat` | Daily config (lighter build) |

### Optional: GPU capture in iris-gui

```powershell
$env:IRIS_GUI_GL = "1"
.\target\release\iris-gui.exe
```

Uses `GlCompositor` on the refresh thread instead of the CPU path. Still slower than native CLI OpenGL, but better for GUI-only workflows.

---

## Config files

| File | Role |
|------|------|
| `irix-install/iris-windows.toml` | **Shared canonical TOML** for CLI / premiere.bat (export from GUI to keep in sync) |
| `%APPDATA%\iris\gui.json` | **GUI system of record** — named machines, autosaved edits |
| `irix-install/iris-wsl.toml` | WSL / Linux CLI |

**Ports (native Windows):**

- IRIX (NAT forward): `telnet 127.0.0.1 2323`
- Monitor: `telnet 127.0.0.1 8888`
- Serial console: `telnet 127.0.0.1 8881`

---

## RAM layouts

| Goal | `banks` | Guest RAM (typical) |
|------|---------|---------------------|
| Authentic Indy max | `[128, 128, 0, 0]` | 256 MB |
| IRIX 6.5 extended | `[128, 128, 64, 64]` | 384 MB — use `iris-windows-384.toml` |
| IP24/IP22 IRIX 5.3 maximum | `[128, 128, 128, 128]` | 512 MB — **not for IRIX 6.5** |

For IP28, two 512 MB banks (`[512, 512, 0, 0]`) provide 1 GB and have separate
PROM/IRIX validation; the IP24/IP22 four-bank warning above does not apply.
See [IP28 guide](../docs/indigo2-ip28.md).

After any change: **Stop → cold Start** (fully quit iris, relaunch). Verify:

1. Monitor: `mc regs` — banks 2–3 should show **VLD=1** when extended RAM is configured
2. IRIX: `hinv -t memory` or System Manager → About This System

For **IRIX 6.5**, prefer **384 MB** over 512 MB. The 512 MB preset is documented for IRIX 5.3.

### Silent app quits (debug)

See [rules/testing/silent-app-quit-debug.md](../rules/testing/silent-app-quit-debug.md).

1. **Capture log:** `wsl\capture-app-crash.ps1`
2. **RAM A/B:** `--config irix-install\iris-windows-384.toml` if you were on 512 MB

Send `premiere-debug.log`, monitor `status`/`bt`/`dt 80`, and `hinv -t memory` after a quit.

### R5000 Indy

The CPU is a runtime setting, not a build feature: every build contains the
R4400, R5000, and R10000 (IP28 default). Pick **R5000** in iris-gui (Machine menu or General tab),
or in the TOML / on the command line:

```toml
[machine]
cpu = "r5000"
```

```powershell
target\release\iris.exe --config irix-install\iris-windows.toml --cpu r5000
```

The emulated R5000 has 32 KB 2-way L1 caches and **no** secondary cache. The
R5000SC variant (external L2) is not available: `iris/r5ksc` deliberately fails
to build until its L1I cache bugs are fixed (`rules/testing/r5k-l1i-cache-bugs.md`).
The old `iris/r5k` feature is removed. A snapshot refuses to restore onto a
different CPU model.

---

## GUI ↔ TOML ↔ CI sync

The GUI and CLI use the same **`MachineConfig`** schema but **different files** unless you sync them. **`iris-ci` loads no config** — it only drives an already-running `iris` process.

| Layer | Stores |
|-------|--------|
| **iris-gui** | `%APPDATA%\iris\gui.json` |
| **iris CLI** | `iris.toml` via `--config` |
| **Debug settings** | `[debug]` TOML section (`gui_gl_capture`, `no_idle`, `debug_log`) — GUI Debug tab edits it; maps to `IRIS_GUI_GL`/`IRIS_NO_IDLE`/`IRIS_DEBUG_LOG` at Start |
| **iris-ci** | Nothing (socket client only) |

### Recommended: one canonical TOML

Treat **`irix-install/iris-windows.toml`** as the shared file:

```
GUI edits  →  File → Export  →  iris-windows.toml  →  premiere.bat / iris.exe
     ↑                                                      ↓
     └──────── File → Import when TOML changed on disk ─────┘
```

### Path A — GUI → CLI (after tuning in the UI)

1. **Stop** the VM.
2. Set RAM, disks, network in the GUI (Memory tab presets, etc.).
3. **File → Export current to iris.toml…** → save as `irix-install/iris-windows.toml`.
4. Run CLI from repo root (paths in TOML are relative to cwd):

   ```powershell
   wsl\run-iris-premiere.bat
   ```

5. Optional: enable **CI mode** on the CI / Automation tab before export if you need `ci = true` in TOML.

**Export/Import** appear under **File** in source builds (hidden in App Store `bundled` builds).

### Path B — TOML → GUI

1. **File → Import iris.toml…** → pick `irix-install/iris-windows.toml`.
2. **Stop → Start** so guest RAM matches imported `banks`.

### Debug parity (GUI vs CLI)

Mirror the GUI **Debug** tab (`gui_gl_capture`, `no_idle`, `debug_log`) in your launch script if you need the same env at CLI:

```powershell
$env:IRIS_GUI_GL = "1"
$env:IRIS_NO_IDLE = "1"
$env:IRIS_DEBUG_LOG = "all"
```

### CI (`iris-ci`)

The CI server and `iris-ci` support both Unix-domain and loopback TCP sockets.
Native Windows uses `127.0.0.1:19851` by default; Linux/macOS use
`/tmp/iris.sock`. For example, with the native Windows emulator running in CI
mode:

```powershell
.\target\release\iris-ci.exe --socket 127.0.0.1:19851 ping
```

**WSL workflow:**

1. TOML includes `ci = true` (and `ci_socket` if non-default).
2. Start iris in WSL with that config.
3. In another WSL terminal: `iris-ci ping`, `iris-ci boot`, etc.

See [rules/snapshot/iris-ci-is-the-canonical-ci-socket-interface.md](../rules/snapshot/iris-ci-is-the-canonical-ci-socket-interface.md).

### Sync checklist

- [ ] Same `banks` in exported TOML and GUI Memory tab
- [ ] Same `nvram`, `scsi.*.path`, `prom` paths (run from repo root or use absolute paths)
- [ ] **Stop → Start** in GUI after RAM changes
- [ ] Same Debug env for CLI as GUI Debug tab
- [ ] `mc regs` shows expected MEMCFG for extended RAM
- [ ] Re-export after GUI changes you want CLI/CI to keep

---

## iris-gui first time

1. Run `run-iris-gui-windows.bat`
2. **File → Import iris.toml…** → `irix-install/iris-windows.toml` (or create a machine and configure)
3. **Memory** tab: pick **256 MB** (authentic) or **384 MB** (IRIX 6.5 extended)
4. **Debug** tab: set GL capture / no-idle / debug-log if needed (defaults are fine for normal use)
5. **Start**
6. After further edits: **File → Export** so CLI/premiere.bat stay aligned

---

## From Ubuntu terminal (WSL)

```bash
cd ~/iris-wsl-build
chmod +x wsl/run-iris.sh wsl/run-iris-gui.sh
./wsl/run-iris.sh          # CLI
./wsl/run-iris-gui.sh      # GUI
```

---

## Rebuild

**Windows (native):**

```powershell
cargo +nightly-x86_64-pc-windows-msvc build --release --bin iris --features lightning,rex-jit,idle-pause
cargo +nightly-x86_64-pc-windows-msvc build -p iris-gui --release
```

**WSL:**

```bash
cd ~/iris-wsl-build
cargo build --release --bin iris --features lightning,rex-jit,idle-pause
cargo build -p iris-gui --release
```

Add `,jitv2` to try the experimental region JIT instead of (or alongside) `rex-jit`.

---

## Sync source from Windows → WSL copy

```bash
rsync -a --exclude target /mnt/c/Users/chron/CURSOR-PROJECTS/iris-main/ ~/iris-wsl-build/
chmod +x ~/iris-wsl-build/wsl/*.sh
```

After syncing, rebuild in WSL if you changed Rust sources.
