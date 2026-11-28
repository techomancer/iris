# New iris-gui framework: shared MVC core plus native shells

Status: **in progress** (2026-11-27): see the GUI code row below. This is the working
brief for whoever implements it. Decisions are the user's unless marked
*proposed*. Start at [Where things stand](#where-things-stand), then
[Phases](#phases).

Reference implementation of the pattern: `../rusty-backup` (CONTRIBUTING.md
"Architecture policy (MVC)", `docs/native_frontends_plan.md`,
`docs/appkit_shell_plan.md`, `frontends/appkit`, `frontends/mac-screens`).
Same layering, adapted for an emulator with a live display. Read its
CONTRIBUTING section before writing a controller.

## Where things stand

| Thing | State |
|---|---|
| Branch `core-activity-counters` | Merged into `main` as PR techomancer/iris#797 (2026-11-22). |
| Branch `iris-newgui` | Rebased onto `main` after #797 merged (2026-11-22). Local only, never pushed. Holds this plan and the GUI work. |
| Branch `iris-gtk` | Cut from `iris-newgui` (2026-11-27). Local only, never pushed. Holds the GTK 4 shell (Phase 5). |
| GUI code | Phase 4 started: the AppKit shell (`iris-gui/frontends/appkit`, binary `iris-appkit`) has slices 0 to 9: app, window, toolbar, menu bar, wake, library, effects, quit through the controller, the summary page, the Metal presenter with physical-key input, capture, full screen and detach (presenter and key path checked by running; mouse capture, full screen and detach not yet tried by hand), and the settings window over the schema (every field kind drawn; port forward, CD-ROM, copy-on-write and remove edits driven by the script hook reached the machine's TOML). Slice 3: the status lights are buttons; `AppController::light_command` maps a light's id to the command it stands for (disk: browse, CD-ROM: insert, network: check, serial: console, capture), in egui too. The New Machine window draws `NewMachineSpec::fields` (the settings field kinds) and creates through `NewMachineSpec::create`; Saved States is a pop-up menu of per-item ids (built, not yet seen: a pop-up menu cannot be dismissed by the script hook). Slice 5's disk browser: `AppController` owns the browser (`OpenBrowser`, `BrowseFiles`, `Browser`, `CloseBrowser`; power on and switching machines close it first, and wait while it writes), `disk_browser::{tools, columns, action_for, ids}` name its controls, and the AppKit window mirrors `app.browser()`. Seen on a copy of the IRIX 5.3 disk: partitions, listing, path bar, selection, the editor (read-only). Not driven from the shell: the write actions (each starts with a modal prompt the script hook cannot answer; the controller paths were tried on copies earlier) and drag-and-drop import (not built). egui's disk lights stay unclickable until egui has a browser window. Slices 6 to 8: Check Networking, the serial console and the camera test are windows that follow their controllers; the shell owns a `BenchController` (as egui does); Diagnostics is an alert with Copy; Preferences draw `controller::preferences` (language and the app-wide display scaling). Seen running on a PROM-only Indy: all of them, a full quick benchmark run streaming its console; the camera fell back to its black source (no camera opened on this Mac). Slice 9: `iris-gui/frontends/appkit/scripts/bundle.sh` builds `IRIS.app` (`io.techomancer.iris.appkit`, the egui release's speed features) and a DMG, signs with a Developer ID and notarizes when asked; `build.yml` gains a `macos-appkit` job (continue-on-error, so it cannot hold up a release) whose DMG the release publishes as `dist-macos-appkit`. Built and launched here ad hoc, arm64 only; the CI job, Developer ID signing and notarization have not run. After slice 9 (2026-11-22): a Devices menu from `AppController::disk_menu` (disc insert/change/eject/mount, browse, copy-on-write commit and discard by per-item ids); Video In, Performance and the NFS share in the settings schema (Perfetto, Debug and CI stay TOML-only as developer settings); Help pages from `controller::help` (About, camera and networking, NFS with the live gateway, Ultra64, Licenses, Privacy); drag-and-drop import in the browser; test hooks `IRIS_APPKIT_ANSWERS`, `drop:`, `type:`, `dump:` (clanker/macos/drive-the-appkit-shell-with-the-screen-locked.md). Every browser write action was driven from the AppKit shell on a fresh copy of the IRIX 5.3 disk (Edit Anyway, new folder, drop import, chmod, rename, chown, edit and save, export, delete) and the copy booted to the IRIX login in the bundled app. Not done: the hand checks (capture, full screen, detach, a real Finder drag), the SVG icon fallback for the toolbar and controls, the CI job's first run. Machine pictures (2026-11-23, answers the IP-tile circles question): one front-view line drawing per case in `iris-gui/assets/icons/machine-*.svg`, black on transparent so each shell tints it, drawn from reference photos except the Origin/Onyx 3000 rack (no photo found yet); `MachineIcon::svg` embeds them, AppKit renders them with resvg into template images (library rows, the current-state row), so they show below macOS 11 too. Driven on the release bundle the same day: the library row's Start, Devices > insert and eject a disc on a running PROM-only Indy (the menu now follows the live disc), Power Off through the writable-disk question. Phase 2 done (boundary count 0; egui no longer counted, D1): App, Session, Display, Settings (schema), Network, Serial, Bench, Camera, Debug, new-machine and disk-browser controller slices done (library, folders, duplicate, quit guard; power on/off with the preflights as controller questions, saved states, discs, copy-on-write, NVRAM reset, status-bar lights from the core counters; egui rewired). Phases 0 and 1 done: `iris-gui/core` (`iris-gui-core`) with the controller contract, ids, keys, labels, icons, `model::job`, and `iris-gui/scripts/check-boundaries.sh`; the toolkit-free modules live in `core/src/model` (plus `hardware.rs`, the pure profile helpers from `extra_ui.rs`). |
| Mock-ups | **Not in the repository, by the user's decision; do not commit mock-ups or screenshots.** Seven boards (stopped, running and settings in light and dark, and the macOS menu bar) are in the git-ignored `ignore/mockups/` on the user's machine, with their sources in `src/` (`node expand.js <in> <out.html> [light|dark]` makes plain HTML) and the annotated feedback screenshot; ask the user if the folder is absent. They predate the feedback in [Design changes from feedback](#design-changes-from-feedback-2026-11-21), so [The window](#the-window) is the specification. |
| Unrelated finding | `kona-tile-handoff-race.md`: a real hang in the KONA tile scheduler, test ignored, not fixed. Not part of this work. |

When #797 merges, rebase `iris-newgui` onto `main`. Do not put GUI commits on
`core-activity-counters`, and do not put this plan there.

## Rules for this work

- All new code lives under `iris-gui/`. The approved exceptions are in
  [Outside iris-gui](#outside-iris-gui); anything else needs the user's OK.
- Read first: `clanker/CLANKER.md`, `HACKING.md`, `iris-gui-README.md`,
  `clanker/gui/01-overview.md`, and every note in `clanker/gui/` and
  `clanker/macos/` (keyboard capture, physical keys, winit/main-thread rules,
  GL teardown, the frame sink's damaged band, profile menus from core tables,
  `machines::visit_paths`, sandbox grants). They are the gotchas the new
  shells must not rediscover.
- Never kill iris while IRIX runs on a real disk image; stop the guest or use
  an overlay.
- Tests: `cargo test -p iris-gui`, `cargo test --lib <filter>` for the core.
  A full `cargo test --workspace` in a debug build takes very long here, and
  `cargo nextest` (what CI runs) is not installed locally. CI is the full
  check: it runs on pull requests into `main`.
- Commit when asked; never push or open a PR unasked.

## Decisions

| # | Decision |
|---|---|
| D1 | *Changed 2026-11-22 (user):* egui is to be removed. It still builds and ships until then, but gets no more rewiring: the boundary script no longer counts it, and nothing new is built for it. (Was: egui stays and ships, rewired slice by slice.) |
| D2 | One machine runs at a time. The library lists all of them; the worker (`handle.rs`) and the cwd-per-machine rule stay. Session state hangs off a `SessionController`, not the app, so several sessions remain possible later. |
| D3 | Layout follows VMware Workstation: library on the left, the selected machine on the right (summary when stopped, guest display when running). The display can be detached into its own window. |
| D4 | macOS presenter: a `CAMetalLayer` view fed by the existing `FrameSink` damaged bands (`replaceRegion` per band). Vsync, integer or linear scaling, and a later path to a shared-device wgpu presenter. |
| D5 | Keep every current indicator and add per-device activity lights ([Status bar](#status-bar)). |
| D6 | Shells are Rust over the platform bindings (`objc2` / `objc2-app-kit`). No Swift, no C ABI. |
| D7 | Every shell has every feature, laid out in native widgets. macOS gets a complete menu bar ([Menu bar](#menu-bar-macos)). |
| D8 | Adding an SGI machine (Octane, O2, Onyx...) must not need shell code: settings are built from the core hardware tables plus one per-profile traits table in `iris-gui-core`. |
| D9 | The first macOS release is a full implementation (parity with egui). The App Store build waits until more platforms are ready. |
| D10 | All new code lives under `iris-gui/`. |
| D11 | Controls are icons with tooltips, not words. The controller gives each action an icon *meaning* (an enum) and its tooltip; the shell picks the glyph. macOS uses SF Symbols (`NSImage(systemSymbolName:)`, macOS 11+). SF Symbols may not ship on other platforms, so GTK4, Win32 and egui use one set of our own SVGs in `iris-gui/assets/icons/`, also the macOS fallback below 11. The accessibility label is the tooltip. |
| D12 | Light and dark appearance, following the system. Shells use the toolkit's semantic colours (AppKit: `NSColor` system colours, no hard-coded hex). The guest display area stays dark in both. |
| D13 | Build features are shown to the user (from `iris::build_features`). Placement is under review, see [Open](#open). |
| D14 | No pause/resume: the core has none. |
| D15 | Saved states: the core already takes a name (`Cmd::SaveState(String)` / `RestoreState`), so each machine has a list of its named states. No snapshot tree. |
| D16 | The egui shell keeps its current look. It is rewired onto the controllers but does not take the Workstation layout. |
| D17 | macOS floor 10.13, the same as today's egui bundle (`scripts/dist/macos-bundle.sh`). Newer features are runtime-checked. Not verified that anything older builds. |
| D18 | Core activity counters for disks and audio: done, on `core-activity-counters` ([Core counters](#core-counters)). No serial counter in the core. |
| D19 | Each native shell crate is one line in the root `Cargo.toml` `members`, with its dependencies and code gated on its own `cfg(target_os)`, so `cargo build --workspace` elsewhere builds an empty binary. |
| D20 | Opening the hung KONA test was out of scope; it is ignored and written up separately. |
| D21 | No importing of CLI `.toml` files into the library (2026-11-21). Machines are created in the GUI; their files remain core-schema TOML. |
| D23 | A disk browser in the GUI, built on the published rusty-backup crates (2026-11-22): open a machine's disk or CD-ROM, look through EFS, XFS and ext partitions, and import, export and edit files. **No provisioning**: a new disk is an empty image and the user runs `fx`, because fx chooses the root and swap split itself (`clanker/irix/fx-label-create-writes-root-and-swap.md`). No formatting either; the OS installer does that. See [Disk browser](#disk-browser-rusty-backup-crates). |
| D22 | Display scale is one setting for the whole application, not per machine (2026-11-21). No core config key. |
| D24 | *2026-11-27:* the GTK shell is built with **libadwaita**, not bare GTK 4. The core's own shapes map onto its widgets (a settings `Section` of `Field`s onto `AdwPreferencesPage`/`Group` and the row types, `ChoiceSpec` onto `AdwAlertDialog`'s responses with its default, dismiss and destructive appearances, `Effect::Toast` onto `AdwToastOverlay`), and `AdwOverlaySplitView` and `AdwToolbarView` are the collapsible library and the sliding status bar the AppKit shell had to hand-build. Floor: libadwaita 1.5 and GTK 4.12 (`v1_5`, `v4_12`), what Ubuntu 24.04 carries — already the host `build.yml` builds the Linux packages on, so the floor costs nothing we ship. |
| D25 | *2026-11-27:* D11 stands for GTK: the shell draws **our own SVG set** in `iris-gui/assets/icons/`, one file per `Icon` meaning, rendered with resvg and tinted to the current foreground colour, rather than mapping onto the Adwaita icon theme's names. One set of glyphs on every platform, and the same files serve Win32 and the pre-macOS-11 AppKit fallback. |

## Layout on disk

```
iris-gui/
  Cargo.toml, src/            the egui shell (as today), becoming a view
  core/                       crate iris-gui-core: no toolkit dependencies
    src/model/                worker handle, machine store, settings, summary, netplan, i18n
    src/controller/           one controller per task; Action / Effect; keys, ids, labels, icons
  frontends/
    appkit/                   crate iris-gui-appkit (objc2), macOS
    gtk4/, win32/             later
  locales/, assets/           shared by every shell (assets/icons/ is the SVG set)
  scripts/check-boundaries.sh shells may import only iris_gui_core::controller
```

`core/` is a path dependency of `iris-gui`, so Cargo makes it a workspace
member with no root edit. It is not called `shared/`: the root `.gitignore`
ignores any folder of that name. In this plan "the core" still means the
emulator crate `iris`; the GUI's crate is always written `iris-gui-core`.

## Layers

- **Model** (`core/src/model/`): today's toolkit-free modules, moved:
  `handle.rs` (`Cmd`/`Evt`/`Status`, the 64 MB worker), `machines.rs`,
  `summary.rs`, `settings.rs`, `netplan.rs`, `netfix.rs`, `safe_stop.rs`,
  `ram.rs`, `single_instance.rs`, `i18n.rs`, `macos_sandbox.rs`,
  `capture_access.rs`, and the frame-sink half of `framebuffer.rs`.
- **Controller** (`core/src/controller/`): rusty-backup's contract.
  `apply(Action) -> Vec<Effect>`, `tick() -> Vec<Effect>`, plain read
  accessors. `Action` and `Effect` are exhaustive enums matched with no `_`
  arm, so a new feature fails to compile in a shell that has not handled it.
  Stable widget ids double as accessibility identifiers. Key bindings are
  one data table (`Primary` = Cmd on macOS, Ctrl elsewhere). Labels and
  tooltips come from the controller through Fluent. No toolkit, no raw threads.
- **View**: egui (`iris-gui/src`), AppKit, later GTK4 and Win32. Renders, maps
  input to actions, fulfils effects (file pickers, confirmations, alerts,
  clipboard, open URL, elevation prompt).

`Notify::wake` replaces polling: egui `request_repaint`, AppKit
`dispatch_async` to the main queue (always queued, never run inline). The
frame path keeps its own wake (`FrameSink`), separate from controller wakes.

### Controllers

| Controller | Owns | Comes from |
|---|---|---|
| `AppController` | library (discover, select, new, rename, delete, duplicate, folders), active machine, dirty/autosave, toasts, quit and close interception (CHD sync), prefs, language | `main.rs`, `machines.rs` |
| `SessionController` | start, stop, halt IRIX, saved states, screenshot, run state, every status-bar light, CD load/eject/remount, COW commit/reset, safe-stop rules | `main.rs`, `handle.rs`, `scsi_menu.rs`, `safe_stop.rs` |
| `DisplayController` | capture/release rules and chords, fullscreen, detach, scaling mode, the "powered off" overlay | `input.rs`, `framebuffer.rs`, `main.rs` |
| `SettingsController` | the machine config as a form schema (below), validation, refit on profile change, missing-disk preflight | `config_ui.rs`, `extra_ui.rs`, `summary.rs` |
| `NetworkController` | NAT/PCAP mode, subnet presets and overlap warnings, live port forwards, "check networking", PCAP elevation | `netplan.rs`, `netfix.rs`, `capture_access.rs` |
| `SerialConsoleController` | console port routing, scrollback, input, the serial light | `serial_console.rs` |
| `DiskBrowserController` | the disk browser: partitions, listing, selection, import and export (files, folders, archives), text edit, chmod, chown, rename, move, delete, check; read-only rules and their reasons; the jobs behind them | new |
| `BenchController` | benchmark run, progress, results | `bench_ui.rs` |
| `DebugController` | monitor port, perfetto start/stop, diagnostics text, build features | `config_ui.rs` Debug tab |
| `CameraTestController` | VINO/IndyCam source test | `camera_test.rs` |
| dialogs | new-machine wizard, rename, about | `dialogs/` |

### Settings as data (D8)

`SettingsController` hands every shell a list of sections, each a list of
fields; a shell implements each field kind once.

```rust
pub struct Section { pub id: SectionId, pub title: String, pub fields: Vec<Field> }
pub enum FieldKind { Choice{options, selected}, Toggle(bool), Text{value, validator},
                     Number{value, min, max, unit}, Path{value, pick: PickSpec},
                     BankGrid(..), DiskTable(..), PortForwardTable(..), Note(Severity, String) }
pub struct Field { pub id: FieldId, pub label: String, pub help: Option<String>,
                   pub kind: FieldKind, pub enabled: Enabled /* with the reason */ }
```

Options come from the core tables (`MachineProfile::ALL`, `cpus()`,
`memory_banks()`, `bank_sizes()`, `GraphicsBoard::ALL`), never from a shell.
What the core does not say goes in **one** `ProfileTraits` table in
`iris-gui-core` (NVRAM kind and whether reset/seed applies, EEPROM MAC, console port,
storage bus, fixed graphics, which sections apply, the library icon), replacing
the `match profile` arms scattered through `config_ui.rs` and `extra_ui.rs`
(`clanker/gui/profile-menus-must-use-core-hardware-tables.md`). Adding a
machine is then: the core profile, one traits row, one icon.

### Configuration files

- A machine is its core-schema TOML at `machines/<name>/<name>.toml`, the
  same file the `iris` CLI reads. That stays the single source of truth; the
  GUI never keeps machine settings anywhere else.
- `gui.json` holds only app preferences: UI scale, display scale, language,
  the selected machine, sandbox grants, and (new) library folders.
- No import of a CLI `.toml` into the library (D21). **File > Open Config
  File** opens the selected machine's TOML in the user's editor, and the CLI
  can run a GUI machine's TOML directly.
- Display scale is an application preference in `gui.json` (D22); it does not
  travel with a machine.

## The window

- **Toolbar** (icons, D11): show/hide library; power on, shut down guest
  (halt IRIX), power off, restart; save state, saved states, screenshot,
  insert disc; then on the right detach display, full screen, serial console,
  machine settings.
- **Library** (left, collapsible from the toolbar and View menu).
- **Content** (right): the summary when stopped, the guest display when
  running.
- **Status bar** under the content.
- **Settings** open as a sheet: device list on the left with a one-line
  summary each, the section's form on the right. Fields that can change live
  stay enabled while running; the rest say "takes effect at next power on".

### Design changes from feedback (2026-11-21)

Community review of the stopped-view mock-up (an annotated screenshot, not
kept in the repository) and chat comments.
The first three were stated in words; the rest are *proposed* readings of
circles and underlines and need the user's confirmation.

| Feedback | Change |
|---|---|
| "Add folders" | The library has user folders. Default view groups by machine family; the user can create folders and drag machines into them. Stored in `gui.json`. |
| "Think of a better way" (saved states) | Replace the row of chips with a list: name, date, size, with Restore, Rename and Delete per row, in its own section of the summary and in a popover from the toolbar button. |
| "stopped" crossed out in a row | Drop the run-state word from library rows. Show a status dot on the icon only for a machine that is running or has a problem. |
| Hide or fold the sidebar | Already designed: toolbar toggle and View > Show Library. |
| Fullscreen | Already exists in egui (F11; Ctrl+Alt+F11 sends F11 to the guest) and is kept: toolbar button and View > Enter Full Screen. |
| TOML as the one config | Already true for machine settings; importing a CLI file was considered and dropped (D21). See [Configuration files](#configuration-files). |
| Circles on the IP24/IP22/IP28 tiles *(proposed reading)* | Replace board-number badges with a small picture of each machine; the IP number moves to the subtitle. |
| Underlines on the spec line, "missing", "copy-on-write", PROM *(proposed reading: said twice)* | Shorten the spec line under the name to machine and IRIX version; drop the "missing" tag where the warning banner already says it. |
| Box round "This build" and the `jitv2` chip *(proposed reading: shown twice)* | Keep the executor chip in the status bar; move the full feature list to About and Help > Diagnostics, with at most one summary line on the page. |
| Box round the empty status-bar hint | Show the device lights greyed out when stopped, so the bar is the same in both states and each still opens its settings. |

### Status bar

Left: run state (Stopped / PROM / Booting / Running / Halted / Powered off),
executor chip (interpreter / jitv2 / lightning), "CPU change at next power on"
when pending, unsaved mark, toast.

Right, one clickable icon each with a state dot and a tooltip that says what
it is doing: every disk (read / write; copy-on-write dirty-sector count; CHD
sync pending), every CD-ROM (disc loaded, activity; click to insert or eject),
network (NAT or PCAP, activity, guest IP, fault badge that opens the
diagnosis), serial console, audio, camera, input capture (with the release
chord), display scale, MIPS, kernel Hz. Greyed out when the machine is
stopped. A disk's or CD-ROM's icon opens the [disk browser](#disk-browser-rusty-backup-crates).

Sources: MIPS, Hz, network, PCAP, CHD sync and `dirty_cow` are already in
`handle.rs` `Status`. Disks and audio come from the new core counters. The
serial light comes from the GUI's own console connection (lit only while the
console window is connected), because the core has no serial counter. Camera
"in use" has no source yet: find one in the VINO/camera code or drop the light.

### Core counters

On `core-activity-counters` (and so on `iris-newgui`):

```rust
// Workstation (src/workstation.rs), both lock-free and monotonic
fn disk_activity(&self) -> Vec<Vec<iris::scsi::DiskCounts>>; // per disk bus, then per SCSI id
fn audio_frames(&self) -> Option<u64>;                       // None with no sound card
pub struct DiskCounts { reads, read_bytes, writes, write_bytes } // all u64
```

- Bus order is the machine's SCSI controller order, then NVMe and Fibre
  Channel where it has them. Match a light to a config entry with
  `config::scsi_slot(controller, id)`.
- They only go up. The worker's 500 ms status tick should sample both, keep
  the previous sample, and put "read since last tick / wrote since last tick /
  audio moving" into `Status`. 500 ms is slow for a disk light: *proposed* a
  faster tick (about 100 ms) for these fields only, since reading them costs a
  few atomic loads.
- An id with nothing on it stays at zero; show lights only for configured
  devices.

### Menu bar (macOS)

Every item runs a controller action and greys out with it. Shortcuts are
proposals and live in the controller key map.

- **IRIS**: About, Check for Updates, Settings (app prefs, language), Quit.
- **File**: New Machine, New Folder, Duplicate, Rename, Delete, Show in
  Finder, Open Config File, Close.
- **Edit**: standard items; Copy Screen; Paste Text to Guest.
- **View**: Show Library, Show Status Bar, Enter Full Screen, Detach Display,
  Scale (Fit, 1x, 2x, Integer Only), Filter (Nearest, Linear), Serial
  Console, Log.
- **Machine**: Power On, Shut Down Guest, Power Off, Restart, Save State,
  Saved States, Take Screenshot, Capture Input, Send Key (F11, Ctrl+Alt+Del,
  Esc), Machine Settings.
- **Devices**: per CD-ROM (Insert Disc, Eject, Remount in IRIX), per disk
  (Apply Changes, Roll Back), Synchronize Disks, Network (NAT, Bridged, Check
  Networking, Port Forwards), Camera Source, Mute Audio.
- **Tools**: Benchmark, Camera Test, Monitor Console, Perfetto trace,
  Automation, Diagnostics, Reset NVRAM, Set MAC Address.
- **Window**, **Help** (guide, report an issue).

## Disk browser (rusty-backup crates)

One feature (D23), in `iris-gui-core` so every shell gets it. The create-disk
dialog is unchanged: it still makes an empty image.

### What the user sees

- A "browse files" icon opens the browser for one disk or CD-ROM. It lives
  in four places (reviewed in mock-ups, 2026-11-22): a main-toolbar button
  beside Insert Disc (a short list when the machine has several disks); a
  button at the right edge of each disk and CD-ROM row in the summary's
  hardware list (the row itself still opens settings); first item of the menu
  a status-bar disk light opens (then Apply Changes, Roll Back, Disk
  Settings); and first item of each disk's submenu in Devices.
- Left: the device's partitions from the SGI volume header, each with its
  type, size and filesystem (EFS, XFS, ext2/3/4, or "not browsable" with the
  reason: swap, volume header, unformatted, unsupported).
- Right: the selected partition's files as a list with name, size, mode,
  owner, group and date; folders open in place; a path bar.
- Actions, each a controller `Action` with a toolbar icon, a menu item and a
  context-menu entry:

| Action | Notes |
|---|---|
| Export | selected files or folders to a host folder (`rb-fsexporter`), with progress and cancel |
| Export as archive | `.tar` or `.tgz` (`fs-tar` `export_tar`, `TarCompression::None` / `Gzip`) |
| Import | host files or folders into the current folder (`rb-fsimporter`), after a preflight that checks free space and name collisions |
| Import archive | a `.tar`, `.tgz` or `.zip` expanded into the current folder (the importer expands these itself) |
| Import options | what to do with existing names, symlinks, ownership and mode for new files (default root:sys, the host's mode bits), archive expansion on or off |
| Edit text | open a file in a plain editor, save back. Saving writes a new file and replaces the old one, then restores owner, mode and dates. Refuse files over a size limit or that are not text. |
| Permissions | chmod (`set_permissions`) and chown (`set_owner`), on one entry or recursively |
| Rename | `rename`, within the folder |
| Move | to another folder. **Not available yet**: the filesystem trait only renames within a folder. Wait for the rusty-backup release that adds a move; show the action disabled with that reason until then. Do not fake it with copy and delete. |
| New folder, Delete | `create_directory`, `delete_entry` / `delete_recursive`, with a confirmation that names what goes |
| Check | `Filesystem::fsck`, report only |

- **Drag and drop.** Dropping files or folders from the host's file manager
  onto the list imports them into the folder being browsed (onto a folder
  row: into that folder). The drop does not start copying: it opens the
  Import dialog with the source and destination filled in, so the options
  and the free-space check are seen first. A drop is refused, with the
  reason, wherever Import is disabled.
- **Line endings.** Import offers "Convert text files to Unix line endings"
  (dos2unix), off by default. It applies only to files that look like text
  (no NUL bytes in the first block) and never to archives copied as one
  file. The text editor shows the file's line endings and can convert on
  save. `rb-fsimporter` has no such option today; a release
  that adds it is assumed to land before the browser ships (requested
  2026-11-22). Use that option; do not build a staging-folder conversion in
  iris. Until the release exists, leave the checkbox out.
- Decided defaults (2026-11-22): new files take the destination folder's
  owner and group; owner and group names are read from the browsed disk's
  own `/etc/passwd` and `/etc/group`, with numbers always accepted; the text
  editor refuses files over 1 MB or that look binary and offers Export
  instead; a `.tardist` is a tar and is unpacked like one unless the user
  chooses "copy the archive as one file" (what you want before running
  `inst` on it); the volume-header partition is listed and browsable through
  `iris::sgi_vh`, read-only.

### Read-only cases

The controller decides, and gives the reason the view shows:

- **The machine is running**: read-only. Browse, export and export as
  archive work; nothing that writes is offered. The view is of a live disk,
  so it can be a few seconds stale or briefly inconsistent; say so in the
  window.
- **CD-ROMs**: always read-only.
- **XFS the crate cannot edit**: IRIX volumes with version-1 directories, and
  Linux v5 volumes with features it does not maintain. `fs-xfs` reports this;
  pass its reason on.
- **A filesystem whose check fails**: read-only until the user chooses to
  continue.

### The crates

All MIT, all on crates.io (the user's statement; `sgi-disk` 0.1.0 and
`fs-xfs` 0.2.0 checked 2026-11-22). The crates' own summary for iris is
`../rusty-backup/docs/iris_sgi_crates.md`: read it for exact signatures.

| Crate | Version | Use |
|---|---|---|
| `sgi-disk` | 0.1.0 | `read_volume_header`, `filesystem_partitions`, `open_partition` / `open_partition_editable`, `open_kind` / `open_kind_editable` for EFS and XFS. Its disk builders are not used. |
| `disk-core` | 0.3.0 | `Filesystem`, `EditableFilesystem`, `FileEntry` |
| `fs-unix` | 0.3.0 | features `efs` and `ext`. ext is opened by iris directly, below. |
| `fs-xfs` | 0.2.0 | through `sgi-disk`'s `xfs` feature. MIT from 0.2.0; never depend on 0.1.x (GPL). |
| `rb-fsexporter` | 0.2.0 | `extract` with a `Host` for progress and cancel |
| `rb-fsimporter` | 0.1.0 | `import_dir_with`, `preflight_dir_with`, `measure_dir_for`, `DirImportOptions`, a `Host` for progress and cancel. Choose `native-zstd` or `pure-zstd` by what builds cleanest on every target. |
| `fs-tar` | 0.2.0 | `export_tar` |

Depend on released versions in `iris-gui/core/Cargo.toml` with
`default-features = false` and only the features needed. No path or git
dependencies, and never rusty-backup's main crate (AGPL-3.0). Add the crates
to iris's third-party notices.

**ext partitions** (Linux guests): `sgi-disk` lists only EFS and XFS, and its
partition-type list has no Linux type, so a Linux partition parses as
`Unknown`. That needs no change in rusty-backup. iris takes the full header
from `read_volume_header` (every entry has its start and size), and for a
partition that is not EFS or XFS probes for an ext superblock and opens it
with `fs_unix::ext::ExtFilesystem::open(reader, offset)`, which is both
`Filesystem` and `EditableFilesystem`.

### How iris reaches the disk

- **Stopped machine**: open the image with `iris::block_node::open_image`
  and wrap the `BlockNode` (`read` / `write` at a byte offset, `flush`) as
  `Read + Write + Seek + Send`. The browser then sees what the guest would
  (a raw image, a CHD with its diff, a copy-on-write overlay), and writes
  land where the guest's would: on a copy-on-write disk they go to the
  overlay, where Apply Changes and Roll Back already work, and an edited CHD
  leaves a pending sync. Never open the file behind iris's back.
- **Running machine**: the current contents are partly in the running
  machine (overlay, CHD diff), so the browser must read through it. This
  needs a small read-only hook in the core, approved 2026-11-22:
  *proposed* `Workstation::read_disk(bus, id, offset, &mut [u8]) ->
  io::Result<()>`, reading the target's block node under the bus lock in
  short holds so it never stalls the guest for long, with no write
  counterpart. Like the counters, it goes on its own branch off `main` with
  its own pull request, not on `iris-newgui`.
- **Slow work is a job**: listing a big tree, import, export, archive,
  recursive chmod and check run on a worker with progress and cancel, never
  in `apply` or `tick` (rusty-backup's `model::job` is the pattern).
- Closing the window, or powering the machine on, first finishes or cancels
  the job and calls `sync_metadata`; power-on is refused while a write job
  runs.

### Verified (2026-11-22)

- Every action (open, list, owner names from `/etc/passwd`, edit text and
  save, new folder, rename, chmod, chown, import a folder and a file, export
  to a folder and to `.tgz`, delete, check) ran on copies of two real
  installs from the NAS: `Indy-IRIX53_dev.chd` (IRIX 5.3, EFS root) and
  `irix65_Indy.chd` (IRIX 6.5, XFS root, which the crate opened for editing).
  The crates' check found nothing new afterwards; both disks already had
  orphan inodes (and the 6.5 disk 24 leaked blocks) before any edit, so the
  open-time check makes them read-only until Edit Anyway. Writes went to the
  CHD's `.diff.chd`, as the guest's would. The test is
  `controller::disk_browser::tests::real_irix_disk` (ignored; needs
  `IRIS_BROWSER_TEST_IMAGE`). Found and fixed: new inodes need the file-type
  bits (`clanker/gui/disk-core-create-modes-carry-the-file-type.md`).

- **Booted and checked by IRIX** (2026-11-22). The edited IRIX 5.3 copy
  booted to the desktop, showed the edited `/etc/motd` at login, listed the
  imported files and read them back. Attached unmounted to a pristine copy,
  IRIX's `fsck -n` reported exactly what it reports for an untouched copy
  (orphan inode 371, a free-inode count), nothing new. The edited 6.5 XFS
  copy, attached the same way: `xfs_repair -n` matched the untouched copy,
  and after IRIX replayed the log on a read-write mount every edit was
  there. Both 6.5 copies have an unreplayed log, which fs-xfs cannot see;
  XFS therefore opens read-only until Edit Anyway
  (`clanker/gui/disk-browser-xfs-log-replay.md`).
- Found by the IRIS boot and fixed: folder imports took the host's owner;
  they now take the destination folder's, as decided.

- **Editor saves, re-checked after the review** (2026-11-26). A second save
  with the editor open deleted by the entry it opened with (EFS removes the
  name before checking the inode), and saves went into the folder shown, not
  the file's. Fixed (the editor keeps its folder and the fresh entry);
  `real_irix_disk` now saves twice and from the root, and passed on fresh
  copies of `Indy-IRIX53_dev.chd` (EFS) and `irix65_Indy.chd` (XFS). The 5.3
  copy then booted to the desktop: `/etc/motd` held the last save, with its
  mode and owner, no `.iris-edit` file in `/etc` or `/`, the imports there;
  `shutdown -p` powered it off.
  - A headless (`headless = true`) Indy stays in the PROM: it does not
    autoboot. Boot checks need a display (the monitor's `screen shot` then
    works) and `key`/`ptr` to log in.
  - That 5.3 disk has no network (no telnet) and takes about 10 minutes to
    show xdm's login after "Reboot to start using the reconfigured kernel";
    log in as root on the screen, then raise the Console window by its
    title bar (its left button minimizes it) before typing.

### Verified 2026-11-27 (fs-xfs 0.2.3, disk-core 0.3.3)

`real_irix_disk` (every write action, the tardist prompt included) on copies
of `Indy-IRIX53_dev.chd` (EFS) and `irix65_Indy.chd` (XFS, unclean log
replayed first), then IRIX itself, with the edited disk attached as a second,
unmounted drive beside an unedited copy as the baseline:

- EFS: `fsck -n` on the edited disk reports exactly the baseline's problems
  (one unreferenced inode, a wrong free-inode count), with 12 more files. The
  content reads back in IRIX, and the edited disk boots as root.
- XFS: the content reads back in IRIX, and the root boots after the replay
  and the edits. `xfs_check` finds two faults the baseline does not have; see
  `clanker/gui/fs-xfs-deletes-leave-freed-inodes-half-cleared.md`. IRIX
  6.5's directories are editable (the same short-form and block forms).

### Not verified

- `fs-xfs`'s directory hash on large IRIX-made directories (leaf and node
  form) beyond these edits.

### Overlap with what iris has

`iris::sgi_vh` and the `mkvh` tool build PROM-bootable volume-header images
and `mkvh dump` prints a header. Keep them; they are not part of the browser.

## Phases

Small commits per slice. egui builds and `cargo test -p iris-gui` passes after
each one.

0. **Scaffold.** Create `iris-gui/core` (`iris-gui-core`, depends on
   `iris`, no toolkit crates) and make `iris-gui` depend on it. Add
   `controller/mod.rs` with the `Controller` trait, `Effect`, `Notify`,
   `ids`, `keys`, `labels`, `icons`; add `scripts/check-boundaries.sh`. No
   behaviour change.
1. **Move the model down.** Move the toolkit-free modules to
   `core/src/model`; egui imports them from there. Check each for egui
   types first (`framebuffer.rs` splits; `i18n.rs` and the `t!` macro must
   work from both crates).
2. **Controllers, egui rewired per slice**, in this order: App (library) ->
   Session (including the counter sampling above) -> Display -> Settings
   (schema and `ProfileTraits`) -> Network -> Serial -> Bench, Debug, Camera
   -> dialogs -> Disk browser last (a new feature, not parity: read-only
   browse and export first, then import and the edits, then the running-
   machine read hook once it has landed in the core). Exit: the boundary
   script reports 0 for the shells it counts. *Done 2026-11-22:* egui is no
   longer counted (D1 changed), and the AppKit shell is clean.
3. *(dropped, D16: egui keeps its look.)*
4. **AppKit shell.** 0 app, window, menu bar, wake, library; 1 summary;
   2 Metal presenter, physical-key input (macOS key codes to the
   `winit::keyboard::KeyCode` the PS/2 layer takes), capture, fullscreen,
   detach; 3 status bar; 4 settings sheet (the field kinds); 5 disks, CD,
   COW, sync on quit, then the disk browser; 6 network; 7 serial console; 8 benchmark, debug,
   camera; 9 bundle, signing, notarized DMG, its own bundle id beside the
   egui app.
5. **GTK 4 shell** (`iris-gui/frontends/gtk4`, binary `iris-gtk`), the native
   Linux front end, which replaces the egui app on Linux as the AppKit shell
   replaced it on macOS. libadwaita (D24), our own SVG icons (D25). Slices,
   one commit each:
   0. **App.** The crate, `AdwApplication`, single instance, the wake (`Notify`
      onto the GLib main context) and the timed tick, the
      action -> effects -> refresh loop, the window with its header bar, quit
      through the controller, and the `IRIS_GTK_SCRIPT` hook with `shot:`
      (the window paints itself into a PNG through its own GSK renderer:
      Wayland has no way to grab another process's window).
   1. **Icons.** The SVG set (D25) and its loader.
   2. **Chrome.** The app menu (`GMenu`) over the command ids, enablement
      through `GSimpleAction`, accelerators from `controller::keys`, and the
      dialog host: `AdwAlertDialog` for Confirm, Choose, AskText and Alert,
      `GtkFileDialog` (the xdg portal) for the pickers. Every GTK dialog is
      asynchronous, so an answer comes back as an action rather than from the
      effect call; `IRIS_GTK_ANSWERS` answers them unattended.
   3. **Library** in an `AdwOverlaySplitView`: families, folders, the machine
      pictures, row Start buttons and the row menu.
   4. Machine page; 5. status bar and the device lights; 6. the guest display;
   7. keyboard and mouse (physical keys, pointer lock where the compositor
      offers it, Follow mode where it does not); 8. settings over the schema;
   9. New Machine; 10. Preferences; 11. disk browser; 12. serial console,
   Check Networking, camera test, benchmark, diagnostics, Help;
   13. packaging (AppImage and deb/rpm ship the GTK app; ask the user before
   changing what CI ships).

   Then the **parity check and release** beside egui, and Win32 over the same
   controllers.

Testing: controller unit tests with no window; the existing egui headless
tests; later a flow driver over the accessibility ids, as rusty-backup's
tier 8.

When behaviour a human guide describes changes, update it in the same change:
`iris-gui-README.md`, `FEATURES.md`, `HELP.md`, and a `CHANGELOG.md` line.

## Outside iris-gui

Approved:

- Root `Cargo.toml`: one `members` line per shell crate (D19).
- The disk and audio counters in the core (D18), done.
- Human docs at the repository root, and the release workflow for the new
  bundle, when it ships.

- The rusty-backup crates as dependencies of `iris-gui-core` (D23), and the
  matching licence entries in iris's third-party notices.
- A read-only disk hook on `Workstation` for browsing a running machine's
  disk (2026-11-22), on its own branch and pull request.

Not approved, ask first: a serial counter in the core; anything else in
`src/`; any change in `../rusty-backup` (it is a separate repository).

## Implementer decisions (2026-11-22)

- **Controller composition.** `AppController` is the root each shell owns. It
  owns the `SessionController` (which owns the worker handle), so library
  changes can wait for the worker. Window-like tasks with their own lifetime
  (serial console, benchmark, camera test, disk browser, dialogs) are separate
  controllers the app creates and the shell holds and ticks.
- **Answers.** Every effect that asks something carries a `purpose`; the shell
  answers with `Answer` (`Picked`, `Confirmed`, `Text`, `Chose`) in the asking
  controller's `Answer(..)` action. One generic effect host per shell.
- **Toasts** are `Effect::Toast`; each shell shows them in its status line. The
  controller keeps no toast state.
- **Library folders** live in `gui.json` (`library_folders`: name, machines,
  collapsed). A machine in no folder shows under its family
  (`controller::profile::Family`).
- **Duplicate** copies the TOML and every file under 4 MB in the machine's
  folder (NVRAM, EEPROM, flash); larger files (disk images) stay where they are
  and the copy refers to them by absolute path. The toast says how many are
  shared. Copying disk images silently could fill a disk.
- **Library icons** (the user's open question, proposed answer used): each
  profile has a `MachineIcon` in the `ProfileTraits` table; a shell maps it to
  a picture, falling back to a generic glyph. Changing the answer is one match
  per shell.
- **Session actions go through the app** (`AppAction::Session`): power on
  reads the TOML and checks it, a disc insert edits the config when stopped,
  a folder grant writes the preferences. The preflights (missing disks, CHD
  folders the app cannot write, safe stop) are `Choose` questions.
- **Status tick split** (implementer question): the worker ticks every 100 ms
  and samples the disk and audio counters each tick; MIPS and Hz are measured
  over at least 0.45 s and carried between. A `Status` is sent only when it
  changed, and every event wakes the shell through `Notify`, so an idle guest
  costs no wakes and no shell polls.
- **Saved states** (implementer question) are listed by scanning
  `saves/<name>/` in the machine folder (`snapshot.toml`'s time, the folder's
  size); no index. Restore while stopped powers on, then restores.
- **Lights**: one per configured SCSI disk and CD-ROM (bus = controller,
  then id), network, audio. NVMe and Fibre Channel disks have no light yet.
  The camera light is dropped: nothing in the core says the camera is in use.
  The serial light comes with the serial console controller.
- **Display**: the shell sends physical keys and pointer deltas only while
  captured; the release chord, Ctrl+Alt+Esc, F11 / Ctrl+Alt+F11 and the
  focus-loss grace are `DisplayController` rules. Capture, full screen and
  detach are controller state the shell mirrors (it grabs the cursor or
  resizes the window when the state changes). Scaling is app-wide
  (`display_mode` Fit or Scale with `vm_scale`, `integer_only`,
  `display_filter` in `gui.json`); `display::draw_size` is the one sizing
  rule. egui has no detached window (it keeps its look, D16).
- **Settings schema scope.** `SettingsController` builds Machine, Memory,
  Display, Disks, Network and Advanced; Advanced is the complete TOML
  (validated by `hardware::parse_complete`), so every key the schema does not
  model is still reachable from any shell. NAT subnet, port forwards and the
  PCAP interface apply live. egui still draws its own tabs over
  `config_mut()` (D16) and will not be moved onto the schema (D1 changed).
- **Commands by id** (`controller::commands`): the toolbar's and menu bar's
  tooltips, enablement and routing are the controller's; a shell maps a
  sender's identifier to `AppController::command(id)` and only does the
  `ShellCommand`s (its own views). Per-item buttons encode the item in the id
  (`ids::item`).
- **AppKit quit** goes through `applicationShouldTerminate:`, which asks the
  controller; Cmd+Q now folds CHD diffs like a window close.
- **AppKit test hook**: `IRIS_APPKIT_SCRIPT` runs command ids (and
  `select:NAME`, `page:SECTION`, `set:CONTROL=VALUE`) a few seconds apart, so
  the shell can be driven by its controller ids with the screen locked.
- **Settings controls by id**: a shell names each settings control
  `settings::control_id(field, row, part)` and turns a control's report into
  an action with `settings::action_for`, so table rows need no per-shell
  routing. The AppKit settings are a window, not a sheet: edits apply at
  once and autosave, so there is nothing to confirm, and the machine stays
  visible beside it.
- **Autosave needs a timed tick** in a shell that sleeps until woken: a
  stopped machine sends no wakes, so AppKit schedules a tick at
  `autosave_due_in()`.
- **Reset NVRAM** now asks first; it loses PROM `setenv` settings.
- **Interim accessors.** `AppController::config_mut`, `prefs_mut` and
  `SessionController::handle{,_mut}` are `#[doc(hidden)]` bridges for egui
  code not yet on a controller. Only egui uses them; they go with egui.

## Open

Parked by the user (2026-11-25), from the testers' feedback:

- Viewing other machines' pages while one runs (Start greyed). Selecting a
  machine also changes the working directory the running machine uses, so it
  needs "the machine shown" apart from "the machine running".
- Removing the serial-console Shut Down and graceful Restart until the core
  has a front-panel power button (Power Off main, Reset in its menu). The
  tester's view: a graceful action that needs a logged-in console should not
  ship at all. The core work: a power-button press per machine (Indy,
  Indigo2 first), which IRIX answers with an orderly shutdown.
- Running several machines at once: core work first, and a measurement of
  what two machines cost on ordinary hosts.
- *Decided 2026-11-27:* the shortcut proposal is accepted: Show/Hide
  Library ⌃⌘S, Full Screen ⌃⌘F only, no shortcut for Shut Down or Detach,
  Serial Console ⌥⌘C.
- *Decided 2026-11-27:* several machines at once stays unsupported.
- *Decided 2026-11-27:* the serial-console Shut Down and Restart stay as
  they are until IRIS can press a machine's power button for IRIX.

Answered by the user (2026-11-27):

- The circles on the IP tiles do not mean machine pictures; the photos
  stay as they are.
- "This build": nothing to do. The machine page has no build section; the
  feature list is in Help > Diagnostics and the executor chip in the status
  bar.
- Disk browser: importing a `.tardist` asks whether to unpack it.
- Disk browser: Move waits for a rusty-backup release (checked 2026-11-27).

For the implementer:

- The read hook's exact shape, and a text-editor size limit.
- *Decided:* the read hook waits on its own branch; the text editor's limit
  is 1 MB (plan default). A partition that will not open is marked with the
  reason and the browser moves to the next.

- Where a machine's saved states are listed from: a scan of the machine
  folder, or an index the controller keeps.
- A source for the camera "in use" light.
- Whether the status tick is split into a slow and a fast part for the lights.
