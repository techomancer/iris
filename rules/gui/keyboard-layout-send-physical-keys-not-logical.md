# Non-US keyboard layouts: send physical key positions, never logical keys

The guest does its own layout translation — the SGI PROM reads `keybd=` (e.g.
`setenv keybd de_CH`, then `rtc save`) and IRIX layers X11 keymaps on top.
`keybd` appears **nowhere** in the iris source: iris must feed the guest raw
**scancodes for physical key positions** and let the guest apply the layout.

## The bug (fixed, issue #72)

`src/ui.rs` (CLI) always did this right — it uses `KeyEvent::physical_key`.

`iris-gui/src/input.rs` did not: it read `egui::Event::Key { key, .. }`, which is
the **logical** key the host OS already produced from the host's layout, then
reverse-mapped it to a `KeyCode` assuming a US keyboard. That applies the layout
twice: what you see is `guest_layout(US_position_labelled_with(host_layout(key)))`.

On a German host + `keybd=de`, that predicted every symptom in #72 exactly:

| Pressed | egui logical `Key` | sent as | guest showed |
|---|---|---|---|
| `-` (at US `/`) | `Minus` | US `Minus` | `ß` |
| Shift-7 = `/` | `Slash` | US `Slash` | `_` |
| Shift-, = `;` | `Semicolon` | US `Semicolon` | `Ö` |
| Shift-0 = `=` | `Equals` | US `Equal` | `` ` `` |
| `Z` (at US `Y`) | `Z` | US `Z` | `y` |

**Fix:** prefer `physical_key` (`Event::Key` carries it), fall back to `key`.

### Why umlauts "worked" while ASCII punctuation didn't

Not a contradiction — it's the giveaway. egui-winit builds the event as
`key: logical_key.or(physical_key)` (egui-winit `src/lib.rs`). `egui::Key` has no
variant for `ä ö ü ß`, so those keys fell through to the *physical* key and landed
correctly by accident. Only characters egui can name got relocated. If a bug
report says "umlauts fine, `/` and `;` broken", this is the cause.

## The ISO 102nd key (`< > |`)

Separate, independent bug: `KeyCode::IntlBackslash` — the extra key left of `Z`
on every European ISO keyboard — had no entry in any scancode set, so it was dead
in the CLI too. Added:

| Set | Code | Cross-check |
|---|---|---|
| 1 | `0x56` | standard `KEY_102ND` |
| 2 | `0x61` | standard; arrows are `E0`-prefixed in set 2, so `0x61` is free |
| 3 | `0x13` | Linux `atkbd_set3_keycode[0x13] == KEY_102ND (86)`; sits in the leftmost column between `LCtrl 0x11`, `LShift 0x12`, `CapsLock 0x14` |

Do **not** reuse set 2's `0x61` for set 3 — in set 3 that is the Left arrow.
IRIX drives the keyboard in **set 3**, so set 3 is the one that matters.

## Why iris-gui is on egui 0.35 (do not downgrade)

**The 0.29 → 0.35 bump is load-bearing for keyboard correctness, not cosmetic.**
On **0.29** egui-winit threw away information no iris-side code could recover:

- **`IntlBackslash` is unrepresentable.** `key_from_key_code` has no arm for it and
  `egui::Key` has no `Less`/`Greater`, so *both* `logical_key` and `physical_key`
  are `None` and **no event is emitted at all**. `<` `>` do nothing in the GUI.
- **AltGr is unreachable.** `egui::Modifiers` has only `alt`, no left/right split,
  and modifier keys produce no `Key` event, so `input.rs` can only ever send
  `AltLeft`. On DE/de_CH that kills the whole AltGr level (`@ \ | { } [ ] ~ €`).
  de_CH `\` is AltGr+`<` — hit by both gaps at once.
- **Numpad is collapsed** into the main row (`Numpad0-9→Num0-9`,
  `NumpadDivide→Slash`, `NumpadEnter→Enter`), so `KP_*` keysyms never reach X11.

eframe 0.29 exposes no raw winit window-event hook (`window_event` is
`pub(crate)`; `raw_input_hook` only sees already-converted `egui::Event`).

**egui 0.35.0 closed all three** — this is why we bumped. `Key::IntlBackslash`
exists, and `key_from_key_code` gained discrete `ShiftLeft/Right`,
`ControlLeft/Right`, `AltLeft/Right`, `SuperLeft/Right` arms, so `pump()` now
forwards modifiers as **real press/release key events** and the
`egui::Modifiers`-diff synthesis is gone. `InputState` tracks `held_mods:
Vec<KeyCode>` instead of `last_mods: Modifiers`.

That also enabled the **two-key release chord** (`RELEASE_CHORD`): hold left
Ctrl+Alt (left Option+Cmd on macOS) and press nothing else. It fires on *key-up*
and only if `chord_consumed` is false — i.e. no other key was pressed while the
chord was held — which is what preserves Ctrl+Alt+F11. Ctrl+Alt+Esc is kept as an
explicit fallback. Impossible on 0.29: `egui::Modifiers` has no left/right split.

**The Mac App Store winit patch is *not* a blocker.** egui 0.35 pins
`winit = "0.30.13"` — the exact version vendored at `third_party/winit-0.30.13`,
so `[patch.crates-io]` still applies (verify: the `winit 0.30.13` block in
`Cargo.lock` has **no `source` line**).
See [../macos/appstore-private-api.md](../macos/appstore-private-api.md).

### 0.29 → 0.35 port notes (53 errors, all mechanical)

| Old (0.29) | New (0.35) |
|---|---|
| `ui.close_menu()` | `ui.close()` (41 of the 53) |
| `SidePanel::left/right`, `TopBottomPanel::bottom` | unified `egui::Panel::left/right/bottom` |
| `.exact_width()` / `.default_width()` | `.exact_size()` / `.default_size()` |
| `.show(ctx, …)` on panels | `.show(ui, …)` |
| `.show_animated(ctx, bool, …)` | `.show_collapsible(ui, &mut bool, …)` |
| `App::update(&mut self, ctx, frame)` | `App::ui(&mut self, ui, frame)` |
| `ctx.screen_rect()` | `ctx.viewport_rect()` |
| `ctx.style()` | `ui.style()` |
| `Frame::none()` | `Frame::new()` |
| `Margin::symmetric(f32, f32)` | `Margin::symmetric(i8, i8)` |
| `Image::rounding()` | `Image::corner_radius()` |
| `push_id(impl Hash)` | needs `AsIdSalt` = `Hash + Debug` |

`App::ui` gives a `&mut Ui`, not a `&Context`. Keep the old body working with
`let ctx = &ui.ctx().clone();` at the top (temporary lifetime extension), so the
`&Context` call sites are unchanged and panels take `ui`.

iris-gui depends on no third-party egui plugin crates, so nothing else gates the
bump.
