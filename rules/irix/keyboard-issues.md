# IRIX Keyboard Issues

## Alt-tab corrupts X11 keyboard input

After alt-tabbing away from the Rex window and returning, IRIX X11 terminal
apps (Console, Terminal, xterm) show escape codes instead of typed characters.
The IRIX login dialog still works (different input path).

**Cause:** The Alt key release event from alt-tab confuses IRIX's X keyboard
state machine. The PS/2 scancode for LAlt (0x19 in set 3) is delivered as a
release without a matching press.

**Workarounds:**
1. Don't alt-tab while interacting with IRIX GUI — use Right Ctrl to ungrab mouse
2. Use telnet via port forwarding (host 2323 -> guest 23) for terminal access
3. Mount the disk image directly to edit files from the host

**Status:** Pre-existing emulator issue, not introduced by any recent changes.
Proper fix would require filtering or suppressing stale modifier key events
in the UI event handler when focus is regained.

**Possibly addressed in iris-gui (2026-08-03, UNVERIFIED).** `pump()` no longer
synthesises modifiers from `egui::Modifiers` diffs; it forwards real
press/release key events and tracks `held_mods`, and `release_capture()` lifts
exactly what it recorded as pressed — so an orphan release without a matching
press should no longer be generated on focus loss. **Not tested against IRIX X11**
— if you can reproduce the original alt-tab corruption, check whether it still
happens before spending time here. Note this applies to iris-gui only; the CLI
(`src/ui.rs`) always forwarded real key events.
See [../gui/keyboard-layout-send-physical-keys-not-logical.md](../gui/keyboard-layout-send-physical-keys-not-logical.md).
