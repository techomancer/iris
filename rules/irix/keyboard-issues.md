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
