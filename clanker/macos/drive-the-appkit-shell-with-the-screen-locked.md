# Drive the AppKit shell with the screen locked

The AppKit shell (`iris-gui/frontends/appkit`) has test hooks so an agent
can drive it and see it without the screen unlocked.

- `IRIS_APPKIT_SCRIPT="step;step;..."` runs one step every
  `IRIS_APPKIT_SCRIPT_DELAY` seconds (default 3). A step is a command id
  (`toolbar.power-on`, `menu.close`, `help.nfs`, `browse:1`, a status light
  id), or `select:NAME`, `page:SECTION` (settings), `set:CONTROL=VALUE`
  (`i:N`, `b:0/1`, `t:TEXT`, `n:NUM`, as the control would report),
  `open:N`/`pick:N,M` (browser rows), `drop:PATH`, `type:TEXT` (the browser's
  editor, `\n` for a newline), `at:X,Y[,BUTTONS]` (Follow mode's pointer on a
  guest pixel), `dump:browser`, `dump:devices`, `hover:library|panel-out|status|status-out`
  (the pointer over a hover zone: the hidden library's edge, leaving its
  floating panel, full screen's bottom edge, leaving its status bar),
  `split:W` (the sidebar's divider to W points; prints the width it settles at).
- The app's monitor has `screen`, `ptr` and `key` (one HostInput per
  machine): `ptr pos` reads where the guest pointer is, `key type` types.
  The Indy PROM on a blank disk stops at a menu with a cursor: enough to
  test pointer steering with no IRIX to shut down.
- `IRIS_APPKIT_ANSWERS="yes;text:foo;choose:1;pick:/path"` answers the next
  prompts (confirm, text, choice, file or folder picker) in order; later
  prompts show their real panels. Every disk-browser write starts with one.
- Run with `HOME=<scratch>` so the library, settings and window frames are
  separate from the user's.
- See windows with `screencapture -x -l <window id>`. Get ids from
  `CGWindowListCopyWindowInfo` filtered on the owner name, case-insensitively:
  the bundled app's owner is "IRIS", the debug binary's "iris-appkit".
  Window frames are autosaved per window, so a window that grew once stays
  large in later runs.

Gotchas:

- A script step that opens a modal panel or a pop-up menu stalls the script
  (`performSelector:afterDelay:` does not fire in those run-loop modes):
  answer panels through `IRIS_APPKIT_ANSWERS`, and do not script pop-ups.
- With the screen locked the app is never active, so there is no key window.
- Full screen does not engage while the app is in the background (launched
  from a shell, not brought to the front): `toggleFullScreen:` is called and
  nothing happens. A run that did enter it had the window in front. Check
  full-screen behaviour by hand, or with the app active.
- Shut Down sends `halt` to the
  serial console, which does nothing on a guest whose console is the
  graphics screen. A guest booted in the app with the screen locked cannot be
  shut down cleanly; do not start IRIX from a script unless something can
  log in (a telnet forward to a guest whose network is up).
- Stop every run you start: a forgotten instance keeps its disk browser's
  image open.
- The app's monitor `screen shot` / `screen pix` read a black frame (2026-11-27:
  `screen info` stayed at frame 3 while the window showed IRIX's desktop).
  `ptr` and `key` do reach the guest. Look with `screencapture -l <id>` of
  the window instead; the guest fills the view scaled, so map window points
  back to guest pixels before a `ptr click`.
- `key type` sends `\n` as two characters when the shell passes it through
  single quotes, so nothing is entered. Send `key Return` as its own command.
- A guest with no working telnet can still be shut down cleanly: on IRIX 5.3,
  `ptr click` the root icon and `key Return` logs in; close Software Manager
  (double-click its window-menu button), click into the Console window, type
  `sync; shutdown -y -g0 -p`, `key Return`. The app folds a compressed CHD's
  diff after the power-off; wait for `*.synctmp.chd` to go before ending it.
