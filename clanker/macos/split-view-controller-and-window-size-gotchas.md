# NSSplitViewController: constraints reach the window, and autosave restores collapse

Status: rule (2026-11-25), the Mac app's main window (`iris-gui/frontends/appkit/src/window.rs`).

- **Constraints inside the split now size the window.** With a plain
  NSSplitView the pane's content constraints stayed inside it. Under an
  NSSplitViewController they reach the window: a required or high-priority
  constraint in the page (the machine page's "fit without scrolling",
  `page.height <= clip.height`) above the window's own size priority
  (`NSLayoutPriorityWindowSizeStayPut`, 500) made the window grow taller at
  launch. Keep such page constraints below 500 (the page uses 490 to fit).
- **...and they move the divider.** A sidebar holds its width at 260. The
  preview's "as wide as the page" at 480 outranked it, so AppKit widened the
  sidebar to its maximum to make the page narrower, and a dragged divider
  snapped back. Anything in the pane that prefers a size stays below 260
  (the preview's growth is 200).
- **Stack views hug at 750 horizontally.** A vertical NSStackView page in the
  content pane pulled the divider toward itself and gave the window's extra
  width to the sidebar. Set the page's (and wide rows') horizontal hugging to 1.
- **The split's autosave restores a collapsed sidebar the item does not know
  about.** After a run that ended with the library hidden, the next launch
  showed it but `NSSplitViewItem.isCollapsed()` was true, so the first toggle
  did nothing. Call `setCollapsed(false)` after setting up the split (the
  library always starts shown; the autosave keeps only its width).
- The sidebar's scroll view adjusts its content insets for a full-size
  content view the window does not have; the list sat under the toolbar.
  `setAutomaticallyAdjustsContentInsets(false)` and zero insets.

## The glass sidebar needs three things

On macOS 26 and later AppKit puts a sidebar item in an `NSGlassEffectView`
and floats it over the content only when:

1. the split view controller is the window's `contentViewController` (not a
   view inside another content view; the status bar, hover zones and floating
   library moved into the machine's column);
2. the window has `FullSizeContentView`, and the toolbar carries
   `NSToolbarSidebarTrackingSeparatorItemIdentifier` after the items that sit
   over the sidebar (the library button; it is no longer `navigational`);
3. the content item sets `automaticallyAdjustsSafeAreaInsets` (guard it with
   `respondsToSelector:`; older systems lack it). The content then runs under
   the sidebar, so everything in it keeps to its `safeAreaLayoutGuide`
   (left: the sidebar's width; top: the toolbar).

The sidebar's own table must be clear (`setBackgroundColor(clear)`) or it
covers the glass. Collapse and expand with the controller's `toggleSidebar:`:
an item's `animator().setCollapsed(false)` expands without animating.
`IRIS_APPKIT_SCRIPT` step `split:W` moves the divider and prints the width
it settles at.

Over a plain dark window that glass reads as no edge at all, and agent
captures (always of a background window) cannot show the front-window look.
The sidebar is drawn as Finder's instead: an `NSBox` (custom, no border)
filling it in `quaternaryLabelColor`, a 1 pt `tertiaryLabelColor` line down its
trailing edge, `titlebarAppearsTransparent`, and the sidebar's
`titlebarSeparatorStyle` none. Two AppKit layers fought that tone:

- a source-list table makes its scroll view add an `NSVisualEffectView` of
  its own behind the rows, darker than the fill, so only the strip under the
  window buttons showed the fill (a "band"). The window hides that subview
  on every refresh (AppKit adds it again when the table reloads);
- floating group rows draw their own strip: `setFloatsGroupRows(false)`.

The table uses `NSTableViewStyle::SourceList` (rounded, inset selections);
the deprecated source-list highlight only below macOS 11.
`find what draws a region`: walk the theme frame (`contentView().superview()`)
and print each view's class and window rect.

## Debug builds never start the CPU

`Machine::start` autostarts the CPU only without `debug_assertions` (and
without `developer`). A debug `iris-appkit` therefore shows a machine whose CPU
waits at the reset vector until the monitor's `start`
(`python3 tools/mon.py start`). The session controller counts a guest power-off
only on a running -> stopped change, so that waiting CPU is no longer stopped
two seconds later; drive tests with a monitor `start` after Power On.
