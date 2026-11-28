# A shell that sleeps until woken needs a timed tick for autosave

**Symptom.** In the AppKit shell, settings edits showed in the window but
never reached the machine's TOML file while the machine was stopped.

**Cause.** `AppController` autosaves from `tick()` once an edit is
`AUTOSAVE_AFTER` (600 ms) old. egui repaints on its own schedule, so a tick
always came. The AppKit shell ticks only on a `Notify` wake, and a stopped
machine sends none, so the pause never ended.

**Rule.** After every refresh, ask `AppController::autosave_due_in()` and,
when it is `Some`, schedule a tick that far ahead (AppKit:
`performSelector:withObject:afterDelay:` on the shared target's `tickNow:`).
A tick that comes early just schedules again. Any new shell (GTK4, Win32)
needs the same.
