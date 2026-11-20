# Sync compile gates a page with code on the pool's hot count

With `jitv2.compile = "sync"` the dispatch miss compiles inline only when the page has no code
or `note_waiting_arrival()` says this arrival makes it hot; any other new entry is interpreted.
Compiling at every entry re-emitted the page until the arena flushed, and IRIX 6.5.22 never
reached login.

The gate must not call `try_schedule_page()`: nothing is queued, so `handle_request`'s
`clear_scheduled` never runs for an interpreted arrival and the flag stays set. `j2 pcp` then
shows a false in-flight page, and `j2 inline off` finds the threaded path blocked. A compile
resets `hot_wait` and `promoted` through `clear_scheduled`, so the count restarts after each one.
