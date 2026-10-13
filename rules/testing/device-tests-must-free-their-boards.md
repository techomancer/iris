# Device tests must free their boards (`TestBoard`), not `Box::leak` them

**Symptom.** The GitHub `Rust` workflow dies 1–2.5 min into `Run tests` with
`Process completed with exit code 143` (SIGTERM) or `The operation was
canceled` that nobody requested. No test failure appears in the log; the last
lines are whatever tests happened to be running (often `dev::ng1::rex3`).

**Cause.** The runner (ubuntu-24.04, 16 GB) runs out of memory. Device tests
used `Box::leak` so their threads could hold a `'static` reference. A REX3 is
~75 MB and a GR2 ~25 MB; a few hundred leaked boards took the lib tests to a
12 GB peak (REX3 alone 7.7 GB, GR2 2.9 GB) — even with `--test-threads=1`,
which is how you tell a leak from concurrency.

**Rule.** Build a started board as `crate::test_board::TestBoard<T>` (holds an
`Arc<T>`; drop calls `Device::stop()`, which joins the threads, then frees).
Bind it as `let rex = &make_rex3();` so helpers taking `&Rex3` coerce. Wrap
before `start()` — the threads keep that address. After the change the lib
tests peak at 1.3 GB.

**Stop must not hang mid-command.** Freeing boards runs `stop()` at the end of
every test, including ones that leave work in flight. That exposed a GR2 race:
RE3 exits on its sentinel, HQ2 then queues more RE3 work and spun forever in
`finish()` (`gl_context_save_read_waits_for_hq`). Any wait one device thread
does on another must also give up once `running` is cleared.

**Measuring.** `/usr/bin/time -f %M target/debug/deps/iris-<hash> <filter>
--test-threads=4` per module; per-test with `--exact`.
