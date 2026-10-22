# The REX3-Processor thread must park when the gfifo is idle

**Symptom:** in *graphical* mode an otherwise-idle IRIX desktop (e.g. sitting at
the xdm login, or any static screen) pins a host core at ~100%. It does **not**
happen headless (`headless = true`), because headless constructs no REX3 at all
(`machine.rs`), so there is no REX3 thread.

**Diagnosis trail (worth remembering):**

- The guest MIPS-CPU thread is *not* the spinner — the idle-park detector in
  `mips_exec.rs` works fine here (verified: at the xdm idle loop the MIPS-CPU
  thread parks, host ~0%). Don't chase the CPU/idle-park code for this.
- `ps -M <pid>` (per-thread CPU) is the key tool. It showed the **REX3-Processor**
  thread at ~98% while MIPS-CPU was ~0. `sample <pid>` maps the hot thread id to
  its name.
- Headless-vs-windowed is the discriminator: headless parks (no REX3), windowed
  spins (REX3 present). That alone localises it to a REX3 thread, not the CPU.

**Root cause:** `Rex3::register_processor` (the gfifo command consumer) backed
off an empty fifo with `crossbeam_utils::Backoff::snooze()`, which escalates to
`std::thread::yield_now()` but **never sleeps**. An idle desktop leaves the
gfifo empty indefinitely, so the thread yield-spins a whole core forever.

**Fix:** once emptiness is *sustained* (`backoff.is_completed()`), `park_timeout`
instead of yielding, and have the single producer choke point (`gfifo_push`)
`unpark` the consumer when it pushes. Correctness does not depend on the unpark —
the (short, 2 ms) `park_timeout` is a backstop, so a missed wakeup costs only
latency, never a hang. During active drawing the fifo is never sustained-empty,
so the loop stays hot (`backoff.reset()` on every consumed entry) — the park
path only triggers at true idle.

**Race note (the ordering that makes the unpark lossless):** the consumer sets
`processor_parked = true` *before* its final `gfifo.peek()` emptiness check, so a
concurrent `gfifo_push` either (a) lands before the peek and is seen, or (b) sees
`parked == true` and unparks — and an unpark token delivered before `park` makes
`park_timeout` return immediately. `wait_idle` on the producer side already gates
on `gfxbusy && gfifo.is_empty()`, so a queued-but-not-yet-processed command never
looks "done" even if the consumer is briefly parked.

**Related — REX3-Refresh idle skip:** the REX3-Refresh thread separately did a
full-framebuffer `copy_from_slice` (~16 MB) + RGBA conversion + GL upload *every*
frame at 60 Hz regardless of whether anything changed (~25-30% of a core at idle).
It now renders only when something visible changed: a new `fb_dirty` atomic (set
by the gfifo consumer on any processed entry) covers all REX3 drawing, and the
palette/cursor/mode mutexes' existing `dirty` flags are peeked (not cleared —
`refresh()` clears them when it actually runs). A ~10 Hz heartbeat keeps the live
status bar moving and bounds any missed-dirty staleness. Crucially the **VBLANK
tick (STATUS_VRINT + cursor-Y latch + vblank_cb) still runs every frame** even
when the render is skipped — the guest's vsync timing must not depend on host
rendering. Skipped frames don't swap, so the front buffer just stands.

Combined with the parking fix, idle graphical host CPU dropped ~130% → ~9%.

Note a *live* VINO camera source genuinely keeps the guest busy (continuous video
DMA → the guest blits each frame via REX3 → `fb_dirty` every frame), so "idle"
with `[vino] source = "camera"` is not truly idle and will render at full rate —
that is correct.
