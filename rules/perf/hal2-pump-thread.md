# HAL2 dedicated pump thread — reverted, do not reintroduce as-is

**Keywords:** hal2,audio,scratch,stutter,pump,hptimer,sleep,windows,chime,startup

Phase 3 moved Codec A output off `TimerManager`/`hptimer` onto a dedicated
`HAL2-Pump` thread, first with `thread::sleep(period - elapsed)` at
`period ≈ 1/44100 s` (~23 µs), later "fixed" to spin like `hptimer` does.
Both versions were reverted; `src/dev/hal2.rs` is back on `TimerManager` via
`arm_codeca`/`disarm_codeca` (see `HAL2_I_...` timer wiring in that file).

## Why the sleep-based version failed

`hptimer` (`src/hptimer.rs`) **spins** for waits under ~200 µs so 44.1 kHz
audio ticks stay on schedule. `thread::sleep` on Windows — even with
`timeBeginPeriod(1)` from `Hal2::start()` — cannot reliably sleep for 23 µs;
effective wakeups land near **1 ms**, so DMA was drained ~40× too slowly. The
cpal ring buffer underran constantly (scratchy/crackling audio).

## Why the whole pump-thread approach was reverted, not just re-fixed

A dedicated, bespoke thread bolted directly onto `Hal2` — even once made
spin-based and correctly paced — still broke the IRIX startup chime: a
short, one-shot Codec A playback that fires very early in boot, before the
rest of the audio path's steady-state assumptions hold. The ad-hoc thread
didn't get the same startup-ordering/prebuffering guarantees that
`TimerManager` already provides for every other periodic device callback in
the emulator, and re-deriving those guarantees one-off for HAL2 wasn't worth
it.

Slow/incorrect PDMA drain also stresses the IRIX audio path in ways that look
like unrelated kernel instability; treat audio regressions as P0 before
chasing unrelated kernel panics.

## Correct approach (current)

- Codec A runs on the shared `TimerManager`/`hptimer` via `Hal2::arm_codeca`,
  exactly like every other periodic device callback in the emulator. No
  dedicated thread.
- **If a dedicated audio thread is revisited:** a plug-in fix is fine — just
  give it its own `TimerManager`-style instance (i.e. another copy of
  `hptimer`'s spin/park policy: spin under ~200 µs, park above it), not a
  bespoke sleep/spin loop written from scratch for HAL2. The bug was the
  ad-hoc timing logic and missing startup guarantees, not the idea of a
  separate thread per se.

## Verify after HAL2 changes

- Telnet monitor `hal2 status` — `cpal underruns (samples)` should stay low
  during desktop idle and under `glxgears`-style load.
- **Boot a fresh IRIX image and confirm the startup chime plays correctly**
  (not silent, not garbled/truncated) — this was the concrete symptom that
  caught the pump-thread regression; a quiet desktop-idle underrun count
  alone is not sufficient to catch startup-path audio bugs.
- Premiere: `wsl\run-iris-premiere.bat`.

## Kernel panic after earlier crashes

If you see `Read TLB Miss` at `0xff800000` repeatedly, the **root disk may be
corrupted** from a prior panic — not a JIT bug. Enable `overlay = true` on
`[scsi.1]` and restore from a clean `scsi1.raw` copy. See
`rules/testing/disk-image-hygiene.md`.
