# A mid-session IMPACT recording needs the live DRBsize to replay

Status: rule, 2026-11-08 (`mgrasq1.rec`, sgiquake on the Octane's Maximum
IMPACT at 1024x768).

**Symptom.** A recording started with `mgras rec` during a session
replayed on the GPU raster unit to a different final state on every run,
and never to the CPU engine's. The checkpoint hashes agreed up to
checkpoint 181 and then differed now and then (`MGRAS_REPLAY_HASHES_OUT`
/ `_IN`). Pixel memory there showed 12:12:12 colour in the depth
buffer's pages 3-4, 10-11, 17-18, and so on, on the GPU only.

**Cause.** The X server writes `DRBsize` (0x16e) once, at start-up. A
recording that starts later never contains it, so the replay board keeps
the PROM's layout (DRBsize 0: 7 tiles a row, one engine), while the
stream's `DRBpointers` (A 0x2E0, B 0x370) were chosen for the live layout
(0x9000025A: 6 tiles a row, two engines, 144 pages a buffer). In the
PROM's layout buffer A needs 448 pages from 0x2E0, runs past page 1023
and wraps onto the depth buffer at page 0. The CPU engine settles the
aliased stores in scanline order, the GPU unit's tiles race
(`clanker/wgpu/aliased-pixels-race-across-tiles.md`), so the GPU result
varies. The replay is also not the session that was recorded.

**Rule.** Replay a mid-session recording with the registers the session
had: `MGRAS_REPLAY_INIT=16e=9000025a` (Maximum IMPACT, 1024x768, IRIX
6.5.22) writes them before the stream. Octane recordings also want
`MGRAS_REPLAY_OCTANE=1` and `MGRAS_REPLAY_BOARD=max`. The replay's report
prints `DRBsize` and `DRBpointers` at the end: DRBsize 0 means the
recording never set it. With the live layout the sgiquake recording's
518 checkpoints are equal on both engines and the GPU unit is
deterministic.

**The recorded hashes never match.** A mid-session recording's own
checkpoints hashed a board whose earlier state (pixels, TRAM) is not in
the recording, so a plain checked replay fails all of them, from the
first. Replay with `MGRAS_REPLAY_NOCHECK=1` and compare the engines with
each other: the CPU engine with `MGRAS_REPLAY_HASHES_OUT=f`, then the GPU
unit (`MGRAS_REPLAY_GPU=1`) with `MGRAS_REPLAY_HASHES_IN=f`, which
reports `first differing None` when every checkpoint agrees (about 4 and
2.5 minutes for sgiquake).
