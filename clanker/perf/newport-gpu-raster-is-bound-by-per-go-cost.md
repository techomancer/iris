# The Newport GPU raster is bound by per-GO CPU cost, not state upload

Status: measured 2026-11-23 (after 758468e43, with the per-GO changes that
followed it).

## Symptom

On GL demos (Ideas in Motion, powerflip, solidview) the resident raster's
processor thread was the bottleneck: the GFIFO stayed full. The guest's GL
library walks every triangle into spans, so the board sees millions of GOs
of 3-30 pixels (powerflip: 8.4M in 25 s, median 3 pixels).

## What it was not

Building and uploading the 256-byte state slot: `RexStateSlot::with_source_color`
was about 1% of the thread, and the upload did not show in the profile. On
average 5.6-9.6 of the slot's 64 words change between primitives (coordinates
50-54, start colours 23/25/26, source colour 63, z-pattern 9; slopes per
triangle), so a dirty-word delta would cut the upload eightfold, but the
upload was not what cost.

## What it was (raster thread, powerflip replay)

- clock reads: four a GO (15%);
- the shared status mutex four or five times a GO, the raster mutex three
  times, `Gpu::check` locking and cloning its error (13%);
- 1024 slots a batch ended almost every batch at a quarter of `MAX_PRIMS`,
  so the per-batch submission (about 130 us, much of it the amdgpu CS ioctl)
  came four times as often;
- a `Vec` allocated per shaded span for its shade checkpoints.

Fixed: one clock read per 32 GOs while a batch is open, the fault check as an
atomic, position and accept under one status lock, `Gpu::check` with an atomic
fast path, `MAX_SLOTS = MAX_PRIMS`, inline shade payloads, and generation
timing and slot-change counting only with `IRIS_REX3_GPU_STATS`. Ideas replay
15.0 s -> 10.2 s; the powerflip replay went from raster-bound (6.9 s) to
bound by the replay's own producer (4.5 s, GFIFO median 8191 of 65535).

What remains, by share of the thread: span setup
(`prepare_resident_flat_fill` with the fill walk, 17%), the register decode
in `register_processor` (16%), `execute_go` (10%), GFIFO peek and consume
(10%).

## How to measure

- Record on the CPU raster (`rex rec FILE` needs `gputrace`; recording is
  refused under the resident raster), replay with
  `REX3_REPLAY_ENGINE=resident` in a `--features wgpu,gputrace` test binary.
  Checkpoints read the GPU planes back and must match the CPU recording.
- Time with `REX3_REPLAY_NOCHECK=1` (checkpoint readbacks dominate
  otherwise). Build both sides with `--profile developer` (thin LTO, a few
  minutes instead of 17) from a worktree for the baseline.
- Wall time is only the raster's cost while the GFIFO stays full: check
  `gfifo_p50` in the telemetry line. The thread's CPU time is no measure
  either, since it spins while the GFIFO is empty.
- `perf record -g --call-graph dwarf -t <tid>` on the `REX3-Resident-P`
  thread alone; a whole-process dwarf profile is swamped by the main thread
  loading the recording and loses the raster thread's samples.

## Shade DDA (2026-11-23)

`prepare_resident_flat_fill` then spent a third of the thread stepping the
shade DDA once per pixel to make 32-pixel checkpoints for the GPU. The clamp
is the identity while a component's bit 31 is clear and its low 20 bits stay
in [0, 0x7FFFF], and with a slope under 0x40000 the first step out lands where
the clamp pins it for good, so the value after n steps is a clamped line.
IRIX leaves junk in bits 20-30 of COLORRED (`0x903d33`): testing the whole
value against 0x7FFFF matched no GO at all, and the stepping fallback then
made everything slower. Testing the low 20 bits, 99.5-99.96% of shaded GOs
take the closed form (iconbook 92%). Ideas replay 10.2 -> 7.6 s; the
producer-bound replays then needed the GFIFO linger
(`gfifo-consumer-that-catches-up-slows-the-producer.md`). Live, powerflip
went from about 193k to 360k GOs a second; ideas is bound by the guest CPU
(about 290k either way, GFIFO median 63 entries).
