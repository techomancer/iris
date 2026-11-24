# A GFIFO consumer that catches up slows the producer

Status: measured 2026-11-23 (Newport GPU raster, 2990WX).

## Symptom

Making the Newport GPU raster's processor thread faster (closed-form shade,
see `newport-gpu-raster-is-bound-by-per-go-cost.md`) made the producer-bound
replays slower: powerflip 4.6 -> 5.2 s, solidview 3.3 -> 3.7-4.0 s, with
large run-to-run spread. The raster-bound one (ideas) got 25% faster.

## Cause

Once the consumer keeps pace, the GFIFO sits nearly empty (median 4095 of
65535 instead of 32767, backpressure 1.6k instead of 1.1M) and the consumer
reads `tail` and each slot right after the producer writes them. Every push
then takes a cache-line transfer back to the producer's core; across dies on
a Threadripper that is more than the ~60 ns the replay spends per entry.
`perf` on the producer: `gfifo_try_push` samples +33%, the whole difference.
In the live emulator the producer is the guest CPU thread.

## Fix / rule

Linger before going idle: on an empty ring the processor thread waits for
`GFIFO_LINGER_ENTRIES` (256) to collect, for at most `GFIFO_LINGER` (20 us),
reading `tail` (`GFifo::pending_at`, from the consumer's own head) only every
`GFIFO_LINGER_SPINS` (512) spin hints (`rex3.rs`, `register_processor`).
64 entries / 5 us / 64 spins recovered only part of it; 256 / 20 / 512 beat
the old build on every replay (powerflip 4.59 -> 4.45 s, solidview 3.33 ->
3.24 s, ideas 10.26 -> 7.85 s).

Any other GFIFO consumer that becomes faster than its producer has the same
problem: measure the producer, not just the consumer.
