# KONA tile scheduler: a tile's hand-over can strand a job

Status: **open, not fixed** (found 2026-11-20). The test that shows it,
`dev::kona::back::tiles::tests::a_tiled_run_is_bit_exact_with_the_inline_one`,
is `#[ignore]`d until the fix lands. A candidate fix is below; it was tried
locally and not committed.

Code: `src/dev/kona/back/tiles.rs`, added 2026-11-10 in `30fc7236` ("kona: add
the pixel store, raster stage, tile scheduler and display"). The lines involved
are unchanged since.

## Symptom

On an Apple M1 (macOS, debug build, `cargo test`), the test never finishes: one
thread at 100% CPU, no output. Run alone it hangs every time (0 of 6 runs
passed; it hung for 2.5 hours inside a full `cargo test --workspace`). CI is
green on every platform, macOS arm64 included, so it depends on timing the
runners do not produce.

It hangs in the first threaded configuration (one worker, ring of 4). Stacks:

- the test thread spins in `Tiles::submit`'s wait loop (`yield_now`, `retire`
  breaking on `pending != 0`);
- `kona-tile-0` sleeps in `Shared::worker` on the condition variable.

State dumped from the stuck producer:

```
head=271 tail=267 inflight=3385 budget=16384 ring=4 workers=1 sleepers=1 ready_len=0
slot 3 pending=1      (slots 0,1,2 pending=0, waiting behind it to retire in order)
tile 60 head=30 tail=31 scheduled=false
```

Tile 60 has a job in its queue, is not marked scheduled and is not on the
ready queue. Nothing will ever run it, so slot 3 never retires, the ring stays
full and the producer waits for ever. `Drop for Tiles` calls `drain`, so a
panic in that state hangs too.

## Cause

The hand-over of a tile between its owner and the producer is a
store-then-load on each side (Dekker's pattern):

| | writes | then reads |
|---|---|---|
| owner, `run_tile` | `scheduled.store(false, SeqCst)` | `is_empty()`: `tail.load(Acquire)` |
| producer, `submit` | `Tile::push`: `tail.store(.., Release)` | `scheduled.swap(true, SeqCst)` |

It is correct only if at least one side sees the other's write. That needs all
four operations to be `SeqCst`. With the tail store `Release` and the tail
load `Acquire`, both may miss: the owner reads the old tail, takes the queue
for empty and returns with `scheduled == false`; the producer's swap reads
`true` and leaves the job to the owner. The final state above is exactly that,
and no sequentially consistent interleaving reaches it.

Not established: why the generated code differs in practice on aarch64 (both
orderings were expected to use the same instructions), and whether a running
Onyx2 guest hits it. It is the live InfiniteReality raster path, so it would
look like a frozen display with `kona-hip` spinning.

## Candidate fix

Two orderings in `Tile`:

```rust
// push
self.tail.store(t.wrapping_add(1), SeqCst);      // was Release
// is_empty
self.head.load(Relaxed) == self.tail.load(SeqCst) // was Acquire
```

Measured on the same machine, same build:

| Variant | Runs passed |
|---|---|
| original | 0 of 6 |
| `fence(SeqCst)` after `scheduled.store(false)` in `run_tile` | 6 of 6 |
| the two `SeqCst` orderings above, no fence | 12 of 12 (about 0.05 s each) |

`pop`'s `tail.load(Acquire)` is not part of the hand-over and can stay.

## Before closing this

- Apply the fix, remove the `#[ignore]`, and run the test in a loop (a few
  hundred runs) on Apple silicon; run the rest of `dev::kona::back`.
- The sleep path has the same shape (`sleepers` against the ready queue) and
  is already `SeqCst` on both sides; `loom` would check both hand-overs
  properly (`TESTING.md`, "Loom models").
- Only this one test was run against the candidate fix.
