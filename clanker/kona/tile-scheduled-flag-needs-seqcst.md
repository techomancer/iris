# A tile's `scheduled` hand-off needs SeqCst on the queue's tail

Status: rule, fixed on 2026-11-20 (`fix/kona-tiles-lost-wakeup`).

## Symptom

`dev::kona::back::tiles::tests::a_tiled_run_is_bit_exact_with_the_inline_one`
sometimes never finishes. The test thread spins in `Tiles::submit` or `drain`
(waiting for a slot to retire), and every `kona-tile-N` worker sleeps on the
wake condvar. Four `cargo test --workspace` runs on Apple Silicon each hung
for more than an hour at 100% of one core.

## Cause

Producer and tile owner hand a tile off through two variables, the tile
queue's `tail` and the tile's `scheduled` flag:

- producer (`submit`): `tail.store` (push), then `scheduled.swap(true)`; it
  puts the tile on the ready queue only if the swap returned `false`.
- owner (`run_tile`): after the last pop, `scheduled.store(false)`, then
  `is_empty()` loads `tail`; it takes the tile back only if that load sees
  the new slot.

That is the store-buffering pattern. With `tail` stored Release and loaded
Acquire, the owner's load can complete before its own `scheduled` store
(AArch64 `LDAPR` may pass an earlier `STLR`). The owner then reads the old
`tail` and leaves, while the producer's swap reads the old `true` and leaves
the tile to the owner. Nobody runs the slot, its `pending` count never
reaches zero, and the producer waits forever.

## Rule

All four accesses in a store-buffering hand-off must be SeqCst: here the
`tail` store in `Tile::push` and the `tail` load in `Tile::is_empty`, as well
as the `scheduled` swap and store, which already were. Release/Acquire is
enough only for the one-way publication of the slot index (`pop`).
