# HOSTRW tests must use rows that do not fill a whole 64-bit word

Every HOSTRW test in `rex3_tests.rs` used `DM0_HOSTW_BLOCK` /
`DM0_READ_BLOCK`, which carry `STOPONXY`, and CI8 widths of 8 or 16 px. Both
choices hide bugs, and between them they hid two real ones.

## Width: 8 and 16 px are the two widths that cannot fail

A CI8 row of 8 px is exactly one 64-bit word and 16 px is exactly two. The row
boundary therefore always lands on a word boundary, the shifter is never left
holding a partial word, and the row-boundary flush is never exercised.

Widths like **12, 20, 33 and 4** leave a partial trailing word on every row.
That is where a skew appears — and it is the shape a real taskbar or icon blit
uses.

Sabotaging the row flush in `draw_block_g`:

```rust
-if stop_on_word && ctx.hostcnt > 0 {
+if stop_on_word && ctx.hostcnt > 0 && host_batch_drained(ctx) {
```

left the whole suite green. With a 12 px width the readback shows the merge
directly — row 1's pixels packed into row 0's partial word:

```
got  030f091505140c1d
want 030f091500000000
```

## STOPONY: the `!stopony` branch returns before the flush

The flush lives *after* the `!stopony` early break in `draw_block_g`, so a
non-STOPONY shape never reaches it. Covering the flush needs **STOPONY set**;
covering the batch resume loop needs it **clear**. Neither shape covers both,
so both belong in the suite —
`test_host{w,r}_stopony_flushes_partial_word_at_row_boundary` and
`test_host{w,r}_bulk_matches_scalar_and_counter_pattern`.

## Verify against generated data, not against the other engine

Batch-vs-scalar comparison cannot catch a fault in the walker both engines
share: they agree, and both are wrong. The sweep tests pin every pixel to a
32-bit counter spread over 4 CI8 pixels, so a pixel's value says which pixel it
should have been. `counter_explain()` turns a mismatch into
`(value belongs to pixel 1014, skew +1006)` rather than two hex blobs.

The counter is offset so no legal pixel is ever `0x00` — otherwise a cleared
framebuffer reads as correctly-transferred data and the comparison is vacuous.

## Feeding the stream: rows are word-padded, not continuous

A 12 px CI8 row consumes **2 whole words**, with the 4-byte tail discarded —
not 1.5 words with the next row packed in behind it. A generator that packs
continuously across rows produces a stream the hardware never sends, and the
resulting "failure" is the test's own bug. Confirmed by feeding a padded stream
and checking every pixel lands where the counter says.

## Related

- `rules/rex3/hostrw-row-boundary-forces-a-word-flush.md` — why the flush is
  unconditional.
- `rules/rex3/hostrw-batching-design.md` — the batch token/payload protocol.
- `src/mc_vdma.rs` `rex3_e2e_tests` — the same pattern driven through the real
  VDMA engine into a live REX3, which is what catches an MC/REX3 disagreement
  that neither component's own tests can see.
