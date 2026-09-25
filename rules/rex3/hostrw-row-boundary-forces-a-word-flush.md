# A row boundary forces a HOSTRW word flush — unconditionally

In host mode (`COLORHOST`/`ALPHAHOST`, or a `READ` opcode), the end of a
scanline is always a word boundary, even when the row's pixel count does not
divide evenly into `host_count`.

A 17-pixel-wide CI8 row at 4 pixels per 32-bit word sends 4 full words plus a
5th word holding **one** pixel, zero-padded. The partial word is sent; it does
not carry over and accumulate pixels from the next row.

## Where this lives

`rex3_generic.rs`, in both the block and span walkers:

```rust
// Host mode: a row boundary is always a forced word boundary too.
if stop_on_word && ctx.hostcnt > 0 {
    break;
}
```

`ctx.hostcnt > 0` means "a word is open but not full". The `hostcnt == 0` check
further down never fires for such a word, which is why this separate check
exists.

## Do not add conditions to it

When HOSTRW batching landed, the "one word per GO" stop condition correctly
became "one *transfer* per GO" (a batch carries N words behind a single GO).
The same `host_batch_drained()` guard was applied to this row-boundary check as
well — which is wrong. The flush must happen at every row boundary, mid-transfer
or not. Gating it merges each row's partial word into the following row.

Symptom, from `jit_hostr_ci8_stress_odd_width_block` (17x11 CI8):

```
interp = [.. 191a1b1c, 1d1e1f20, ..]   // rows run together
jit    = [.. 11000000, 12131415, ..]   // correct: partial word, zero-padded
```

The JIT's `emit_shader` flushes at the row boundary unconditionally, so the
interpreter silently diverged from it. The JIT/interpreter equivalence stress
tests are what caught this; a batching test alone would not have, because the
merged output is self-consistent.

## Related

- `rules/rex3/hostrw-batching-design.md` — the batching work, and the
  one-word-per-GO → one-transfer-per-GO change that this rule constrains.
- `docs/rex3.pdf` §HOSTRW / DRAWMODE1 — `HOSTDEPTH`/`RWPACKED` packing rules.
  The markdown summary in `docs/rex3.md` does not state the row rule; the PDF
  and the JIT's own behaviour are the references.
