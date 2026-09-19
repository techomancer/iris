# Batched HOSTRW in the REX3 JIT — and how the equivalence test lied twice

Batched HOSTRW transfers used to bypass the JIT entirely: `execute_go` forced
`entry = None` whenever `host_len > 1`. That sidestepped the JIT/interpreter
equivalence rule rather than satisfying it, and left the batched path — the one
IRIX actually uses for pixmaps and tiled fills — running on the slower engine.

## What the shader needed

Three changes, all mirroring interpreter accessors exactly:

1. **Load from the cursor's slot**, not element 0. `emit_hostrw_slot_ptr`
   reimplements `Rex3Context::hostrw_index` including both clamps:
   `host_len.saturating_sub(1)` and `HOSTRW_BUF_QWORDS - 1`. The clamps are not
   decoration — in generated code an out-of-range index is an unchecked write
   past a 1 MiB array.
2. **Advance after the load**, mirroring `fetch_host_pixel`'s
   `hostrw_get()` + `hostrw_advance()`. A shader consumes exactly one word per
   call, so the step belongs at the load.
3. **Store to the cursor's slot and advance**, mirroring `send_host_word`.

## The part that is easy to miss

A shader stops at a word boundary (`host_xstop`) and **returns**. `execute_go`
returned immediately after calling it, so a batch of N words painted one word
and dropped the rest. The interpreter path had a resume loop; the shader path
needed the same one. Without it, lifting the bypass is silent data loss.

## The equivalence test passed while proving nothing — twice

Both failures are worth knowing, because both look like success.

**First: the shape never compiled under the key the test waited on.**
`wait_compiled` was passed the *raw* `dm1`, but `execute_go` and
`compile_shader` both key on `rex3_shape::normalize_dm1(dm1, opcode)`. The
lookup waited on a key nothing was ever filed under, timed out, and the test
happily compared the interpreter against itself.

**Second: compilation is asynchronous.** The first run only *requests* the
compile; it still executes on the interpreter. A test that runs once and
compares is comparing two interpreter runs even when the key is right.

Both were caught the same way — a `rexdiag` probe printing `jit_go_count`
deltas, which read `hostw jit_gos=0 hostr jit_gos=0` while the test was green.
The fix is to assert dispatch actually happened:

```rust
assert!(after >= before + n as u64,
    "only {} GOs reached compiled code for {n} words", after - before);
```

Once that assertion was in place the test immediately found a real divergence:
row 1 of every image repeated row 0's pixels, because the load-side advance was
missing. **A JIT equivalence test without a dispatch-count assertion is not an
equivalence test.** See also
`memory/feedback_verify_instrumentation_fires.md` — a zero counter is usually
broken wiring, and here a *green test* was the broken wiring.

## Sabotage results

Each mutation fails exactly the direction it breaks, which is what makes the
pair trustworthy:

| mutation | HOSTW | HOSTR |
|---|---|---|
| remove load-side `emit_hostrw_advance` | **FAIL** | ok |
| store to fixed `ctx_off!(hostrw)` | ok | **FAIL** |

## Related

- `rules/rex3/hostrw-batching-design.md` — the batch token/payload protocol.
- `rules/rex3/hostrw-tests-must-use-partial-word-rows.md` — why test widths must
  not be multiples of 8.
- `rules/irix/vdma-gio-side-is-always-whole-qwords.md` — the MC side that
  produces these batches.
