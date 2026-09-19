# Batched HOSTRW / VDMA — design sketch

**Status: built** (bulk trait methods + REX3 batch tokens + MC staging buffer).
Two things landed differently from this sketch — see "What actually got built"
at the bottom. The cost analysis below is the pre-batching baseline, kept
because it is what motivated the design; it has not been re-measured since.

## What it costs today

Per 8 bytes transferred, derived from the 58.3 ns push cost measured this
session (~9.52 ns per guest instruction at ~105 MHz emulated):

**GIO -> Mem (reading pixels out of REX3)** — `mc.rs` `dma_read64` path:

| step | ns |
|---|---|
| `gfifo_push(REX3_DMA_PURE_GO)` | 58.3 |
| **`wait_idle()` — full pipeline drain, per qword** | ~200 (unmeasured) |
| 8x `translate_addr` + `write8`, **per byte** | ~200 |
| **total** | **~458 ns/qword ≈ 17 MB/s** |

**Mem -> GIO (writing pixels in)**:

| step | ns |
|---|---|
| 8x `translate_addr` + read, per byte | ~200 |
| `dma_write64` -> gfifo push | 58.3 |
| **total** | **~258 ns/qword ≈ 31 MB/s** |

Two separate pathologies:

1. **`wait_idle()` per qword.** The read path pushes a GO, drains the *entire*
   pipeline, reads one latched word, and repeats. That is the double round-trip:
   every 8 bytes costs a full producer/consumer synchronisation.
2. **Per-byte TLB translation.** Both directions walk one byte at a time calling
   `translate_addr` for each, even when the whole line sits in one page.

## The design

Move the bulk transfer *into* the shader, and let VDMA hand it a buffer rather
than a word.

**HOSTRW becomes an array, and a single transfer is length 1.** No batched-vs-
single special case anywhere: the pixel bodies index `buf[cursor]` instead of
touching a scalar `ctx.hostrw`, and a PIO register access is simply `cursor = 0,
len = 1`. One code path, one shader, no const-selected arm — which is the whole
point of doing it this way rather than adding a parallel batched pipeline.

### Where the buffer lives — not in `Rex3Context`

`Rex3Context` is `#[derive(Copy)]` and is serialised field-by-field to TOML on
every snapshot (`save_rex3_context`). A multi-megabyte array inside it would make
each context copy a multi-megabyte memcpy and write ~131k qwords of TOML per
snapshot per megabyte.

So: **the buffer sits on `Rex3`, beside `fb_rgb`/`fb_aux`; only the cursor and
length live in the context.** The shader already receives the two framebuffers as
pointer arguments — the host buffer is the third, and reaching it costs the same
as reaching a framebuffer. `Rex3Context` grows by ~8 bytes and the snapshot
format is untouched.

That also keeps the JIT working unchanged: `offset_of!(hostrw)` stays valid for
the cursor field, and Cranelift's existing HOSTRW handling can keep using element
0 until it is taught about the buffer.

### Writes (Mem -> GIO)

- Extend HOSTRW backing store from one qword to a real buffer (a few MB).
- VDMA leads the transfer with a **batch-write command** pushed through the
  GFIFO: `(buffer, length)` rather than N separate qword pushes.
- The consumer streams the payload into the buffer and hands it to the draw
  engine **once**, so the shader consumes host pixels from memory instead of
  being fed one qword at a time through `fetch_host_pixel`.

### Reads (GIO -> Mem)

- VDMA pushes a **batch-read GO** with the destination buffer and length.
- The shader fills the buffer directly — it already computes every pixel; it just
  writes them contiguously instead of latching one at a time into `ctx.hostrw`.
- VDMA consumes the whole buffer in **one** `dma_read` call.

That removes `wait_idle()` from the per-qword path entirely: one synchronisation
per *batch* instead of per 8 bytes.

### While in there: fix the per-byte TLB walk

Translate once per page and copy the run, rather than per byte. A 4KB page holds
512 qwords; at ~25 ns per translation that is most of the remaining cost in both
directions.

## Constraints this must respect

- **The pixel bodies already take the host pixel from `ctx`** (`fetch_host_pixel`
  / `store_host_pixel`, now in `rex3_generic.rs` and shape-driven). They index
  `buf[cursor]` instead of a scalar — and because a single transfer is just
  `len = 1`, there is no batched-vs-single branch to select. Same code, same
  shader, whatever the length.
- **`host_count`/`host_shift` still govern packing** — 1 to 16 pixels per 64-bit
  word depending on HOSTDEPTH/RWPACKED/RWDOUBLE. Batching changes where the words
  come from, not how they unpack.
- **`dma_read64`'s ordering inversion must be preserved.** CPU-driven PIO gets
  read-then-advance for free; VDMA has no software discard loop, so the current
  code pushes a GO, waits, *then* reads. A batch read has the same requirement at
  batch granularity: the buffer must be filled before VDMA consumes it.
- **BUS_BUSY/retry discipline.** `dma_write64` spins on BUS_BUSY because the DMA
  worker has no EXEC_RETRY. A batch command must either be accepted whole or
  rejected whole — the same rule `try_push2` enforces for the 64-bit pair, and
  the same bug class if it is got wrong (partial commit + retry = duplicate).
- **Two producers.** CPU and the VDMA worker both push. A batch command occupies
  one queue entry, so this does not change the producer-lock story, but the
  buffer itself needs a clear owner while in flight.

## Expected win

If the batch amortises both `wait_idle()` and the TLB walk over, say, a 4KB page
(512 qwords):

- read: ~458 ns/qword -> roughly the memcpy cost plus one sync per batch
- write: ~258 ns/qword -> likewise

Both directions should land in the hundreds of MB/s rather than tens. Worth
measuring rather than promising: the numbers above have one unmeasured term
(`wait_idle`), and the real gain depends on typical transfer length, which the
corpus does not record.

## Why it also cleans up `mc.rs`

The current `dma_loop` interleaves address translation, byte packing, direction
handling, zoom/stride bookkeeping and BUS_BUSY spinning in one nest. Batching
splits it: translate a run, hand a slice to the device, advance. The zoom/stride
logic stays, but it stops being tangled with per-byte bus access.


## What actually got built

Implemented across four layers.

- `traits.rs`: `dma_write64_bulk` / `dma_read64_bulk` beside the scalar pair.
  Defaults are scalar loops, so other devices are unaffected. `BUS_ERR` means
  "not batchable here, use scalar" and the MC path falls back on it.
  `physical.rs` forwards both — otherwise the default loop runs against
  `Physical` and the device's own override never sees the slice.
- `rex3.rs`: **`Rex3Context::hostrw` is now an array**, `HOSTRW_BUF_QWORDS`
  (131072) u64s = 1 MiB, inline in the context. Not a pointer, not a separate
  buffer on `Rex3`: the port itself is the array, it always exists, and it can
  never be null. `host_cursor` picks the element and `host_len` is the count.
  `REX3_DMA_BATCH_W` / `REX3_DMA_BATCH_R` tokens, plus `GFIFO_PAYLOAD`.
- `rex3_generic.rs`: `fetch_host_pixel` / `send_host_word` index
  `hostrw[host_cursor]` and step.
- `mc_vdma.rs`: `VDMA_CHUNK_QWORDS` (32768) staging buffer; the flat 64-bit
  paths gather/scatter through it. Chunking is unavoidable at any buffer size,
  so the sizes trade round trips against footprint, nothing more.

### The count is the only thing that distinguishes a batch

`host_len` defaults to **1**, set by `power_on_default` and re-armed by every
PIO register write (`hostrw_arm_single`) and at the end of `execute_go`. A batch
token sets it to the transfer size. Nothing downstream branches on "is this a
batch" — there is only a count, which is the point of making the port an array.

Corollaries that bit during implementation:

- The walkers' "one word per GO" stop became "one *transfer* per GO"
  (`hostrw_drained()`). At `host_len == 1` that is true after one word, exactly
  the old rule.
- **The row-boundary flush stayed unconditional.** Applying the same drained
  guard there merges each row's partial word into the next row. See
  `rules/rex3/hostrw-row-boundary-forces-a-word-flush.md`.
- The JIT bypass is `host_len > 1`, not `> 0`. Gating on `> 0` disables the JIT
  for *every* draw, since 1 is the resting state — caught by
  `cid_write_masks_jit` asserting `jit_go_count == 768` and getting 0.

### Payload travels through the GFIFO

`dma_write64_bulk` pushes the token and then one GFIFO entry per payload word,
all under one producer lock with a single Release on tail (`GFifo::push_batch`).
The register processor's `REX3_DMA_BATCH_W` arm calls `drain_payload`, which
streams those entries straight into `ctx.hostrw[]`.

Keeping the payload *in the queue* rather than writing the array behind the
queue's back is what orders the transfer correctly against concurrent CPU
register writes: the words arrive in the stream at the point the producer put
them. `push_batch` checks capacity for token + payload before writing any slot,
so it is all-or-nothing like `try_push2`.

`drain_payload` must step over the token first — the consumer loop `peek`s and
only `consume()`s after `process_register` returns, so `local_head` still points
at the token. It then rewinds one so the caller's `consume()` retires the last
payload entry, making the total head advance exactly `1 + taken`.

### The array must be the last field in `Rex3Context`

`repr(C)` lays fields out in declaration order, and the JIT addresses every
scalar by `offset_of!` as a Cranelift `Offset32`. With a megabyte mid-struct,
every field after it lands at a ~1 MB offset. Keep `hostrw` last.

### Stacks had to grow

`Rex3Context` is `Copy` and construction moves it through temporaries, so a
1 MiB member overflows the old stacks. Raised to 64 MiB in `main.rs`,
`bench_runner.rs` and both `rex3_tests.rs` construction helpers (the GUI already
used 64 MiB for this reason). Virtual address space, lazily committed.

### Batched draws bypass the JIT

Compiled shaders address `hostrw[0]` by fixed offset and never step the cursor,
so a multi-word transfer would consume one word and repeat it. `execute_go`
forces the interpreter when `host_len > 1`. A correctness guard, not a design
choice, and it caps the win: the batched path is the slower engine until
`emit_store_hostrw` and the shader-entry load learn to walk the array.

### Still not done

- **Per-page TLB translation.** The staging loops translate once per qword; the
  generic byte engine still does it once per *byte*. See
  `rules/irix/vdma-transfers-are-always-translated.md`.
- **Measurement.** No before/after numbers taken. The table at the top is the
  pre-batching baseline and one of its terms (`wait_idle`) was never measured.

### Tests

`rex3_tests.rs`: `test_hostw_batch_matches_scalar_writes` (multi-word write
equivalence — this is the one that caught the one-word-per-GO stop),
`test_hostw_batch_of_one_equals_single_write` (len-1 identity),
`test_pio_write_after_batch_is_unaffected` (no leak into PIO),
`test_hostr_batch_matches_scalar_dma_reads` (read equivalence). The existing
JIT/interpreter HOSTR stress tests are what caught the row-flush regression.
