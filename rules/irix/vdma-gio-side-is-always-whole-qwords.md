# The GIO side of a VDMA transfer is always whole qwords — only memory is ragged

A VDMA transfer's byte count says nothing about how many 64-bit transactions
reach the device. The generic engine packs a short tail with
`byte_count.min(8)`, zero-fills the rest and still issues a **full**
`dma_write64`; the read direction consumes a **full** qword and scatters only
the valid bytes, discarding the rest.

So a 164-byte line is **21 wire transactions** (20 full plus one carrying 4
valid bytes), never 20.5. The partial access is a *memory-side* operation and
never changes the transaction count.

## The bug this hid

`qword_flat()` gated batching on `flat_len() & 7 == 0`. The Start-menu blit is
239 lines x 164 bytes = 39196 bytes, and `39196 & 7 == 4`, so the whole transfer
fell to the byte engine over a **4-byte remainder** — 4899 GFIFO round trips to
save nothing. Measured cost in `vdma.log`: **916 ms** for one gio->mem read.

The gate was testing the wrong property. Alignment and length are memory-side
concerns; neither disqualifies batching the wire traffic.

## The worse bug: `flat_len & 7` can pass by coincidence

Checking the *aggregate* byte count is not merely too strict — it is **unsound**,
and it fails silently.

The IRIX screensaver save/restore over SoftWindows sends 642 lines x 964 bytes.
`964 & 7 == 4`, so every line straddles a qword. But the 642 four-byte
remainders sum to 2568 bytes, which *is* a whole number of qwords, so
`flat_len = 618888` divides by 8 and the old gate let it through.

The linear engine then flattened it to `618888 / 8 = 77361` qword transactions.
The wire needs `ceil(964 / 8) * 642 = 121 * 642 = 77682`. That is **321 qwords —
2568 bytes — never sent**, with every line after the first progressively skewed
by its predecessor's missing tail. The chunk sums in `vdma.log` confirm it:
`32768 + 32768 + 11825 = 77361`.

The fix is a per-line test, not an aggregate one:

```rust
&& (self.line_count <= 1 || (self.line_width & 7) == 0)
```

A **single** ragged line is still linear-safe: there is no following line to
skew, so its one padded tail is simply the end of the transfer. That is why the
check is gated on `line_count <= 1`.

`ragged_line_regression_tests` pins all three cases, and states the 321-qword
shortfall numerically so neither engine can quietly reintroduce it.

## The per-line bulk engine

`dma_lines_bulk` handles everything the flat path declines except descending
transfers and fills: unaligned start and end, non-zero stride, and zoom repeats.

* **Chunks are whole numbers of lines**, never split mid-line. A line carries
  the zoom rewind and the stride step, so splitting one would mean
  reconstructing that state mid-chunk for nothing.
* `lines_per_chunk = (VDMA_CHUNK_QWORDS / qwords_per_line).max(1)`, so a chunk
  can never exceed the staging buffer — which is itself asserted at compile time
  to fit REX3's `HOSTRW_BUF_QWORDS` and the GFIFO depth.
* A line longer than the whole staging buffer is refused by `line_bulk_ok()` and
  falls to the generic engine.

### The repeat counter must cross the chunk boundary

A chunk flush can land in the middle of a line's zoom repeats. The scatter side
has to resume the nest exactly where the gather left it, so `chunk_reps` is
carried across the flush rather than re-derived from the line index.

Resetting it to `line_zoom` at each flush instead — which looks harmless —
corrupts **174778 bytes** in the `two lines per chunk, zoomed 3x` shape.
`line_bulk_chunk_tests::many_chunks_match_the_reference` catches it.

## Zoom on reads is incoherent

`line_zoom` means "send this source line N times". On a **write** that is
well-defined and IRIX uses it for tiled fills. On a **read** the source is
REX3's destructive-advancing pipeline: a line cannot be re-read, so "repeat line
N times" would mean filling N memory destinations from N *different* pipeline
words — not a zoom at all.

IRIX's `yzoom` (`sys/vdma.h`, packed into `DMA_STRIDE[25:16]`) is a
plumbed-through field only ever written from a caller-supplied value; nothing in
the tree sets it for a read. The engine handles it symmetrically anyway, because
matching the reference engine is cheaper than special-casing a shape that may
still arrive.

## How to test a change here

Differentially, against `dma_generic_bytes`. It is the definition of correct
behaviour, so the only property worth asserting is that the new engine is
indistinguishable from it — wire stream, guest memory, fault flag and final
`mem_vaddr`.

`line_bulk_tests_support::{compare_write, compare_read}` do exactly that.
Include ragged widths (164), non-zero stride, and zoom repeats; a shape whose
width divides evenly into 8 proves nothing, the same way 8/16px CI8 rows hid the
REX3 row-flush bug (`rules/rex3/hostrw-tests-must-use-partial-word-rows.md`).

Both comparison helpers assert the reference engine actually moved data, so a
shape that transfers nothing fails instead of passing vacuously.

## A spec question left open

rex3.pdf §3.10 says for *linear* block DMA "the width per scanline must be an
integer number of bus words... the end pixel of a block row must never be packed
into the same GIO word as the start pixel of the next row: this is a REX3
restriction; to overcome this limitation, stride block DMA is used."

164 bytes violates that, yet IRIX sends it with `stride=0`. Either the driver is
in span mode (where "arbitrary byte count and start byte" is explicitly allowed)
or this is one flat run that happens to be 239x164. The packing we emit is
byte-identical to what the byte engine already produced, so this is a note rather
than a blocker — but it is worth resolving before trusting the shape.
