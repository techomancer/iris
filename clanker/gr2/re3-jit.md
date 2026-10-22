# The RE3 span JIT (`gr2-jit`)

`src/dev/gr2/re3_jit` compiles RE3 SHADED / FLAT / FLAT4 spans. Its
pattern follows `gr4-jit` (clanker/mgras/gr4-jit.md: key, store, compiler
thread, sweep test). These are the GR2-specific gotchas.

## The ABI is the `Re3` layout

A shader's only argument is `*mut Re3`. It reads registers, iterators and
private-op state at `offset_of!(Re3, ctx...)`, and VRAM and the Z buffer at
`offset_of!(Re3, vram/zbuf)`. Reordering `Re3` or `Re3Context` is safe,
because offsets are computed at compile time. Changing what a field means
is not safe unless the emitter changes too.

## Invalidation

A memoised shader is only dropped when key-shaping state changes:

- the registers in `re3_jit::shapes_key` (FUNC, ALIGNPAT, ENABPAT,
  ENABDITH, ENABWID, DEPTHFN, FBOPTION);
- the `Re3::set_pixfmt / set_zctl / set_stencil / set_blend` setters.

Rules for adding state:

- **Any new state the key depends on must go through one of these.** A
  direct `re3.ctx.pixfmt = ..` would keep running the old shader.
- **Data the shader reads live needs no invalidation:** masks, scissor,
  WID, colours, stencil reference/masks, the Z write mask. It is loaded
  per span.
- **Invalidate only on an actual change.** `gl_setup` rewrites FUNC, the
  pixel format and the Z/stencil ops on every primitive, so an
  unconditional invalidate would recompute the key per primitive.

Mode 0 is off, so a zeroed `Re3` (`bare_re3`, `Box::new_zeroed`) never runs
shaders. `Gr2::new` sets `IRIS_GR2_JIT`'s mode.

## Exactness

- The emitters mirror `draw_shaded_span` / `draw_flat_span` and their
  helpers. Change the interpreter and the emitter together. Then run
  `cargo test --release --features gr2-jit --lib re3_jit`.
- The sweep compares whole VRAM, the Z buffer and the iterators. Mutating
  the XOR op, the dither threshold or stencil DECR each made it fail at
  once (checked when it landed).
- Final iterators are written in closed form: start + steps × step
  (FLAT: steps = count / run). Stepping only adds, and both sides wrap.
  The loops step only the iterators a pixel reads.
- Row spans (DX = 0x4000, DY = 0) and every FLAT span are clipped to the
  scissor once. Their x is `X + n` exactly, because load_iterators puts the
  subpixel fraction in y when |DX| = 1.0. FLAT positions are always
  `x0 + i`, whatever DX says. Other SHADED spans clip per pixel.

## A faster RE3 can make the board slower

At first the JIT made long Z-buffered spans about 35% slower (gltest
--tribench --trisize 300 --depth: 1,210 tris/s against 1,870). The shaders
were not slow; the HQ2 thread was paying for RE3 being idle:

- **Every HQ2 push took RE3's thread-slot mutex and unparked it.**
- **A consumer that keeps catching up parks and re-reads the producer's
  FIFO lines far more often.**

The JIT made RE3 idle much more of the time. The fix is the mgras one
(6e76140b): a thread raises a cache-padded `*_asleep` flag before it parks,
and producers unpark only when that flag is up (`Gr2::wake` / `sleep`).
With it the JIT is 12% faster on the same test (about 2,050 against 1,830).

When a raster speed-up shows up as a slowdown, look at the wake path
first.

## HQ2 -> RE3: batched, and unchanged registers skipped

The span-building thread was slow, but not because of the span arithmetic.
Every RE3 FIFO entry (about 13 per span) took the FIFO's producer lock and
a SeqCst fence for the wake check. In the host benchmark
(`hq2_tests::bench_gl_smooth_triangles`) that came to about 75% of HQ2's
time; the float setup was about 6%.

`hq_loop`'s Sink now does two things:

- **Batching.** It stages entries and pushes them as one unit
  (`GFifo::try_push_slice`: one lock, one tail release, one wake check).
  It flushes at each IR write (span end), when the stage is full, before
  anything that waits on RE3 (finish, reads, GEDMA) and when HQ2 runs out
  of work.
- **Skipping unchanged writes.** It keeps a shadow of the registers it has
  sent and skips writes that would leave a register unchanged. The
  exceptions always go through:
  - IR and RWDATA (RWDATA is also written by RE3);
  - the RGB / YX fan-outs, which drop the shadow of R, G, B / X, Y;
  - anything after a CPU write to the RE3 window, which bumps
    `re3_cpu_writes` and makes the shadow forget everything.

This is exact: RE3 holds every register until its next write.

Ordering: everything HQ2 sends stays in order. A CPU write straight to RE3
registers can now land ahead of HQ2 entries still staged, which IRIX does
not do while the GE is busy.

Host benchmark, smooth Z-buffered triangles, JIT on:

| Triangle legs | Before | After |
|---|---|---|
| 8 px | 64k tris/s | 390k |
| 32 px | 36k | 138k |
| 128 px | 6.9k | 16.3k |

In IRIX (gltest):

| Benchmark | Before | After |
|---|---|---|
| 8-px triangles | 66k tris/s | 153k (the guest CPU feeding GL is now the limit) |
| 32-px triangles | 20k | 74k |
| 300-px Z-buffered triangles, JIT | 2,050 | 3,790 |

## Where the time goes

Before the batching (8-px triangles):

- **HQ2 thread: one core, saturated** in `raster_poly`, most of it the
  per-entry FIFO handshake.
- **RE3: about half a core.**

After the batching (host benchmark, 32 px):

- **RE3: one core.** About two thirds of it is the shader's pixel work.
- **HQ2: waits on a full FIFO.** Its own span setup is about 5%.

Fills gained 3× in the guest (78 to 245 Mpx/s, Z-buffered 58 to 115).
