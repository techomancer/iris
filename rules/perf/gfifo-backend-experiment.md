# GFIFO backend: swappable ring for benchmarking — 2026-09-14

> **Not in this tree.** The swappable backend below (`src/gfifo.rs` and the
> `gfifo-*` features) was an experiment that did not land; the GFIFO is the
> hand-written `GFifo` in `src/rex3.rs`. The measurements are kept because they
> are why the custom ring stayed.

The CPU→REX3-painter command queue (`src/gfifo.rs`) can be built on any of four
ring implementations, picked at compile time:

```
cargo build --release                             # custom (hand-written, default)
cargo build --release --features gfifo-rtrb       # rtrb::RingBuffer
cargo build --release --features gfifo-heapless   # heapless::spsc::Queue
cargo build --release --features gfifo-ringbuf    # ringbuf::HeapRb (agerasev)
```

`rex status` prints the live backend, so a number can never be misattributed to
the wrong build. Enabling two at once is a `compile_error!`, not a silent pick.

## Why compile-time and not an env var

A runtime enum would put a dispatch branch on `push`/`peek`/`consume` — the
exact three functions being measured. Feature flags cost four builds but leave
each variant as direct as a hand-written ring. If you want to A/B without
rebuilding, that convenience is paid for out of the measurement itself.

## Benchmark with `--tribench`, not `--bench`

`gltest --bench N` measures **fill rate**: a handful of GFIFO commands per
full-screen quad, then ~480k rasterized pixels. That is rasterizer-bound, so it
is nearly blind to the queue — expect it to move very little between backends,
and do not read a flat result there as "the backends are equivalent".

`gltest --tribench N [--trisize PX]` (added for this) draws many *small*
triangles instead. Per-primitive GL work — and so the REX3 register writes
queued through the GFIFO — dominates, while pixels per primitive stay small.
That is the workload whose cost actually lands on the queue. Default leg length
is 8px; shrink it to push the ratio further toward queue-bound.

## Things that are easy to get wrong here

- **`len()` is called off-thread.** The perf monitor, `rex status`, and the
  refresh thread all ask for occupancy, and they are neither the producer nor
  the consumer. All three third-party backends split into `!Sync` halves that
  cannot answer from another thread, so `GFifo` keeps a shared counter for them.
  The custom ring still derives `len()` from its own indices (`TRACKS_LEN`), so
  the default path pays nothing and keeps its original arithmetic exactly.

- **The API must stay peek-then-consume, never pop.** The consumer has to
  process an entry *completely* — including `execute_go()`, which can rasterize
  a whole primitive — before it stops counting as pending, because REX3 register
  reads are gated on "is the queue drained". Popping up front would report the
  queue empty mid-draw. All four backends support peek-without-advance; that was
  the gating requirement when choosing candidates.

- **Producers are not strictly single.** Usually only the CPU thread pushes, but
  MC's VDMA worker does too. They do not overlap in practice (the CPU does not
  write REX3 registers mid-DMA) — but three of the four backends are strictly
  SPSC, so every push takes `push_lock`. It is uncontended in the common case,
  which is the case that matters for throughput.

- **heapless 0.9 erases the capacity const-generic** from the split halves:
  `Producer<'a, T>`, not `Producer<'a, T, N>` — only `Queue` carries `N`. Its
  `split()` also borrows the queue, so the queue is boxed and leaked once per
  process to give the halves `'static`. That matches the real lifetime (`Rex3`
  lives until exit, and `reset()` reuses the queue rather than rebuilding it).

- **`rtrb` is a hard dependency regardless** — hal2, seeq8003 and net use it —
  so `gfifo-rtrb` enables no `dep:`. Making it optional breaks those three.

## Test coverage

`cargo test --release [--features gfifo-*] gfifo_tests` runs the same suite
against whichever backend compiled in: SPSC and MPSC ordering/integrity, plus
`len()`-tracks-occupancy, peek-does-not-advance, reset-discards-and-stays-usable,
wraparound (3× depth in lockstep) and fill-to-capacity. The last five were added
with this work — the original two covered ordering only, and `len()`/`reset()`/
`flush_head()` were untested, which is precisely where the backends differ.
