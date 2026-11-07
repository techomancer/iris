# 2D fills are one record per buffer, not a blit pixel each

Status: rule, 2026-11-07.

**Symptom.** Small X and PROM redraws were slow, slower on the GPU unit than
on the CPU unit. The PROM's power-on screen took 4.2 s of drawing on the GPU
unit (1.0 s on the CPU) in replay; a file manager's minimize and restore
took 0.78 s (0.18 s). The GPU unit had 734k records and 2.7M blit pixels for
the minimize and restore.

**Cause.** `fill_rect`, `clear_planes` and `pattern_fill` called
`paint_integer` for every pixel. Each pixel rebuilt the state slot
(`Slot::new`, `pack_view`, `DestMem::of`), ran `pixel_allowed` and went
through `run_now` and `blit_pixel`, which staged its word. On the CPU unit
each pixel went through `planes_in`, which means `dest_mem` and a raster
wait.

**Rule.**
- A fill is `Board::fill_box(box, FillOp)`.
- On the GPU unit it is `GpuRaster::blit_fill`: one REC_BLIT per buffer
  with BLIT_CONST, so B_OFF holds the word, plus BLIT_WINDOW, so the kernel's
  `owned()` runs the bound, clip and id tests. Nothing is staged.
- The front end still mirrors stores into byte fields (the window-id
  bytes), as `window-id-bytes-are-written-through-to-the-mirror.md`
  requires.
- The per-pixel path remains for one case: an id test with both buffers
  selected and one of them a byte field. There each pixel must see the test
  before both stores. It is also used when the destination has no id plane,
  because the kernel's test does not fail a pixel without an id byte as the
  CPU does.
- `gpu_dest_slot` caches the slot in `Board::slot_cache`, keyed on the six
  views, the clip rectangles, `clip_kept` and `id_compare`. Any per-pixel
  path that needs a slot gets it from there.

**Check.**
- `odyssey::tests::fills` compares fills with per-pixel paints under random
  clips, id tests, operations, plane masks, buffers and id-plane
  destinations. Run it with and without `VPRO_GPU=1`. A deliberate
  plane-mask bug fails both of its tests.
- Recordings in `ignore/recordings/fills/` (`fm-minrestore.rec`,
  `prom-poweron.rec`): the replay prints "(x s drawing)", which is time
  without the checkpoint hashing. Use it to compare builds.

**Results.** Drawing time with the same memory hash before and after:

| Recording | CPU unit | GPU unit |
|---|---|---|
| prom-poweron | 0.98 s to 0.16 s | 4.2 s to 0.15 s |
| fm-minrestore | 0.175 s to 0.047 s | 0.78 s to 0.12 s |

About 0.07 s of the GPU unit's fm-minrestore time is the harness's
`gpu_sync` at its 3 marks: a GPU wait plus a readback of every dirty block,
which a live machine never does. On the CPU unit, a plain fill with no id
test stores raw words a run at a time (`Rows::fill` encodes the word once).

**What stays slow in the PROM is the guest.** It draws the logo as 1x1
quads, about 20 FIFO words each, and the board finishes each word in about
1 µs. The gaps of about 300 µs fall between vertex words, so the board is
waiting on the CPU emulation.
