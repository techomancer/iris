# SolidView profile (2026-11-07)

Status: measure / open.

Recording: `ignore/recordings/solid.rec`: startup, small window, then
maximised, about 76 frames, 7.69M records. It starts mid-session, and
the replay starts from `test_screen`. A frame is about 9,300 aliased
depth-tested lines (half blended), each its own GL_BEGIN/END, plus about
2,100 lit GL_POLYGON triangles with per-vertex normals. 465k lines in the
whole session, 8.7 pixels each, only 1,050 smoothed. The guest polls the
FIFO level and finds it full (`0x106c` level field saturated), so the
board, not the guest, sets the pace.

Drawing time (replay "(x s drawing)"):

| | before | after |
|---|---|---|
| CPU unit, inline | 2.55 s | 2.49 s |
| CPU unit, threads (`VPRO_REPLAY_THREADS`) | 3.90 s | 3.22 s |
| GPU unit | 6.09 s | 2.51 s |
| GPU records / batches | 4.18M / 1218 | 345k / 285 |

Done: host-memory buffers and the static ring
(`clanker/wgpu/cpu-written-buffers-live-in-host-memory.md`), packed
fragment records (REC_FRAGS, 19 pixels a record), and the aliased line's
single minor candidate (`centre_minor`), which halved `line_projected`.

Open:
- **GPU line kernel.** Lines are still set up and shaded on the CPU. A
  REC_LINE kernel would move them to the GPU, but `line_projected`
  interpolates in f64 and WGSL has none. It needs an integer or exact-f32
  reference formulation first, which changes the CPU reference and its
  goldens. A user decision.
- **Off-by-one on the GPU unit.** One pixel of one lit triangle differs
  by 1 in one channel (0xff30386c CPU, 0xff30386d GPU). It is first
  visible after record 7,611,589, the GL_END of a GL_POLYGON triangle;
  word offset 0xb363ec of board memory. `ignore/perf/bisect.sh` found it
  (it bisects VPRO_REPLAY_LIMIT on the final memory hash), and
  `VPRO_REPLAY_DUMP` diffs the boards. Everything before agrees. Likely
  smooth colour interpolation or rounding in the triangle kernel.
- **The threaded CPU unit is slower than inline** (3.2 s against 2.5 s).
  Single-pixel line quads through `QuadRing` cost more in hand-off than
  they save; packing them, as the GPU path does, is the obvious fix.
- On the CPU unit the per-frame clears (`ClearDraw::span`, `Rows::fill`,
  about 13%) are now the top cost.
