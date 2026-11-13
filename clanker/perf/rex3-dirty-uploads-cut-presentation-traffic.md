# REX3 dirty uploads cut source-plane presentation traffic

Measured 2026-11-05 on Windows 11 with a Ryzen 7 9800X3D, GeForce RTX
5070 (NVIDIA 617.42), Core OpenGL 3.2, and a release `perfetto,gpujit`
build.

The legacy Newport GL path submitted both physical 2048x1024 `R32UI`
planes on every composition: 16,777,216 bytes per frame. Tracking
independent RGB and auxiliary damage in conservative 64x32 tiles lets the
refresh thread copy only dirty rectangles into its cached snapshot and lets
the GL compositor submit the same rectangles with strided
`glTexSubImage2D` calls.

`REX3_FORCE_FULL_UPLOAD=1` restores the legacy path for comparison and
fallback. Both traces ran the same guest workload, contained 5,705 raster
slices, and reported zero dropped Perfetto events. Frame counts were
host-paced rather than equal, so all transfer results are normalized per
composed frame.

| Metric | Partial default | Forced full | Change |
|---|---:|---:|---:|
| Composed frames | 1,036 | 1,095 | informational |
| RGB + aux upload bytes/frame | 30,886 | 16,777,216 | 99.816% less (543.2x) |
| Upload bytes p50 / p95 | 0 / 0 | 16,777,216 / 16,777,216 | zero at p95 |
| RGB + aux submission time/frame | 3.36 us | 1.213 ms | 99.723% less (360.8x) |
| Submission p50 / p95 | 0.16 / 0.28 us | 1.123 / 1.527 ms | |

In the partial trace, 1,032 of 1,036 frames uploaded neither plane. Two
frames uploaded RGB only and two uploaded both planes. A separate settled
67-frame trace reported zero dirty tiles, snapshot-copy bytes, and source
upload bytes throughout.

The upload slices measure CPU/driver submission time, not asynchronous GPU
completion; the implementation deliberately does not add `glFinish`.
Texture creation still forces one complete upload of both planes.

Fifteen stable captures from each A/B run had the same SHA-256:
`777841377fa028632c7f4a652759a3e3c9cc38dc26f789e4a825dc01e63cdade`.
The matched 1282x1024 image is the non-black IRIX startup screen and contains
165 distinct RGB colors. The guest later reaches the existing Xsgi failure
around undefined REX3 offset `0x007c`; that prevents a later desktop/cursor
workload but does not affect the exact startup-screen comparison.
