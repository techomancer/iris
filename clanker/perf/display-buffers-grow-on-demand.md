# Display buffers grow on demand

`Rex3Screen` used to reserve four 2048-row planes per head: RGB, auxiliary,
DID and composed readback (about 52 MiB). Newport's physical planes are only
1024 rows, and most other boards never use Newport's auxiliary or DID planes.
The software compositor and debug overlay also reserved 2048-row buffers.

The shared RGB/readback buffers now start at 1024 rows. Newport allocates its
auxiliary and DID planes on its first display-state refresh. IMPACT, VPro,
InfiniteReality and O2 grow the composed frame buffer before writing taller
modes. Readback and presentation paths grow `rgba` before writing it. The
software compositor, iris-gui GL capture buffer and debug overlay allocate to
the active height.

Buffers only grow, so a mode change does not repeatedly reallocate them. Any
new direct writer must call `ensure_fb_rows` or `ensure_rgba_rows` first; do
not reintroduce a fixed 2048-row allocation to avoid that call.
