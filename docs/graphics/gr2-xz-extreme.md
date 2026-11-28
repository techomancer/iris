# GR2 XZ and Extreme graphics

IRIS implements the Express-family graphics path in `src/dev/gr2/` for Indy
XZ, Indigo2 XZ, and Indigo2 Extreme.
There is no separate Elan setting in the configuration.

## Select a board

```toml
[machine]
profile = "indigo2_ip22"    # indy_ip24 also supports xz

[[devices]]
type  = "express"
model = "xz"               # elan and extreme are restricted to indigo2_ip22
```

Select **GR2 XZ** or **GR2 Extreme** in iris-gui's General tab. No graphics
Cargo feature is needed; `gr2-jit` adds an optional span JIT (see [Express (GR2)](gr2.md)). Non-Newport boards require one head and Guest
resolution; Newport resolution presets do not apply. Extreme presents eight
GE7 engines, XZ two. The model shares rendering code rather than reproducing
the physical engines' timing.

## Implementation

The board owns the 4 MB GIO graphics slot at `0x1F000000`:

| Board offset | Block |
|---|---|
| `0x00000–0x1FFFF` | Shared RAM |
| `0x40000–0x5FFFF` | Command FIFO |
| `0x60000–0x67FFF` | HQ2 microcode storage |
| `0x68000–0x69FFF` | GE7 diagnostic windows |
| `0x6A000` | HQ2 registers |
| `0x6C000` | Board version |
| `0x6C040` | VC1 display controller |
| `0x6C0A0` | Bt457 RAMDAC |
| `0x6C100` | XMAP5 |
| `0x6C200–0x6C2FF` | RE3 raster registers |
| `0x6C600–0x6C603` | RE3 read/write data port |

HQ2 interprets textport and irisGL command tokens. Geometry, lighting,
homogeneous clipping, polygons, glyphs, context state, and pixel DMA are
implemented in software; uploaded GE7 microcode is stored for diagnostics
rather than executed. The RE3 thread owns VRAM and draws from its FIFO.
VC1, XMAP5, and Bt457 feed a software compositor and a display thread.

Both frontends obtain frames through `Machine::get_display()` and `GfxDisplay`.
Graphics do not require HostGL; `hostgl` is a separate optional service for
programs using the replacement guest libGL.

## Snapshots and reset

Snapshot schema 10 stores XZ and Extreme as exact named `state`, `vram` and
`zbuf` content-addressed parts. State includes all board registers, SRAM and
microcode, display devices, complete HQ2 parser/GL continuation, RE3 context,
FIN2 waits and unread GEDMA output. Disk snapshots, CI rollback and live
debugger checkpoints use the same representation. Restore rejects malformed
parts and XZ/Extreme or board-presence mismatches before changing live state.

Stop drains accepted HQ2 and RE3 work before the workers exit. Power-on clears
the complete architectural board state; it retains only the configured
`gr2-jit` mode and rebuilds runtime threads, queues, traces, presentation
objects, telemetry and JIT memoization.

## Inspect and trace

Use `gr2 help` for the full command list:

```text
gr2 status
gr2 hq
gr2 metrics
re3 regs
gr2 trace gr2.log hq,re3,cpu    # needs a build with the gputrace feature
gr2 trace flush
gr2 trace off
gr2 fbdump gr2dump
```

`gr2 metrics` reports span-size buckets, consecutive spans with the same
pipeline shape, synchronization boundaries, candidate framebuffer/Z traffic,
host reads, presentation scans and, with `gr2-jit`, JIT hits and misses. The
counters are runtime diagnostics and reset on power-on or snapshot restore.
Ordinary presentation currently composes from host VRAM; there is no resident
RE3 GPU raster or scanout path.

Coverage and synchronization rules are in
[GR2 design](../../clanker/gr2/DESIGN.md),
[bring-up notes](../../clanker/gr2/bringup.md), and the individual notes under
`clanker/gr2/`. Treat those recorded workloads as validation of specific paths,
not a claim that every Express application or diagnostic is implemented.
