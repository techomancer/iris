# GR2 XZ and Extreme graphics

IRIS implements the Express-family graphics path in `src/dev/gr2/` for Indy
XZ, Indigo2 XZ, and Indigo2 Extreme. It replaces the old probe-only XZ stub.
The placeholder register map from that stub is not the implemented map.
There is no separate Elan setting in the configuration.

## Select a board

```toml
[machine]
profile = "indigo2_ip22"    # indy_ip24 also supports xz

[graphics]
board = "xz"               # extreme is restricted to indigo2_ip22
heads = 1
resolution = "guest"
```

Select **GR2 XZ** or **GR2 Extreme** in iris-gui's General tab. No graphics
Cargo feature is needed. Non-Newport boards require one head and Guest
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

HQ2 interprets textport and irisGL command tokens. Geometry, lighting,
homogeneous clipping, polygons, glyphs, context state, and pixel DMA are
implemented in software; uploaded GE7 microcode is stored for diagnostics
rather than executed. The RE3 thread owns VRAM and draws from its FIFO.
VC1, XMAP5, and Bt457 feed a software compositor and a display thread.

Both frontends obtain frames through `Machine::get_display()` and `GfxDisplay`.
Graphics do not require HostGL; `hostgl` is a separate optional service for
programs using the replacement guest libGL.

## Snapshot limitation

GR2 save/load covers selected registers, microcode, SRAM, and display tables.
VRAM and full drawing/context state are not saved yet, so snapshots do not
reproduce the complete rendered scene. See [TODO.md](../TODO.md).

## Inspect and trace

Use `gr2 help` for the full command list:

```text
gr2 status
gr2 hq
re3 regs
gr2 trace gr2.log hq,re3,cpu
gr2 trace flush
gr2 trace off
gr2 fbdump gr2dump
```

Coverage and synchronization rules are in
[GR2 design](../rules/gr2/DESIGN.md),
[bring-up notes](../rules/gr2/bringup.md), and the individual notes under
`rules/gr2/`. Treat those recorded workloads as validation of specific paths,
not a claim that every Express application or diagnostic is implemented.
