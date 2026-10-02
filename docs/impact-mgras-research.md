# IMPACT / MGRAS graphics — research notes

IRIS emulates **Newport (REX3)** on Indy and Indigo2 XL paths. IMPACT is a separate
post-1995 architecture (on-board geometry + TRAM). A **preview register stub** lives in
`src/dev/mgras.rs` (Wave 5).

## Implementation status (IRIS)

| Component | Status |
|-----------|--------|
| `src/dev/mgras.rs` | Preview — per-slot register file, GIO mapping for gfx/exp0/exp1 |
| `config::ImpactSection` | `[impact]` TOML: `gfx`, `exp0`, `exp1` → `none` / `solid` / `high` / `max` |
| `physical.rs` | Maps MGRAS stub across populated GIO slots when `profile = indigo2_ip22` |
| Command processing / TRAM / GL | **Not implemented** |
| IRIX `impact` driver attach | **Not expected** to reach a working console |

Monitor commands: `mgras` (slot summary), `impact` (hinv-style inventory preview).

Example (preview only — Indigo2 profile):

```toml
[machine]
profile = "indigo2_ip22"

[impact]
gfx = "solid"
# exp0 = "high"   # second board for High IMPACT
# exp1 = "max"    # third board for Maximum IMPACT
```

## Hardware families

| Option | GIO64 slots | Geometry | Texture RAM | IRIX driver class |
|--------|-------------|----------|-------------|-------------------|
| Newport XL | 1 | CPU (FPU) | system RAM Z-buffer | `gfx` / REX3 |
| Solid IMPACT | 1 | on-board GE | TRAM | `impact` / MGRAS |
| High IMPACT | 2 | on-board GE | TRAM | `impact` |
| Maximum IMPACT | 3 | dual GE | TRAM | `impact` |

Valid dual-head combos with Newport: Solid+Solid, Solid+High, Solid+Max — not High+High.

## Reference sources

- MAME: `impact.c`, `newport.cpp`, `ip22.cpp` — register maps and IRQ fan-out
- Linux MIPS: `arch/mips/include/asm/sgi/ip22.h`, `drivers/video/impact`
- IRIX: `impact` kernel module, `libGL` IMPACT path, `hinv -c graphics`
- Hardware overview: [Wikipedia SGI Indigo2](https://en.wikipedia.org/wiki/SGI_Indigo2), [sgistuff.net Newport](http://www.sgistuff.net/hardware/graphics/newport.html)

## Estimated emulation scope (if pursued)

1. GIO multi-slot board model + IMPACT PSU/riser constraints
2. MGRAS geometry engine + raster engine + TRAM ASICs
3. IRIX PROM revision checks for IMPACT-ready systems
4. Dual-head policy separate from dual Newport

**Recommendation:** Complete Indigo2 Newport boot + dual-head before pursuing full IMPACT.
Orders of magnitude more work than REX3.

## Stub register map (`src/dev/mgras.rs`)

Each populated GIO slot exposes an 8 KB window at `slot_base + 0x0F0000` (same offset
pattern as Newport/XZ until a verified MGRAS map is available):

| Offset | Name | Notes |
|--------|------|-------|
| `0x0000` | `BOARD_ID` | ASCII-tagged placeholder per slot kind |
| `0x0004` | `REVISION` | class revision placeholder |
| `0x0008` | `STATUS` | idle / FIFO-empty defaults |
| `0x000C` | `INTR_STATUS` | write-1-to-clear storage |
| `0x0010` | `INTR_ENABLE` | stored |
| `0x0018` | `FIFO_WRITE` | accepted, not executed |
| `0x0024` | `SLOT_ROLE` | slot index (0=gfx, 1=exp0, 2=exp1) |

GIO physical bases (IP22):

| Slot | Range |
|------|-------|
| gfx  | `0x1F000000`–`0x1F3FFFFF` |
| exp0 | `0x1F400000`–`0x1F5FFFFF` |
| exp1 | `0x1F600000`–`0x1F9FFFFF` |
