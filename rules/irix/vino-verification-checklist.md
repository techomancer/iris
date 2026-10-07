# VINO / IndyCam verification checklist

Consolidates guidance from `rules/irix/vino-*.md`.

## Prerequisites

- `[vino]` configured in TOML; host camera support is built in (no `camera` feature)
- MC SYSID bit 4 set (`rules/irix/vino-attach-via-sysid-bit4.md`)
- GIO alias `0x1F080000` → VINO (`rules/irix/vino-gio-alias-offset.md`)

## Per-channel routing (implemented)

- **SELECT_D1 clear:** composite / SAA7191 path → `source_d0` (default black NTSC field)
- **SELECT_D1 set:** IndyCam / CDMC path → `source_d1` (`set_source` from machine)

## I2C

- Repeated-start reads: START → addr → subaddr → RE-START → read-addr → READ (`src/dev/vino.rs` tests)
- Mid-transaction streaming: `I2C_DATA` writes while `NOT_IDLE` set

## Frame rate mask

- `CH_FRAME_RATE` write resets `field_counter` so the 12/10-field mask applies from field 0

## Manual IRIX tests

1. `vlinfo` — `vino 0` with 5 nodes
2. IRIX 5.3: `indycam_eoe` / capture per `rules/irix/indycam-end-to-end-capture.md`
3. IRIX 6.5: `videod` + `vl_eoe` / `vino_eoe` per `rules/irix/vino-capture-on-6.5-progress.md`
4. `vino status` in monitor — D0/D1 source lines

## Known gaps

- 6.5 interlace capture may show a thin diagonal artifact (`src/dev/vino.rs` comments)
- HPC1 region black-holed to avoid capture panic (`src/physical.rs`)

## CDMC → VideoSource (implemented)

- `CdmcAdjustedSource` applies gain, colour balance, saturation, and shutter exposure
  to UYVY fields from `source_d1` (`cdmc.rs` → `video_source.rs`)
- Manual IRIX capture still required to close the checklist end-to-end
