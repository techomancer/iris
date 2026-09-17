# VINO GIO64 alias offset — must subtract 0x1F000000, not 0x01000000

**Keywords:** vino, gio, alias, physical, ENODEV, /dev/vino, videod, wrapping_add
**Category:** physical, vino
**Status:** Fixed in `src/physical.rs` (the `vino_gio_alias` `AliasBus::new(...)` call).

## Symptom

`videod -verboseLoad` on a fully-installed IRIX 6.5 prints:

    dso /usr/lib32/dmedia/video/vino.so: builtin bind func:
        Unable to open device /dev/vino for vino.
    no device found.

even when the kernel has the real `vino` driver loaded (i.e. after
`vino_eoe` is installed from Foundations 1.1; this bug is independent of
the missing-CD issue documented in `rules/irix/irix-install.md`).

## Root cause

VINO is mapped at two physical addresses in iris:

- `0x00080000` — primary registers (`crate::vino::VINO_BASE`), used by
  diag/PROM paths via `PHYS_TO_K1(0x00080000)`.
- `0x1F080000` — the GIO64 aperture, which is what the IRIX kernel uses
  to reach the chip after `ioremap`-style mapping.

`vino_gio_alias` is an `AliasBus` that should rewrite GIO-aperture reads
back to the primary address (`0x1F080000` → `0x00080000`). It does so by
`wrapping_add`'ing a fixed offset to the incoming address. The intended
translation is a wrapping subtraction of `0x1F000000`:

    0x00080000 - 0x1F080000  ==  -(0x1F000000)  ==  0xE1000000   (mod 2^32)

The previous offset was `0xFF000000`, which is `-(0x01000000)` — a 16 MB
subtraction, not a 528 MB subtraction. With that offset the alias mapped
`0x1F080000` → `0x1E080000`, which lands in the reserved/GIO-timeout
region (`gio_err_ptr` from layer 2 of `build_device_map`). Reads
returned `0xFFFFFFFF` and writes were swallowed, so the IRIX vino
driver's `REV_ID` probe saw `0xFF` (chip_id=0xF, rev=0xF) instead of
the expected `0xB0` (chip_id=0xB, rev=0) and refused to attach.

## Fix

In `src/physical.rs`, change the `vino_gio_alias` offset:

    -let vino_gio_alias = AliasBus::new(std::ptr::null::<ErrorBus>(), 0xFF000000u32);
    +let vino_gio_alias = AliasBus::new(std::ptr::null::<ErrorBus>(), 0xE1000000u32);

Verify: `0x1F080000u32.wrapping_add(0xE1000000)` == `0x00080000`.

## How this slipped in

The original comment said "wrapping subtraction of 0x1F000000" but the
literal `0xFF000000` is the two's-complement of `0x01000000`, not
`0x1F000000`. Easy off-by-an-F mistake — `0xFF` looks like "negate
upper byte" but the upper *nibble* needed to be `0xE`, not `0xF`.

## End-to-end caveat

Even with this fix, `videod` still fails to open `/dev/vino` on a stock
6.5.22 install because the kernel was built with `vidstubs.a` instead
of the real `vino` driver — see `rules/irix/irix-install.md` for the
missing Foundations 1.1 CD problem. To verify this fix end-to-end you
need `vino_eoe` (kernel driver) + `vl_eoe` (vlserver) + `indycam_eoe`
(IndyCam-specific bits) installed, all of which live on that missing
CD.

## See also

- `src/vino.rs` — `VINO_BASE = 0x00080000`, `REV_ID @ 0x0000` returns `0xB0`.
- `src/physical.rs` — `build_device_map` mapping of `0x1F080000` to
  `vino_gio_alias`.
- `docs/vino/` — VINO ASIC datasheets.
- `rules/irix/irix-install.md` — install procedure + missing-CD note.
