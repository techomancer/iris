# IRIX vino driver attach gated on SYSID bit 4 — set it

**Keywords:** vino, indycam, vid, vino_init, SYSID, MC, EISA, no boards found, videod, vlinfo
**Category:** vino, irix, mc
**Status:** Fixed in `src/dev/mc.rs` (`init_registers`): SYSID now has bit 4 set
on Indy (`0x13` for Guinness, `0x10` otherwise). With this, `vlinfo` on a
fresh IRIX 5.3-for-Indy install reports `vino 0` with 5 nodes (digital +
analog input, two memory drains, device controls). Without it, `videod`
prints `Fatal server error: no boards found` and exits.

## What was happening

After the [`vino_gio_alias` offset
fix](vino-gio-alias-offset.md) the IRIX kernel could reach VINO at the
GIO64 aperture, but `videod` still reported `no boards found`. Tracing
every iris-vino bus access with `eprintln!` showed that across an entire
boot, login, and `videod` invocation, **exactly one VINO read happened**
— a `REV_ID` read returning `0xB0`, then total silence. The driver
attached chip-presence detection succeeded, but nothing followed.

Disassembling `/var/sysgen/boot/vino.a` from the running IRIX (extracted
via `uuencode` over the serial console; capstone for MIPS-II BE) found
`vino_init` at offset `0x100002ac` in the ECOFF `.text`:

```
0x100002b4: jal      vinoInitialize          ; stub, returns 0
0x100002bc: bnez     $v0, exit
0x100002c0: lui      $t6, 0xbfa0             ; MC base in KSEG1
0x100002c4: lw       $v0, 0x1c($t6)          ; v0 = MC[0x1C] = SYSID
0x100002c8: andi     $t7, $v0, 0x10          ; t7 = SYSID & 0x10
0x100002cc: beql     $t7, $zero, epilogue    ; if t7==0, bail
0x100002d0: lw       $ra, 0x1c($sp)          ; (likely-branch delay slot)
0x100002d4: jal      0x10000664              ; first call into real probe
0x100002d8: lui      $a0, 0xa008             ; KSEG1(0x00080000) = VINO_BASE
... [board allocation, register, vinoInitializeHW, etc.] ...
```

`beql $t7, $zero, epilogue` is "branch likely equal". If SYSID bit 4 is
clear, the branch is taken and the function returns immediately —
silently, no `printf`, no other bus traffic. This is the silent-exit we
were chasing.

The single REV_ID read in the no-bit-4 case isn't from `vino_init` at
all; it's from another piece of inventory code (`vinoUpdateInventory` /
`_replace_in_inventory`) that runs regardless of the EISA gate and
populates `hinv`'s "Vino video: unit 0, revision 0" line. So `hinv`
lying about a present chip while `vid` has zero registered boards was a
red herring — they come from independent code paths.

## What bit 4 of SYSID means

`docs/mc.pdf` (the SGI MC ASIC datasheet) documents SYSID bit 4 as
"EISA bus present. Determined by `eisa_present_n` pin." Indy is **not**
an EISA system, so per the spec this bit should be clear on Indy and
set on Indigo2. But the IRIX 5.3 vino driver (the same `vino.a` is used
for both IP22 variants) gates the entire vino attach path on it, so on
spec-correct Indy the driver would never attach. That contradicts
real-world IndyCam working out of the box on Indy — meaning either
(a) the bit is repurposed on Indy MC silicon as a "vino-board-present"
strap and the docs/silicon-spec drift, or (b) the spec is right and
this particular `vino_init` only attaches on Indigo2 with vino addon
and a different code path covers Indy. The pragmatic answer for the
emulator is: real IndyCam-equipped Indys behave as if this bit is set,
so iris should report it set.

## Fix

`src/dev/mc.rs::init_registers`:

```rust
regs[(REG_SYSID / 4) as usize] = if guinness { 0x00000013 } else { 0x00000010 };
```

Was `0x00000003 / 0x00000000`. Bit 4 is now set in both modes. The low
nibble (revision) is unchanged.

## Verification

On the running IRIX after `boot` and `root` login:

```
IRIS # /usr/etc/videod &
IRIS # vlinfo
name of server:
number of devices on server:    1

device: vino 0
        nodes = 5
        VINO Device Controls, type = Device, kind = 0, number = 0
        Digital Video Input, type = Source, kind = Video, number = 0
        Analog Video Input, type = Source, kind = Video, number = 1
        Memory Drain 0, type = Drain, kind = Memory, number = 0
        Memory Drain 1, type = Drain, kind = Memory, number = 1
```

`vlinfo -l` enumerates 67 controls (`h_clamp_begin`, `odd_offset`, …).

## What still doesn't work

`vidtomem -f /tmp/cap -v 0` hangs without producing `/tmp/cap-00000.rgb`
— frame DMA isn't completing. That's a separate VINO emulation gap
(DMA descriptors, frame-ready interrupt, the camera test-pattern
pipeline) and is its own investigation, tracked in TODO.md / a future
rule.

## Reproduction history

- `IRIX 5.3 for Indy.iso` (from jrra.zone/sgi) installed onto a fresh
  4 GB raw disk via the usual PROM → fx → Install System Software path.
- Default install set + `vino.sw.eoe` + `vino.sw.diag` selected; `go`.
- After install + reboot, `videod` failed with `no boards found`.
- Disassembly + bit-4 hypothesis + fix as above.

## See also

- [vino-gio-alias-offset.md](vino-gio-alias-offset.md) — the prior alias
  fix that made REV_ID reachable in the first place.
- `docs/mc.pdf` §5.4 — SYSID register layout (documented).
- `src/dev/mc.rs` — `REG_SYSID` and `init_registers`.
- `src/dev/vino.rs` — `REV_ID @ 0x0000` returns `0xB0` (chip_id=0xB rev=0).
