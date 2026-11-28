# Indigo2 IP22

Select in the GUI (**Platform → SGI Indigo2 (IP22)**) or set:

```toml
[machine]
profile = "indigo2_ip22"
```

No separate build is required — the same `iris` / `iris-gui` binary supports Indy IP24 and Indigo2 IP22 (`--ip22` on the command line).

## PROM, NVRAM and storage

- **PROM:** iris carries the Indigo2 PROM (5.3 Rev E, not the Indy's). Left
  at its default, `prom` tries `prom.bin` and then `070-1367-012.bin` in the
  working directory and otherwise runs the built-in PROM; `prom = "(embedded)"`
  selects the built-in one outright. A file `prom` names must load, or iris
  stops with an error rather than falling back.
- **NVRAM and MAC:** the Indigo2 keeps its PROM environment and Ethernet address
  in a 93CS56 serial EEPROM, not the Indy's DS1386. It is loaded from `nveeprom`
  (default `nveeprom.bin`, `--nveeprom`) and written back with the monitor's
  `nveeprom save` or automatically on normal Stop. `[network] mac` is
  injected into a blank slot before boot; an existing MAC is retained.
- **SCSI:** two WD33C93A controllers. A device goes on the second one at
  `scsi1/tN` (so `scsi1/t1` is id 1 there, and id 1 can be on both);
  `scsi0` / `scsi1` in the monitor address each controller.
- **Interrupts:** INT2 and the fullhouse cascade are modelled; the IRIX kernel
  never hooks GIO interrupt 2, so vertical retrace is delivered the way the
  table below describes.

## Hardware deltas vs Indy IP24

| Item | Indy (Guinness) | Indigo2 (fullhouse) |
|------|-----------------|---------------------|
| MC SYSID | `0x00000013` | `0x00000010` |
| MC GIO64_ARB ONE_GIO | set (`0x400`) | clear (dual GIO64 bus) |
| IOC sys_id | `0x26` | `0x11` |
| MAP bits 6–7 | GIO expansion IRQs | GFX DRAIN0/1 feedback |
| Primary vblank IRQ | L1 `VERTICAL_RETRACE` (extio `SG_RETRACE` on fullhouse) | same — MAP `GFX_DRAIN0` is FIFO drain only |
| Graphics | Newport or GR2 XZ @ `0x1F000000` | Newport XL, GR2 XZ/Extreme, or IMPACT |
| Audio | HAL2 (shared A2 architecture) | HAL2 |

## Boot checklist

1. Build: `cargo build --release --features lightning,rex-jit` (same as Indy);
   select the profile in the config or with `--machine indigo2_ip22`
2. **Stop → Start** after changing platform (cold start picks up IRQ + VC2 bootstrap)
3. Monitor `mc regs` → SYSID `00000010`, GIO64_ARB without `0x400`
4. Monitor `ioc status` → `sys_id=11`, `gc_select`/`extio` visible on fullhouse
5. Guest `hinv` → one XL graphics board
6. X11 login on primary head (`/dev/gfx` / head 0)

With **Guest** display resolution, iris bootstraps **1280×1024** on fullhouse at Start so the GUI shows a framebuffer before IRIX programs VC2 (the PROM often delays or skips graphics initialisation on Indigo2-class hardware).

7. Dual-head (a `newport` device entry at `gio0`): guest `hinv` shows two XL boards; iris-gui shows side-by-side heads; CI `screenshot` accepts `"head": 0` or `1`

## Headless smoke (CI)

```powershell
iris.exe --config irix-install/iris-indigo2-smoke-ci.toml
iris-ci ping
# monitor: mc regs → SYSID 00000010
```

## Dual-head Newport

```toml
[graphics]
heads = 2
```

The second board sits in GIO expansion slot 0, with its REX3 at `0x1F4F0000`: the one place IRIX looks for a second Newport. The Ultra64 board uses the same slot, so `heads = 2` turns it off, and `test_device` is refused. Snapshots include `rex3_head1.bin` and `rex3_head1_rgb.bin` / `rex3_head1_aux.bin`.

IRIX's Newport handler reads REX3 STATUS and acts on whatever it finds there, so each board only has to raise an interrupt the driver registered for its slot:

| | Retrace | FIFO |
|---|---|---|
| Head 0, Indy | vertical retrace (local 1) | FIFO full (local 0) |
| Head 1, Indy | slot 0's single pin, `GIO_EXP0` (map status bit 6), shared by both sources | same pin |
| Head 0, Indigo2 | graphics interrupt, EXT_IO `SG_IRQ_1` | EXT_IO `SG_IRQ_2` |
| Head 1, Indigo2 | graphics interrupt, EXT_IO `S0_IRQ_1` | EXT_IO `S0_IRQ_2` |

On Indigo2 the two slots share the local 0 bits; each bit is set while either slot's EXT_IO bit is low.

## IRQ routing (fullhouse)

Shared GIO interrupt lines (FIFO full, graphics, retrace) are latched in the IOC `extio` shadow and muxed by `gc_select` bit 0 (0 = graphics/SG slot, 1 = expansion S0). MAP bits 6–7 carry GFX drain feedback for Newport heads.

## GR2 and IMPACT graphics

Add a `[[devices]]` entry of type `express` (`model` `xz`, `elan` or `extreme`) or
`impact` (`model` `solid`, `high` or `max`). The GUI exposes the same choices in General. GR2 and IMPACT have
command interpreters, software rasterizers, and a display path; they are no
longer preview stubs. Neither takes a Newport resolution preset or a second head. There is no separate `[impact]` section. See
[GR2 graphics](graphics/gr2-xz-extreme.md) and [MGRAS design](../clanker/mgras/DESIGN.md).

For the R10000 Indigo2 profile and larger RAM banks, see
[Indigo2 IP28](indigo2-ip28.md). IP28 has its own built-in PROM.

## Remaining gaps

- Full EXTIO bus-error and EISA interrupt paths

See also [`docs/interrupt_map.md`](interrupt_map.md) and [`clanker/gui/machine-profile-vs-guest-ip22.md`](../clanker/gui/machine-profile-vs-guest-ip22.md).

## Chips

- [REX3 graphics](dev/rex3.md), the Newport board
- [HAL2 audio](dev/hal2.md)
- [WD33C93A SCSI controllers](dev/wd33c93a.md)
