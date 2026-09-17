# Indigo2 IP22

Select in the GUI (**Platform → SGI Indigo2 (IP22)**) or set:

```toml
[machine]
profile = "indigo2_ip22"
```

No separate build is required — the same `iris` / `iris-gui` binary supports Indy IP24 and Indigo2 IP22 (`--ip22` on the command line).

## PROM, NVRAM and storage

- **PROM:** IRIS tries `prom` from the config, then `070-1367-012.bin` in the
  working directory, then falls back to an embedded Indigo2 PROM
  (`src/prombini2.rs`) — not the Indy one.
- **NVRAM and MAC:** the Indigo2 keeps its PROM environment and Ethernet address
  in a 93CS56 serial EEPROM, not the Indy's DS1386. It is loaded from `nveeprom`
  (default `nveeprom.bin`, `--nveeprom`) and written back with the monitor's
  `nveeprom save`. `[network] mac` is injected into it before boot.
- **SCSI:** two WD33C93A controllers. Put a device on the second one with
  `controller = 1` in its `[scsi.N]` section; `scsi0` / `scsi1` in the monitor
  address each controller.
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
| Graphics | Integrated Newport @ `0x1F000000` | XL card in GIO slot (same REX3 stack) |
| Audio | HAL2 (shared A2 architecture) | HAL2 |

## Boot checklist

1. Build: `cargo build --release --features lightning,rex-jit` (same as Indy), or
   pass `--ip22` on the command line
2. **Stop → Start** after changing platform (cold start picks up IRQ + VC2 bootstrap)
3. Monitor `mc regs` → SYSID `00000010`, GIO64_ARB without `0x400`
4. Monitor `ioc status` → `sys_id=11`, `gc_select`/`extio` visible on fullhouse
5. Guest `hinv` → one XL graphics board
6. X11 login on primary head (`/dev/gfx` / head 0)

With **Guest** display resolution, iris bootstraps **1280×1024** on fullhouse at Start so the GUI shows a framebuffer before IRIX programs VC2 (embedded Indy PROM often delays or skips gfx init on Indigo2-class hardware).

7. Dual-head (`graphics.heads = 2`): guest `hinv` shows two XL boards; iris-gui shows side-by-side heads; CI `screenshot` accepts `"head": 0` or `1`

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

Second REX3 maps at GIO slot 1 (`0x1F600000`). Snapshots include `rex3_head1.bin` and `rex3_head1_rgb.bin` / `rex3_head1_aux.bin`.

Head 0 vblank → `VerticalRetrace` (Indy direct L1; fullhouse via extio `SG_RETRACE`). Head 1 vblank → `GioExp1` (Indy) or `GioExp0Retrace` / extio `S0_RETRACE` (fullhouse dual-head).

## IRQ routing (fullhouse)

Shared GIO interrupt lines (FIFO full, graphics, retrace) are latched in the IOC `extio` shadow and muxed by `gc_select` bit 0 (0 = graphics/SG slot, 1 = expansion S0). MAP bits 6–7 carry GFX drain feedback for Newport heads.

## Not implemented

- IMPACT / MGRAS preview stub (`src/mgras.rs`, `[impact]` config) — see `docs/impact-mgras-research.md`
- Full EXTIO bus-error and EISA interrupt paths

See also [`docs/interrupt_map.md`](interrupt_map.md) and [`rules/gui/machine-profile-vs-guest-ip22.md`](../rules/gui/machine-profile-vs-guest-ip22.md).
