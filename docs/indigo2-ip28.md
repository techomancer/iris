# Indigo2 IMPACT IP28

IP28 pairs the Indigo2 fullhouse devices with an R10000 CPU and IMPACT graphics.
The same binary supports IP24, IP22, and IP28; no `ip28` Cargo feature is needed.
IRIX 6.5 desktop boot and a two-bank 1 GB configuration have been recorded in
the repository's validation notes. Device and graphics coverage is still evolving.

## Configure the machine

IP28 includes the embedded `070-1477-002` PROM in `src/prombinip28.rs`.
The loader tries the configured `prom` path, then `070-1477-002.bin` in the
working directory, then the embedded IP28 image. An external PROM is optional.
Set top-level paths and banks before the TOML sections:

```toml
prom = ""                 # use the default-file / embedded IP28 fallback
nvram = "ip28-nvram.bin"
nveeprom = "ip28-nveeprom.bin"
banks = [512, 512, 0, 0]

[machine]
profile = "indigo2_ip28"
cpu = "r10000"

[graphics]
board = "highimpact"       # also solidimpact or maximpact
heads = 1
resolution = "guest"

[scsi.1]
path = "irix65.chd"
```

In iris-gui, select **SGI Indigo2 IMPACT (IP28)**. New Machine selects R10000,
uses the embedded PROM by default, and defaults to IMPACT Solid. Choose High
or Maximum in General if needed. Stop and start to apply hardware changes.
IP28's IRIX kernel has no Newport driver; choose an IMPACT board.

## CPU, memory, and clock

R10000 reports PRId `0x00000925`, FIR `0x00000900`, MIPS IV, 64 TLB entries,
and 44-bit virtual addresses. Loads, stores, and fetches access memory directly;
shadow cache arrays model CACHE operations, tags, data, ECC, and PROM tests.
This is functional emulation, not a timing or out-of-order CPU model.

RAM begins at `0x20000000`. MEMCFG has a 16 MB granule; IP28 accepts 256 MB
and 512 MB banks in addition to the smaller sizes. Two 512 MB banks provide
1 GB, verified through PROM POST and IRIX `hinv -t memory`. This does not
change the IP22/IP24 four-bank IRIX 6.5 limitation. See
[IP28 512 MB banks](../rules/irix/ip28-512mb-banks.md).

CP0 Count defaults to 97.5 MHz, which IRIX reports as 195 MHz CPU inventory.
Override with `[clock] fixed_mhz` or `--clock-fixed-mhz`. MIPS in the status
bar measures host emulation throughput; Hz counts the guest kernel's ticks.

## PROM settings and graphics

PROM environment and MAC live in the motherboard EEPROM (`nveeprom`), while
the DS1386 uses `nvram`. Normal Stop saves both. `nveeprom save` and `rtc save`
save immediately before forced termination. Fresh IP28 EEPROMs get PROM
defaults, including a boot-tune volume; existing settings are retained. See
[Battery-backed state on Stop](../rules/irix/nvram-persistence.md).

IMPACT uses HQ3/GE11 command interpretation, RSS software rasterization, TE1
texturing, and VC3/XMAP/colormap/DAC display composition. The GUI and CLI use
the same `GfxDisplay` interface. Select Solid, High, or Maximum through
`[graphics] board`; there is no separate `[impact]` config section. The model
defaults to 1280×1024 before the guest programs VC3; scanout then follows the
guest's timing and DID tables, backed by 2048×2048 raster storage. Newport
resolution presets and dual-head mode do not apply. IMPACT board state is not
saved in snapshots yet. See [MGRAS design](../rules/mgras/DESIGN.md) for coverage and
synchronization, and `mgras help` in the monitor for tracing and framebuffer
inspection.
