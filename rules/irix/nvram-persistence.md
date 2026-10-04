# Battery-backed state on Stop

`Machine::stop` stops the CPU before stopping HPC3. HPC3 then saves the
DS1386 NVRAM to `nvram` and, on Indigo2, the motherboard EEPROM to `nveeprom`.
GUI Stop and normal GUI Quit use this path. Manual `rtc save` and
`nveeprom save` still work. An empty path disables automatic saving; the CPU
daughtercard EEPROM has no backing file.

Both automatic and manual saves write a temporary file alongside the target,
flush it, and rename it over the previous image. A failed write leaves the
previous image intact and reports an error. A forced process termination does
not run Stop.

## Fresh IP28 EEPROMs

The IP28 PROM reads `volume` for the boot tune before it initializes an erased
EEPROM. Erased bytes therefore silence the first boot even with working audio.
Before starting the CPU, IRIS initializes an erased IP28 motherboard EEPROM
with PROM defaults, layout revision 9, volume `80`, and boottune `1`. It retains
the MAC and calculates the checksum after any MAC injection. Stop saves this
image. Existing non-erased settings, including a deliberately muted volume,
are preserved. GUI-created and reset EEPROM files follow the same path.

The layout and defaults come from SGI's `irix/kern/sys/IP22nvram.h` and
`stand/arcs/lib/libsk/ml/nvram.c`; the checksum is the XOR/rotate algorithm in
`stand/arcs/IPXXprom/IPXXnvram.c`. IP28 audio also requires the separate
`ip28-boot-chime` CPU-revision and PBUS register fixes.
