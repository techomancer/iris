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

## Fresh settings blocks

The Indy, Indigo2 and IP28 PROMs read `volume` for the boot tune before they
initialize an erased settings block. Erased bytes therefore silence the first
boot even with working audio. Before starting the CPU, IRIS fills an erased
block with the PROM's own defaults (`src/dev/sgi_nvram.rs`): volume `80`,
layout revision 8 on the Indy and Indigo2, and revision 9 plus boottune `1` on
the IP28. On the Indy the block is the DS1386 NVRAM after the clock registers;
on the Indigo2 it is the motherboard EEPROM. It retains the MAC and calculates
the checksum after any MAC injection. Stop saves this image. Existing
non-erased settings, including a deliberately muted volume, are preserved.
GUI-created and reset EEPROM files follow the same path.

The layout and defaults come from SGI's `irix/kern/sys/IP22nvram.h` and
`stand/arcs/lib/libsk/ml/nvram.c`; the checksum is the XOR/rotate algorithm in
`stand/arcs/IPXXprom/IPXXnvram.c`. IP28 audio also requires the separate
`ip28-boot-chime` CPU-revision and PBUS register fixes.
