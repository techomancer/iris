DCB device numbers: trust the kernel's dcbctrl writes, not header names

The display control bus maps device N at 0x60000 + N*0x400, with its
protocol timing register at 0x68000 + N*4. MgrasInitBoardGfx
(mgras_init.c ~line 650) writes the timing words:

  slot 3,4,5  0x108C  -> CMAP broadcast, CMAP0, CMAP1 (identical timing)
  slot 6      0x1084  -> DAC
  slot 7      0x318C  -> XMAP (in the PP1s; slowest CS hold)
  slot 8      0x210A  -> VC3 (the only async-ack device)
  slot 9      0x1090  -> board version
  slot 0      never written

An RE pass renumbered the devices from MGRAS.h (XMAP 0, DAC 4, CMAPs 5/6/7)
and broke the display. MGRAS.h's struct and the decompile's field names
(dcbctrl_cmap1 at 0x6801C, ...) are wrong: the binaries have no type DWARF,
so those names came from the generated header itself.

General rule: in ignore/gr4 decompiles, check raw offsets and values; a
field name is only the header's guess.
