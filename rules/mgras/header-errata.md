ignore/gr4 headers: known errata (fold back into the headers)

The headers were partly generated from this emulator's early code and from
guesses, and no binary carries type DWARF, so they are hints until checked
against raw offsets in the decompiles or against what boots. Found so far:

- DCB device numbers (MGRAS.h struct, decompile field names): were wrong. Real:
  CMAP all/0/1 = 3/4/5, DAC = 6, XMAP = 7, VC3 = 8, BDVERS = 9. See
  dcb-device-numbers.md.
- Command FIFO ports (HQ3.h HQ3_CFIFO_USER 0x700F0, HQ3_CFIFO_PRIV
  0x500F0): the drivers write 0x70080/0x70084 (user) and 0x50080/0x50084
  (privileged).
- Context switch request (HQ3.h HQ3_CTXSW 0x500F8): the kernel writes
  0x50050.
- Raster commands through the FIFO (HQ3.h HQ_CMD_RSS_WRITE(reg) =
  0x1000 | reg << 2): the command word is (0x1000 | reg | 0x400 execute)
  << 8 | byte count.
- IR opcodes (RE4.h had TRIANGLE 1, LINE 2, BLOCK 3, CHAR 4, SPAN 5): 4 is a
  point (at xline_xystarti), 5 the X line (IR_OP_X_LINE, with Setup), 8 the
  block (IR_OP_BLOCK). Fixed in RE4.h.
- Block type 0 is a fill, not "wait for char data"; char stipples are block
  type 1 with CharStipEn (fill mode bit 3). Fill mode bit 10 skips a line's
  last pixel. Fixed in RE4.h.
- The mgras_hw struct had the DCB devices and the FIFO ports in the 1994
  diagnostic order/addresses. Fixed in MGRAS.h (offsets checked with
  offsetof); HQ3.h's FIFO section rewritten (word formats, command space,
  CP_* and GL token tables).

What the headers get right and the code relies on: the rss_single register
file layout (names and offsets of every register the model uses).
