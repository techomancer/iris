# Mapping GL state queries to the GE (glstate, statemap, probes)

`test/gltest/glstate.c` sets every queryable OpenGL state to a value no
default has, reads it back with the matching getter, and checks the
answer. It is board-neutral (GL, GLX, Xlib only), so the same program
maps GR2 later; only the decoders are per board.

As of 2026-10-13 on MaxImpact (6.5.22): 397 pass, 1 fail (selection-mode
name stack, not modelled), 15 XFAIL (libGLcore's own departures, listed in
the test), 20 skipped (extensions IMPACT rejects).

The map itself is in ignore/gr4/GE11.h "STATE READBACK". The model is
`src/dev/mgras/glquery.rs`: a query shadow of the raw values, kept beside
the render state.

## Running

- In the guest: `sh irix.sh glstate`, then `./glstate -v [--only S]...`. `-v`
  lists every mismatch of a case, not just the first.
- Recording: `mgras rec F`, run glstate, `mgras rec off`. Then
  `tools/mgras-statemap.py F --results glstate.txt` prints, for each case,
  the commands sent and the answers read back, followed by the readback
  map: each token and GE address, with the GL queries that use it.
- Formats: `tools/mgras-probe-states.py OUTDIR [--only S]...` drives
  `mgras probe LO HI VALUE` (the GE answers VALUE for readbacks in that
  range). It runs five bit-mask runs, all-ones, zero, and answers 1..9.
  The bit pattern across the runs gives the bit a boolean query tests; the
  small answers give an enum's decode table.

## Gotchas

- **Never probe 0x2A0.** It is the GE error word that glGetError reads,
  and odd values make libGLcore spin forever (glstate hung at 100% CPU).
- **IRIX telnet truncates long command lines.** A long glstate command line
  arrives cut off (`/shared/b`). Put the command in a script under
  /shared and run the script.
- **glPushAttrib reads GE state back.** Each case brackets its work between
  a start marker and an end marker (glLineWidth(1 + n/1000) and
  glLineWidth(1 + (n + 0.5)/1000)), so the next case's push does not count
  as this case's queries.
- **"Bit 31" in the bit runs is ambiguous.** Every mask has bit 31 set, so
  a non-zero test looks the same. The answer-1 run tells them apart: blend,
  scissor and line smooth test non-zero; clip-plane enables test bit 31.
- **Old placeholders win.** `Gl::state_word` matched first and still held
  stubs (0x1FC as the dither flag only, 0x2BF as 0.0). Grep for a fixed
  address before adding one to `query_word`.
- **The raster colour query wants OpenGL's latch.** Drawing deliberately
  follows later colours, for IRIS GL bitmaps (gr_osview). So the query
  answers from `Query::raster_color`, set at RASTER_POS.
- **glMatrixMode(GL_COLOR) used to fall through to the modelview stack.**
  It now has its own stack (`Gl::cm`).
