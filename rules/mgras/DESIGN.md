MGRAS (IMPACT / GR4) design

References: ignore/gr4/*.h (hardware headers), decompiled kernel modules in
ignore/gr4/mgras_kernel/decomp, libGLcore and the X DDX under ignore/gr4.
The headers were partly generated from this emulator's own early code, and
none of the binaries carry type DWARF, so struct field names in the
decompiles are the headers' guesses. Check raw offsets, not names (see
dcb-device-numbers.md). Hardware findings go back into the headers.

Same shape as GR2 (rules/gr2/DESIGN.md): CPU thread, a frontend thread, a
backend thread and a display thread, with two FIFOs.

blocks (src/dev/mgras):
  mod.rs        Mgras: bus decode, shared state, threads, sync rules
  hq3.rs        HQ3 engine, frontend thread: CFIFO word parser, command
                processor tokens (HLE), host DMA (HAG), formatter, context
                switch. GE11 HLE goes here later, as GE7 lives in hq2.rs.
  ge11.rs       GE11 diagnostic port and microcode storage (inert)
  rss.rs        raster subsystem, backend thread: RE4 register file and
                primitives (block, line, stipple, transfers; later triangles),
                PP1 pixel ops, the framebuffer planes. TE1 joins it later.
  dcb.rs        display control bus decode; vc3/xmap/cmap/dac chips (inert,
                CPU thread)
  disp.rs       display thread: retrace, then a frame snapshot + compose, present
  frame.rs      display composition, Newport-style: snapshot after vblank
                decodes planes, per-pixel window IDs (VC3 runs), per-DID
                descriptors (XMAP), cmap, 8-bit gamma and cursor into flat
                buffers; one pass per pixel (cursor, overlay, main, gamma)
                makes the frame. Shader-ready: per-pixel work reads only its
                own position in the buffers plus small tables.
  debug.rs      monitor commands and annotated trace
  record.rs     replayable recording + replay test (regression goldens)

data path:
  CPU -> DCB (0x60000), dcbctrl (0x68000), HQ3 ucode RAM, privileged flag
         and enable registers, GE diag port: direct, CPU thread
  CPU -> CFIFO (user 0x70080, privileged 0x50080, 0x80000 window)
         -> hq_fifo -> HQ3 thread -> rss_fifo
  CPU -> RSS registers (0x7C000, +0x1000 = execute) -> hq_fifo too: the
         HQ3 forwards them, so they stay in program order with CFIFO
         raster commands (the old model ran both synchronously)
  RSS thread owns the framebuffer; the display thread reads it unlocked
  (tearing tolerated, as REX3 and GR2)

sync rules:
  - every producer pushes while holding `submit` (the CPU's bus accesses,
    the display thread's frame tick). Holding `submit` with the board idle
    means nothing can change engine state: that is how reads of drawing
    state, checkpoints, composites and the monitor touch the engines. The
    HQ3 thread never takes `submit` (its holder may be waiting for idle);
    DMA recording goes through `mem_rec`.
  - idle = hq_fifo empty, HQ3 not busy, then rss_fifo empty, RSS not busy,
    checked in that order (whatever the HQ3 forwarded before going idle is
    already in rss_fifo).
  - RSS register reads, the PIO read pair and anything whose answer depends
    on drawing wait (bus busy) until both FIFOs are idle. A PIO high read
    then queues an advance op, like GR2's READ_ADVANCE.
  - flags are atomic: the HQ3 thread raises them (done, swap, context
    loaded), the CPU sets and clears them. Whoever changes them recomputes
    the general interrupt level.
  - host DMA runs on the HQ3 thread. Host to board: memory is read there
    and pixel lines go to the RSS as payload entries. Board to host: drain
    the RSS, read the framebuffer, write memory, then raise the flag.
  - ordering markers go through hq_fifo: context switch, PIO read advance.
    Host GL composite waits for idle under `submit` and draws directly.
    (Frame ticks are gone: block type 0 turned out to be a plain fill, so
    nothing waits for "char data that may follow" any more.)
  - status, FIFO status, DMA busy, DMA/RE-interface context and raster reads
    wait for idle; flag reads do not (drivers poll flags until the HQ3
    raises them, as on hardware).
    A threaded desktop boot polls the flags ~640k times (single-threaded:
    a few hundred) at no cost in boot time. If it ever matters, stall the
    flag read while the HQ3 has work, as GR2 does for FIN2.

state: RSS, HQ3 and DCB state are plain data (no Box/Vec/HashMap), built
in place with Arc::new_zeroed, C-style structs in case of a JIT later.

regression: tools/mgras-capture.py records a session from power-on
(IRIS_MGRAS_REC) with hash checkpoints (scripted marks plus one every 250k
records); record.rs replays it through the bus interface and checks every
checkpoint. Goldens live in ignore/mgras-golden (too big for git):
desktop.rec and dense.rec were recorded by the single-threaded model.
Current golden: desktop2.rec (after the points / skip-last / block type 0
fixes). desktop, dense, threaded and flat predate them.
Deliberate behaviour changes (2D fixes) make old goldens fail from the
first checkpoint after the changed drawing: check the replayed screen
against the capture's PNGs (diff the changed pixels), then re-capture.
Every refactor step must replay them bit-exact:
  MGRAS_REPLAY=ignore/mgras-golden/dense.rec cargo test --release mgras_replay_file -- --nocapture

phases:
  - goldens from the single-threaded model
  - split into the files above, no threads, goldens pass
  - HQ3 + RSS + display threads, goldens pass, live boot to desktop
  - flatten state to plain data, goldens pass
  - monitor commands and annotated trace (gr2 parity): `mgras help`,
    `mgras stats` (command/primitive counters), `mgras trace <file>
    [hq,rss,cpu|all]` (IRIS_MGRAS_TRACE from power-on), `rss regs|pix`
  - trace-replay test architecture:
      rss.rs tests            RSS semantics on a bare `Rss` (no threads)
      mgras_tests.rs          the live board through the bus: FIFO/direct
                              ordering, PIO reads, DMA both ways (fake
                              memory + page table), flags, context switch,
                              GE11 readback, text-recording replay
      testdata/*.txt          small captures in the recording's text form
                              (mgras rec <file>.txt, or tools/mgras-rec.py
                              text to slice a binary one); replayed with
                              record::tests::replay_recs, no checkpoints
                              (they start mid-session), then asserted
      ignore/mgras-golden     power-on goldens, every checkpoint checked
    Not yet: snapshots (save_state is empty). With a non-COW disk a restore
    would leave the filesystem behind RAM, and the framebuffer belongs in
    the machine's chunk manifest like REX3's; do both together.
  - 2D: tools/xwd-compare.py checks XGetImage (xwd -root, fetched over the
    built-in NFS share) against `mgras fbdump`; 8-bit dumps match exactly,
    and xwud (XPutImage) redisplays them exactly apart from the cursor
  - framebuffer 2048x2048 (for 1600x1200 and 1920x1035), indexed by the
    hardware's (x, y) with the screen bottom-left. state_bytes hashes the
    old 1280x1024 region, plus the rest only when it is non-zero, so older
    goldens keep their hashes. Shared display buffers/textures:
    disp::FB_MAX_H rows (Newport's input textures stay 1024). Big state is
    built in place (Arc/Box::new_zeroed), never by value: never derive
    Clone/Default on Rss/Frame or build them as struct literals.
  - OpenGL (2026-10-04): GE11 HLE in the HQ thread (gl.rs) on the shared
    GL core (src/dev/gl: matrices/stacks, clipping, viewport, lighting,
    fog; copied from gr2, gr2 moves onto it next). It sets up triangles
    through the RE4 area registers like SGI's diag _mg0_FillTriangle;
    rss.rs rasterizes them (GL half-open rule, colour planes rebuilt from
    start/dx/de). Area IR opcodes 0/1 are ours (real numbers unknown).
    The GL window (origin, masks, DRB pointers) comes from the kernel:
    token 0xE4 or the 63-word context image with word 1 bit 29 set (layout
    in ignore/gr4/HQ3.h). Each GL batch loads it inside a gl_enter /
    gl_leave bracket that the RSS uses to save and restore the raster
    registers the X server set (GL_BRACKET in rss.rs).
  - contexts (HQ3.h "CONTEXT MANAGEMENT"): real hardware keeps each
    context's GE state on the board, in an ERAM slot whose address is image
    word 0, and the HQ saves its host-interface (parser) state with it. So
    do we: Hq3Engine::eram (2 x 64K words; only GE0's is used) holds, per
    slot, the Gl as raw words and the user-port parser. Saved at the switch
    request, loaded from the image's slot (fresh on word 1 bit 31). The GE
    pass-through tokens (0xFA-0xFD) move ERAM to and from host memory by
    DMA or FIFO pixel data, which is how the kernel evicts slots when more
    than about ten contexts are open (tested: ideas + atlantis + 8 gltests,
    thousands of evictions, state intact). Gl must stay plain numbers:
    ERAM is guest-visible, and loads clamp what indexes arrays.
  - raster state across switches: the hardware saves none. The load writes
    the incoming window registers (plus, for GL, the X server's
    screen-wide registers from image words 18-39); the GE regenerates GL
    raster state; the X server sets its mode registers per operation. We
    apply only the window from the image: each GL batch loads its full
    raster state from Gl (begin_raster), and the RSS's gl_enter / gl_leave
    bracket keeps the X server's registers as X left them, so X never sees
    GL's values. The bracket covers every raster register except the TE's
    (the GE keeps those loaded across batches) and the model-private ones:
    X sets the IR, colours and modes only when they change, so anything GL
    leaves behind gets used. A hand-picked list missed the IR; after a
    switch back from gltest, X's line executes redrew GL's last 800x600
    triangle over the desktop (gl_batch_leaves_x_instruction_alone).
    glFinish = __MGR_RETURN_MODE round trip (answered 0, address logged).
    glprim flat and smooth triangles render live on 6.5.22.
  - then 3D

pixel memory (Octane technical report section 4.4.14, Table 4.2; our
reading): RDRAM bytes are 9 bits, and the framebuffer is built from 36-bit
buffers and one 9-bit buffer a pixel. The 9-bit one is only an 8-bit CI
overlay plus 1 spare bit. Each 36-bit buffer is one of 12/12/12 RGB,
8888 RGBA, 5551 or 4444 double buffered, 12-bit CI double buffered, or
ZST (24-bit Z + 8-bit stencil). Spare bits (not 12/12/12) are per-pixel
tags: the displayed half of independently swapped double-buffered windows,
fast clears. Count by resolution (High/Maximum): 1024x768 3/6,
1280x1024 2/4, 1600x1200 and 1920x1035 1/2. 12MB x 9 bits fits
1280x1024 x (2x36 + 9) exactly. Confirmed from the IDE diags'
_xy_to_rdram and the PROM timing tables (full layout in ignore/gr4/PP1.h):
there is no linear framebuffer; a buffer is a run of 2KB RDRAM pages
(448 at 1280x1024, 288 at 1024x768, tiles of 192x16 pixels), and
DRBpointers / the XMAP DIB pointer hold page numbers. Off-screen memory
(back buffer, pbuffer) is therefore another buffer's pages on the same
(x, y) grid, not rows beyond the screen. Pixels are still stored as the
old u32 values in a word's low bits; double buffering, Z and 12-bit
colour need the 36-bit word formats per buffer use.

pixel memory model (pixmem.rs, 2026-10-03): 1024 pages x 3072 u64 words,
a page = one tile; a 36-bit buffer reads a page as 192x16 pixels, the
overlay as 768x16, four 9-bit pixels a word (shift 9 * (x & 3)). Drawing
goes to the page in DRBpointers[9:0], DrawBuffer picking overlay vs wide;
scanout to the XMAP DIB pointers ([9:0] main, [31:20] overlay; PROM value
until written); DRBsize gives tiles per row (PROM 1280x1024 layout until
written). state_bytes rebuilds the old 1280x1024 fb/overlay views through
the scanout pointers plus a hash of any other set bit, so desktop2.rec
still replays 35/35. Not yet: the second pointer field ([19:10]), double
buffering, Z/stencil, the raw RDRAM window (device space is still a
register map), two raster engines.
