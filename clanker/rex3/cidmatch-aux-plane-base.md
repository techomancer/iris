# CIDMATCH probe used the wrong framebuffer base on aux-plane draws — 2026-09-18

**Status: FIXED.** `Dm1::use_aux()` now picks the base for the CID probe in both shader
emitters; regression test `jit_cidmatch_aux_plane_matches_interp`, verified to fail
without the fix.

A SIGSEGV on the `REX3-Processor` thread after ~5 hours of IRIX 6.5 with X11 running.
Worth keeping for two reasons: the fix is one line per site, and the *diagnosis* was done
entirely from a macOS crash report with no symbols — the faulting frame is JIT code, so
there is nothing to symbolicate. The method below generalises to any rex-jit or jitv2
crash report.

## The bug

`emit_calculate_fb_address` picks its base by plane:

```rust
let fb_ptr = if use_aux { pctx.fb_aux } else { pctx.fb_rgb };  // OLAY/PUP/CID → fb_aux
let px_ptr = b.ins().iadd(fb_ptr, byte_off64);
```

The CIDMATCH probe then re-derived the byte offset — unconditionally against `fb_rgb`:

```rust
let byte_off64 = b.ins().isub(px_ptr, pctx.fb_rgb);   // wrong base when px_ptr is aux
b.ins().iadd(pctx.fb_aux, byte_off64)
```

For an RGB/RGBA draw the two cancel and the probe is right, which is why every existing
CIDMATCH test passed: `check_cid_write_masks` sweeps all 16 masks, 4 CIDs and 3 adrmodes,
but only ever with `DM1_RGB24_SRC`. For an OLAY/PUP/CID draw `px_ptr` is *already*
`fb_aux + off`, so the probe read

    fb_aux + (fb_aux - fb_rgb) + off

`fb_rgb` and `fb_aux` are two independent `Box<[u32]>` allocations, so `fb_aux - fb_rgb`
is whatever the allocator chose — 0x29C0_0000 (700 MB) in the crashed process. The
consequence therefore ranges from reading a stray mapped word to SIGSEGV, with nothing in
the shader to distinguish the cases.

X11 draws menus, popups and the cursor into the overlay planes with CID checking live, so
the shape is common; it needs the *pair* (aux plane, `cidmatch != 0xF`) to fire, which is
why it survived this long. Both emitters carried the probe — `emit_shader` and the line
emitter — so both had to be fixed.

## Reading the crash report

The report gives `pc` in no mapped image and a thread name, and that is the whole of it:

```
Thread 33 Crashed:: REX3-Processor
0   ???        0x117b141fc ???
1   iris-gui   ..._4iris4rex3..Rex3..Device5start + 1980
KERN_INVALID_ADDRESS at 0x0000007d3188c028      esr 0x92000006 (translation fault)
x1 0x7cde000000  x2 0x7d07c00000  x3 0x29c8c028  x24 0x800  x26 0x8c028  x27 0x46
```

The lever is `instructionByteStream` in the JSON tail: `beforePC` and `atPC`, 40 bytes
each, base64. Decode them (`base64 -d`, then read 4-byte little-endian words) and
disassemble by hand or with `llvm-objdump -d --triple=arm64`:

```
-12  8b3a4043   add  x3, x2, w26, uxtw      ; px_ptr = fb_ptr + byte_off
 -8  f84003e1   ldur x1, [sp]               ; reload the spilled fb_rgb
 -4  cb010063   sub  x3, x3, x1             ; "byte offset" = px_ptr - fb_rgb
 +0  b8636843   ldr  w3, [x2, x3]           ; ← fault: base is fb_ptr, not fb_rgb
 +4  12000463   and  w3, w3, #3             ; cid = aux_raw & 3
 +8  5280009b   mov  w27, #4                ; cidmatch nibble, folded in as a constant
+12  1ac32763   lsrv w3, w27, w3            ; cidmatch >> cid
+16  360000a3   tbz  w3, #0, ...            ; bit clear → skip the pixel
```

`and #3` / `lsrv` / `tbz` is the CIDMATCH block and nothing else in the shader, so the
site is identified without symbols. The folded `mov w27, #4` even gives the register
state: `CIDMATCH=0b0100`, i.e. only CID 2 permitted.

Then the arithmetic confirms it rather than merely suggesting it:

- `x2 - x1 = 0x29C0_0000` — two allocation bases, 700 MB apart.
- `x26 = 0x8C028` → `/4 = 143370` → `y = 143370 / 2048 = 70`, `x = 10`. A pixel comfortably
  inside a 1280x1024 screen: **the coordinates were never out of bounds**, which rules out
  a clipping or DDA bug and points at the addressing itself.
- `x2 + (x2 - x1 + x26) = 0x7D31_88C0_28` = `far`, exactly. The faulting address is
  reproduced from the registers, so the mechanism is not a guess.

Two things made this fast, and both are worth repeating: **the shader emitters are
straight-line IR, so a short instruction window is enough to name the block**, and
**reproducing `far` arithmetically from the register file turns a hypothesis into a
proof.** If the recomputation had not matched, the story would have been wrong.

## Guard against the general shape

Any pointer formed in the shader has to pick its base with the *same* predicate
`emit_calculate_fb_address` used. `Dm1::use_aux()` now exists so the choice is made in one
place; the inlined `matches!(dm1.planes(), OLAY | PUP | CID)` copies (there were three)
are gone. A second base derived from a pointer is the smell — prefer carrying the offset.
