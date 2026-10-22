# Lines with COLORHOST/ALPHAHOST: legal, but Cranelift refuses them

`rex shaders list` reports these as `failed`:

```
failed  0x0000008a  0x3165f011  0x00001e03   DRAW ILINE ALPHAHOST | RGB 12bpp host:12bpp DITHER BLEND(SA+MSA)
failed  0x0000008a  0x3165f031  0x00001e03   ... + DBLSRC
```

`dm0 = 0x8a` decodes to OPCODE=DRAW, ADRMODE=I_LINE, ALPHAHOST=1. They are
rejected by the guard at `rex3_jit/compiler.rs`:

```rust
if is_line && (is_hostr || is_hostw) {
    eprintln!("REX JIT: not jittable (line+host) ...");
    return None;
}
```

This is a **deliberate gap in Cranelift's line emitter**, not a compile error and
not an invalid draw mode.

## The mode is legal — do not sanitise it away

`rex3.pdf` §3.5.1 (Lines: Overview) and the framebuffer-access paragraph above it:

> The framebuffer may be accessed as points, **lines**, spans, or blocks of data.
> […] In the following subsections, **"Segments I" refers to packed host data,
> using the HOSTRW registers with COLORHOST or ALPHAHOST set**; "Segments II"
> refers to remaining cases which have DRAWMODE0 bit LENGTH32 set.

Host-sourced pixels are an access *pattern* available to the framebuffer
generally — lines included — not a span-only feature. DRAWMODE0 bits 6/7 are
documented plainly as "RGB/CI draw source: 0=DDAs; 1=HOSTRW register" and "Alpha
draw source", with no adrmode restriction.

Nothing in §3.5.1, §3.6 (Line Draw Instructions) or the DRAWMODE0 table excludes
lines. So **`unpack` must not fold COLORHOST/ALPHAHOST to 0 for line adrmodes.**
Doing so would silently drop the alpha source from a blended, dithered Gouraud
line — which is what these two shapes are, i.e. GL drawing antialiased or
translucent lines.

## What actually runs them

The generic path handles them correctly and always has: `pixel_draw` fetches the
host pixel on its own terms, and the walker (line/span/block) is orthogonal to
the pixel body. The same is true of a generated LLVM shader — `ConstMode` with
`ALPHAHOST=1` and `ADRMODE=I_LINE` is an ordinary instantiation.

So the dispatch order already does the right thing:

1. generated LLVM shader, if the corpus covered the shape
2. Cranelift — declines these
3. generic `DynMode` path — correct for everything

The only cost of the Cranelift gap is that these shapes miss the *JIT* tier. Once
the corpus records them (it does now — see below) the generator emits real
shaders and they get the fastest tier anyway.

## Why they were missing from the corpus

`save_profile` used to persist `cache.keys()` — only shapes Cranelift
successfully compiled. A shape Cranelift *refused* could therefore never enter
the corpus, so the generator never saw it, so it never got an LLVM shader either.
A rejected shape was permanently stuck on the slowest tier.

Fixed by recording at dispatch (`Rex3::seen_shapes`) rather than at compile, and
by saving in every build rather than only under `rex-jit`.

## MAME is not a reference here

`newport.cpp`'s `do_iline`/`do_fline` take a precomputed `color` argument and
never consult the HOSTRW registers — it models no host-on-line support at all.
Consistent with its divergence from the spec on
[shading](shade-dda-clamp-open-question.md) and
[fastclear](fastclear-cid-divergence.md): MAME is the reference for TLB, not for
the REX3 pixel pipeline.

## If someone wants to close the Cranelift gap

The line emitter would need the host-pixel fetch the span/block emitter already
has. Low value: the LLVM table covers these shapes now, and the generic path is
correct meanwhile. Worth doing only if a workload appears that draws host-sourced
lines in shapes the table does not cover.
