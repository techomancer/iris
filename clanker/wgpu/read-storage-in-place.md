# Read records, slots and tables from their buffers in place: a large struct copied by value gave silently wrong results

Status: 2026-11-06, rule, measured on an RX 6600 XT (RADV, Mesa 26.1.8,
ACO) with the VPro kernels.

## Symptom

The textured-triangle parity test passed, then failed on 10,626 words
(texels about two thirds of the CPU's) after a *fourth arm* was added to
the raster hook's dispatch (`switch recs[pi].kind`), with nothing on the
executed path changed. Any fourth arm did it, a `switch` or an if/else
chain, with or without a new storage binding. `RADV_DEBUG=noopt` and the
shader cache made no difference; the result was deterministic.

## Cause

The texture code copied the 4.8 KB `TexSlot` by value (`let t =
tex_slots[i]`) and passed it by value through three call levels, and
each lane carried a 1 KB `var<private>` log2 table with an initialiser.
Those copies live in per-lane private (scratch) memory. Reading the slot
fields in place (`tex_slots[ti].field`) and moving the table to a small
storage buffer made the test pass with the four arms; converting the
352-byte record and 368-byte slot copies to in-place reads followed as a
precaution. The failing shape's SPIR-V (naga-cli 30.0.0 from the dumped
WGSL) passes `spirv-val --target-env vulkan1.2`, so the fault is in the
driver's handling of that much private memory, or in a naga construct
that validates but means something else; either way the rule below
removes the exposure and is faster.

## Rule

- In a kernel, never bind a large struct to a `let` or pass it by value.
  Pass the buffer index and read `buffer[i].field` where it is used; copy
  only small things (a `TexLevel`, a decoded `Backend`).
- Tables go in storage buffers, not `var<private>` arrays with
  initialisers (every lane initialises its own copy on entry).
- Initialise every function-scope array an unused tap or texel might be
  read from, even where a zero weight discards it: WGSL promises zero,
  but keep the promise visible.
- When a parity test that passed starts failing after an unrelated
  change, suspect private memory first and bisect with the shader alone
  (`IRIS_WGPU_DUMP=<dir>` writes the composed WGSL of every kernel;
  `naga` and `spirv-val` take it from there).
