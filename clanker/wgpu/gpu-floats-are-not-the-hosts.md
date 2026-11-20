# GPU float division is not correctly rounded, and products get fused: guard both before comparing with the CPU model

Status: 2026-11-16, rule, measured on an RX 6600 XT (RADV, Vulkan 1.4) and
an RTX 5070 (Vulkan and D3D12) with the VPro kernels in
`src/dev/odyssey/wgpu`.

## Symptom

The GPU raster unit's depth words differed from the CPU pipeline's by one
in the lowest mantissa bit on a few percent of pixels, under any depth
range, with every other byte of the pixel equal.

## Cause

Two things the host's f32 arithmetic does that a GPU's need not:

- **Division.** Vulkan allows `x / y` 2.5 ULP; RADV computes it as a
  reciprocal and a multiply. `z / d` in the depth ratio came out as
  `z * rcp(d)` (the diagnostic matched that variant in most cases, and
  neither variant where `rcp` itself was a unit off).
- **Contraction.** SPIR-V without `NoContraction` lets the driver fuse
  `a * b + c` into one rounding, naga emits no such decoration, and RADV
  fuses. D3D12 also contracted through the original `x * nc` guard even
  though `nc` came from a storage record: nine parity tests failed, with
  1,914 differing depth words in the flat-triangle test alone.

Add, subtract and multiply are correctly rounded everywhere wgpu runs;
those never differed.

## Rule

- Divide with `div_exact` from `src/wgpu/shaders/float.wgsl`: the two
  mantissas divided as integers, rounded to nearest even, which is what
  the host does. Use it wherever a quotient feeds a value the hashes see
  (the depth ratio, the colour weights' `W_ONE / sum`; later the texture
  unit's divides by Q and the level of detail).
- Keep a product apart from a sum by making its f32 bits observable through
  a one the compiler cannot see. Each board kernel reads `1.0` from its
  record (`Rec::one`); `p(x)` bitcasts `x` and XORs it with
  `bitcast<u32>(nc) ^ 0x3f800000`. The mask is zero at runtime, but the
  integer dependency prevents D3D12 and Vulkan compilers from contracting
  the operation that produced `x` into the following add. A plain
  `p(x) = x * nc` is not a sufficient barrier on D3D12. Any product that
  feeds an add or a subtract, including `1.0 - l1`, needs the bit-observable
  wrap.
- Match the CPU's operation order exactly (the lanes code in
  `pipe/raster.rs` and `pipe/interp.rs` is the reference) and reproduce
  Rust's saturating `as i32` (`sat_i32`) and `as f32` of an i64
  (`i64_to_f32` in `src/wgpu/shaders/i64.wgsl`, correctly rounded) rather
  than trusting the driver's conversions.
- Expect denormals to flush on the GPU; nothing in the pipelines produces
  them.

The parity tests (`cargo test --features wgpu --lib odyssey::wgpu`) are
the check: they compare whole board memories word for word.
