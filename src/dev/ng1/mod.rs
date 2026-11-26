//! Newport graphics (NG1: Indy XL / Indigo2 XL).
//!
//!   rex3.rs          REX3 raster engine: registers, GFIFO, VRAM, drawing
//!   rex3_generic.rs  generic draw path (interpreter, JIT fallback)
//!   rex3_shape.rs    canonical DrawMode decoding
//!   rex3_shaders.rs  generated, precompiled draw functions
//!   rex3_profile.rs  draw-shape corpus persistence
//!   rex3_rec.rs      replayable bus recording (`rex rec`), replay tests
//!   rex3_trace.rs    annotated GFIFO/CPU trace (`rex trace`)
//!                    (both are submodules of rex3.rs, for its private state)
//!   rex3_plane.rs    page-isolated CPU plane shadows for the resident tests
//!   rex3_frame.rs    resident frame cuts, direct presentation, capture
//!                    (submodules of rex3.rs; test-only)
//!   rex3_dirty.rs    per-tile damage tracking of the planes
//!   rex3_flat_fill.rs flat fills the GPU raster paths share (`wgpu`)
//!   rex3_jit/        Cranelift shader JIT (`rex-jit`)
//!   rex3_jit_queue.rs its priority compile queue, free of Cranelift types
//!   wgpu/            the display composed on the host GPU (`wgpu`), and the
//!                    test-only resident raster
//!   vc2.rs           VC2 video controller (+ vc2_timings.rs presets)
//!   xmap9.rs         XMAP9 cross-map
//!   cmap.rs          CMAP colour map
//!   bt445.rs         Bt445 RAMDAC

pub mod rex3;
pub(crate) mod rex3_dirty;
#[cfg(feature = "wgpu")]
pub(crate) mod rex3_flat_fill;
pub mod rex3_generic;
pub mod rex3_shaders;
pub mod rex3_shape;
/// Draw-shape corpus persistence. Deliberately NOT behind `rex-jit`: the corpus
/// records what the guest draws, and the generated shader table serves draws in
/// builds with no Cranelift at all.
pub mod rex3_profile;
/// Priority compile queue for the REX3 JIT. Split out of `rex3_jit` so it
/// carries no Cranelift types: it is compiled (and unit-tested) without the
/// `rex-jit` feature.
#[cfg(any(feature = "rex-jit", test))]
pub mod rex3_jit_queue;
#[cfg(feature = "rex-jit")]
pub mod rex3_jit;
pub mod vc2;
/// The display composed on the host GPU (the wgpu window renderer).
#[cfg(feature = "wgpu")]
// Resident parity tests retain the existing unsafe Box-style plane indexing.
#[cfg_attr(test, allow(dangerous_implicit_autorefs))]
pub mod wgpu;
pub mod vc2_timings;
pub mod xmap9;
pub mod cmap;
pub mod bt445;
