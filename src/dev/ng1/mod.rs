//! Newport graphics (NG1: Indy XL / Indigo2 XL).
//!
//!   rex3.rs          REX3 raster engine: registers, GFIFO, VRAM, drawing
//!   rex3_generic.rs  generic draw path (interpreter, JIT fallback)
//!   rex3_shape.rs    canonical DrawMode decoding
//!   rex3_shaders.rs  generated, precompiled draw functions
//!   rex3_profile.rs  draw-shape corpus persistence
//!   rex3_jit/        Cranelift shader JIT (`rex-jit`)
//!   vc2.rs           VC2 video controller (+ vc2_timings.rs presets)
//!   xmap9.rs         XMAP9 cross-map
//!   cmap.rs          CMAP colour map
//!   bt445.rs         Bt445 RAMDAC

pub mod rex3;
pub mod rex3_generic;
pub mod rex3_shaders;
pub mod rex3_shape;
/// Draw-shape corpus persistence. Deliberately NOT behind `rex-jit`: the corpus
/// records what the guest draws, and the generated shader table serves draws in
/// builds with no Cranelift at all.
pub mod rex3_profile;
#[cfg(feature = "rex-jit")]
pub mod rex3_jit;
pub mod vc2;
pub mod vc2_timings;
pub mod xmap9;
pub mod cmap;
pub mod bt445;
