//! Board-independent OpenGL core for the high-level geometry engines: what
//! the GE microcode of every SGI board does the same way. GR4 (IMPACT's
//! GE11) builds on it; GR2 (GE7, `src/dev/gr2/gl.rs`) is where it was copied
//! from and moves onto it next.
//!
//! - `math`: 4x4 matrices (column-major, glLoadMatrix order), the GL matrix
//!   builders and matrix stacks.
//! - `vertex`: the transformed vertex, view-volume and user clip planes,
//!   polygon clipping, the viewport mapping.
//! - `light`: the OpenGL 1.1 lighting equation and fog.
//!
//! Board specifics stay with the boards: how state arrives (FIFO token
//! encodings), the raster engine's input formats, window clipping.

pub mod light;
pub mod math;
pub mod vertex;
