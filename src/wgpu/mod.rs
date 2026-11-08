//! Host-GPU rasterisation on wgpu.
//!
//! The boards' raster units (IMPACT's RSS engines, the GR2's RE3s, the
//! VPro's raster unit) are replaced by compute kernels, and the board's
//! memory lives in a GPU buffer that nothing on the CPU maps: every read of
//! it is a readback. This module is the part every board shares; the
//! board-specific kernels and the glue that turns its register words into
//! batch records live in `src/dev/<board>/wgpu`. The design is in
//! `clanker/plans/wgpu-raster.md`.
//!
//! The shape of the work:
//!
//! - A **batch** is a run of primitives in submission order with the state
//!   each one was issued under. The CPU side (`batch`) appends a fixed-size
//!   *setup record* per primitive and a *state slot* per distinct state;
//!   a record names its slot. A batch closes when something has to observe
//!   memory (a readback, a frame end), when it is full, or on a time cap.
//! - The **bin** kernel (`shaders/bin.wgsl`) gives every 8x8 tile a bitmask
//!   over the batch's primitives and appends each touched tile to a busy
//!   list. Setting bits commutes, and the bit position is the primitive
//!   index, so the tile's list is in submission order without a sort.
//! - The **raster** kernel (`shaders/raster.wgsl` around the board's
//!   `arch_*` hooks) runs one 64-lane workgroup per busy tile, dispatched
//!   indirectly from the busy count. The workgroup is the tile's only
//!   writer for the batch, walks its mask in order and stores pixels
//!   itself, so primitive order per pixel needs no atomics and no
//!   cross-workgroup synchronisation.
//!
//! `device` opens the adapter and holds the queue; `shader` assembles WGSL
//! from the files on disk (`include_str!`, with `#include`); `tiles` is the
//! tile geometry and the buffer layouts both sides agree on; `memory` is a
//! board's memory on the GPU with uploads and readbacks; `batch` is the
//! recorder and the two dispatches.

pub mod batch;
pub mod device;
pub mod host;
pub mod memory;
pub mod present;
pub mod scanout;
pub mod shader;
pub mod tiles;

#[cfg(test)]
mod tests;

/// Library fragments a board's shaders may `#include`.
pub mod library {
    use super::shader::Fragment;

    /// 64-bit integers on two u32 halves, with a correctly rounded
    /// conversion to f32 (`shaders/i64.wgsl`).
    pub const I64: Fragment = Fragment::new("i64.wgsl", include_str!("shaders/i64.wgsl"));
    /// A correctly rounded f32 division (`shaders/float.wgsl`).
    pub const FLOAT: Fragment = Fragment::new("float.wgsl", include_str!("shaders/float.wgsl"));
}

pub use batch::{ArchShaders, Batch, BatchPipelines, Full};
pub use device::{Gpu, GpuOptions};
pub use memory::{GpuMemory, Staging};
pub use present::{Frame, Placement, Presenter};
pub use scanout::{Scanout, ScanoutPipeline, ScanoutShaders};
pub use shader::{Fragment, Source};
pub use tiles::{TileBox, TileGrid, TILE_H, TILE_W};
