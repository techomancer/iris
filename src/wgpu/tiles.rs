//! Tile geometry and the batch buffer layouts the Rust side and the WGSL
//! side agree on. `shaders/common.wgsl` declares the same structs; the
//! constants reach it through `Source::define`.

use bytemuck::{Pod, Zeroable};

/// A tile is 8x8 pixels: one 64-lane workgroup, one pixel a lane.
pub const TILE_W: u32 = 8;
pub const TILE_H: u32 = 8;
pub const LANES: u32 = TILE_W * TILE_H;

/// Primitives a batch can hold, and the mask words per tile that implies.
pub const MAX_PRIMS: u32 = 4096;
pub const MASK_WORDS: u32 = MAX_PRIMS / 32;

/// wgpu's cap on workgroups along one dispatch axis. The raster dispatch
/// folds the busy count over x and y when it is larger.
pub const MAX_GROUPS: u32 = 65535;

/// The tiles over a target of `width` x `height` pixels at `origin`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGrid {
    pub origin_x: i32,
    pub origin_y: i32,
    pub width: u32,
    pub height: u32,
}

impl TileGrid {
    pub fn new(width: u32, height: u32) -> TileGrid {
        TileGrid { origin_x: 0, origin_y: 0, width, height }
    }

    pub fn cols(&self) -> u32 {
        self.width.div_ceil(TILE_W)
    }

    pub fn rows(&self) -> u32 {
        self.height.div_ceil(TILE_H)
    }

    pub fn tiles(&self) -> u32 {
        self.cols() * self.rows()
    }

    /// The tiles a pixel box `x0..=x1`, `y0..=y1` (inclusive, in target
    /// pixels) touches, or None when it misses the grid.
    pub fn tile_box(&self, x0: i32, y0: i32, x1: i32, y1: i32) -> Option<TileBox> {
        let (w, h) = (self.width as i64, self.height as i64);
        let x0 = (x0 as i64 - self.origin_x as i64).max(0);
        let y0 = (y0 as i64 - self.origin_y as i64).max(0);
        let x1 = (x1 as i64 - self.origin_x as i64).min(w - 1);
        let y1 = (y1 as i64 - self.origin_y as i64).min(h - 1);
        if x0 > x1 || y0 > y1 {
            return None;
        }
        Some(TileBox {
            x0: (x0 / TILE_W as i64) as u16,
            y0: (y0 / TILE_H as i64) as u16,
            x1: (x1 / TILE_W as i64) as u16,
            y1: (y1 / TILE_H as i64) as u16,
        })
    }
}

/// An inclusive box of tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileBox {
    pub x0: u16,
    pub y0: u16,
    pub x1: u16,
    pub y1: u16,
}

impl TileBox {
    pub fn count(&self) -> u32 {
        (self.x1 - self.x0 + 1) as u32 * (self.y1 - self.y0 + 1) as u32
    }

    /// As the bin kernel reads it: x0, y0, x1, y1.
    pub fn packed(&self) -> [u32; 4] {
        [self.x0 as u32, self.y0 as u32, self.x1 as u32, self.y1 as u32]
    }
}

/// `BatchHeader` in common.wgsl: the batch's uniform.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct BatchHeader {
    pub prim_count: u32,
    pub tile_cols: u32,
    pub tile_rows: u32,
    pub mask_words: u32,
    pub origin_x: i32,
    pub origin_y: i32,
    pub _pad: [u32; 2],
}

/// The head of `BusyList` in common.wgsl: the busy count, then the tile
/// indices.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct BusyHead {
    pub count: u32,
    pub _pad: [u32; 3],
}

/// `DispatchArgs` in bin.wgsl: the raster pass's indirect arguments, in
/// a buffer of their own.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DispatchArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub _pad: u32,
}

impl DispatchArgs {
    /// A dispatch of no groups, for the bin pass to raise.
    pub const EMPTY: DispatchArgs = DispatchArgs { x: 0, y: 1, z: 1, _pad: 0 };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_box_clips_to_the_grid() {
        let g = TileGrid::new(1280, 1024);
        assert_eq!(g.cols(), 160);
        assert_eq!(g.rows(), 128);
        assert_eq!(g.tile_box(-5, -5, 7, 7), Some(TileBox { x0: 0, y0: 0, x1: 0, y1: 0 }));
        assert_eq!(g.tile_box(8, 8, 15, 15), Some(TileBox { x0: 1, y0: 1, x1: 1, y1: 1 }));
        assert_eq!(g.tile_box(1279, 1023, 5000, 5000), Some(TileBox { x0: 159, y0: 127, x1: 159, y1: 127 }));
        assert_eq!(g.tile_box(1280, 0, 1300, 10), None);
        assert_eq!(g.tile_box(0, 0, -1, 5), None);
    }
}
