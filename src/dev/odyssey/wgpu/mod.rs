//! The VPro raster unit on the host GPU (`clanker/plans/wgpu-raster.md`,
//! the VPro section).
//!
//! The quad ring's three commands (pipe/ring.rs: a triangle's record and
//! box, a quad the raster thread shaded, a clear's rectangle) become
//! records of a `crate::wgpu::Batch`, and the back-end state a draw's
//! slots name becomes a batch slot: the back-end words as the command FIFO
//! delivered them, the destination's views, the window's clip list and id
//! test, and the polygon stipple. Board memory is a `GpuMemory`; the
//! kernels in `shaders/` store into it with the model's arithmetic
//! (pipe/raster.rs, interp.rs, fixed.rs, backend.rs), and every CPU read of
//! it is a readback after a flush.
//!
//! This is the raster unit alone: `GpuRaster` records, flushes, waits and
//! reads back. With `[graphics] raster = "wgpu"`, `gpu_unit` routes the
//! board's raster and blit operations here; CPU consumers use the VRAM mirror
//! and synchronize the affected blocks before reading them.

#![allow(dead_code, reason = "GPU raster helper entry points")]

pub(crate) mod scanout;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use crate::wgpu::host::HostBuffer;

use bytemuck::{Pod, Zeroable};

use crate::wgpu::shader::Source;
use crate::wgpu::{library, ArchShaders, Batch, BatchPipelines, Fragment, Full, Gpu, GpuMemory, TileGrid};

use super::consts::{dispmode, write, ODSY_DID_MAIN_SHIFT};
use super::field::Codec;
use super::gpu_unit::{BlitKey, BlitView, CopyOp, CopySelect, LineWords};
use super::pipe::backend::{ClearDraw, DestMem};
use super::pipe::ring::BackQuad;
use super::pipe::setup::Tri;
use super::pipe::state::{BackendWords, RasterWords};
use super::pipe::texture;
use super::view::{Kind, Rows, View, MAX_DIM};
use super::vram::{VramPtr, DIRTY_BLOCK_SHIFT};

/// The plane the batch grid covers, per axis, in pixels: records beyond
/// it are dropped and counted.
pub(crate) const PLANE_SPAN: u32 = 2048;

pub(crate) const VRAM: Fragment = Fragment::new("vram.wgsl", include_str!("shaders/vram.wgsl"));
/// The word operations vram.wgsl includes: plain atomics on board memory.
pub(crate) const VRAM_MEM: Fragment = Fragment::new("vram_mem.wgsl", include_str!("shaders/vram_mem.wgsl"));
/// The raster kernel's alternative under the same name: the tile cache,
/// which holds each tile's blocks in workgroup memory for the batch.
const TILE_CACHE: Fragment = Fragment::new("vram_mem.wgsl", include_str!("shaders/tile_cache.wgsl"));

/// Whether the raster kernel keeps each tile's blocks in workgroup memory
/// (`shaders/tile_cache.wgsl`): `VPRO_TILE_CACHE=0` or `1`.
fn tile_cache_default() -> bool {
    std::env::var("VPRO_TILE_CACHE").map(|v| v != "0").unwrap_or(false)
}
/// Whether the bin pass tests a triangle's edges at each tile
/// (`shaders/vpro_bin.wgsl`) or bins it by its box: `VPRO_BIN_EDGES=0`
/// turns the edge test off.
fn bin_edges_default() -> bool {
    std::env::var("VPRO_BIN_EDGES").map(|v| v != "0").unwrap_or(true)
}
pub(crate) const VIEWS: Fragment = Fragment::new("views.wgsl", include_str!("shaders/views.wgsl"));
const BACKEND: Fragment = Fragment::new("backend.wgsl", include_str!("shaders/backend.wgsl"));
const TEXTURE: Fragment = Fragment::new("texture.wgsl", include_str!("shaders/texture.wgsl"));
const VPRO: Fragment = Fragment::new("vpro.wgsl", include_str!("shaders/vpro.wgsl"));
const VPRO_REC: Fragment = Fragment::new("vpro_rec.wgsl", include_str!("shaders/vpro_rec.wgsl"));
const VPRO_BIN: Fragment = Fragment::new("vpro_bin.wgsl", include_str!("shaders/vpro_bin.wgsl"));
const COPY: Fragment = Fragment::new("copy.wgsl", include_str!("shaders/copy.wgsl"));

/// `fixed::log2_q8`'s table: log2(1 + i/256) in 1/256, for the kernels'
/// level of detail (a small buffer of their own).
fn log2_table() -> [u32; 256] {
    std::array::from_fn(|i| ((1.0 + i as f64 / 256.0).log2() * 256.0).round() as u16 as u32)
}

/// Record kinds (`shaders/vpro.wgsl`).
const REC_TRI: u32 = 1;
const REC_FRAG: u32 = 2;
const REC_RECT: u32 = 3;
const REC_BLIT: u32 = 4;
const REC_FRAGS: u32 = 5;
const REC_LINE: u32 = 6;

const TRI_SMOOTH: u32 = 1;
const TRI_TEXTURED: u32 = 2;
const TRI_SPECULAR: u32 = 4;
const LINE_X_MAJOR: u32 = 8;
// Line word offsets (shaders/vpro.wgsl L_*).
const L_LA: usize = 0;
const L_NA: usize = 1;
const L_DU: usize = 2;
const L_DN: usize = 3;
const L_MAJOR: usize = 4;
const L_MINOR: usize = 6;
const L_WIDTH: usize = 8;
const L_INV: usize = 9;
// The kernel's LINE_SUB.
const _: () = assert!(super::pipe::setup::SUB == 256);

/// Payload words of a record.
const REC_WORDS: usize = 84;

// Triangle word offsets.
const T_E0: usize = 0;
const T_DX: usize = 6;
const T_DY: usize = 12;
const T_NEED: usize = 18;
const T_INV_AREA: usize = 19;
const T_Z: usize = 20;
const T_D: usize = 23;
const T_IW: usize = 26;
const T_SLACK: usize = 29;
const T_RGBA12: usize = 32;
const T_FLAT12: usize = 44;
const T_STEPS: usize = 48;
const T_BOX: usize = 49;
const T_ORIGIN: usize = 53;
const T_ALPHA: usize = 55;
const T_MODE: usize = 56;
/// S/W, T/W, R/W, Q/W at the three vertices (TriTex::values), 12 f32.
const T_TEX: usize = 57;
const T_TEXSLOT: usize = 69;
/// The separate specular colours at the vertices (3 x 3) and the
/// provoking one (3), in 12.12.
const T_SPEC12: usize = 70;
const T_FLAT_SPEC: usize = 79;

// Fragment word offsets.
const F_X: usize = 0;
const F_Y: usize = 1;
const F_COV: usize = 2;
const F_ZQ: usize = 3;
const F_RGBA: usize = 7;
// Packed fragment word offsets: the pixels' box, their count, then each
// pixel as x | y << 16, its depth word and its colour in two words.
const P_BOX: usize = 0;
const P_COUNT: usize = 4;
const P_PIXELS: usize = 5;
const P_WORDS: usize = 4;
/// Pixels a packed fragment record holds.
const P_MAX: usize = (REC_WORDS - P_PIXELS) / P_WORDS;

// Rectangle word offsets and flags.
const R_BOX: usize = 0;
const R_COLOUR: usize = 4;
const R_COLOUR_MASK: usize = 5;
const R_DEPTH: usize = 6;
const R_STENCIL: usize = 7;
const R_ID: usize = 8;
// Blit word offsets and flags.
const B_BOX: usize = 0;
const B_OFF: usize = 4;
const B_MASK: usize = 5;
const B_VIEW: usize = 6;
const B_OP: usize = 7;
const B_AND: usize = 8;
const B_OR: usize = 9;
const B_OWN: usize = 10;
const BLIT_WINDOW: u32 = 1;
const BLIT_OWN_VIEW: u32 = 2;
/// The record stores one word, B_OFF itself, at every pixel of its box.
const BLIT_CONST: u32 = 4;

/// Words the stage buffer holds (the pixels of a batch's blits).
const STAGE_WORDS: usize = 1 << 20;

/// Pixels the copy kernel's scratch holds (two words each): a larger copy
/// stays on the CPU.
const COPY_SCRATCH_PIXELS: u32 = 1 << 21;
// Copy flags (shaders/copy.wgsl).
const COPY_MIRROR_X: u32 = 1;
const COPY_MIRROR_Y: u32 = 2;
const COPY_RAW: u32 = 4;
const COPY_FILTER: u32 = 8;
const COPY_SIGNED: u32 = 16;
const COPY_WINDOW: u32 = 32;
const COPY_SELECT_PIXEL: u32 = 64;
const COPY_SELECT_B: u32 = 128;
const COPY_BYTE_MASKS: u32 = 256;
const COPY_SELECT_FRONT: u32 = 512;

/// How long a batch stays open with the board busy before `tick` submits it.
const MAX_OPEN: std::time::Duration = std::time::Duration::from_millis(4);

const CLEAR_COLOUR: u32 = 1;
const CLEAR_DEPTH: u32 = 2;
const CLEAR_STENCIL: u32 = 4;
const CLEAR_TARGET0: u32 = 8;

// Slot flags.
const SLOT_HAS_STENCIL: u32 = 1;
const SLOT_RGBA: u32 = 2;
const SLOT_TO_AUX: u32 = 4;
const SLOT_STIPPLE: u32 = 8;
const SLOT_BACKGROUND: u32 = 16;

// View kinds.
const KIND_NONE: u32 = 0;
const KIND_RGBA8: u32 = 1;
const KIND_Z24: u32 = 2;
const KIND_BYTE: u32 = 3;
/// A texel field: the byte count in bits 15:8, FIELD_FMT in 23:16.
const KIND_TEXEL: u32 = 4;
const KIND_LANES: u32 = 5;

/// A setup record, as `shaders/vpro.wgsl` reads it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct Rec {
    kind: u32,
    slot: u32,
    flags: u32,
    one: u32,
    w: [u32; REC_WORDS],
}

const _: () = assert!(size_of::<Rec>() == 352);

impl Rec {
    fn new(kind: u32, flags: u32) -> Rec {
        Rec { kind, slot: 0, flags, one: 1.0f32.to_bits(), w: [0; REC_WORDS] }
    }

    fn set_i(&mut self, i: usize, v: i32) {
        self.w[i] = v as u32;
    }

    fn set_f(&mut self, i: usize, v: f32) {
        self.w[i] = v.to_bits();
    }

    fn set_i64(&mut self, i: usize, v: i64) {
        self.w[i] = v as u32;
        self.w[i + 1] = (v >> 32) as u32;
    }

    /// A filled triangle: its setup record `t` from a draw whose window
    /// origin is (ox, oy), walked within plane box `bounds` (inclusive;
    /// `walk_bounds` plus the origin), with `steps` of polygon offset,
    /// under depth range and slack `depth`, with the alpha test's words.
    #[allow(clippy::too_many_arguments, reason = "the record's sources")]
    pub fn tri(t: &Tri, ox: i32, oy: i32, bounds: (i32, i32, i32, i32), steps: i64, depth_range: [f64; 2], depth_slack: f64, mode: u32, alpha: u32, textured: bool, specular: bool) -> Rec {
        let mut flags = if t.smooth { TRI_SMOOTH } else { 0 };
        if textured {
            flags |= TRI_TEXTURED;
        }
        if specular {
            flags |= TRI_SPECULAR;
        }
        let mut r = Rec::new(REC_TRI, flags);
        for (a, axis) in t.tex.values().iter().enumerate() {
            for (v, value) in axis.iter().enumerate() {
                r.set_f(T_TEX + 3 * a + v, *value);
            }
        }
        for v in 0..3 {
            for k in 0..3 {
                r.set_i(T_SPEC12 + 3 * v + k, t.spec12[v][k]);
            }
            r.set_i(T_FLAT_SPEC + v, t.flat_spec12[v]);
        }
        let tri = &t.tri;
        for k in 0..3 {
            // The edge values at the centre of plane pixel (0, 0): window
            // pixel (-ox, -oy).
            let e0 = tri.e00[k].wrapping_sub(tri.dx[k].wrapping_mul(ox as i64)).wrapping_sub(tri.dy[k].wrapping_mul(oy as i64));
            r.set_i64(T_E0 + 2 * k, e0);
            r.set_i64(T_DX + 2 * k, tri.dx[k]);
            r.set_i64(T_DY + 2 * k, tri.dy[k]);
        }
        r.w[T_NEED] = (0..3).fold(0, |m, k| m | ((tri.need[k] != 0) as u32) << k);
        r.set_f(T_INV_AREA, tri.inv_area);
        // The planes as depth4_lanes() takes them.
        r.set_f(T_Z, tri.z[0] as f32);
        r.set_f(T_Z + 1, (tri.z[1] - tri.z[0]) as f32);
        r.set_f(T_Z + 2, (tri.z[2] - tri.z[0]) as f32);
        r.set_f(T_D, tri.d[0] as f32);
        r.set_f(T_D + 1, (tri.d[1] - tri.d[0]) as f32);
        r.set_f(T_D + 2, (tri.d[2] - tri.d[0]) as f32);
        for k in 0..3 {
            r.set_f(T_IW + k, t.inv_w[k] as f32);
        }
        let (n, f) = (depth_range[0] as f32, depth_range[1] as f32);
        r.set_f(T_SLACK, depth_slack as f32);
        r.set_f(T_SLACK + 1, n);
        r.set_f(T_SLACK + 2, f - n);
        for v in 0..3 {
            for k in 0..4 {
                r.set_i(T_RGBA12 + 4 * v + k, t.rgba12[v][k]);
            }
        }
        for k in 0..4 {
            r.set_i(T_FLAT12 + k, t.flat12[k]);
        }
        r.set_i(T_STEPS, steps.clamp(-(1 << 25), 1 << 25) as i32);
        r.set_i(T_BOX, bounds.0);
        r.set_i(T_BOX + 1, bounds.1);
        r.set_i(T_BOX + 2, bounds.2);
        r.set_i(T_BOX + 3, bounds.3);
        r.set_i(T_ORIGIN, ox);
        r.set_i(T_ORIGIN + 1, oy);
        r.w[T_ALPHA] = alpha;
        r.w[T_MODE] = mode;
        r
    }

    /// A quad shaded on the CPU (`QuadRing::push_shaded`).
    pub fn frag(q: &BackQuad) -> Rec {
        let mut r = Rec::new(REC_FRAG, 0);
        r.set_i(F_X, q.x);
        r.set_i(F_Y, q.y);
        r.w[F_COV] = q.cov as u32;
        for i in 0..4 {
            r.w[F_ZQ + i] = q.zq[i];
            let c = q.rgba[i];
            r.w[F_RGBA + 2 * i] = c[0] as u32 | (c[1] as u32) << 16;
            r.w[F_RGBA + 2 * i + 1] = c[2] as u32 | (c[3] as u32) << 16;
        }
        r
    }

    /// A clear's rectangle, (x0, y0)-(x1, y1) inclusive in plane pixels
    /// (`QuadRing::push_rect`).
    pub fn rect(x0: i32, y0: i32, x1: i32, y1: i32, c: &ClearDraw) -> Rec {
        let mut flags = 0;
        let mut r = Rec::new(REC_RECT, 0);
        r.set_i(R_BOX, x0);
        r.set_i(R_BOX + 1, y0);
        r.set_i(R_BOX + 2, x1);
        r.set_i(R_BOX + 3, y1);
        if let Some(colour) = c.colour {
            flags |= CLEAR_COLOUR;
            r.w[R_COLOUR] = colour;
        }
        r.w[R_COLOUR_MASK] = c.colour_mask;
        if let Some(d) = c.depth {
            flags |= CLEAR_DEPTH;
            r.w[R_DEPTH] = d;
        }
        if let Some((v, m)) = c.stencil {
            flags |= CLEAR_STENCIL;
            r.w[R_STENCIL] = v as u32 | (m as u32) << 8;
        }
        r.w[R_ID] = c.id_value as u32 | (c.id_mask as u32) << 8;
        for b in 0..2 {
            if c.targets[b] {
                flags |= CLEAR_TARGET0 << b;
            }
        }
        r.flags = flags;
        r
    }

    /// A row run of pixels at (x0, y) to (x1, y) from stage word `off`,
    /// stored as `key` says.
    fn blit(x0: i32, y: i32, x1: i32, off: u32, key: &BlitKey) -> Rec {
        Rec::blit_box(x0, y, x1, y, off, key)
    }

    /// The same over a box of rows (inclusive): a fill, with BLIT_CONST.
    fn blit_box(x0: i32, y0: i32, x1: i32, y1: i32, off: u32, key: &BlitKey) -> Rec {
        let mut flags = if key.window { BLIT_WINDOW } else { 0 };
        let mut r = Rec::new(REC_BLIT, 0);
        r.set_i(B_BOX, x0);
        r.set_i(B_BOX + 1, y0);
        r.set_i(B_BOX + 2, x1);
        r.set_i(B_BOX + 3, y1);
        r.w[B_OFF] = off;
        r.w[B_MASK] = key.mask;
        r.w[B_OP] = key.op;
        r.w[B_AND] = key.and;
        r.w[B_OR] = key.or;
        match key.view {
            BlitView::Slot(v) => r.w[B_VIEW] = v,
            BlitView::Own(v) => {
                flags |= BLIT_OWN_VIEW;
                r.w[B_OWN..B_OWN + 4].copy_from_slice(&v);
            }
        }
        r.flags = flags;
        r
    }

    /// The box a record of this kind touches, from its own words.
    fn pixel_box(&self) -> (i32, i32, i32, i32) {
        match self.kind {
            REC_TRI | REC_LINE => (self.w[T_BOX] as i32, self.w[T_BOX + 1] as i32, self.w[T_BOX + 2] as i32, self.w[T_BOX + 3] as i32),
            REC_FRAG => {
                let (x, y) = (self.w[F_X] as i32, self.w[F_Y] as i32);
                (x, y, x + 1, y + 1)
            }
            // The clear's, the blit's and the packed fragments' boxes sit
            // at the same words.
            _ => (self.w[R_BOX] as i32, self.w[R_BOX + 1] as i32, self.w[R_BOX + 2] as i32, self.w[R_BOX + 3] as i32),
        }
    }
}

/// A state slot, as `shaders/backend.wgsl` reads it: words 0 to 15 the
/// back-end words and flags, then the six views, the clip rectangles, the
/// stipple and the opaque stipple's background.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct Slot {
    words: [u32; 16],
    views: [[u32; 4]; 6],
    clip: [[i32; 4]; 4],
    pattern: [u32; 32],
    bg12: [i32; 4],
}

const _: () = assert!(size_of::<Slot>() == 368);

/// A plane the kernels cannot address yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Unsupported(pub String);

/// A coded field's kind word (`views.wgsl`).
pub(crate) fn codec_kind(c: Codec) -> Result<u32, Unsupported> {
    Ok(match c {
        Codec::Byte => KIND_BYTE,
        Codec::Lanes => KIND_LANES,
        Codec::Texel { bytes, field } => {
            if !matches!(bytes, 1 | 2 | 4) {
                return Err(Unsupported(format!("a {bytes}-byte texel field")));
            }
            KIND_TEXEL | (bytes as u32) << 8 | (field as u32) << 16
        }
    })
}

fn pack_view(v: &View) -> Result<[u32; 4], Unsupported> {
    let kind = match v.kind {
        Kind::None => KIND_NONE,
        Kind::Rgba8 => KIND_RGBA8,
        Kind::Z24 => KIND_Z24,
        Kind::Byte => KIND_BYTE,
        Kind::Coded(c) => codec_kind(c)?,
    };
    let (base, pitch, log2, at) = v.parts();
    if base >= 1 << 32 || pitch >= 1 << 16 || v.width >= 1 << 16 || v.height > MAX_DIM {
        return Err(Unsupported(format!("a view at {base:#x} of pitch {pitch}, {} x {}", v.width, v.height)));
    }
    Ok([base as u32, pitch | (log2 as u32) << 16 | (at as u32) << 24, kind, v.width | v.height << 16])
}

impl Slot {
    /// The slot of a draw under back-end words `w` into `planes`, in index
    /// mode or not, for an overlay context or not; `raster` for the window's
    /// clip list and id test; the polygon stipple's rows where it is on,
    /// and an opaque stipple's background colour.
    pub fn new(w: &BackendWords, planes: &DestMem, index: bool, to_aux: bool, raster: &RasterWords, stipple: Option<&[u32; 32]>, background: Option<[f64; 4]>) -> Result<Slot, Unsupported> {
        let mut flags = 0;
        if planes.stencil.v.kind != Kind::None {
            flags |= SLOT_HAS_STENCIL;
        }
        if !index {
            flags |= SLOT_RGBA;
        }
        if to_aux {
            flags |= SLOT_TO_AUX;
        }
        let mut s = Slot::zeroed();
        s.words[..9].copy_from_slice(&[w.mode, w.depth_func, w.write_mask, w.stencil, w.blend[0], w.blend[1], w.blend[2], w.logic_op, w.draw_mask]);
        s.words[10] = raster.clip_kept;
        s.words[11] = raster.id_compare;
        for (i, r) in [&planes.colour[0], &planes.colour[1], &planes.depth, &planes.stencil, &planes.did, &planes.overlay].iter().enumerate() {
            s.views[i] = pack_view(&r.v)?;
        }
        for (i, c) in raster.clip_rect.iter().enumerate().take(4) {
            s.clip[i] = [c.x0, c.y0, c.x1, c.y1];
        }
        if let Some(rows) = stipple {
            flags |= SLOT_STIPPLE;
            s.pattern = *rows;
            if let Some(bg) = background {
                flags |= SLOT_BACKGROUND;
                s.bg12 = bg.map(super::pipe::fixed::from_f64);
            }
        }
        s.words[9] = flags;
        Ok(s)
    }

    /// The slot of a clear: the planes and the window's id test, no
    /// back-end words.
    pub fn clear(planes: &DestMem, raster: &RasterWords) -> Result<Slot, Unsupported> {
        Slot::new(&BackendWords { write_mask: write::INDEX_MASK, ..Default::default() }, planes, false, false, raster, None, None)
    }

    /// No state at all: for a record that carries its own view and tests
    /// nothing (a DMA put's blits).
    pub fn none() -> Slot {
        Slot::zeroed()
    }

    /// A view the slot holds, for a blit of its own view into a surface
    /// the kernels address (`BlitView::Own`).
    pub fn own_view(rows: &Rows) -> Result<[u32; 4], Unsupported> {
        pack_view(&rows.v)
    }
}

impl Slot {
    /// The byte span view `i` may be written in over plane box `b`
    /// (inclusive): the tiles from the box's first to its last, which the
    /// tile-major layout keeps contiguous. None for a view the slot lacks.
    fn view_span(&self, i: usize, b: (i32, i32, i32, i32)) -> Option<(u64, u64)> {
        let v = self.views[i];
        if v[2] == KIND_NONE {
            return None;
        }
        let (width, height) = ((v[3] & 0xffff) as i32, (v[3] >> 16) as i32);
        let (x0, y0, x1, y1) = (b.0.max(0), b.1.max(0), b.2.min(width - 1), b.3.min(height - 1));
        if x0 > x1 || y0 > y1 {
            return None;
        }
        let (base, pitch, log2) = (v[0] as u64, (v[1] & 0xffff) as u64, (v[1] >> 16) & 0xff);
        let tile = |x: i32, y: i32| (y as u64 >> 4) * pitch + (x as u64 >> 4);
        let lo = base + ((tile(x0, y0) << 8) << log2);
        let hi = base + (((tile(x1, y1) + 1) << 8) << log2);
        Some((lo, hi - lo))
    }
}

/// A copy's operands as `shaders/copy.wgsl` reads them (`CopyOp` there).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CopyUniform {
    src_a: [u32; 4],
    src_b: [u32; 4],
    dst: [u32; 4],
    did: [u32; 4],
    /// width, height, source x, source y
    geo: [i32; 4],
    /// destination x, y, the flags, the display's shown-buffer bits
    geo2: [i32; 4],
    /// clip rectangles kept, the id compare word, the byte mask low and high
    misc: [u32; 4],
    /// the destination plane's width
    misc2: [u32; 4],
    clip: [[i32; 4]; 4],
    /// scale and bias pairs: (sR, bR, sG, bG), (sB, bB, sA, bA)
    sb: [[i32; 4]; 2],
}

const _: () = assert!(size_of::<CopyUniform>() == 224);

impl CopyUniform {
    fn of(op: &CopyOp) -> Result<CopyUniform, Unsupported> {
        let mut flags = 0;
        for (on, f) in [(op.mirror.0, COPY_MIRROR_X), (op.mirror.1, COPY_MIRROR_Y), (op.raw, COPY_RAW), (op.byte_mask.is_some(), COPY_BYTE_MASKS), (op.window.is_some(), COPY_WINDOW)] {
            if on {
                flags |= f;
            }
        }
        let mut did = [0u32; 4];
        let mut displayed = 0;
        match op.select {
            CopySelect::Fixed(0) => {}
            CopySelect::Fixed(_) => flags |= COPY_SELECT_B,
            CopySelect::PerPixel { did: d, displayed: shown, front } => {
                flags |= COPY_SELECT_PIXEL | if front { COPY_SELECT_FRONT } else { 0 };
                did = pack_view(&d)?;
                displayed = shown;
            }
        }
        let mut sb = [[0i32; 4]; 2];
        if let Some((v, signed)) = op.filter {
            flags |= COPY_FILTER | if signed { COPY_SIGNED } else { 0 };
            sb = [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]]];
        }
        let (mut clip, mut clip_kept, mut id_compare, mut width) = ([[0i32; 4]; 4], 0, 0, 0);
        if let Some(w) = op.window {
            (clip, clip_kept, id_compare, width) = (w.clip, w.clip_kept, w.id_compare, w.width);
            did = pack_view(&w.did)?;
        }
        let mask = op.byte_mask.unwrap_or(0);
        Ok(CopyUniform {
            src_a: pack_view(&op.src[0])?,
            src_b: pack_view(&op.src[1])?,
            dst: pack_view(&op.dst)?,
            did,
            geo: [op.size.0 as i32, op.size.1 as i32, op.src_xy.0, op.src_xy.1],
            geo2: [op.dst_xy.0, op.dst_xy.1, flags as i32, displayed as i32],
            misc: [clip_kept, id_compare, mask as u32, (mask >> 32) as u32],
            misc2: [width, 0, 0, 0],
            clip,
            sb,
        })
    }
}

/// The copy kernel (`shaders/copy.wgsl`): its two passes, its operand
/// buffer and scratch, and its groups; the memory group binds the board's
/// memory alone.
struct CopyKernel {
    gather: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    uniform: wgpu::Buffer,
    #[allow(dead_code, reason = "held by the group; kept for its lifetime to be plain")]
    scratch: wgpu::Buffer,
    group: wgpu::BindGroup,
    empty: wgpu::BindGroup,
    vram: wgpu::BindGroup,
}

impl CopyKernel {
    fn new(gpu: &Arc<Gpu>, words: u64, vram: &wgpu::Buffer) -> Result<CopyKernel, String> {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let op_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vpro copy"),
            entries: &[entry(0, wgpu::BufferBindingType::Uniform), entry(1, wgpu::BufferBindingType::Storage { read_only: false })],
        });
        // views.wgsl binds the memory at group 2: group 1 is empty.
        let empty_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("vpro copy none"), entries: &[] });
        let vram_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vpro copy vram"),
            entries: &[entry(0, wgpu::BufferBindingType::Storage { read_only: false })],
        });
        let src = Source::new("vpro copy")
            .define("VRAM_WORD_MASK", (words - 1) as u32)
            .define("MAX_DIM", MAX_DIM)
            .define("DID_MAIN_SHIFT", ODSY_DID_MAIN_SHIFT)
            .define("DISPMODE_COUNT", dispmode::COUNT)
            .library(VRAM)
            .library(VRAM_MEM)
            .library(VIEWS)
            .body(COPY);
        let layouts = [&op_layout, &empty_layout, &vram_layout];
        let gather = src.compute_pipeline(&gpu.device, "gather", &layouts)?;
        let scatter = src.compute_pipeline(&gpu.device, "scatter", &layouts)?;
        let uniform = gpu.buffer("vpro copy op", size_of::<CopyUniform>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let scratch = gpu.buffer("vpro copy scratch", COPY_SCRATCH_PIXELS as u64 * 8, wgpu::BufferUsages::STORAGE);
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vpro copy"),
            layout: &op_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch.as_entire_binding() },
            ],
        });
        let empty = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("vpro copy none"), layout: &empty_layout, entries: &[] });
        let vram = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vpro copy vram"),
            layout: &vram_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: vram.as_entire_binding() }],
        });
        Ok(CopyKernel { gather, scatter, uniform, scratch, group, empty, vram })
    }
}

/// A draw's texture on the GPU (`shaders/texture.wgsl` `TexSlot`): the
/// registers the unit reads at sample time and what the CPU latches from
/// them once (`texture::Latch`), in the kernel's layout.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct TexSlot {
    head: [u32; 44],
    levels: [[u32; 12]; 32],
    detail: [u32; 12],
    colour_table: [u32; 512],
    detail_fn: [u32; 256],
}

const _: () = assert!(size_of::<TexSlot>() == 4 * (44 + 384 + 12 + 512 + 256));

const TEX_LOD: u32 = 1;
const TEX_POST: u32 = 2;
const TEX_STORED_BORDER: u32 = 4;
const TEX_TABLE: u32 = 8;
const TEX_TABLE_ALPHA: u32 = 16;
const NO_CLAMP: i32 = 0x7fff_ffff;

fn pack_level(l: &texture::Level) -> [u32; 12] {
    let bytes = 1u32 << l.bpp;
    [l.addr as u32, l.pitch, l.bpp as u32, l.field as u32, bytes, l.ox, l.oy, l.wl, l.hl, l.dl, l.slice as u32, 0]
}

impl TexSlot {
    pub fn new(r: &texture::TexRegs, l: &texture::Latch) -> TexSlot {
        let mut t = TexSlot::zeroed();
        let mut flags = 0;
        if l.lod {
            flags |= TEX_LOD;
        }
        if l.post {
            flags |= TEX_POST;
        }
        if r.stored_border() {
            flags |= TEX_STORED_BORDER;
        }
        let (table, alpha) = r.colour_table();
        if table {
            flags |= TEX_TABLE;
        }
        if alpha {
            flags |= TEX_TABLE_ALPHA;
        }
        let h = &mut t.head;
        h[0] = r.sampler[0];
        h[1] = r.sampler[4];
        h[2] = r.target().axes() as u32;
        h[3] = flags;
        h[4] = l.count;
        h[5] = l.min_lod as u32;
        h[6] = l.max_lod as u32;
        h[7] = l.mag_below as u32;
        for a in 0..3 {
            h[8 + a] = l.lod_scale[a].to_bits();
            h[12 + a] = l.scale[a].to_bits();
            h[20 + a] = if l.clamp[a] == i64::MAX { NO_CLAMP as u32 } else { l.clamp[a] as i32 as u32 };
        }
        h[16] = l.detail_scale[0].to_bits();
        h[17] = l.detail_scale[1].to_bits();
        h[23] = NO_CLAMP as u32;
        h[24..28].copy_from_slice(&l.border);
        for k in 0..8 {
            h[28 + k] = super::bits::sextract32(r.scale_bias[k], 0, super::consts::tex::SCALE_BIAS_BITS) as u32;
        }
        h[36] = match r.env() {
            texture::Env::Replace => 0,
            texture::Env::Modulate => 1,
            texture::Env::Decal => 2,
            texture::Env::Blend => 3,
            texture::Env::Add => 4,
        };
        let ec = r.env_colour();
        for k in 0..4 {
            h[37 + k] = super::pipe::fixed::field12_full(ec[k]) as u32;
        }
        for i in 0..texture::LEVELS {
            t.levels[i] = pack_level(&l.level[i]);
        }
        t.detail = pack_level(&l.detail);
        t.colour_table[..256].copy_from_slice(&r.colour_table[0]);
        t.colour_table[256..].copy_from_slice(&r.colour_table[1]);
        t.detail_fn = r.detail;
        t
    }

    fn hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for w in bytemuck::cast_slice::<TexSlot, u32>(std::slice::from_ref(self)) {
            h = (h ^ *w as u64).wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
}

/// Texture slots a batch may hold.
const MAX_TEX_SLOTS: u32 = 256;

/// A run of consecutive pixels of one row being collected into one blit.
struct BlitRun {
    slot: Slot,
    key: BlitKey,
    y: i32,
    x0: i32,
    next_x: i32,
    off: u32,
}

/// The raster unit: the batch recorder over the board's memory on the
/// GPU. The front end records in command order and flushes when it would
/// observe memory; `wait` is `Board::raster_wait`'s equivalent.
pub(crate) struct GpuRaster {
    gpu: Arc<Gpu>,
    pipelines: BatchPipelines,
    batch: Batch<Rec, Slot>,
    vram: GpuMemory,
    /// The board's group per batch set (the batch's `next_set`): the
    /// memory and log2 table, shared, and that set's texture slots and blit
    /// stage, in host memory the kernels read directly. The layout and the
    /// table make more when the batch adds a set.
    sets: Vec<RasterSet>,
    layout: wgpu::BindGroupLayout,
    log2_buf: wgpu::Buffer,
    /// The batch's texture slots and their indices by hash.
    tex_slots: Vec<TexSlot>,
    tex_map: std::collections::HashMap<u64, u32>,
    /// The batch's blit pixels and their buffer, and the run of
    /// consecutive pixels being collected (`blit_pixel`): its slot, view,
    /// mask and window flag, its row, first and next x, and where its
    /// pixels start in the stage.
    stage: Vec<u32>,
    run: Option<BlitRun>,
    /// The packed fragment record being collected (`push_frag`) and its
    /// slot.
    frags: Option<(Slot, Rec)>,
    /// A rectangle of a view the front end is storing into the mirror
    /// untracked (a scratch image through the FIFO), to upload when it is
    /// done: the view and its inclusive pixel box.
    region: Option<(super::view::View, i32, i32, i32, i32)>,
    /// The slot last issued in this batch and its index.
    last: Option<(Slot, u32)>,
    pending: Option<wgpu::SubmissionIndex>,
    /// When the open batch took its first record or blit pixel; None while
    /// it is empty. `tick` submits a batch that has been open too long.
    opened: Option<std::time::Instant>,
    /// The command stream position the board's thread had retired at the
    /// last `tick`, and where it is published once everything recorded up
    /// to it is submitted (`DisplayShared::covered`).
    stream_now: u64,
    stream: Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
    /// The CPU's mirror of the memory (`Vram` with its dirty map): what
    /// the front end's own paths read and write. Its dirty blocks go up
    /// before each batch; the blocks the batches wrote come back on
    /// `read_back`.
    mirror: VramPtr,
    /// Blocks (`DIRTY_BLOCK_SHIFT`) the batches since the last readback may
    /// have written, from the records' boxes and their slots' views.
    gpu_dirty: Vec<u64>,
    gpu_dirty_any: bool,
    /// Records whose box missed the grid.
    pub dropped: u64,
    /// Bytes moved each way so far; the uploads split into the mirror's
    /// dirty blocks and the regions writers named.
    pub uploaded: u64,
    pub uploaded_blocks: u64,
    pub uploaded_tiles: u64,
    pub read_back_bytes: u64,
    pub readbacks: u64,
    /// Waits for a submission (`wait`), and batches `tick` submitted
    /// because the board went idle or the batch had been open too long.
    pub waits: u64,
    pub flushes_idle: u64,
    pub flushes_timed: u64,
    /// Measurement only: `read_back` flushes and returns without waiting
    /// or reading, so the front end's paths see a stale mirror and the
    /// GPU path's own time shows; `read_back_now` still does the work.
    pub lazy: bool,
    /// `VPRO_GPU_TRACE_WAITS=1`: readbacks counted by the board function
    /// that caused them (the first `odyssey::` frame outside this module
    /// and the raster unit's plumbing).
    pub waits_by: Option<std::collections::BTreeMap<String, u64>>,
    /// With VPRO_GPU_TRACE_WAITS: batches submitted, by the line that
    /// called `flush`.
    pub flushes_by: Option<std::collections::BTreeMap<u32, u64>>,
    copy: CopyKernel,
    /// Copies the kernel took (`copy`).
    pub copies: u64,
}


/// Slots a batch may hold.
const MAX_SLOTS: u32 = 1024;

/// A batch set's own buffers on the board's side and its group 2.
struct RasterSet {
    tex: HostBuffer,
    stage: HostBuffer,
    group: wgpu::BindGroup,
}

impl RasterSet {
    fn new(gpu: &Arc<Gpu>, layout: &wgpu::BindGroupLayout, vram: &wgpu::Buffer, log2: &wgpu::Buffer, n: usize) -> RasterSet {
        let tex = HostBuffer::new(gpu, &format!("vpro texture slots {n}"), MAX_TEX_SLOTS as u64 * size_of::<TexSlot>() as u64, wgpu::BufferUsages::STORAGE);
        let stage = HostBuffer::new(gpu, &format!("vpro blit stage {n}"), STAGE_WORDS as u64 * 4, wgpu::BufferUsages::STORAGE);
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vpro vram"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: vram.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tex.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: stage.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: log2.as_entire_binding() },
                // The memory again, as the texel reads' plain words.
                wgpu::BindGroupEntry { binding: 4, resource: vram.as_entire_binding() },
            ],
        });
        RasterSet { tex, stage, group }
    }
}

impl GpuRaster {
    /// A raster unit over `vram_bytes` of board memory (a power of two)
    /// whose plane coordinates span `width` x `height` pixels.
    pub fn new(gpu: &Arc<Gpu>, vram_bytes: u64, width: u32, height: u32) -> Result<GpuRaster, String> {
        GpuRaster::with_tile_cache(gpu, vram_bytes, width, height, tile_cache_default())
    }

    /// `new`, choosing whether the raster kernel caches each tile's blocks
    /// in workgroup memory.
    pub fn with_tile_cache(gpu: &Arc<Gpu>, vram_bytes: u64, width: u32, height: u32, tile_cache: bool) -> Result<GpuRaster, String> {
        if !vram_bytes.is_power_of_two() || vram_bytes < 8 {
            return Err(format!("vram size {vram_bytes:#x} is not a power of two"));
        }
        let vram = GpuMemory::new(gpu, "vpro vram", vram_bytes)?;
        let entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vpro vram"),
            entries: &[entry(0, false), entry(1, true), entry(2, true), entry(3, true), entry(4, false)],
        });
        let log2_buf = gpu.buffer("vpro log2 table", 256 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        gpu.write(&log2_buf, 0, bytemuck::cast_slice(&log2_table()));
        let words = vram_bytes / 8;
        let pipelines = BatchPipelines::new(
            gpu,
            &ArchShaders {
                label: "vpro",
                raster: &[VPRO],
                bin_accept: bin_edges_default().then_some(VPRO_BIN),
                library: &[library::I64, library::FLOAT, VPRO_REC, VRAM, if tile_cache { TILE_CACHE } else { VRAM_MEM }, VIEWS, BACKEND, TEXTURE],
                defines: &[("VRAM_WORD_MASK", (words - 1) as u32), ("MAX_DIM", MAX_DIM)],
                layouts: &[&layout],
            },
        )?;
        let batch = Batch::new(gpu, &pipelines, TileGrid::new(width, height), MAX_SLOTS);
        let copy = CopyKernel::new(gpu, words, &vram.buffer)?;
        let blocks = (vram_bytes >> DIRTY_BLOCK_SHIFT).max(1) as usize;
        Ok(GpuRaster {
            gpu: Arc::clone(gpu),
            pipelines,
            batch,
            vram,
            sets: Vec::new(),
            layout,
            log2_buf,
            tex_slots: Vec::new(),
            tex_map: std::collections::HashMap::new(),
            stage: Vec::new(),
            run: None,
            frags: None,
            region: None,
            last: None,
            pending: None,
            opened: None,
            stream_now: 0,
            stream: None,
            mirror: VramPtr::null(),
            gpu_dirty: vec![0; blocks.div_ceil(64)],
            gpu_dirty_any: false,
            dropped: 0,
            uploaded: 0,
            uploaded_blocks: 0,
            uploaded_tiles: 0,
            read_back_bytes: 0,
            readbacks: 0,
            waits: 0,
            flushes_idle: 0,
            flushes_timed: 0,
            lazy: false,
            waits_by: std::env::var("VPRO_GPU_TRACE_WAITS").is_ok().then(std::collections::BTreeMap::new),
            flushes_by: std::env::var("VPRO_GPU_TRACE_WAITS").is_ok().then(std::collections::BTreeMap::new),
            copy,
            copies: 0,
        })
    }

    /// Attribute a readback to the board function that asked for it.
    fn note_wait(&mut self) {
        let Some(by) = &mut self.waits_by else { return };
        let bt = std::backtrace::Backtrace::force_capture().to_string();
        let mut who = String::from("?");
        for line in bt.lines() {
            let Some(i) = line.find("odyssey::") else { continue };
            let name = line[i..].trim_end();
            if name.contains("::wgpu::") || name.contains("raster_unit::") || name.contains("::raster_wait") || name.contains("::gpu_sync") || name.contains("::dest_mem") || name.contains("::run_now") {
                continue;
            }
            who = name.split(" at ").next().unwrap_or(name).to_string();
            break;
        }
        *by.entry(who).or_insert(0) += 1;
    }


    /// Mirror the memory in `m` (a `Vram` with its dirty map on): upload
    /// all of it now, and from here on its dirty blocks before each batch.
    pub fn attach_mirror(&mut self, m: VramPtr) {
        assert_eq!(m.size(), self.vram.size(), "the mirror is the memory's size");
        self.mirror = m;
        self.vram.write(0, bytemuck::cast_slice(m.words_at(0, m.size())));
        self.uploaded += m.size();
        m.clear_dirty();
    }

    /// Note that the batch may write `s`'s views over plane box `b`.
    fn mark_gpu(&mut self, s: &Slot, b: (i32, i32, i32, i32)) {
        let size = self.vram.size();
        for i in 0..6 {
            let Some((lo, len)) = s.view_span(i, b) else { continue };
            if lo + len > size {
                // Past the end (an address that wraps): everything.
                for w in self.gpu_dirty.iter_mut() {
                    *w = u64::MAX;
                }
                self.gpu_dirty_any = true;
                return;
            }
            let (b0, b1) = ((lo >> DIRTY_BLOCK_SHIFT) as usize, ((lo + len - 1) >> DIRTY_BLOCK_SHIFT) as usize);
            for blk in b0..=b1 {
                self.gpu_dirty[blk >> 6] |= 1 << (blk & 63);
            }
            self.gpu_dirty_any = true;
        }
    }

    /// The marked blocks as byte runs, unmarked.
    fn take_gpu_dirty(&mut self) -> Vec<(u64, u64)> {
        let mut runs: Vec<(u64, u64)> = Vec::new();
        let block = 1u64 << DIRTY_BLOCK_SHIFT;
        let blocks = self.vram.size() >> DIRTY_BLOCK_SHIFT;
        for i in 0..self.gpu_dirty.len() {
            let mut bits = std::mem::take(&mut self.gpu_dirty[i]);
            while bits != 0 {
                let b = (i as u64) * 64 + bits.trailing_zeros() as u64;
                bits &= bits - 1;
                if b >= blocks {
                    break;
                }
                let off = b * block;
                match runs.last_mut() {
                    Some((o, l)) if *o + *l == off => *l += block,
                    _ => runs.push((off, block)),
                }
            }
        }
        self.gpu_dirty_any = false;
        runs
    }

    /// Upload the mirror's bytes of view `v` over its pixel box (inclusive)
    /// now, tile row by tile row, ahead of what is submitted next: for a
    /// writer that stored into memory no pending drawing touches (a
    /// texture or scratch upload) through an untracked pointer, so the
    /// upload is the pixels it wrote and not their whole 16 KB blocks,
    /// which may hold GPU-written bytes the mirror has not read back.
    pub fn upload_region(&mut self, v: &super::view::View, x0: i32, y0: i32, x1: i32, y1: i32) {
        // An upload lands ahead of the batch: the pixels recorded so far
        // must be in a batch of their own first.
        self.flush();
        self.upload_tiles(v, x0, y0, x1, y1);
    }

    /// `upload_region`'s copies, with nothing submitted first.
    fn upload_tiles(&mut self, v: &super::view::View, x0: i32, y0: i32, x1: i32, y1: i32) {
        let (x0, y0, x1, y1) = (x0.max(0), y0.max(0), x1.min(v.width as i32 - 1), y1.min(v.height as i32 - 1));
        if v.kind == Kind::None || x0 > x1 || y0 > y1 {
            return;
        }
        let (base, pitch, log2, _) = v.parts();
        let (pitch, log2) = (pitch as u64, log2 as u32);
        let size = self.vram.size();
        let (tx0, tx1) = ((x0 >> 4) as u64, (x1 >> 4) as u64);
        for ty in (y0 >> 4) as u64..=(y1 >> 4) as u64 {
            let lo = base + (((ty * pitch + tx0) << 8) << log2);
            let hi = base + (((ty * pitch + tx1 + 1) << 8) << log2);
            if hi > size {
                // Wrapping addresses: let the block map carry it.
                self.mirror.take_dirty_runs();
                continue;
            }
            self.vram.write(lo, bytemuck::cast_slice(self.mirror.words_at(lo, hi - lo)));
            self.uploaded += hi - lo;
            self.uploaded_tiles += hi - lo;
        }
    }

    /// The end of a run of the board's stores. The mirror's dirty blocks go
    /// up so the scanout, which composes from the GPU's memory, sees what
    /// the CPU paths drew; and the open batch is submitted when the board
    /// is going idle (`idle`: no store waits) or when it has been open for
    /// `MAX_OPEN`, so what the guest drew reaches the screen about when it
    /// would on the board. A busy guest (a GL game) fills batches faster
    /// than that and is not slowed.
    pub fn tick(&mut self, idle: bool, retired: u64) {
        self.stream_now = retired;
        let open = self.opened.is_some_and(|t| idle || t.elapsed() >= MAX_OPEN);
        if open {
            if idle {
                self.flushes_idle += 1;
            } else {
                self.flushes_timed += 1;
            }
            self.flush();
        } else {
            self.upload_dirty();
            if self.opened.is_none() {
                // Nothing recorded since the last submission: it covers
                // everything retired.
                self.publish_covered();
            }
        }
    }

    /// Publish the stream position the submitted work covers through `at`
    /// (`DisplayShared::covered`), from now on.
    pub fn publish_stream(&mut self, at: std::sync::Arc<std::sync::atomic::AtomicU64>) {
        at.store(self.stream_now, std::sync::atomic::Ordering::Release);
        self.stream = Some(at);
    }

    fn publish_covered(&self) {
        if let Some(s) = &self.stream {
            s.store(self.stream_now, std::sync::atomic::Ordering::Release);
        }
    }

    /// Count readbacks by the board function that causes them (`waits_by`)
    /// from now on, or stop and drop the counts.
    pub fn trace_waits(&mut self, on: bool) {
        self.waits_by = on.then(std::collections::BTreeMap::new);
    }

    /// The unit's counters as text for the monitor (`vpro status`, `vpro
    /// gpu`): batches and records, how the batches were submitted, bytes
    /// each way, waits and readbacks, and the readbacks by cause when
    /// traced.
    pub fn report(&self) -> String {
        let (flushes, records) = self.counts();
        let mut out = format!(
            "  gpu unit: {records} records in {flushes} batches ({} on idle, {} timed, {} dropped), {} open now; {} MiB up ({} MiB blocks, {} MiB regions), {} waits, {} readbacks ({} MiB), {} copies\n",
            self.flushes_idle,
            self.flushes_timed,
            self.dropped,
            self.batch.len(),
            self.uploaded >> 20,
            self.uploaded_blocks >> 20,
            self.uploaded_tiles >> 20,
            self.waits,
            self.readbacks,
            self.read_back_bytes >> 20,
            self.copies,
        );
        if let Some(by) = &self.waits_by {
            let mut v: Vec<_> = by.iter().collect();
            v.sort_by(|a, b| b.1.cmp(a.1));
            for (who, n) in v {
                out.push_str(&format!("    {n:>8} readbacks from {who}\n"));
            }
        }
        out
    }

    /// Upload the mirror's dirty blocks, ahead of what is submitted next.
    fn upload_dirty(&mut self) {
        for (off, len) in self.mirror.take_dirty_runs() {
            self.vram.write(off, bytemuck::cast_slice(self.mirror.words_at(off, len)));
            self.uploaded += len;
            self.uploaded_blocks += len;
        }
    }

    /// Submit and wait, then bring every block the batches may have written
    /// back into the mirror: `Board::raster_wait` for the GPU unit. Cheap
    /// when nothing was recorded since the last readback.
    pub fn read_back(&mut self) -> Result<(), String> {
        if self.lazy {
            self.flush();
            return self.gpu.check();
        }
        self.read_back_now()
    }

    /// `read_back` whether lazy or not.
    pub fn read_back_now(&mut self) -> Result<(), String> {
        self.flush();
        if !self.gpu_dirty_any {
            return self.gpu.check();
        }
        self.note_wait();
        let runs = self.take_gpu_dirty();
        let bytes = self.vram.read_ranges(&runs)?;
        self.pending = None;
        let words: &[u64] = bytemuck::cast_slice(&bytes);
        let mut at = 0usize;
        for (off, len) in runs {
            let n = (len >> 3) as usize;
            self.mirror.write_words_untracked(off, &words[at..at + n]);
            at += n;
        }
        self.read_back_bytes += bytes.len() as u64;
        self.readbacks += 1;
        self.gpu.check()
    }

    /// Bring the mirror current over view `v`'s pixel box (inclusive): the
    /// blocks of its tile rows the batches may have written, and only
    /// those. Nothing waits when none of them is marked, which is the
    /// common case for a front end reading pixels one at a time.
    pub fn read_back_view(&mut self, v: &super::view::View, x0: i32, y0: i32, x1: i32, y1: i32) -> Result<(), String> {
        // A blit run still being collected has marked nothing yet: close
        // it, so its blocks count.
        self.end_pending();
        let (x0, y0, x1, y1) = (x0.max(0), y0.max(0), x1.min(v.width as i32 - 1), y1.min(v.height as i32 - 1));
        if v.kind == Kind::None || x0 > x1 || y0 > y1 || !self.gpu_dirty_any {
            return Ok(());
        }
        let (base, pitch, log2, _) = v.parts();
        let (pitch, log2) = (pitch as u64, log2 as u32);
        let size = self.vram.size();
        let (tx0, tx1) = ((x0 >> 4) as u64, (x1 >> 4) as u64);
        // The marked blocks under the box, as runs.
        let mut runs: Vec<(u64, u64)> = Vec::new();
        let block = 1u64 << DIRTY_BLOCK_SHIFT;
        for ty in (y0 >> 4) as u64..=(y1 >> 4) as u64 {
            let lo = base + (((ty * pitch + tx0) << 8) << log2);
            let hi = base + (((ty * pitch + tx1 + 1) << 8) << log2);
            if hi > size {
                // Wrapping addresses: everything.
                return self.read_back_now();
            }
            for b in (lo >> DIRTY_BLOCK_SHIFT)..=((hi - 1) >> DIRTY_BLOCK_SHIFT) {
                let (w, bit) = ((b >> 6) as usize, 1u64 << (b & 63));
                if self.gpu_dirty[w] & bit == 0 {
                    continue;
                }
                self.gpu_dirty[w] &= !bit;
                let off = b * block;
                match runs.last_mut() {
                    Some((o, l)) if *o + *l == off => *l += block,
                    _ => runs.push((off, block)),
                }
            }
        }
        if runs.is_empty() {
            return Ok(());
        }
        // Those blocks were marked by records: submitted or not, they are
        // drawn once the flush below has run.
        self.flush();
        self.note_wait();
        let bytes = self.vram.read_ranges(&runs)?;
        let words: &[u64] = bytemuck::cast_slice(&bytes);
        let mut at = 0usize;
        for (off, len) in &runs {
            let n = (len >> 3) as usize;
            self.mirror.write_words_untracked(*off, &words[at..at + n]);
            at += n;
        }
        self.read_back_bytes += bytes.len() as u64;
        self.readbacks += 1;
        self.gpu_dirty_any = self.gpu_dirty.iter().any(|w| *w != 0);
        self.gpu.check()
    }

    /// The copy `op` (block 0x2031) on the unit: the records before it are
    /// submitted first, then a gather of the source into the scratch and a
    /// scatter into the destination, in order on the queue; nothing waits
    /// and nothing is read back. False when the unit cannot take it (a view
    /// the kernels cannot address, more pixels than the scratch holds): the
    /// CPU then copies on a mirror brought current.
    pub fn copy(&mut self, op: &CopyOp) -> bool {
        let (w, h) = op.size;
        if w == 0 || h == 0 || w as u64 * h as u64 > COPY_SCRATCH_PIXELS as u64 {
            return false;
        }
        let Ok(u) = CopyUniform::of(op) else { return false };
        // What was recorded goes first, and the mirror's changes with it
        // (the source may be the CPU's).
        self.flush();
        self.gpu.write(&self.copy.uniform, 0, bytemuck::bytes_of(&u));
        let mut enc = self.gpu.encoder("vpro copy");
        for p in [&self.copy.gather, &self.copy.scatter] {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("vpro copy"), timestamp_writes: None });
            pass.set_pipeline(p);
            pass.set_bind_group(0, &self.copy.group, &[]);
            pass.set_bind_group(1, &self.copy.empty, &[]);
            pass.set_bind_group(2, &self.copy.vram, &[]);
            pass.dispatch_workgroups((w * h).div_ceil(64), 1, 1);
        }
        self.pending = Some(self.gpu.submit(enc));
        let mut s = Slot::zeroed();
        s.views[0] = u.dst;
        let (x, y) = op.dst_xy;
        self.mark_gpu(&s, (x, y, x.wrapping_add(w as i32 - 1), y.wrapping_add(h as i32 - 1)));
        self.copies += 1;
        true
    }

    // The quad ring's commands (pipe/ring.rs) as records.

    /// A filled triangle (`QuadRing::push_tri`): see `Rec::tri`. With a
    /// texture slot the kernel textures it; `specular` adds the separate
    /// specular colour after.
    #[allow(clippy::too_many_arguments, reason = "the record's sources")]
    pub fn push_tri(&mut self, t: &Tri, ox: i32, oy: i32, bounds: (i32, i32, i32, i32), steps: i64, depth_range: [f64; 2], depth_slack: f64, mode: u32, alpha: u32, slot: &Slot, tex: Option<&TexSlot>, specular: bool) {
        let r = Rec::tri(t, ox, oy, bounds, steps, depth_range, depth_slack, mode, alpha, tex.is_some(), specular);
        self.push_with_tex(r, slot, tex);
    }

    /// An aliased line (`Draw::line_projected`): its triangle (a, b, a)'s
    /// record with the line's own words where a triangle keeps its edges.
    /// `bounds` is the plane box its pixels can reach.
    #[allow(clippy::too_many_arguments, reason = "the record's sources")]
    pub fn push_line(&mut self, t: &Tri, l: &LineWords, ox: i32, oy: i32, bounds: (i32, i32, i32, i32), steps: i64, depth_range: [f64; 2], depth_slack: f64, mode: u32, alpha: u32, slot: &Slot, tex: Option<&TexSlot>, specular: bool) {
        let mut r = Rec::tri(t, ox, oy, bounds, steps, depth_range, depth_slack, mode, alpha, tex.is_some(), specular);
        r.kind = REC_LINE;
        if l.x_major {
            r.flags |= LINE_X_MAJOR;
        }
        r.set_i(L_LA, l.la);
        r.set_i(L_NA, l.na);
        r.set_i(L_DU, l.du);
        r.set_i(L_DN, l.dn);
        r.set_i(L_MAJOR, l.major.0);
        r.set_i(L_MAJOR + 1, l.major.1);
        r.set_i(L_MINOR, l.minor.0);
        r.set_i(L_MINOR + 1, l.minor.1);
        r.set_i(L_WIDTH, l.width);
        r.set_f(L_INV, l.inv);
        self.push_with_tex(r, slot, tex);
    }

    /// The batch's index for texture slot `t`, or Full.
    fn tex_index(&mut self, t: &TexSlot) -> Result<u32, Full> {
        let h = t.hash();
        if let Some(&i) = self.tex_map.get(&h) {
            if self.tex_slots[i as usize] == *t {
                return Ok(i);
            }
        }
        if self.tex_slots.len() as u32 >= MAX_TEX_SLOTS {
            return Err(Full);
        }
        self.tex_slots.push(*t);
        let i = self.tex_slots.len() as u32 - 1;
        self.tex_map.insert(h, i);
        Ok(i)
    }

    /// `push` with the record's texture slot resolved in the same batch.
    fn push_with_tex(&mut self, mut r: Rec, s: &Slot, tex: Option<&TexSlot>) {
        self.end_pending();
        loop {
            let ok = match tex {
                Some(t) => match self.tex_index(t) {
                    Ok(i) => {
                        r.w[T_TEXSLOT] = i;
                        true
                    }
                    Err(Full) => false,
                },
                None => true,
            };
            if ok && self.push_once(r, s) {
                return;
            }
            self.flush();
        }
    }

    /// A quad shaded on the CPU (`QuadRing::push_shaded`). Its covered
    /// pixels join the packed fragment record being collected under the
    /// same slot: a line's pixels, each a quad of its own, take one record
    /// per P_MAX pixels instead of one each.
    pub fn push_frag(&mut self, q: &BackQuad, slot: &Slot) {
        if q.cov == 0b1111 {
            self.push(Rec::frag(q), slot);
            return;
        }
        for i in 0..4 {
            if q.cov & (1 << i) == 0 {
                continue;
            }
            let (x, y) = (q.x + (i & 1) as i32, q.y + (i >> 1) as i32);
            let joins = self.frags.as_ref().is_some_and(|(s, r)| s == slot && (r.w[P_COUNT] as usize) < P_MAX);
            if !joins {
                self.end_pending();
                let mut r = Rec::new(REC_FRAGS, 0);
                r.set_i(P_BOX, x);
                r.set_i(P_BOX + 1, y);
                r.set_i(P_BOX + 2, x);
                r.set_i(P_BOX + 3, y);
                self.frags = Some((*slot, r));
                self.opened.get_or_insert_with(std::time::Instant::now);
            }
            let r = &mut self.frags.as_mut().expect("the record").1;
            let n = r.w[P_COUNT] as usize;
            let at = P_PIXELS + n * P_WORDS;
            let c = q.rgba[i];
            r.w[at] = (x as u32 & 0xffff) | (y as u32) << 16;
            r.w[at + 1] = q.zq[i];
            r.w[at + 2] = c[0] as u32 | (c[1] as u32) << 16;
            r.w[at + 3] = c[2] as u32 | (c[3] as u32) << 16;
            r.w[P_COUNT] = n as u32 + 1;
            r.set_i(P_BOX, (r.w[P_BOX] as i32).min(x));
            r.set_i(P_BOX + 1, (r.w[P_BOX + 1] as i32).min(y));
            r.set_i(P_BOX + 2, (r.w[P_BOX + 2] as i32).max(x));
            r.set_i(P_BOX + 3, (r.w[P_BOX + 3] as i32).max(y));
        }
    }

    /// Close the packed fragment record being collected, if any.
    fn end_frags(&mut self) {
        if let Some((s, r)) = self.frags.take() {
            self.push_record(r, &s);
        }
    }

    /// A clear's rectangle (`QuadRing::push_rect`).
    pub fn push_rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: &ClearDraw, slot: &Slot) {
        self.push(Rec::rect(x0, y0, x1, y1, c), slot);
    }

    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }

    /// The memory buffer and its size in words, for the scanout's own
    /// bind group.
    pub fn vram_buffer(&self) -> (wgpu::Buffer, u64) {
        (self.vram.buffer.clone(), self.vram.size() / 8)
    }

    /// Record `r` under slot `s`, flushing a full batch first.
    pub fn push(&mut self, r: Rec, s: &Slot) {
        self.end_pending();
        self.push_record(r, s);
    }

    /// Close whatever the front end was collecting: the blit run into its
    /// record, the untracked region into its upload.
    fn end_pending(&mut self) {
        self.end_frags();
        self.end_run();
        self.close_region();
    }

    /// One pixel the front end stored into the mirror untracked at (x, y)
    /// of view `v` (a scratch image: memory no record of the batch
    /// touches): the rectangle grows and goes up when it closes, after the
    /// records before it and before any after it.
    pub fn note_region(&mut self, v: &super::view::View, x: i32, y: i32) {
        match &mut self.region {
            Some((rv, x0, y0, x1, y1)) if *rv == *v => {
                *x0 = (*x0).min(x);
                *y0 = (*y0).min(y);
                *x1 = (*x1).max(x);
                *y1 = (*y1).max(y);
            }
            _ => {
                self.end_pending();
                // The records so far go first; the upload lands between
                // their batch and the next.
                self.flush();
                self.region = Some((*v, x, y, x, y));
            }
        }
    }

    /// Upload the region being collected, if any.
    fn close_region(&mut self) {
        if let Some((v, x0, y0, x1, y1)) = self.region.take() {
            self.upload_tiles(&v, x0, y0, x1, y1);
        }
    }

    fn push_record(&mut self, r: Rec, s: &Slot) {
        while !self.push_once(r, s) {
            self.flush();
        }
    }

    /// One pixel of a CPU path that stores pixels in rows (an image put,
    /// an integer primitive, a window-id write): `word` at plane (x, y),
    /// stored as `key` says under slot `s`. Consecutive pixels of a row
    /// with one key join one blit record; anything else the unit does
    /// first closes the run, so the pixels keep their place in the order.
    pub fn blit_pixel(&mut self, s: &Slot, key: &BlitKey, x: i32, y: i32, word: u32) {
        let joins = self.run.as_ref().is_some_and(|r| r.y == y && r.next_x == x && r.key == *key && r.slot == *s);
        if !joins {
            self.end_pending();
            if self.stage.len() + 1 > STAGE_WORDS {
                self.flush();
            }
            self.run = Some(BlitRun { slot: *s, key: *key, y, x0: x, next_x: x, off: self.stage.len() as u32 });
        } else if self.stage.len() + 1 > STAGE_WORDS {
            // The stage is full mid-row: close the run and start the rest
            // of the row in the next batch.
            self.end_run();
            self.flush();
            self.run = Some(BlitRun { slot: *s, key: *key, y, x0: x, next_x: x, off: 0 });
        }
        self.stage.push(word);
        self.run.as_mut().expect("the run").next_x = x + 1;
        self.opened.get_or_insert_with(std::time::Instant::now);
    }

    /// A 2D fill: `word` at every pixel of the inclusive box, stored as
    /// `key` says under slot `s`, in one record and with nothing staged.
    #[allow(clippy::too_many_arguments, reason = "a box, its word, its key and slot")]
    pub fn blit_fill(&mut self, s: &Slot, key: &BlitKey, x0: i32, y0: i32, x1: i32, y1: i32, word: u32) {
        if x0 > x1 || y0 > y1 {
            return;
        }
        let mut rec = Rec::blit_box(x0, y0, x1, y1, word, key);
        rec.flags |= BLIT_CONST;
        self.push(rec, s);
    }

    /// Close the blit run being collected, if any, into its record.
    fn end_run(&mut self) {
        let Some(r) = self.run.take() else { return };
        let rec = Rec::blit(r.x0, r.y, r.next_x - 1, r.off, &r.key);
        // The record and its pixels go in the same batch: the stage is
        // uploaded by the flush that submits them.
        if !self.push_once(rec, &r.slot) {
            // The batch is full of records: the pixels must move to the
            // next batch's stage with the record.
            let words: Vec<u32> = self.stage.drain(r.off as usize..).collect();
            self.flush();
            let rec = Rec::blit(r.x0, r.y, r.next_x - 1, 0, &r.key);
            self.stage.extend_from_slice(&words);
            self.push_record(rec, &r.slot);
        }
    }

    /// Record `r` under slot `s` in the current batch; false when the batch
    /// is full.
    fn push_once(&mut self, mut r: Rec, s: &Slot) -> bool {
        let slot = match &self.last {
            Some((last, i)) if last == s => *i,
            _ => match self.batch.slot(*s) {
                Ok(i) => {
                    self.last = Some((*s, i));
                    i
                }
                Err(Full) => return false,
            },
        };
        r.slot = slot;
        let (x0, y0, x1, y1) = r.pixel_box();
        match self.batch.prim(r, x0, y0, x1, y1) {
            Ok(Some(_)) => {
                self.opened.get_or_insert_with(std::time::Instant::now);
                if r.kind == REC_BLIT && r.flags & BLIT_OWN_VIEW != 0 {
                    let mut own = Slot::zeroed();
                    own.views[0].copy_from_slice(&r.w[B_OWN..B_OWN + 4]);
                    self.mark_gpu(&own, (x0, y0, x1, y1));
                } else {
                    self.mark_gpu(s, (x0, y0, x1, y1));
                }
                true
            }
            Ok(None) => {
                self.dropped += 1;
                true
            }
            Err(Full) => false,
        }
    }

    /// Submit what was recorded, the mirror's changes ahead of it; nothing
    /// waits.
    #[track_caller]
    pub fn flush(&mut self) {
        if let Some(by) = &mut self.flushes_by {
            if !self.batch.is_empty() {
                *by.entry(std::panic::Location::caller().line()).or_insert(0) += 1;
            }
        }
        self.end_pending();
        // The mirror's changes go up whether or not records follow: the
        // scanout reads the GPU's memory directly.
        self.upload_dirty();
        if self.batch.is_empty() {
            self.publish_covered();
            return;
        }
        let k = self.batch.next_set(&self.pipelines);
        while self.sets.len() <= k {
            let set = RasterSet::new(&self.gpu, &self.layout, &self.vram.buffer, &self.log2_buf, self.sets.len());
            self.sets.push(set);
        }
        if !self.stage.is_empty() {
            self.sets[k].stage.write(0, bytemuck::cast_slice(&self.stage));
            self.stage.clear();
        }
        if !self.tex_slots.is_empty() {
            self.sets[k].tex.write(0, bytemuck::cast_slice(&self.tex_slots));
            self.tex_slots.clear();
            self.tex_map.clear();
        }
        if let Some(sub) = self.batch.flush(&self.pipelines, &[&self.sets[k].group]) {
            self.pending = Some(sub);
        }
        self.last = None;
        self.opened = None;
        // Everything recorded from the stores retired by the last tick is
        // now submitted; the display may latch flips armed after them.
        self.publish_covered();
    }

    /// Submit and wait until the GPU has stored everything recorded, so
    /// memory can be read.
    pub fn wait(&mut self) -> Result<(), String> {
        self.flush();
        // Nothing submitted since the last wait: nothing to wait for, and
        // a poll would also wait for the display's scanout.
        let Some(sub) = self.pending.take() else {
            return self.gpu.check();
        };
        self.waits += 1;
        let r = self.gpu.wait(Some(sub));
        self.gpu.check()?;
        r
    }

    /// Batches flushed and records drawn so far.
    pub fn counts(&self) -> (u64, u64) {
        (self.batch.flushes, self.batch.prims_total)
    }

    /// The batch's GPU times, when measured (WGPU_BATCH_TIMING).
    pub fn timing_report(&self) -> Option<String> {
        self.batch.timing.as_ref().map(|t| t.report())
    }

    /// Write `words` at byte offset `offset` of board memory, in the
    /// model's word order (vram.rs: a `Vram`'s words as bytes).
    pub fn write_words(&mut self, offset: u64, words: &[u64]) {
        self.flush();
        self.vram.write(offset, bytemuck::cast_slice(words));
        self.uploaded += 8 * words.len() as u64;
    }

    /// Zero all of board memory (the mirror has been zeroed by the caller).
    pub fn clear_all(&mut self) {
        self.flush();
        self.vram.clear(0, self.vram.size());
        for w in self.gpu_dirty.iter_mut() {
            *w = 0;
        }
        self.gpu_dirty_any = false;
        self.mirror.clear_dirty();
    }

    /// Board memory as the model's words, after a wait.
    pub fn read_words(&mut self) -> Result<Vec<u64>, String> {
        self.wait()?;
        let bytes = self.vram.read_all()?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// blake3 of board memory in address order, the lowest address the
    /// highest byte of each word (`VramPtr::address_bytes`, what
    /// `debug::memory_hash_bytes` hashes).
    pub fn hash(&mut self) -> Result<[u8; 32], String> {
        let words = self.read_words()?;
        let mut h = blake3::Hasher::new();
        let mut buf = Vec::with_capacity(1 << 16);
        for w in words {
            buf.extend_from_slice(&w.to_be_bytes());
            if buf.len() == buf.capacity() {
                h.update(&buf);
                buf.clear();
            }
        }
        h.update(&buf);
        Ok(*h.finalize().as_bytes())
    }
}
