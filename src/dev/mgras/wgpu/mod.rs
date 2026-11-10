//! IMPACT's raster engine on the host GPU (`clanker/plans/wgpu-raster.md`,
//! the IMPACT section).
//!
//! The boundary is the raster FIFO. The CPU keeps what an RSS does with a
//! FIFO entry up to the point of drawing: the register file, `execute()`'s
//! decode, the transfer and character-stipple state machines, the TE's
//! tables and the triangle and GL line setup (`Rss::tri_setup`,
//! `Rss::gl_line_setup`). Where the interpreter would walk spans and store
//! pixels, an `Rss` with a unit attached (`Rss::gpu`) hands the primitive
//! to it as one record: a block fill or point, an X line, a chunk of
//! character stipple, one line of a pixel transfer (its pixels decoded into
//! a staged buffer), a GL triangle, a GL line. The registers the pixel
//! processors and the texture engine read become the record's slot.
//!
//! Nothing walks scanlines. The kernels (`shaders/`) run the common layer's
//! tile pass: one lane a framebuffer pixel, which asks whether its pixel is
//! covered (the triangle's edges at its row, a line's closed-form
//! Bresenham step, a block's rectangle) and then runs the pixel through the
//! model's arithmetic (rss.rs, te1.rs, fixed.rs) function for function. The
//! model walks triangle edges and GL lines in 64-bit integers
//! (`rss::edge_bound`, `rss::gl_line_q0`), so a lane finds its row's span
//! bounds with the interpreter's exact arithmetic.
//!
//! Memory: the pixel pages (`PixMem`: two engines' 12 MB as u64 words,
//! binding 0), the clip-ID planes (binding 1) and TRAM (binding 2) are
//! three buffers. The kernels store into the first two; TRAM is only
//! sampled, and its one writer, the texture loader, runs on the CPU in
//! FIFO order: a load marks TRAM pages, which go up before the next batch,
//! after the batch holding any textured primitive issued before the load
//! is submitted. The board's `Planes` stay as the CPU's mirror, brought
//! current by `sync` for the readers (PIO and DMA reads, hashes, display).
//!
//! One unit draws every scanline: Maximum Impact's second engine is only
//! the memory layout here (`DRBsize` bits 7:6, which the kernels' locate
//! follows), as it is for one CPU engine with `IRIS_MGRAS_ENGINES=1`.

pub(crate) mod scanout;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::wgpu::host::HostBuffer;
use crate::wgpu::{library, ArchShaders, Batch, BatchPipelines, Fragment, Full, Gpu, GpuMemory, TileGrid};

use super::fixed;
use super::pixmem::{self, Buffer, Kind, PixMem, PAGES, PAGE_WORDS, RES};
use super::rss::{self, reg, Block, CidPlanes, GlLineSetup, Rss, Stipple, TriSetup, Xfer, HEIGHT, WIDTH};
use super::te1::{self, reg as te_reg, Te1};

const PIXMEM: Fragment = Fragment::new("pixmem.wgsl", include_str!("shaders/pixmem.wgsl"));
const PP1: Fragment = Fragment::new("pp1.wgsl", include_str!("shaders/pp1.wgsl"));
const TE1: Fragment = Fragment::new("te1.wgsl", include_str!("shaders/te1.wgsl"));
const RSS: Fragment = Fragment::new("rss.wgsl", include_str!("shaders/rss.wgsl"));
const RSS_REC: Fragment = Fragment::new("rss_rec.wgsl", include_str!("shaders/rss_rec.wgsl"));
const RSS_BIN: Fragment = Fragment::new("rss_bin.wgsl", include_str!("shaders/rss_bin.wgsl"));

/// Whether the bin pass tests a GL triangle's spans at each tile
/// (`shaders/rss_bin.wgsl`) or bins it by its box: `MGRAS_BIN_EDGES=0`
/// turns the test off.
fn bin_edges_default() -> bool {
    std::env::var("MGRAS_BIN_EDGES").map(|v| v != "0").unwrap_or(true)
}

// ── records ──────────────────────────────────────────────────────────────────

/// Record kinds (`shaders/rss.wgsl`).
const REC_FILL: u32 = 1;
const REC_LINE: u32 = 2;
const REC_STIPPLE: u32 = 3;
const REC_XFER: u32 = 4;
const REC_TRI: u32 = 5;
const REC_GLLINE: u32 = 6;

/// Payload words of a record.
const REC_WORDS: usize = 76;

/// A setup record as `shaders/rss.wgsl` reads it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct Rec {
    kind: u32,
    slot: u32,
    flags: u32,
    pad: u32,
    w: [u32; REC_WORDS],
}

const _: () = assert!(size_of::<Rec>() == 320);

// Word offsets, as the kernel's defines (`DEFINES`).
// Fill: the framebuffer box (inclusive), the colour `put` takes, or a
// 12:12:12 fast fill's 36-bit word.
const F_BOX: usize = 0;
const F_COLOUR: usize = 4;
const F_V36: usize = 5;
const FILL36: u32 = 1;
// X line: the endpoints in window coordinates, the colour, the stipple
// pattern and the opaque stipple's background.
const L_P0: usize = 0;
const L_P1: usize = 2;
const L_COLOUR: usize = 4;
const L_PATTERN: usize = 5;
const L_BG: usize = 6;
const LINE_STIPPLE: u32 = 1;
const LINE_OPAQUE: u32 = 2;
const LINE_SKIP_LAST: u32 = 4;
// Block rows (character stipple and transfer lines): the block's first
// pixel and its directions, in window coordinates, then the row.
const B_XS: usize = 0;
const B_YS: usize = 1;
const B_DX: usize = 2;
const B_DY: usize = 3;
const B_ROW: usize = 4;
// Stipple: the first column, the bits' count, the bits (high word first),
// the colour and background.
const S_COL: usize = 5;
const S_COUNT: usize = 6;
const S_BITS: usize = 7;
const S_COLOUR: usize = 9;
const S_BG: usize = 10;
const STIPPLE_OPAQUE: u32 = 1;
// Transfer: the pixels' count and their first word in the stage.
const X_COUNT: usize = 5;
const X_OFF: usize = 6;
const XFER_PUT: u32 = 0;
const XFER_FRAG: u32 = 1;
const XFER_DEPTH: u32 = 2;
const XFER_STENCIL: u32 = 3;
const XFER_ALPHA: u32 = 4;
// Triangle (`TriSetup`): the edges (C, B as i64s, low word first: major,
// upper minor, lower minor) and the row from which the upper serves.
const T_EDGES: usize = 0;
const T_JMID: usize = 12;
const T_J0: usize = 18;
const T_J1: usize = 19;
const T_XS: usize = 20;
const T_YREFI: usize = 21;
const T_C: usize = 22;
// Words 34..37 are free (the old fog plane).
const T_Z: usize = 37;
const T_T: usize = 43;
const TRI_LTOR: u32 = 1;
const TRI_TEX: u32 = 2;
const TRI_FOG: u32 = 4;
const TRI_STIPPLE: u32 = 8;
// GL line (`GlLineSetup`): the minor DDA (qk, qs as i64s).
const G_QK: usize = 0;
const G_QS: usize = 2;
const G_P0: usize = 6;
const G_P1: usize = 7;
const G_DIR: usize = 8;
const G_WIDTH: usize = 9;
const G_CBASE: usize = 10;
const G_CSTEP: usize = 14;
// Words 18 and 19 are free (the old fog steps).
const G_ZBASE: usize = 20;
const G_ZSTEP: usize = 22;
const G_TBASE: usize = 24;
const G_TSTEP: usize = 30;
const G_POS: usize = 36;
const G_PATTERN: usize = 37;
const G_REPEAT: usize = 38;
const GL_XMAJOR: u32 = 1;
const GL_TEX: u32 = 2;
const GL_FOG: u32 = 4;
const GL_STIPPLE: u32 = 8;

impl Rec {
    fn new(kind: u32, flags: u32) -> Rec {
        Rec { kind, slot: 0, flags, pad: 0, w: [0; REC_WORDS] }
    }

    fn set_i(&mut self, i: usize, v: i32) {
        self.w[i] = v as u32;
    }

    fn set_i64(&mut self, i: usize, v: i64) {
        self.w[i] = v as u32;
        self.w[i + 1] = (v >> 32) as u32;
    }

}

// ── slots ────────────────────────────────────────────────────────────────────

/// Slot word offsets: the registers the pixel processors and the texture
/// environment read, then the sampler as `Te1::sampler` resolves it.
const SL_PP1: usize = 0;
const SL_FILL: usize = 1;
const SL_LSBA: usize = 2;
const SL_LSBB: usize = 3;
const SL_MSBS: usize = 4;
const SL_DRB: usize = 5;
const SL_DRBSIZE: usize = 6;
const SL_XYWIN: usize = 7;
const SL_CONFIG: usize = 8;
const SL_CLIPMODE: usize = 9;
/// Screen mask n: x at SL_CLIP + 2n, y at SL_CLIP + 2n + 1.
const SL_CLIP: usize = 10;
const SL_WINMODE: usize = 18;
const SL_ZMODE: usize = 19;
const SL_STMODE: usize = 20;
const SL_STMASK: usize = 21;
const SL_AFUNC: usize = 22;
const SL_BLEND: usize = 23;
const SL_BLENDRG: usize = 24;
const SL_BLENDBA: usize = 25;
const SL_FOGRG: usize = 26;
const SL_FOGB: usize = 27;
const SL_TM1: usize = 28;
const SL_TXENVRG: usize = 29;
const SL_TXENVB: usize = 30;
const SL_XFERZ: usize = 31;
const SL_MODE2: usize = 32;
const SL_LS: usize = 33;
const SL_LT: usize = 34;
const SL_MAXLEVEL: usize = 35;
const SL_NC: usize = 36;
const SL_SEL: usize = 37;
const SL_D: usize = 38;
const SL_BORDER: usize = 39;
const SL_PAGES: usize = 48;
const SL_BPAGES: usize = 64;
const SL_STIPPLE: usize = 80;
/// The detail texture (`te1::Detail`): on, page, log2 N, shift S, shift T,
/// the TXLOD cutoff index, then its 32 weights at SL_DTABLE.
const SL_DETAIL: usize = 112;
const SL_DTABLE: usize = 120;
/// The fog table (`rss::reg::FOG_LUT`) and its index's scale and offset
/// (FOG_IDX_K, FOG_IDX_B), when fog is on.
const SL_FOGLUT: usize = 152;
const SL_FOGIK: usize = SL_FOGLUT + super::rss::FOG_LUT_WORDS;
const SL_FOGIB: usize = SL_FOGIK + 1;
const SLOT_WORDS: usize = 284;

/// A state slot, as the kernels read it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct Slot {
    w: [u32; SLOT_WORDS],
}

impl Slot {
    /// The registers as they stand, with the sampler where `tex` and the
    /// polygon stipple's rows where `stipple`.
    fn of(r: &Rss, tex: bool, stipple: bool) -> Slot {
        let mut s = Slot::zeroed();
        let w = &mut s.w;
        for (k, n) in [
            (SL_PP1, reg::PP1FILLMODE),
            (SL_FILL, reg::FILLMODE),
            (SL_LSBA, reg::COLORMASKLSBSA),
            (SL_LSBB, reg::COLORMASKLSBSB),
            (SL_MSBS, reg::COLORMASKMSBS),
            (SL_DRB, reg::DRBPOINTERS),
            (SL_DRBSIZE, reg::DRBSIZE),
            (SL_XYWIN, reg::XYWIN),
            (SL_CONFIG, reg::CONFIG),
            (SL_CLIPMODE, reg::CLIP_MODE),
            (SL_WINMODE, reg::PP1WINMODE),
            (SL_ZMODE, reg::ZMODE),
            (SL_STMODE, reg::STENCILMODE),
            (SL_STMASK, reg::STENCILMASK),
            (SL_AFUNC, reg::AFUNCMODE),
            (SL_BLEND, reg::BLENDFACTOR),
            (SL_BLENDRG, reg::BLENDCOLOR_RG),
            (SL_BLENDBA, reg::BLENDCOLOR_BA),
            (SL_FOGRG, reg::FOG_RG),
            (SL_FOGB, reg::FOG_B),
            (SL_XFERZ, reg::XFER_Z),
        ] {
            w[k] = r.reg(n);
        }
        for n in 0..4 {
            w[SL_CLIP + 2 * n] = r.reg(reg::CLIP_X + 2 * n as u32);
            w[SL_CLIP + 2 * n + 1] = r.reg(reg::CLIP_Y + 2 * n as u32);
        }
        if tex {
            w[SL_TM1] = r.reg(te_reg::TEXMODE1);
            w[SL_TXENVRG] = r.reg(te_reg::TXENV_RG);
            w[SL_TXENVB] = r.reg(te_reg::TXENV_B);
            let smp = r.sampler();
            w[SL_MODE2] = smp.mode2;
            w[SL_LS] = smp.ls;
            w[SL_LT] = smp.lt;
            w[SL_MAXLEVEL] = smp.max_level;
            w[SL_NC] = smp.nc;
            w[SL_SEL] = smp.sel;
            w[SL_D] = smp.d;
            w[SL_BORDER..SL_BORDER + 4].copy_from_slice(&smp.border_color);
            w[SL_PAGES..SL_PAGES + 16].copy_from_slice(&smp.pages);
            w[SL_BPAGES..SL_BPAGES + 16].copy_from_slice(&smp.border_pages);
            let d = &smp.detail;
            w[SL_DETAIL..SL_DETAIL + 6].copy_from_slice(&[d.on, d.page, d.log, d.shift_s, d.shift_t, d.last]);
            w[SL_DTABLE..SL_DTABLE + 32].copy_from_slice(&d.table);
        }
        if r.reg(reg::FOG_ON) != 0 {
            for n in 0..super::rss::FOG_LUT_WORDS {
                w[SL_FOGLUT + n] = r.reg(reg::FOG_LUT + n as u32);
            }
            w[SL_FOGIK] = r.reg(reg::FOG_IDX_K);
            w[SL_FOGIB] = r.reg(reg::FOG_IDX_B);
        }
        if stipple {
            for j in 0..32 {
                w[SL_STIPPLE + j] = r.poly_stipple_row(j as u32);
            }
        }
        s
    }
}

/// Offsets both sides agree on, as `const` defines in the kernels.
const DEFINES: &[(&str, u32)] = &[
    ("REC_WORDS", REC_WORDS as u32),
    ("REC_FILL", REC_FILL),
    ("REC_LINE", REC_LINE),
    ("REC_STIPPLE", REC_STIPPLE),
    ("REC_XFER", REC_XFER),
    ("REC_TRI", REC_TRI),
    ("REC_GLLINE", REC_GLLINE),
    ("F_BOX", F_BOX as u32),
    ("F_COLOUR", F_COLOUR as u32),
    ("F_V36", F_V36 as u32),
    ("FILL36", FILL36),
    ("L_P0", L_P0 as u32),
    ("L_P1", L_P1 as u32),
    ("L_COLOUR", L_COLOUR as u32),
    ("L_PATTERN", L_PATTERN as u32),
    ("L_BG", L_BG as u32),
    ("LINE_STIPPLE", LINE_STIPPLE),
    ("LINE_OPAQUE", LINE_OPAQUE),
    ("LINE_SKIP_LAST", LINE_SKIP_LAST),
    ("B_XS", B_XS as u32),
    ("B_YS", B_YS as u32),
    ("B_DX", B_DX as u32),
    ("B_DY", B_DY as u32),
    ("B_ROW", B_ROW as u32),
    ("S_COL", S_COL as u32),
    ("S_COUNT", S_COUNT as u32),
    ("S_BITS", S_BITS as u32),
    ("S_COLOUR", S_COLOUR as u32),
    ("S_BG", S_BG as u32),
    ("STIPPLE_OPAQUE", STIPPLE_OPAQUE),
    ("X_COUNT", X_COUNT as u32),
    ("X_OFF", X_OFF as u32),
    ("XFER_PUT", XFER_PUT),
    ("XFER_FRAG", XFER_FRAG),
    ("XFER_DEPTH", XFER_DEPTH),
    ("XFER_STENCIL", XFER_STENCIL),
    ("XFER_ALPHA", XFER_ALPHA),
    ("T_EDGES", T_EDGES as u32),
    ("T_JMID", T_JMID as u32),
    ("T_J0", T_J0 as u32),
    ("T_J1", T_J1 as u32),
    ("T_XS", T_XS as u32),
    ("T_YREFI", T_YREFI as u32),
    ("T_C", T_C as u32),
    ("T_Z", T_Z as u32),
    ("T_T", T_T as u32),
    ("TRI_LTOR", TRI_LTOR),
    ("TRI_TEX", TRI_TEX),
    ("TRI_FOG", TRI_FOG),
    ("TRI_STIPPLE", TRI_STIPPLE),
    ("G_QK", G_QK as u32),
    ("G_QS", G_QS as u32),
    ("G_P0", G_P0 as u32),
    ("G_P1", G_P1 as u32),
    ("G_DIR", G_DIR as u32),
    ("G_WIDTH", G_WIDTH as u32),
    ("G_CBASE", G_CBASE as u32),
    ("G_CSTEP", G_CSTEP as u32),
    ("G_ZBASE", G_ZBASE as u32),
    ("G_ZSTEP", G_ZSTEP as u32),
    ("G_TBASE", G_TBASE as u32),
    ("G_TSTEP", G_TSTEP as u32),
    ("G_POS", G_POS as u32),
    ("G_PATTERN", G_PATTERN as u32),
    ("G_REPEAT", G_REPEAT as u32),
    ("GL_XMAJOR", GL_XMAJOR),
    ("GL_TEX", GL_TEX),
    ("GL_FOG", GL_FOG),
    ("GL_STIPPLE", GL_STIPPLE),
    ("SLOT_WORDS", SLOT_WORDS as u32),
    ("SL_PP1", SL_PP1 as u32),
    ("SL_FILL", SL_FILL as u32),
    ("SL_LSBA", SL_LSBA as u32),
    ("SL_LSBB", SL_LSBB as u32),
    ("SL_MSBS", SL_MSBS as u32),
    ("SL_DRB", SL_DRB as u32),
    ("SL_DRBSIZE", SL_DRBSIZE as u32),
    ("SL_XYWIN", SL_XYWIN as u32),
    ("SL_CONFIG", SL_CONFIG as u32),
    ("SL_CLIPMODE", SL_CLIPMODE as u32),
    ("SL_CLIP", SL_CLIP as u32),
    ("SL_WINMODE", SL_WINMODE as u32),
    ("SL_ZMODE", SL_ZMODE as u32),
    ("SL_STMODE", SL_STMODE as u32),
    ("SL_STMASK", SL_STMASK as u32),
    ("SL_AFUNC", SL_AFUNC as u32),
    ("SL_BLEND", SL_BLEND as u32),
    ("SL_BLENDRG", SL_BLENDRG as u32),
    ("SL_BLENDBA", SL_BLENDBA as u32),
    ("SL_FOGRG", SL_FOGRG as u32),
    ("SL_FOGB", SL_FOGB as u32),
    ("SL_TM1", SL_TM1 as u32),
    ("SL_TXENVRG", SL_TXENVRG as u32),
    ("SL_TXENVB", SL_TXENVB as u32),
    ("SL_XFERZ", SL_XFERZ as u32),
    ("SL_MODE2", SL_MODE2 as u32),
    ("SL_LS", SL_LS as u32),
    ("SL_LT", SL_LT as u32),
    ("SL_MAXLEVEL", SL_MAXLEVEL as u32),
    ("SL_NC", SL_NC as u32),
    ("SL_SEL", SL_SEL as u32),
    ("SL_D", SL_D as u32),
    ("SL_BORDER", SL_BORDER as u32),
    ("SL_PAGES", SL_PAGES as u32),
    ("SL_BPAGES", SL_BPAGES as u32),
    ("SL_STIPPLE", SL_STIPPLE as u32),
    ("SL_DETAIL", SL_DETAIL as u32),
    ("SL_DTABLE", SL_DTABLE as u32),
    ("SL_FOGLUT", SL_FOGLUT as u32),
    ("SL_FOGIK", SL_FOGIK as u32),
    ("SL_FOGIB", SL_FOGIB as u32),
    ("FB_W", WIDTH as u32),
    ("FB_H", HEIGHT as u32),
    ("PAGES", PAGES as u32),
    ("PAGE_WORDS", PAGE_WORDS as u32),
    ("PIX_TILE_W", pixmem::TILE_W as u32),
    ("PIX_TILE_H", pixmem::TILE_H as u32),
    ("OVERLAY_TILE_W", pixmem::OVERLAY_TILE_W as u32),
    ("RE_WORDS", pixmem::RE_WORDS as u32),
    ("TRAM_NIBBLES", te1::TRAM_NIBBLES as u32),
    ("TRAM_PAGE_NIBBLES", te1::PAGE_NIBBLES as u32),
    ("ZST_PAGE", rss::ZST_PAGE),
];

// ── the unit ────────────────────────────────────────────────────────────────

/// Words the transfer stage holds (a batch's transfer pixels).
const STAGE_WORDS: usize = 1 << 20;
/// Slots a batch may hold.
const MAX_SLOTS: u32 = 1024;
/// RECIP (1024 words), then log2 in Q8 (256), then 2^x in Q8 (256,
/// `te1::exp2_q8`).
const TABLE_WORDS: usize = 1024 + 256 + 256;
/// TRAM pages, as the dirty map counts them (`Te1::dirty`).
const TRAM_PAGES: usize = te1::TRAM_BYTES / (te1::PAGE_NIBBLES / 2);

/// A batch set's own buffers on the board's side and its group 2.
struct RasterSet {
    stage: HostBuffer,
    group: wgpu::BindGroup,
}

/// The unit: the batch recorder over the board's memories on the GPU.
pub(crate) struct GpuRaster {
    gpu: Arc<Gpu>,
    pipelines: BatchPipelines,
    batch: Batch<Rec, Slot>,
    pix: GpuMemory,
    cid: GpuMemory,
    tram: GpuMemory,
    tables: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    sets: Vec<RasterSet>,
    /// The batch's transfer pixels.
    stage: Vec<u32>,
    /// The slot last issued in this batch and its index.
    last: Option<(Slot, u32)>,
    /// The open batch holds a textured primitive (a TRAM write must submit
    /// it first).
    textured: bool,
    pending: Option<wgpu::SubmissionIndex>,
    /// Pages (engine-major, `RES * PAGES`) and clip-ID rows the batches
    /// since the last `sync` may have written.
    dirty_pages: Vec<u64>,
    dirty_cid: Vec<u64>,
    /// TRAM pages loaded since the last upload.
    tram_dirty: [u64; TRAM_PAGES / 64],
    /// Records whose box missed the framebuffer.
    pub dropped: u64,
    pub records: u64,
    pub uploaded: u64,
    pub read_back: u64,
    pub syncs: u64,
}

fn log2_q8_table() -> [u32; 256] {
    std::array::from_fn(|i| fixed::log2_table()[i] as u32)
}

impl GpuRaster {
    pub fn new(gpu: &Arc<Gpu>) -> Result<GpuRaster, String> {
        let pix = GpuMemory::new(gpu, "mgras pixel pages", (RES * pixmem::RE_WORDS * 8) as u64)?;
        let cid = GpuMemory::new(gpu, "mgras clip ids", (WIDTH * HEIGHT) as u64)?;
        let tram = GpuMemory::new(gpu, "mgras tram", te1::TRAM_BYTES as u64)?;
        let tables = gpu.buffer("mgras tables", (TABLE_WORDS * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let mut t: Vec<u32> = fixed::RECIP.to_vec();
        t.extend_from_slice(&log2_q8_table());
        t.extend_from_slice(te1::exp2_q8());
        gpu.write(&tables, 0, bytemuck::cast_slice(&t));
        let entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mgras memory"),
            entries: &[entry(0, false), entry(1, false), entry(2, true), entry(3, true), entry(4, true)],
        });
        let pipelines = BatchPipelines::new(
            gpu,
            &ArchShaders {
                label: "mgras",
                raster: &[RSS],
                bin_accept: bin_edges_default().then_some(RSS_BIN),
                library: &[library::I64, library::WIDE, PIXMEM, PP1, TE1, RSS_REC],
                defines: DEFINES,
                layouts: &[&layout],
            },
        )?;
        let batch = Batch::new(gpu, &pipelines, TileGrid::new(WIDTH as u32, HEIGHT as u32), MAX_SLOTS);
        Ok(GpuRaster {
            gpu: Arc::clone(gpu),
            pipelines,
            batch,
            pix,
            cid,
            tram,
            tables,
            layout,
            sets: Vec::new(),
            stage: Vec::new(),
            last: None,
            textured: false,
            pending: None,
            dirty_pages: vec![0; (RES * PAGES).div_ceil(64)],
            dirty_cid: vec![0; HEIGHT.div_ceil(64)],
            tram_dirty: [0; TRAM_PAGES / 64],
            dropped: 0,
            records: 0,
            uploaded: 0,
            read_back: 0,
            syncs: 0,
        })
    }

    fn new_set(&self, n: usize) -> RasterSet {
        let stage = HostBuffer::new(&self.gpu, &format!("mgras stage {n}"), (STAGE_WORDS * 4) as u64, wgpu::BufferUsages::STORAGE);
        let group = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mgras memory"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.pix.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.cid.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.tram.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: stage.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.tables.as_entire_binding() },
            ],
        });
        RasterSet { stage, group }
    }

    /// Take the board's planes as the memories' contents: everything goes
    /// up, and the planes become the mirror `sync` keeps current.
    pub fn attach(&mut self, mem: &PixMem, cid: &CidPlanes, te: &mut Te1) {
        self.flush();
        self.write_pix(0, &mem.words);
        self.cid.write(0, &cid[..]);
        self.tram.write(0, te.tram());
        self.uploaded += cid.len() as u64 + te1::TRAM_BYTES as u64;
        self.tram_dirty = [0; TRAM_PAGES / 64];
        te.dirty = [0; 4];
    }

    /// Write pixel memory words at word `at` (after what was recorded).
    pub fn write_pix(&mut self, at: usize, words: &[u64]) {
        self.flush();
        self.pix.write(at as u64 * 8, bytemuck::cast_slice(words));
        self.uploaded += words.len() as u64 * 8;
    }

    /// Write clip IDs at byte `at`.
    pub fn write_cid(&mut self, at: usize, bytes: &[u8]) {
        self.flush();
        self.cid.write(at as u64, bytes);
        self.uploaded += bytes.len() as u64;
    }

    /// Before the texture engine changes TRAM: primitives recorded so far
    /// sample what it holds now, so a batch holding a textured one goes
    /// first (`te` is the TRAM as they see it; its pages loaded since the
    /// last upload go up with that batch).
    pub fn before_tram_write(&mut self, te: &mut Te1) {
        if self.textured {
            self.take_tram(te);
            self.flush();
        }
    }

    /// TRAM pages the texture engine loaded since the last look, into the
    /// unit's map; they go up before the next batch.
    fn take_tram(&mut self, te: &mut Te1) {
        for (d, s) in self.tram_dirty.iter_mut().zip(te.dirty.iter_mut()) {
            *d |= std::mem::take(s);
        }
    }

    /// The loaded TRAM pages, up.
    fn upload_tram(&mut self, te: &Te1) {
        let page = te1::PAGE_NIBBLES / 2;
        for i in 0..self.tram_dirty.len() {
            let mut bits = std::mem::take(&mut self.tram_dirty[i]);
            while bits != 0 {
                let p = i * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                self.tram.write((p * page) as u64, &te.tram()[p * page..(p + 1) * page]);
                self.uploaded += page as u64;
            }
        }
    }

    /// Record `r` (covering the framebuffer box) under slot `s`; `staged`
    /// are its pixels for the stage (a transfer line's), `te` the TRAM a
    /// textured primitive samples.
    fn push(&mut self, mut r: Rec, s: &Slot, b: (i32, i32, i32, i32), staged: &[u32], te: Option<&mut Te1>) {
        if b.0 > b.2 || b.1 > b.3 || b.2 < 0 || b.3 < 0 || b.0 >= WIDTH as i32 || b.1 >= HEIGHT as i32 {
            self.dropped += 1;
            return;
        }
        if let Some(te) = te {
            // A texture is about to be sampled: the TRAM it was loaded into
            // must be up before this batch runs.
            self.take_tram(te);
            if self.tram_dirty.iter().any(|&w| w != 0) {
                self.upload_tram(te);
            }
            self.textured = true;
        }
        if self.stage.len() + staged.len() > STAGE_WORDS {
            self.flush();
        }
        for _ in 0..2 {
            if !staged.is_empty() {
                r.w[X_OFF] = self.stage.len() as u32;
            }
            if self.push_once(r, s, b) {
                self.stage.extend_from_slice(staged);
                return;
            }
            self.flush();
        }
        panic!("mgras gpu: a record does not fit an empty batch");
    }

    fn push_once(&mut self, mut r: Rec, s: &Slot, b: (i32, i32, i32, i32)) -> bool {
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
        match self.batch.prim(r, b.0, b.1, b.2, b.3) {
            Ok(Some(_)) => {
                self.records += 1;
                self.mark(s, b, r.kind);
                true
            }
            Ok(None) => {
                self.dropped += 1;
                true
            }
            Err(Full) => false,
        }
    }

    /// Every page (engine-major index) of buffer `buf` under the
    /// framebuffer box `x0..=x1`, `y0..=y1` (clamped to the framebuffer).
    fn buffer_pages(buf: &Buffer, b: (i32, i32, i32, i32), mut f: impl FnMut(usize)) {
        let (x0, y0, x1, y1) = (b.0.max(0) as u32, b.1.max(0) as u32, b.2.min(WIDTH as i32 - 1), b.3.min(HEIGHT as i32 - 1));
        if x1 < x0 as i32 || y1 < y0 as i32 {
            return;
        }
        let (x1, y1) = (x1 as u32, y1 as u32);
        let tile_w = if buf.kind == Kind::Wide { pixmem::TILE_W } else { pixmem::OVERLAY_TILE_W } as u32;
        let (r0, r1) = ((y0 >> buf.yshift) / pixmem::TILE_H as u32, (y1 >> buf.yshift) / pixmem::TILE_H as u32);
        for e in 0..(1usize << buf.yshift) {
            for pr in r0..=r1 {
                for pc in x0 / tile_w..=x1 / tile_w {
                    f(e * PAGES + (buf.ptr + pr * buf.xtiles + pc) as usize % PAGES);
                }
            }
        }
    }

    /// Bring the mirror `mem` current under framebuffer box `b` of buffers
    /// `bufs` (a transfer's read): the pages there that drawing marked are
    /// read back, after a wait; the others stay marked. Nothing waits when
    /// none is marked.
    pub fn sync_box(&mut self, mem: &mut PixMem, bufs: &[Buffer], b: (i32, i32, i32, i32)) -> Result<(), String> {
        let mut pages = Vec::new();
        for buf in bufs {
            let dirty = &mut self.dirty_pages;
            Self::buffer_pages(buf, b, |p| {
                if dirty[p / 64] & 1 << (p % 64) != 0 {
                    dirty[p / 64] &= !(1 << (p % 64));
                    pages.push(p);
                }
            });
        }
        if pages.is_empty() {
            return Ok(());
        }
        self.wait()?;
        self.syncs += 1;
        pages.sort_unstable();
        let page_bytes = (PAGE_WORDS * 8) as u64;
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for p in pages {
            match runs.last_mut() {
                Some(r) if r.0 + r.1 == p as u64 * page_bytes => r.1 += page_bytes,
                _ => runs.push((p as u64 * page_bytes, page_bytes)),
            }
        }
        let bytes = self.pix.read_ranges(&runs)?;
        self.read_back += bytes.len() as u64;
        let mut at = 0usize;
        for (off, len) in runs {
            let w0 = off as usize / 8;
            mem.words[w0..w0 + len as usize / 8].copy_from_slice(bytemuck::cast_slice(&bytes[at..at + len as usize]));
            at += len as usize;
        }
        Ok(())
    }

    /// Note the pages and clip-ID rows a record may write.
    fn mark(&mut self, s: &Slot, b: (i32, i32, i32, i32), kind: u32) {
        let (x0, y0, x1, y1) = (b.0.max(0) as u32, b.1.max(0) as u32, b.2.min(WIDTH as i32 - 1) as u32, b.3.min(HEIGHT as i32 - 1) as u32);
        let pp1 = s.w[SL_PP1];
        let drbsize = s.w[SL_DRBSIZE];
        let field = rss::draw_buffer(pp1);
        if field == rss::DRAW_CID {
            for y in y0..=y1 {
                self.dirty_cid[y as usize / 64] |= 1 << (y % 64);
            }
            return;
        }
        let drb = s.w[SL_DRB];
        let mut bufs = vec![Buffer::new(drb, if field & 0x70 == 0x40 { Kind::Overlay } else { Kind::Wide }, drbsize)];
        let second = (drb >> 10) & 0x3FF;
        if second != 0 {
            bufs.push(Buffer::new(second, Kind::Wide, drbsize));
        }
        if matches!(kind, REC_TRI | REC_GLLINE | REC_XFER) {
            bufs.push(Buffer::new(rss::ZST_PAGE, Kind::Wide, drbsize));
        }
        for buf in bufs {
            let tile_w = if buf.kind == Kind::Wide { pixmem::TILE_W } else { pixmem::OVERLAY_TILE_W } as u32;
            let (r0, r1) = ((y0 >> buf.yshift) / pixmem::TILE_H as u32, (y1 >> buf.yshift) / pixmem::TILE_H as u32);
            for e in 0..(1usize << buf.yshift) {
                for pr in r0..=r1 {
                    for pc in x0 / tile_w..=x1 / tile_w {
                        let page = e * PAGES + (buf.ptr + pr * buf.xtiles + pc) as usize % PAGES;
                        self.dirty_pages[page / 64] |= 1 << (page % 64);
                    }
                }
            }
        }
    }

    /// Rows `y0..=y1` of buffers `bufs` were written in the mirror `mem`
    /// (a host GL composite, with nothing recorded pending): their pages
    /// go up before the next batch.
    pub fn upload_rows(&mut self, mem: &PixMem, bufs: &[Buffer], y0: u32, y1: u32) {
        self.flush();
        let mut pages = vec![0u64; (RES * PAGES).div_ceil(64)];
        for buf in bufs {
            let tile_w = if buf.kind == Kind::Wide { pixmem::TILE_W } else { pixmem::OVERLAY_TILE_W } as u32;
            let (r0, r1) = ((y0 >> buf.yshift) / pixmem::TILE_H as u32, (y1 >> buf.yshift) / pixmem::TILE_H as u32);
            for e in 0..(1usize << buf.yshift) {
                for pr in r0..=r1 {
                    for pc in 0..(WIDTH as u32).div_ceil(tile_w) {
                        let page = e * PAGES + (buf.ptr + pr * buf.xtiles + pc) as usize % PAGES;
                        pages[page / 64] |= 1 << (page % 64);
                    }
                }
            }
        }
        for i in 0..pages.len() {
            let mut bits = pages[i];
            while bits != 0 {
                let p = i * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                self.pix.write((p * PAGE_WORDS * 8) as u64, bytemuck::cast_slice(&mem.words[p * PAGE_WORDS..(p + 1) * PAGE_WORDS]));
                self.uploaded += (PAGE_WORDS * 8) as u64;
            }
        }
    }

    /// Submit what was recorded; nothing waits.
    pub fn flush(&mut self) {
        if self.batch.is_empty() {
            self.stage.clear();
            self.last = None;
            return;
        }
        let k = self.batch.next_set(&self.pipelines);
        while self.sets.len() <= k {
            let set = self.new_set(self.sets.len());
            self.sets.push(set);
        }
        if !self.stage.is_empty() {
            self.sets[k].stage.write(0, bytemuck::cast_slice(&self.stage));
            self.stage.clear();
        }
        if let Some(sub) = self.batch.flush(&self.pipelines, &[&self.sets[k].group]) {
            self.pending = Some(sub);
        }
        self.last = None;
        self.textured = false;
    }

    /// Submit and wait until everything recorded is stored.
    pub fn wait(&mut self) -> Result<(), String> {
        self.flush();
        if let Some(sub) = self.pending.take() {
            self.gpu.wait(Some(sub))?;
        }
        self.gpu.check()
    }

    /// Bring the CPU's mirror current: wait, then read back the pages and
    /// clip-ID rows drawing may have changed since the last sync.
    pub fn sync(&mut self, mem: &mut PixMem, cid: &mut CidPlanes) -> Result<(), String> {
        let any = self.dirty_pages.iter().chain(&self.dirty_cid).any(|&w| w != 0);
        if !any && self.batch.is_empty() {
            return Ok(());
        }
        self.wait()?;
        self.syncs += 1;
        let page_bytes = (PAGE_WORDS * 8) as u64;
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for i in 0..self.dirty_pages.len() {
            let mut bits = std::mem::take(&mut self.dirty_pages[i]);
            while bits != 0 {
                let p = (i * 64) as u64 + bits.trailing_zeros() as u64;
                bits &= bits - 1;
                match runs.last_mut() {
                    Some(r) if r.0 + r.1 == p * page_bytes => r.1 += page_bytes,
                    _ => runs.push((p * page_bytes, page_bytes)),
                }
            }
        }
        if !runs.is_empty() {
            let bytes = self.pix.read_ranges(&runs)?;
            self.read_back += bytes.len() as u64;
            let mut at = 0usize;
            for (off, len) in runs {
                let w0 = off as usize / 8;
                let n = len as usize / 8;
                mem.words[w0..w0 + n].copy_from_slice(bytemuck::cast_slice(&bytes[at..at + len as usize]));
                at += len as usize;
            }
        }
        let mut rows: Vec<(u64, u64)> = Vec::new();
        for i in 0..self.dirty_cid.len() {
            let mut bits = std::mem::take(&mut self.dirty_cid[i]);
            while bits != 0 {
                let y = (i * 64) as u64 + bits.trailing_zeros() as u64;
                bits &= bits - 1;
                let row = WIDTH as u64;
                match rows.last_mut() {
                    Some(r) if r.0 + r.1 == y * row => r.1 += row,
                    _ => rows.push((y * row, row)),
                }
            }
        }
        if !rows.is_empty() {
            let bytes = self.cid.read_ranges(&rows)?;
            self.read_back += bytes.len() as u64;
            let mut at = 0usize;
            for (off, len) in rows {
                cid[off as usize..(off + len) as usize].copy_from_slice(&bytes[at..at + len as usize]);
                at += len as usize;
            }
        }
        Ok(())
    }

    /// Pixel memory words `range`, after a wait (tests, hashes).
    pub fn read_pix(&mut self, range: std::ops::Range<usize>) -> Result<Vec<u64>, String> {
        self.wait()?;
        let bytes = self.pix.read((range.start * 8) as u64, (range.len() * 8) as u64)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Clip-ID bytes `range`, after a wait.
    pub fn read_cid(&mut self, range: std::ops::Range<usize>) -> Result<Vec<u8>, String> {
        self.wait()?;
        self.cid.read(range.start as u64, range.len() as u64)
    }

    /// Batches flushed and records drawn so far.
    pub fn counts(&self) -> (u64, u64) {
        (self.batch.flushes, self.batch.prims_total)
    }

    /// The batch's GPU times, when measured (WGPU_BATCH_TIMING).
    pub fn timing_report(&self) -> Option<String> {
        self.batch.timing.as_ref().map(|t| t.report())
    }

    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }

    /// The pixel pages' buffer, for the scan-out.
    pub fn pix_buffer(&self) -> wgpu::Buffer {
        self.pix.buffer.clone()
    }
}

// ── primitives, as the RSS hands them over ──────────────────────────────────

/// The window origin and the y direction (`Rss::to_fb`).
fn origin(r: &Rss) -> (i64, i64, bool) {
    let win = r.reg(reg::XYWIN);
    (rss::signed16(win) as i64, rss::signed16(win >> 16) as i64, r.reg(reg::CONFIG) & rss::CONFIG_YFLIP != 0)
}

/// A window box (inclusive, any order) as a framebuffer box, clamped to
/// what an i32 holds.
fn fb_box(r: &Rss, wx0: i64, wy0: i64, wx1: i64, wy1: i64) -> (i32, i32, i32, i32) {
    let (ox, oy, flip) = origin(r);
    let (ax, bx) = (ox + wx0.min(wx1), ox + wx0.max(wx1));
    let (ay, by) = if flip { (oy - wy0.max(wy1), oy - wy0.min(wy1)) } else { (oy + wy0.min(wy1), oy + wy0.max(wy1)) };
    let c = |v: i64| v.clamp(-(1 << 20), 1 << 20) as i32;
    (c(ax), c(ay), c(bx), c(by))
}

impl GpuRaster {
    /// A block fill (`Rss::fill`).
    pub fn fill(&mut self, r: &Rss, b: &Block) {
        let fast36 = r.deep36() && !r.draws_overlay() && r.reg(reg::FILLMODE) & rss::FILL_FAST != 0;
        let mut rec = Rec::new(REC_FILL, if fast36 { FILL36 } else { 0 });
        let bx = fb_box(r, b.xs as i64, b.ys as i64, b.xe as i64, b.ye as i64);
        for (k, v) in [bx.0, bx.1, bx.2, bx.3].into_iter().enumerate() {
            rec.set_i(F_BOX + k, v);
        }
        rec.w[F_COLOUR] = b.color;
        if fast36 {
            let v = rss::pack36([0, 1, 2].map(|k| r.reg(reg::FILL_COLOR_R + k) & 0xFFF));
            rec.set_i64(F_V36, v as i64);
        }
        self.push(rec, &Slot::of(r, false, false), bx, &[], None);
    }

    /// A point (`Rss::point`): a one-pixel fill that never takes the
    /// 12:12:12 fast fill's colour.
    pub fn point(&mut self, r: &Rss) {
        let s = r.reg(reg::LINE_START);
        let (x, y) = (rss::signed16(s >> 16) as i64, rss::signed16(s) as i64);
        let bx = fb_box(r, x, y, x, y);
        let mut rec = Rec::new(REC_FILL, 0);
        for (k, v) in [bx.0, bx.1, bx.2, bx.3].into_iter().enumerate() {
            rec.set_i(F_BOX + k, v);
        }
        rec.w[F_COLOUR] = r.current_block().color;
        self.push(rec, &Slot::of(r, false, false), bx, &[], None);
    }

    /// An X line (`Rss::line`).
    pub fn line(&mut self, r: &Rss) {
        let (s, e) = (r.reg(reg::LINE_START), r.reg(reg::LINE_END));
        let (x0, y0, x1, y1) = (rss::signed16(s >> 16), rss::signed16(s), rss::signed16(e >> 16), rss::signed16(e));
        let fm = r.reg(reg::FILLMODE);
        let mut flags = 0;
        if fm & rss::FILL_LINE_STIPPLE != 0 {
            flags |= LINE_STIPPLE;
            if fm & rss::FILL_LINE_STIPPLE_OPAQUE != 0 {
                flags |= LINE_OPAQUE;
            }
        }
        if fm & rss::FILL_LINE_SKIP_LAST != 0 {
            flags |= LINE_SKIP_LAST;
        }
        let mut rec = Rec::new(REC_LINE, flags);
        rec.set_i(L_P0, x0);
        rec.set_i(L_P0 + 1, y0);
        rec.set_i(L_P1, x1);
        rec.set_i(L_P1 + 1, y1);
        rec.w[L_COLOUR] = r.current_block().color;
        rec.w[L_PATTERN] = r.reg(reg::LINE_STIPPLE);
        if flags & LINE_OPAQUE != 0 {
            rec.w[L_BG] = r.background();
        }
        let bx = fb_box(r, x0 as i64, y0 as i64, x1 as i64, y1 as i64);
        self.push(rec, &Slot::of(r, false, false), bx, &[], None);
    }

    fn block_row(rec: &mut Rec, b: &Block, row: i32) {
        rec.set_i(B_XS, b.xs);
        rec.set_i(B_YS, b.ys);
        rec.set_i(B_DX, b.dx());
        rec.set_i(B_DY, b.dy());
        rec.set_i(B_ROW, row);
    }

    /// The window box of `count` pixels of block row `row` from column
    /// `col`.
    fn row_box(r: &Rss, b: &Block, row: i32, col: i32, count: i32) -> (i32, i32, i32, i32) {
        let y = b.ys as i64 + row as i64 * b.dy() as i64;
        let xa = b.xs as i64 + col as i64 * b.dx() as i64;
        let xb = b.xs as i64 + (col + count - 1) as i64 * b.dx() as i64;
        fb_box(r, xa, y, xb, y)
    }

    /// One chunk of character stipple bits (`Rss::stipple_bits`): `n` bits
    /// from the most significant, into the block's current row; moves the
    /// stipple's position on as the interpreter does.
    pub fn stipple(&mut self, r: &Rss, s: &mut Stipple, bits: u64, n: u32) {
        let count = (n as i32).min(s.block.cols() - s.col);
        let mut rec = Rec::new(REC_STIPPLE, if s.opaque { STIPPLE_OPAQUE } else { 0 });
        Self::block_row(&mut rec, &s.block, s.row);
        rec.set_i(S_COL, s.col);
        rec.set_i(S_COUNT, count);
        rec.set_i64(S_BITS, bits as i64);
        // High word first, as the kernel reads the bits.
        rec.w.swap(S_BITS, S_BITS + 1);
        rec.w[S_COLOUR] = s.block.color;
        if s.opaque {
            rec.w[S_BG] = r.background();
        }
        let bx = Self::row_box(r, &s.block, s.row, s.col, count);
        self.push(rec, &Slot::of(r, false, false), bx, &[], None);
        s.col += count;
        if s.col >= s.block.cols() {
            s.col = 0;
            s.row += 1;
        }
    }

    /// One line of a pixel transfer into the block (`Rss::put_line`): the
    /// host bytes decoded to the values the interpreter stores, staged.
    pub fn transfer_line(&mut self, r: &Rss, x: &Xfer, line: u32, bytes: &[u8]) {
        let bpp = x.bpp as usize;
        let chunks = bytes.chunks(bpp).take(x.width as usize);
        let mut px: Vec<u32> = Vec::with_capacity(x.width as usize);
        let mode = if r.reg(reg::ZST_STENCIL) != 0 {
            px.extend(chunks.map(|c| *c.last().unwrap_or(&0) as u32));
            XFER_STENCIL
        } else {
            let frag = r.transfer_fragments();
            let alpha = if rss::transfer_alpha(x.format) { XFER_ALPHA } else { 0 };
            if rss::generic_format(x.format) {
                px.extend(chunks.map(|c| rss::fb_pixel(x.format, c)));
                if frag { XFER_FRAG | alpha } else { XFER_PUT }
            } else {
                px.extend(chunks.map(|c| rss::from_host(x.format, c.iter().fold(0u64, |a, &b| (a << 8) | b as u64))));
                if x.format == rss::DEPTH_FORMAT {
                    XFER_DEPTH
                } else if frag {
                    XFER_FRAG | alpha
                } else {
                    XFER_PUT
                }
            }
        };
        if px.is_empty() {
            return;
        }
        let count = px.len() as i32;
        let mut rec = Rec::new(REC_XFER, mode);
        Self::block_row(&mut rec, &x.block, line as i32);
        rec.set_i(X_COUNT, count);
        let bx = Self::row_box(r, &x.block, line as i32, 0, count);
        self.push(rec, &Slot::of(r, false, false), bx, &px, None);
    }

    /// A GL triangle from its setup (`Rss::triangle`).
    pub fn triangle(&mut self, r: &mut Rss, t: &TriSetup) {
        let mut flags = 0;
        if t.ltor != 0 {
            flags |= TRI_LTOR;
        }
        if t.tex != 0 {
            flags |= TRI_TEX;
        }
        if t.fog != 0 {
            flags |= TRI_FOG;
        }
        if t.stipple != 0 {
            flags |= TRI_STIPPLE;
        }
        let mut rec = Rec::new(REC_TRI, flags);
        for (k, e) in t.edges.iter().enumerate() {
            rec.set_i64(T_EDGES + 4 * k, e[0]);
            rec.set_i64(T_EDGES + 4 * k + 2, e[1]);
        }
        rec.set_i(T_JMID, t.jmid);
        rec.set_i(T_J0, t.j0);
        rec.set_i(T_J1, t.j1);
        rec.set_i(T_XS, t.xs_i);
        rec.set_i(T_YREFI, t.yref_i);
        for c in 0..4 {
            for k in 0..3 {
                rec.set_i(T_C + 3 * c + k, t.cplanes[c][k]);
            }
        }
        for k in 0..3 {
            rec.set_i64(T_Z + 2 * k, t.zp[k]);
            for c in 0..3 {
                rec.set_i64(T_T + 6 * c + 2 * k, t.tp[c][k]);
            }
        }
        let bx = Self::tri_box(r, t);
        let slot = Slot::of(r, t.tex != 0, t.stipple != 0);
        let te = (t.tex != 0).then_some(&mut *r.te);
        self.push(rec, &slot, bx, &[], te);
    }

    /// A triangle's framebuffer box: its rows, and the edges' extremes at
    /// the first and last of them, a pixel wider (conservative).
    fn tri_box(r: &Rss, t: &TriSetup) -> (i32, i32, i32, i32) {
        if t.j1 <= t.j0 {
            return (1, 1, 0, 0);
        }
        // The rows that can reach the framebuffer.
        let (_, oy, flip) = origin(r);
        let (ra, rb) = if flip { (oy - (HEIGHT as i64 - 1), oy) } else { (-oy, HEIGHT as i64 - 1 - oy) };
        let j0 = (t.j0 as i64).max(ra);
        let j1 = (t.j1 as i64 - 1).min(rb);
        if j0 > j1 {
            return (1, 1, 0, 0);
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for j in [j0, j1] {
            let yc = j as f64 + 0.5;
            for x in [t.x0 + t.s0 * (t.yref - yc), t.x1 + t.s1 * (t.yref - yc), t.x2 + t.s2 * (t.yref2 - yc)] {
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        let c = |v: f64| if v.is_nan() { 0 } else { v.clamp(-(1i64 << 40) as f64, (1i64 << 40) as f64) as i64 };
        fb_box(r, c(lo.floor()) - 1, j0, c(hi.ceil()) + 1, j1)
    }

    /// A GL line from its setup (`Rss::gl_line`); `pos` is the line
    /// stipple's position before it.
    pub fn gl_line(&mut self, r: &mut Rss, l: &GlLineSetup, pos: u32) {
        let mut flags = 0;
        if l.xmajor != 0 {
            flags |= GL_XMAJOR;
        }
        if l.tex != 0 {
            flags |= GL_TEX;
        }
        if l.fog != 0 {
            flags |= GL_FOG;
        }
        if l.stipple != 0 {
            flags |= GL_STIPPLE;
        }
        let mut rec = Rec::new(REC_GLLINE, flags);
        rec.set_i64(G_QK, l.qk);
        rec.set_i64(G_QS, l.qs);
        rec.set_i(G_P0, l.p0);
        rec.set_i(G_P1, l.p1);
        rec.set_i(G_DIR, l.dir);
        rec.set_i(G_WIDTH, l.width);
        for k in 0..4 {
            rec.set_i(G_CBASE + k, l.cbase[k]);
            rec.set_i(G_CSTEP + k, l.cstep[k]);
        }
        rec.set_i64(G_ZBASE, l.zbase);
        rec.set_i64(G_ZSTEP, l.zstep);
        for k in 0..3 {
            rec.set_i64(G_TBASE + 2 * k, l.tbase[k]);
            rec.set_i64(G_TSTEP + 2 * k, l.tstep[k]);
        }
        rec.w[G_POS] = pos;
        rec.w[G_PATTERN] = l.pattern;
        rec.w[G_REPEAT] = l.repeat;
        // The major pixels walked, the minor ones the width reaches at
        // either end (the minor position is linear along the line).
        let last = l.p1 - l.dir;
        let minor = |p: i32| l.b0 + l.slope * ((p as f64 + 0.5) - l.a0);
        let (m0, m1) = (minor(l.p0), minor(last));
        let c = |v: f64| if v.is_nan() { 0 } else { v.clamp(-(1i64 << 40) as f64, (1i64 << 40) as f64) as i64 };
        let (qa, qb) = (c(m0.min(m1).floor()) - l.width as i64 - 1, c(m0.max(m1).ceil()) + l.width as i64 + 1);
        let (pa, pb) = (l.p0.min(last) as i64, l.p0.max(last) as i64);
        let bx = if l.xmajor != 0 { fb_box(r, pa, qa, pb, qb) } else { fb_box(r, qa, pa, qb, pb) };
        let slot = Slot::of(r, l.tex != 0, false);
        let te = (l.tex != 0).then_some(&mut *r.te);
        self.push(rec, &slot, bx, &[], te);
    }
}

/// The pixel an X line (`Rss::line`, Bresenham from (x0, y0) to (x1, y1))
/// draws at window (x, y), as its step along the line: the closed form the
/// kernel evaluates (`shaders/rss.wgsl` `xline_step`).
#[cfg(test)]
fn xline_step(x0: i32, y0: i32, x1: i32, y1: i32, skip_last: bool, x: i32, y: i32) -> Option<u32> {
    let (dx, dy) = ((x1 - x0).abs(), (y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    // Along the major axis k steps; the minor axis has taken
    // floor((2 minor k + major) / (2 major)) of its steps by then.
    let (major, minor, k, at, start, s) = if dx >= dy { (dx, dy, (x - x0) * sx, y, y0, sy) } else { (dy, dx, (y - y0) * sy, x, x0, sx) };
    if k < 0 || k > major || (skip_last && k == major) {
        return None;
    }
    let n = if major == 0 {
        0
    } else {
        let p = minor as u32 * k as u32;
        let (q, rem) = (p / major as u32, p % major as u32);
        q + (2 * rem + major as u32) / (2 * major as u32)
    };
    (at == start + s * n as i32).then_some(k as u32)
}

impl Te1 {
    /// TRAM's bytes (the unit's upload).
    pub(super) fn tram(&self) -> &[u8] {
        // SAFETY: tram_ptr points at TRAM_BYTES bytes owned by self.
        unsafe { std::slice::from_raw_parts(self.tram_ptr(), te1::TRAM_BYTES) }
    }
}

/// The GPU unit as an `Rss` holds it: a pointer the board (or a test)
/// owns, null on the CPU raster path. Plain data, valid zeroed.
#[repr(transparent)]
pub(crate) struct GpuLink(*mut GpuRaster);

impl GpuLink {
    pub fn new(unit: &mut GpuRaster) -> GpuLink {
        GpuLink(unit)
    }

    /// The unit, if one is attached. Not tied to the borrow of the link:
    /// the unit is not part of the `Rss`, which hands itself to it.
    #[inline]
    #[allow(clippy::mut_from_ref, reason = "the unit is the RSS thread's alone, like the RSS itself")]
    pub fn get<'a>(&self) -> Option<&'a mut GpuRaster> {
        // SAFETY: set by the unit's owner, which keeps it alive while the
        // RSS uses it; only the RSS's thread touches it.
        unsafe { self.0.as_mut() }
    }
}
