//! IMPACT's scan-out on the GPU (`crate::wgpu::scanout`): `Frame::snapshot`
//! and `Frame::compose` (frame.rs) as a kernel over the GPU unit's pixel
//! pages, so the display reads the drawing where it is and nothing comes
//! back to the CPU unless someone wants the pixels there.
//!
//! The display slot is the frame's decoded board state, resolved on the CPU
//! as `Frame::snapshot` resolves it: the scan-out buffers, the XMAP modes
//! per window ID, colormap 0, the DAC gamma, the cursor glyph and colours,
//! and the VC3's window-ID runs as a run list per row (the per-pixel ID
//! arrays the CPU builds would be 8 MB a frame).

use std::sync::Arc;

use crate::wgpu::{Fragment, Gpu, Scanout, ScanoutPipeline, ScanoutShaders};

use super::super::dcb::Dcb;
use super::super::frame::{self, CURSOR_MAX};
use super::super::rss::{self, Rss};
use super::{library, DEFINES, PIXMEM, PP1};

const SCANOUT: Fragment = Fragment::new("mgras_scanout.wgsl", include_str!("shaders/scanout.wgsl"));

// Slot word offsets (`shaders/scanout.wgsl` reads them as defines).
const D_WIDTH: usize = 0;
const D_HEIGHT: usize = 1;
const D_MAIN: usize = 2;
const D_BACK: usize = 3;
const D_OVERLAY: usize = 4;
const D_DRBSIZE: usize = 5;
const D_SELECT: usize = 6;
const D_BLANK: usize = 7;
const D_CURSOR_ON: usize = 8;
const D_CURSOR_X: usize = 9;
const D_CURSOR_Y: usize = 10;
const D_CURSOR_SIZE: usize = 11;
const D_CURSOR_RGB: usize = 12;
const D_MAIN_MODE: usize = 16;
const D_OVERLAY_MODE: usize = D_MAIN_MODE + 32;
const D_GAMMA: usize = D_OVERLAY_MODE + 32;
const D_CMAP: usize = D_GAMMA + 256;
const D_GLYPH: usize = D_CMAP + 8192;
const GLYPH_WORDS: usize = CURSOR_MAX * CURSOR_MAX / 16;
/// Per row, the first run of its main list, then of its overlay list
/// (rows + 1 entries each).
const D_ROWS: usize = D_GLYPH + GLYPH_WORDS;
const ROWS: usize = crate::disp::FB_MAX_H;
const D_RUNS: usize = D_ROWS + 2 * (ROWS + 1);
/// Runs the slot holds (x | did << 16), main and overlay together.
const MAX_RUNS: usize = 1 << 16;
const SLOT_WORDS: usize = D_RUNS + MAX_RUNS;
/// No overlay buffer.
const NO_BUFFER: u32 = 0xFFFF_FFFF;

/// Main mode word: bit 0 RGB, bits 2:1 the pair layout (1 4:4:4, 2
/// 5:5:5), bits 15:4 the index mask, 31:16 the colormap base.
fn main_mode_word(m: &frame::MainMode) -> u32 {
    let pair = match m.pair {
        None => 0,
        Some(rss::Pair::P444) => 1,
        Some(rss::Pair::P555) => 2,
    };
    m.rgb as u32 | pair << 1 | (m.index_mask as u32) << 4 | (m.cmap_base as u32) << 16
}

fn defines() -> Vec<(&'static str, u32)> {
    let mut d = DEFINES.to_vec();
    d.extend([
        ("D_WIDTH", D_WIDTH as u32),
        ("D_HEIGHT", D_HEIGHT as u32),
        ("D_MAIN", D_MAIN as u32),
        ("D_BACK", D_BACK as u32),
        ("D_OVERLAY", D_OVERLAY as u32),
        ("D_DRBSIZE", D_DRBSIZE as u32),
        ("D_SELECT", D_SELECT as u32),
        ("D_BLANK", D_BLANK as u32),
        ("D_CURSOR_ON", D_CURSOR_ON as u32),
        ("D_CURSOR_X", D_CURSOR_X as u32),
        ("D_CURSOR_Y", D_CURSOR_Y as u32),
        ("D_CURSOR_SIZE", D_CURSOR_SIZE as u32),
        ("D_CURSOR_RGB", D_CURSOR_RGB as u32),
        ("D_MAIN_MODE", D_MAIN_MODE as u32),
        ("D_OVERLAY_MODE", D_OVERLAY_MODE as u32),
        ("D_GAMMA", D_GAMMA as u32),
        ("D_CMAP", D_CMAP as u32),
        ("D_GLYPH", D_GLYPH as u32),
        ("D_ROWS", D_ROWS as u32),
        ("D_ROW_COUNT", ROWS as u32),
        ("D_RUNS", D_RUNS as u32),
        ("D_SLOT_WORDS", SLOT_WORDS as u32),
        ("CURSOR_MAX", CURSOR_MAX as u32),
        ("NO_BUFFER", NO_BUFFER),
    ]);
    d
}

/// The display thread's scan-out over the unit's pixel pages.
pub(crate) struct MgrasScanout {
    pipeline: ScanoutPipeline,
    scanout: Scanout,
    group: wgpu::BindGroup,
    slot: Vec<u32>,
    runs: Vec<(u16, u8)>,
    /// The last frame's size, and a readback's 0x00RRGGBB words.
    pub width: u32,
    pub height: u32,
    words: Vec<u32>,
}

impl MgrasScanout {
    /// Over the unit's pixel memory buffer (`GpuRaster::pix_buffer`).
    pub fn new(gpu: &Arc<Gpu>, pix: &wgpu::Buffer) -> Result<MgrasScanout, String> {
        let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mgras scanout pages"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let defines = defines();
        let pipeline = ScanoutPipeline::new(
            gpu,
            &ScanoutShaders {
                label: "mgras",
                body: &[SCANOUT],
                library: &[library::I64, library::WIDE, PIXMEM, PP1],
                defines: &defines,
                layouts: &[&layout],
            },
        )?;
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mgras scanout pages"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: pix.as_entire_binding() }],
        });
        let scanout = Scanout::new(gpu, &pipeline, (rss::WIDTH * ROWS) as u32, (SLOT_WORDS * 4) as u64);
        Ok(MgrasScanout { pipeline, scanout, group, slot: vec![0; SLOT_WORDS], runs: Vec::with_capacity(64), width: 0, height: 0, words: Vec::new() })
    }

    pub fn gpu(&self) -> Arc<Gpu> {
        Arc::clone(self.scanout.gpu())
    }

    /// Resolve the frame's slot from the registers and the DCB; the slot's
    /// used length in words.
    fn build(&mut self, r: &Rss, dcb: &Dcb) -> usize {
        let s = &mut self.slot;
        let (w, h) = frame::display_size(dcb);
        let h = h.min(ROWS);
        let (main, overlay) = frame::scanout_buffers(r, dcb);
        (self.width, self.height) = (w as u32, h as u32);
        s[D_WIDTH] = w as u32;
        s[D_HEIGHT] = h as u32;
        s[D_MAIN] = main.ptr;
        s[D_BACK] = (dcb.xmap.dib_pointers() >> 10) & 0x3FF;
        s[D_OVERLAY] = overlay.map_or(NO_BUFFER, |o| o.ptr);
        s[D_DRBSIZE] = r.reg(rss::reg::DRBSIZE);
        s[D_SELECT] = dcb.xmap.buf_select();
        s[D_BLANK] = (dcb.dac.pixmask() == 0) as u32;
        for did in 0..32u32 {
            s[D_MAIN_MODE + did as usize] = main_mode_word(&frame::MainMode::of(dcb.xmap.main_mode(did)));
            let o = dcb.xmap.overlay_mode(did);
            s[D_OVERLAY_MODE + did as usize] = (o != 0) as u32 | (((o >> 3) & 0x1F) * 256) << 16;
        }
        for (k, g) in dcb.dac.gamma.iter().enumerate() {
            s[D_GAMMA + k] = (g[0] >> 2) as u32 | ((g[1] >> 2) as u32) << 8 | ((g[2] >> 2) as u32) << 16;
        }
        s[D_CMAP..D_CMAP + 8192].copy_from_slice(&dcb.cmap[0].pal);
        s[D_CURSOR_ON] = 0;
        if let Some((cx, cy, size, glyph)) = dcb.vc3.cursor() {
            s[D_CURSOR_ON] = 1;
            s[D_CURSOR_X] = cx as u32;
            s[D_CURSOR_Y] = cy as u32;
            s[D_CURSOR_SIZE] = size as u32;
            let sram = &dcb.vc3.sram;
            let words_per_row = size / 16;
            let plane_words = size * words_per_row;
            let bit = |plane: usize, row: usize, col: usize| -> u32 {
                let w = sram[(glyph + plane * plane_words + row * words_per_row + col / 16) & 0x7FFF];
                (w >> (15 - col % 16)) as u32 & 1
            };
            s[D_GLYPH..D_GLYPH + GLYPH_WORDS].fill(0);
            for row in 0..size {
                for col in 0..size {
                    let i = row * CURSOR_MAX + col;
                    s[D_GLYPH + i / 16] |= (bit(0, row, col) | bit(1, row, col) << 1) << (2 * (i % 16));
                }
            }
            let base = dcb.xmap.cursor_cmap_base();
            for c in 1..4 {
                s[D_CURSOR_RGB + c] = dcb.cmap[0].pal.get(base + c).copied().unwrap_or(0xFF_FFFF);
            }
        }
        // The window-ID runs, as `fill_did_row` takes them: a run at x = 0
        // is implied when the row's first is not there.
        let mut n = 0;
        for list in 0..2 {
            for y in 0..=ROWS {
                s[D_ROWS + list * (ROWS + 1) + y] = n as u32;
                if y >= h || y == ROWS {
                    continue;
                }
                if list == 0 {
                    dcb.vc3.main_did_runs(y, &mut self.runs);
                } else {
                    dcb.vc3.overlay_did_runs(y, &mut self.runs);
                }
                if self.runs.first().is_none_or(|r| r.0 != 0) {
                    self.runs.insert(0, (0, 0));
                }
                for &(x, did) in &self.runs {
                    if n < MAX_RUNS {
                        s[D_RUNS + n] = x as u32 | (did as u32) << 16;
                        n += 1;
                    }
                }
            }
        }
        D_RUNS + n
    }

    /// Compose the frame and submit it, no readback; the buffer it is in
    /// (0x00RRGGBB, rows of `width`) and its size.
    pub fn compose(&mut self, r: &Rss, dcb: &Dcb) -> Result<(wgpu::Buffer, u32, u32), String> {
        let n = self.build(r, dcb);
        let (w, h) = (self.width, self.height);
        self.scanout.compose(&self.pipeline, w, h, bytemuck::cast_slice(&self.slot[..n]), &[&self.group])?;
        Ok((self.scanout.output().clone(), w, h))
    }

    /// Compose the frame and read it back as `frame.rs` composes it:
    /// `0xFFBBGGRR`, rows of `stride`, `out` at least `stride * height`.
    pub fn read(&mut self, r: &Rss, dcb: &Dcb, out: &mut [u32], stride: usize) -> Result<(usize, usize), String> {
        let n = self.build(r, dcb);
        let (w, h) = (self.width as usize, self.height as usize);
        self.words.resize(w * h, 0);
        self.scanout.run(&self.pipeline, w as u32, h as u32, bytemuck::cast_slice(&self.slot[..n]), &[&self.group], &mut self.words)?;
        for y in 0..h {
            for (o, v) in out[y * stride..y * stride + w].iter_mut().zip(&self.words[y * w..(y + 1) * w]) {
                *o = 0xFF00_0000 | (v & 0xFF) << 16 | (v & 0xFF00) | (v >> 16) & 0xFF;
            }
        }
        Ok((w, h))
    }

    /// Frames composed and bytes read back.
    pub fn counts(&self) -> (u64, u64) {
        (self.scanout.frames, self.scanout.read_back_bytes)
    }
}
