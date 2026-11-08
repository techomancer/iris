//! End-to-end tests of the common layer on a test board: flat triangles
//! into a u32 plane, checked word for word against a CPU rasteriser with
//! the same arithmetic. They need a GPU (or lavapipe) and skip without
//! one.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::batch::{ArchShaders, Batch, BatchPipelines, Full};
use super::device::{Gpu, GpuOptions};
use super::memory::GpuMemory;
use super::shader::Fragment;
use super::tiles::{TileGrid, MAX_PRIMS};

mod softfloat;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct Rec {
    x: [i32; 3],
    y: [i32; 3],
    colour: u32,
    slot: u32,
}

// WGSL lays the record out as x0,y0,x1,y1,x2,y2: keep the Rust side in
// the same order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct RecWgsl {
    v: [i32; 6],
    colour: u32,
    slot: u32,
}

impl Rec {
    fn wgsl(&self) -> RecWgsl {
        RecWgsl {
            v: [self.x[0], self.y[0], self.x[1], self.y[1], self.x[2], self.y[2]],
            colour: self.colour,
            slot: self.slot,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct Slot {
    mask: u32,
    pad: [u32; 3],
}

const FLAT: Fragment = Fragment::new("flat.wgsl", include_str!("flat.wgsl"));

struct Board {
    gpu: Arc<Gpu>,
    pipelines: BatchPipelines,
    batch: Batch<RecWgsl, Slot>,
    vram: GpuMemory,
    vram_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    stride: u32,
}

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::new(&GpuOptions::default()) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    }
}

impl Board {
    fn new(gpu: Arc<Gpu>, width: u32, height: u32) -> Board {
        let stride = (width + 15) & !15;
        let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("flat vram"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipelines = BatchPipelines::new(
            &gpu,
            &ArchShaders {
                label: "flat",
                raster: &[FLAT],
                bin_accept: None,
                library: &[],
                defines: &[("WIDTH", width), ("HEIGHT", height), ("STRIDE", stride)],
                layouts: &[&layout],
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let vram = GpuMemory::new(&gpu, "flat vram", stride as u64 * height as u64 * 4).unwrap();
        let vram_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat vram"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: vram.buffer.as_entire_binding() }],
        });
        let batch = Batch::new(&gpu, &pipelines, TileGrid::new(width, height), 64);
        Board { gpu, pipelines, batch, vram, vram_group, width, height, stride }
    }

    fn flush(&mut self) {
        self.batch.flush(&self.pipelines, &[&self.vram_group]);
    }

    fn plane(&mut self) -> Vec<u32> {
        let bytes = self.vram.read_all().unwrap();
        bytemuck::cast_slice(&bytes).to_vec()
    }
}

/// The CPU side of flat.wgsl.
struct Reference {
    plane: Vec<u32>,
    width: u32,
    height: u32,
    stride: u32,
}

fn edge(ax: i32, ay: i32, bx: i32, by: i32, px: i32, py: i32) -> i32 {
    (bx - ax) * (py - ay) - (by - ay) * (px - ax)
}

fn inside(e: i32, dx: i32, dy: i32) -> bool {
    e > 0 || (e == 0 && ((dy == 0 && dx > 0) || dy < 0))
}

impl Reference {
    fn draw(&mut self, r: &Rec, mask: u32) {
        let (x, y) = (r.x, r.y);
        let x0 = (x.iter().copied().min().unwrap() >> 4).max(0);
        let x1 = ((x.iter().copied().max().unwrap() + 15) >> 4).min(self.width as i32 - 1);
        let y0 = (y.iter().copied().min().unwrap() >> 4).max(0);
        let y1 = ((y.iter().copied().max().unwrap() + 15) >> 4).min(self.height as i32 - 1);
        for py in y0..=y1 {
            for px in x0..=x1 {
                let (sx, sy) = (px * 16 + 8, py * 16 + 8);
                let e0 = edge(x[1], y[1], x[2], y[2], sx, sy);
                let e1 = edge(x[2], y[2], x[0], y[0], sx, sy);
                let e2 = edge(x[0], y[0], x[1], y[1], sx, sy);
                if inside(e0, x[2] - x[1], y[2] - y[1])
                    && inside(e1, x[0] - x[2], y[0] - y[2])
                    && inside(e2, x[1] - x[0], y[1] - y[0])
                {
                    let i = (py as u32 * self.stride + px as u32) as usize;
                    self.plane[i] = (self.plane[i] & !mask) | (r.colour & mask);
                }
            }
        }
    }
}

/// A deterministic generator.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u32) as i32
    }
}

/// A triangle with positive orientation, vertices in 1/16 pixel around
/// the target (some outside it), of a size drawn from `max`.
fn triangle(rng: &mut Lcg, width: u32, height: u32, max: i32) -> Rec {
    let cx = rng.range(-32, width as i32 + 32) * 16;
    let cy = rng.range(-32, height as i32 + 32) * 16;
    let mut r = Rec::default();
    for i in 0..3 {
        r.x[i] = cx + rng.range(-max, max);
        r.y[i] = cy + rng.range(-max, max);
    }
    if edge(r.x[0], r.y[0], r.x[1], r.y[1], r.x[2], r.y[2]) < 0 {
        r.x.swap(1, 2);
        r.y.swap(1, 2);
    }
    r.colour = rng.next();
    r
}

fn bbox(r: &Rec) -> (i32, i32, i32, i32) {
    (
        r.x.iter().copied().min().unwrap() >> 4,
        r.y.iter().copied().min().unwrap() >> 4,
        (r.x.iter().copied().max().unwrap() + 15) >> 4,
        (r.y.iter().copied().max().unwrap() + 15) >> 4,
    )
}

fn compare(gpu_plane: &[u32], reference: &Reference, what: &str) {
    let mut bad = 0;
    for y in 0..reference.height {
        for x in 0..reference.width {
            let i = (y * reference.stride + x) as usize;
            if gpu_plane[i] != reference.plane[i] {
                if bad < 8 {
                    eprintln!("{what}: ({x},{y}) gpu {:08x} cpu {:08x}", gpu_plane[i], reference.plane[i]);
                }
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "{what}: {bad} pixels differ");
}

fn run(width: u32, height: u32, batches: usize, per_batch: usize, max: i32, seed: u64) {
    let Some(gpu) = gpu() else { return };
    let mut board = Board::new(gpu, width, height);
    let mut reference = Reference { plane: vec![0; (board.stride * height) as usize], width, height, stride: board.stride };
    let mut rng = Lcg(seed);
    let masks = [0xffff_ffff, 0x00ff_ffff, 0xff00_ff00, 0x0000_00ff];
    for b in 0..batches {
        let mut slots = Vec::new();
        for m in masks {
            slots.push(board.batch.slot(Slot { mask: m, pad: [0; 3] }).unwrap());
        }
        for _ in 0..per_batch {
            let mut r = triangle(&mut rng, width, height, max);
            let s = (rng.next() % 4) as usize;
            r.slot = slots[s];
            let (x0, y0, x1, y1) = bbox(&r);
            match board.batch.prim(r.wgsl(), x0, y0, x1, y1) {
                Ok(_) => reference.draw(&r, masks[s]),
                Err(Full) => panic!("batch full at {per_batch}"),
            }
        }
        board.flush();
        if b % 4 == 3 {
            let plane = board.plane();
            compare(&plane, &reference, &format!("after batch {b}"));
        }
    }
    let plane = board.plane();
    compare(&plane, &reference, "final");
    board.gpu.check().unwrap();
}

#[test]
fn small_triangles_match_the_reference() {
    run(320, 200, 8, 500, 16 * 12, 1);
}

#[test]
fn large_overlapping_triangles_keep_order() {
    run(640, 480, 6, 200, 16 * 400, 2);
}

#[test]
fn odd_sized_target_and_offscreen_boxes() {
    run(333, 211, 4, 300, 16 * 60, 3);
}

#[test]
fn full_batches_flush_and_continue() {
    let Some(gpu) = gpu() else { return };
    let (width, height) = (256u32, 256u32);
    let mut board = Board::new(gpu, width, height);
    let mut reference = Reference { plane: vec![0; (board.stride * height) as usize], width, height, stride: board.stride };
    let mut rng = Lcg(7);
    let mut pushed = 0u32;
    let mut flushes = 0;
    while pushed < MAX_PRIMS * 2 + 17 {
        let slot = match board.batch.slot(Slot { mask: !0, pad: [0; 3] }) {
            Ok(s) => s,
            Err(Full) => {
                board.flush();
                flushes += 1;
                continue;
            }
        };
        let mut r = triangle(&mut rng, width, height, 16 * 20);
        r.slot = slot;
        let (x0, y0, x1, y1) = bbox(&r);
        match board.batch.prim(r.wgsl(), x0, y0, x1, y1) {
            Ok(_) => {
                reference.draw(&r, !0);
                pushed += 1;
            }
            Err(Full) => {
                board.flush();
                flushes += 1;
            }
        }
    }
    board.flush();
    assert!(flushes >= 2, "expected the recorder to fill up, flushed {flushes} times");
    let plane = board.plane();
    compare(&plane, &reference, "after refills");
}

#[test]
fn empty_flush_is_a_no_op() {
    let Some(gpu) = gpu() else { return };
    let mut board = Board::new(gpu, 64, 64);
    assert!(board.batch.flush(&board.pipelines, &[&board.vram_group]).is_none());
    assert!(board.plane().iter().all(|&w| w == 0));
}
