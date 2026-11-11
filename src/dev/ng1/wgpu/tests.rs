//! The GPU compose against the software one, word for word.

use std::sync::Arc;

use super::Ng1Scanout;
use crate::compositor::{CompositorSource, SwCompositor};
use crate::dev::ng1::vc2::{
    VC2_CTRL_CURSOR_EN, VC2_CTRL_CURSOR_SIZE, VC2_REG_CURRENT_CURSOR_X, VC2_REG_CURSOR_ENTRY_PTR, VC2_REG_DISPLAY_CONTROL,
    VC2_REG_WORKING_CURSOR_Y,
};
use crate::wgpu::{Gpu, GpuOptions};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// The tests that call this are `#[ignore]`d: a run that asks for them on a
/// host with no usable GPU fails instead of passing unchecked.
fn gpu() -> Arc<Gpu> {
    Gpu::new(&GpuOptions::default()).unwrap_or_else(|e| panic!("no host GPU: {e}"))
}

/// Random planes, window ids, XMAP modes, CMAP, RAMDAC, cursor and
/// scroll, composed by `SwCompositor` and by the kernel: every pixel the
/// same (the software word is 0xAABBGGRR, the kernel's 0x00RRGGBB).
#[test]
#[ignore = "requires a conformant host GPU"]
fn the_gpu_compose_matches_the_software_compositor() {
    let gpu = gpu();
    let mut scan = Ng1Scanout::new(&gpu).unwrap_or_else(|e| panic!("{e}"));
    let mut sw = SwCompositor::new();
    let mut rng = Lcg(7);
    let stride = crate::disp::FB_STRIDE;
    let mut fb_rgb = vec![0u32; stride * crate::disp::FB_MAX_H];
    let mut fb_aux = vec![0u32; stride * crate::disp::FB_MAX_H];
    let mut did = vec![0u8; stride * crate::disp::FB_MAX_H];
    let mut vc2_ram = vec![0u16; 32768];
    for round in 0..12 {
        let (width, height) = [(1280usize, 1024usize), (1024, 768), (640, 480), (333, 211)][round % 4];
        // Planes over the rows the scroll can reach; aux bits that make
        // popups and overlays now and then, not everywhere.
        for v in fb_rgb[..stride * 1024].iter_mut() {
            *v = rng.next();
        }
        for v in fb_aux[..stride * 1024].iter_mut() {
            let r = rng.next();
            *v = if r & 7 == 0 { r } else { r & !0xc };
        }
        for v in did[..stride * height].iter_mut() {
            *v = rng.below(32) as u8;
        }
        for v in vc2_ram.iter_mut() {
            *v = rng.next() as u16;
        }
        let mut xmap = [0u32; 32];
        for m in xmap.iter_mut() {
            *m = rng.next() & 0x00ff_ffff;
        }
        let mut cmap = [0u32; 8192];
        for c in cmap.iter_mut() {
            *c = rng.next() & 0x00ff_ffff;
        }
        let mut ramdac = [0u32; 256];
        for r in ramdac.iter_mut() {
            *r = rng.next() & 0x00ff_ffff;
        }
        let mut vc2_regs = [0u16; 32];
        vc2_regs[VC2_REG_CURRENT_CURSOR_X as usize] = rng.below(width as u32 + 64) as u16;
        vc2_regs[VC2_REG_WORKING_CURSOR_Y as usize] = rng.below(height as u32 + 64) as u16;
        vc2_regs[VC2_REG_CURSOR_ENTRY_PTR as usize] = [0u16, 0x3f00, 0x7fc0, 0x7ff0][rng.below(4) as usize];
        vc2_regs[VC2_REG_DISPLAY_CONTROL as usize] =
            if rng.below(4) != 0 { VC2_CTRL_CURSOR_EN } else { 0 } | if rng.below(2) == 0 { VC2_CTRL_CURSOR_SIZE } else { 0 };
        let src = CompositorSource {
            fb_rgb: &fb_rgb,
            fb_aux: &fb_aux,
            fb_rgb_damage: &[],
            fb_aux_damage: &[],
            did: &did,
            xmap_mode: &xmap,
            xmap_config: 0,
            xmap_cursor_cmap: rng.below(256) as u8,
            xmap_popup_cmap: rng.below(256) as u8,
            cmap: &cmap,
            ramdac: &ramdac,
            vc2_ram: &vc2_ram,
            vc2_regs: &vc2_regs,
            topscan: rng.below(1024) as usize,
            cursor_x_adjust: rng.below(9) as i32 - 4,
            width,
            height,
            status_bar_only: false,
            prebuilt: None,
        };
        sw.compose_pixels(&src);
        let mut out = vec![0u32; width * height];
        scan.run(&src, &mut out).unwrap_or_else(|e| panic!("{e}"));
        let want = sw.pixels();
        let mut bad = 0;
        for y in 0..height {
            for x in 0..width {
                let s = want[y * stride + x];
                let w = ((s & 0xff) << 16) | (s & 0xff00) | ((s >> 16) & 0xff);
                let g = out[y * width + x];
                if g != w {
                    if bad < 4 {
                        eprintln!("round {round} ({x}, {y}): gpu {g:06x} software {w:06x}, mode {:#x}", xmap[(did[y * stride + x] & 31) as usize]);
                    }
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "round {round}: {bad} pixels differ");
    }
}
