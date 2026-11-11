use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

use super::direct::{CaptureFault, Ng1ResidentScanout, ResidentScanoutSource, ResidentScanoutTraffic};
use super::resident::{ObservationCounters, ResidentError, ResidentFaultInjection, ResidentGpu, ResidentRaster};
use super::{Ng1Scanout, Ng1Slot, VC2_WORDS};
use crate::compositor::{CompositorSource, SwCompositor};
use crate::dev::ng1::bt445::Bt445;
use crate::dev::ng1::cmap::Cmap;
use crate::dev::ng1::rex3::{
    DrawMode0, DrawMode1, Rex3, Rex3Context, CLIPMODE_CIDMATCH_SHIFT, DRAWMODE0_ADRMODE_BLOCK_SH,
    DRAWMODE0_OPCODE_DRAW, DRAWMODE1_COMPARE_DISABLE_SH, DRAWMODE1_DRAWDEPTH_24,
    DRAWMODE1_LOGICOP_SRC_SH, DRAWMODE1_PLANES_RGB, DCBMODE_DATAWIDTH_4,
    DCBMODE_DCBADDR_SHIFT, DCBMODE_DCBCRS_SHIFT, REX3_BASE, REX3_CLIPMODE,
    REX3_COLORALPHA, REX3_COLORBLUE, REX3_COLORGRN, REX3_COLORRED, REX3_COORD_BIAS,
    REX3_DCBDATA0, REX3_DCBMODE, REX3_DRAWMODE0, REX3_DRAWMODE1, REX3_TOPSCAN,
    REX3_WRMASK, REX3_XYENDI, REX3_XYSTARTI,
};
use crate::dev::ng1::rex3::frame::ResidentDisplay;
use crate::dev::ng1::rex3_generic::{draw_with_fb, Framebuffers};
use crate::dev::ng1::rex3_shape;
use crate::dev::ng1::vc2::{
    Vc2, VC2_CTRL_BLACKOUT, VC2_CTRL_CURSOR_EN, VC2_CTRL_CURSOR_SIZE, VC2_CTRL_DID_EN,
    VC2_CTRL_VIDEO_TIMING_EN, VC2_REG_CURRENT_CURSOR_X, VC2_REG_CURSOR_ENTRY_PTR,
    VC2_REG_CURSOR_Y_LOC, VC2_REG_DID_ENTRY_PTR, VC2_REG_DISPLAY_CONTROL, VC2_REG_SCANLINE_LEN,
    VC2_REG_VIDEO_ENTRY_PTR, VC2_REG_WORKING_CURSOR_Y, VT_CBLANK_XMAP_N,
};
use crate::dev::ng1::xmap9::Xmap9;
use crate::disp::{Ng1DisplaySource, Rex3Screen, FB_STRIDE};
use crate::gfx_display::GfxDisplay;
use crate::traits::{BusDevice, Device, BUS_BUSY, BUS_OK};
use crate::wgpu::{Gpu, GpuOptions, Staging};

const WIDTH: usize = 64;
const HEIGHT: usize = 48;
const PLANE_WORDS: usize = FB_STRIDE * 1024;
const DID_PARAMS_BYTES: u64 = 8 * size_of::<u32>() as u64;
const DID_LINE: usize = 0x500;
const CURSOR: usize = 0x900;

struct Display {
    screen: Rex3Screen,
    vc2: Mutex<Vc2>,
    xmap: Mutex<Xmap9>,
    cmap: Mutex<Cmap>,
    dac: Mutex<Bt445>,
    diag: AtomicU64,
}

impl Display {
    fn new() -> Self {
        let mut vc2 = Vc2::new();
        vc2.regs[VC2_REG_VIDEO_ENTRY_PTR as usize] = 0x100;
        vc2.regs[VC2_REG_DID_ENTRY_PTR as usize] = 0x300;
        vc2.regs[VC2_REG_SCANLINE_LEN as usize] = (WIDTH << 5) as u16;
        vc2.regs[VC2_REG_CURSOR_ENTRY_PTR as usize] = CURSOR as u16;
        vc2.regs[VC2_REG_CURRENT_CURSOR_X as usize] = 27;
        vc2.regs[VC2_REG_WORKING_CURSOR_Y as usize] = 35;
        vc2.regs[VC2_REG_DISPLAY_CONTROL as usize] =
            VC2_CTRL_BLACKOUT | VC2_CTRL_VIDEO_TIMING_EN | VC2_CTRL_DID_EN | VC2_CTRL_CURSOR_EN;
        vc2.ram[0x100] = 0x200;
        vc2.ram[0x101] = HEIGHT as u16;
        vc2.ram[0x200] = 0x8000 | ((WIDTH as u16 / 2) << 8);
        vc2.ram[0x201] = VT_CBLANK_XMAP_N as u16;
        vc2.ram[0x202] = 0x200;
        for y in 0..HEIGHT {
            vc2.ram[0x300 + y] = DID_LINE as u16;
        }
        vc2.ram[DID_LINE] = 0;
        vc2.ram[DID_LINE + 1] = ((WIDTH as u16 / 2) << 5) | 1;
        vc2.ram[DID_LINE + 2] = 0xffe0;
        for y in 0..32 {
            vc2.ram[CURSOR + y * 2] = if y % 2 == 0 { 0xaaaa } else { 0x5555 };
            vc2.ram[CURSOR + 64 + y * 2] = 0x0f0f;
        }

        let mut xmap = Xmap9::new();
        xmap.mode_table[0] = (1 << 8) | (3 << 10) | (2 << 16);
        xmap.mode_table[1] = (1 << 10) | (2 << 16);
        xmap.cursor_cmap_msb = 3;
        xmap.popup_cmap_msb = 4;
        let mut cmap = Cmap::new(0);
        for (i, color) in cmap.palette.iter_mut().enumerate() {
            let i = i as u32;
            *color = ((i.wrapping_mul(3) & 255) << 16)
                | ((i.wrapping_mul(5) & 255) << 8)
                | (i.wrapping_mul(7) & 255);
        }
        let mut dac = Bt445::new();
        for (i, color) in dac.palette.iter_mut().enumerate() {
            *color = [i as u8; 3];
        }
        let mut screen = Rex3Screen::new();
        // A state-only snapshot must work even when no CPU plane exists.
        screen.fb_rgb = Vec::new();
        screen.fb_aux = Vec::new();
        let mut display = Self {
            screen,
            vc2: Mutex::new(vc2),
            xmap: Mutex::new(xmap),
            cmap: Mutex::new(cmap),
            dac: Mutex::new(dac),
            diag: AtomicU64::new(0),
        };
        assert!(display.refresh());
        assert_eq!((display.screen.width, display.screen.height), (WIDTH, HEIGHT));
        display
    }

    fn refresh(&mut self) -> bool {
        self.screen.refresh_display_state(&self.vc2, &self.xmap, &self.cmap, &self.dac, &self.diag)
    }

    fn refresh_resident(&mut self) -> bool {
        self.screen.refresh_resident_display_state(
            &self.vc2,
            &self.xmap,
            &self.cmap,
            &self.dac,
            &self.diag,
        )
    }
}

fn cpu_source<'a>(state: &'a Ng1DisplaySource<'_>, rgb: &'a [u32], aux: &'a [u32]) -> CompositorSource<'a> {
    CompositorSource {
        fb_rgb: rgb,
        fb_aux: aux,
        fb_rgb_damage: &[],
        fb_aux_damage: &[],
        did: state.did,
        xmap_mode: state.xmap_mode,
        xmap_config: state.xmap_config,
        xmap_cursor_cmap: state.xmap_cursor_cmap,
        xmap_popup_cmap: state.xmap_popup_cmap,
        cmap: state.cmap,
        ramdac: state.ramdac,
        vc2_ram: state.vc2_ram,
        vc2_regs: state.vc2_regs,
        topscan: state.topscan,
        cursor_x_adjust: state.cursor_x_adjust,
        width: state.width,
        height: state.height,
        status_bar_only: false,
        prebuilt: None,
    }
}

fn seed_planes() -> (Vec<u32>, Vec<u32>) {
    let mut rgb = vec![0; PLANE_WORDS];
    let mut aux = vec![0; PLANE_WORDS];
    for y in 0..1024 {
        for x in 0..FB_STRIDE {
            let index = y * FB_STRIDE + x;
            rgb[index] = ((x * 3 + y * 5) as u32 & 255)
                | (((x * 7 + y * 11) as u32 & 255) << 8)
                | (((x * 13 + y * 17) as u32 & 255) << 16);
            aux[index] = if (x + y) % 17 == 0 {
                1 << 2
            } else if (x + y) % 19 == 0 {
                0x12 << 8
            } else {
                0
            };
        }
    }
    (rgb, aux)
}

fn fill_context() -> Rex3Context {
    let mut ctx = Rex3Context::power_on_default();
    ctx.drawmode0 = DrawMode0(DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK_SH | (1 << 8) | (1 << 9));
    ctx.drawmode1 = DrawMode1(
        DRAWMODE1_PLANES_RGB | (DRAWMODE1_DRAWDEPTH_24 << 3) | (1 << 15)
            | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH,
    );
    ctx.clipmode = 0xf << CLIPMODE_CIDMATCH_SHIFT;
    ctx.wrmask = u32::MAX;
    ctx.colorred = 0x22 << 11;
    ctx.colorgrn = 0x77 << 11;
    ctx.colorblue = 0xd4 << 11;
    ctx.coloralpha = 0xff << 11;
    ctx.xstart = (40 + REX3_COORD_BIAS) << 11;
    ctx.xsave = ctx.xstart;
    ctx.ystart = (8 + REX3_COORD_BIAS) << 11;
    ctx.xend = (58 + REX3_COORD_BIAS) << 11;
    ctx.yend = (14 + REX3_COORD_BIAS) << 11;
    ctx
}

fn required_gpu() -> Arc<Gpu> {
    let backend = std::env::var("REX3_RESIDENT_BACKEND").unwrap_or_else(|_| "auto".into());
    let expected = match backend.as_str() {
        "vulkan" => Some(wgpu::Backend::Vulkan),
        "dx12" => Some(wgpu::Backend::Dx12),
        "metal" => Some(wgpu::Backend::Metal),
        "gl" => Some(wgpu::Backend::Gl),
        "auto" => None,
        _ => panic!("unknown required backend {backend}"),
    };
    let gpu = Gpu::new(&GpuOptions { backend: backend.clone(), ..GpuOptions::default() })
        .unwrap_or_else(|error| panic!("required scanout backend {backend}: {error}"));
    if let Some(expected) = expected {
        assert_eq!(gpu.caps.backend, expected, "the requested backend must actually be used");
    }
    assert!(
        matches!(gpu.caps.device_type, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu),
        "scanout gate requires a hardware adapter: {}", gpu.caps,
    );
    eprintln!("direct Newport scanout adapter: {}", gpu.caps);
    gpu
}

fn compare_frame(
    scan: &mut Ng1ResidentScanout,
    resident: &mut impl ResidentScanoutSource,
    display: &mut Display,
    legacy: &mut Ng1Scanout,
    sw: &mut SwCompositor,
    staging: &mut Staging,
    rgb: &[u32],
    aux: &[u32],
) -> Vec<u32> {
    let status = resident.scanout_status().expect("resident status");
    let state = display.screen.display_state();
    let cpu = cpu_source(&state, rgb, aux);
    sw.compose_pixels(&cpu);
    let mut old_gpu = vec![0u32; state.width * state.height];
    legacy.run(&cpu, &mut old_gpu).expect("CPU-source GPU compose");

    let frame = scan.compose(&state, resident).expect("direct GPU compose");
    assert!(Arc::ptr_eq(frame.gpu, legacy.gpu()), "output must use the resident device");
    assert_eq!(frame.position, status.submitted);
    assert_eq!((frame.width as usize, frame.height as usize), (state.width, state.height));
    // This is an explicit readback of the composed output, never a plane.
    let bytes = staging.read(frame.gpu, frame.buffer, &[(0, old_gpu.len() as u64 * 4)])
        .expect("composed frame readback");
    let direct = bytemuck::cast_slice::<u8, u32>(&bytes).to_vec();
    assert_eq!(direct, old_gpu, "direct and CPU-source GPU compositors differ");
    for y in 0..state.height {
        for x in 0..state.width {
            let pixel = sw.pixels()[y * FB_STRIDE + x];
            let expected = ((pixel & 0xff) << 16) | (pixel & 0xff00) | ((pixel >> 16) & 0xff);
            assert_eq!(direct[y * state.width + x], expected, "software parity at ({x}, {y})");
        }
    }
    direct
}

fn assert_no_plane_traffic(traffic: ResidentScanoutTraffic) {
    assert_eq!(traffic.raw_plane_uploads.attempts, 0);
    assert_eq!(traffic.raw_plane_uploads.attempted_bytes, 0);
    assert_eq!(traffic.raw_plane_uploads.submitted, 0);
    assert_eq!(traffic.raw_plane_readbacks.attempts, 0);
    assert_eq!(traffic.raw_plane_readbacks.attempted_bytes, 0);
    assert_eq!(traffic.raw_plane_readbacks.submitted, 0);
    assert_eq!(traffic.did.attempts, 0, "decoded DID map upload attempted");
    assert_eq!(traffic.did.attempted_bytes, 0, "decoded DID map bytes attempted");
    assert_eq!(traffic.did.submitted, 0, "decoded DID map upload submitted");
    assert_eq!(traffic.did.submitted_bytes, 0, "decoded DID map bytes submitted");
}

#[test]
fn direct_ng1_scanout_display_generations_track_real_changes() {
    let mut display = Display::new();
    let first = display.screen.display_state().generations;
    assert_eq!(display.screen.did[0], 0);
    assert_eq!(display.screen.did[WIDTH / 2], 1);
    display.refresh();
    assert_eq!(display.screen.display_state().generations, first);

    // Dirty without a state change must not invalidate an upload.
    display.vc2.lock().dirty = true;
    display.xmap.lock().dirty = true;
    display.cmap.lock().dirty = true;
    display.dac.lock().dirty = true;
    display.refresh();
    assert_eq!(display.screen.display_state().generations, first);

    {
        let mut vc2 = display.vc2.lock();
        vc2.regs[VC2_REG_CURRENT_CURSOR_X as usize] += 1;
        vc2.dirty = true;
    }
    display.refresh();
    let cursor = display.screen.display_state().generations;
    assert!(cursor.vc2_regs > first.vc2_regs);
    assert_eq!(cursor.vc2_ram, first.vc2_ram);
    assert_eq!(cursor.did, first.did, "cursor movement must not upload unchanged DID");

    {
        let mut vc2 = display.vc2.lock();
        vc2.ram[DID_LINE] = 1;
        vc2.dirty = true;
    }
    display.refresh();
    let did = display.screen.display_state().generations;
    assert!(did.vc2_ram > cursor.vc2_ram);
    assert!(did.did > cursor.did);

    display.screen.topscan = 1023;
    let scroll = display.screen.display_state().generations;
    assert!(scroll.geometry > did.geometry);
    assert_eq!(scroll.did, did.did);
    assert_ne!(Display::new().screen.display_state().generations.source, first.source);
}

#[test]
fn direct_ng1_scanout_refresh_skips_cpu_did_decode() {
    let mut display = Display::new();
    display.screen.did.fill(0x5a);
    let before = display.screen.display_state().generations;

    {
        let mut vc2 = display.vc2.lock();
        vc2.ram[DID_LINE] = 7;
        vc2.regs[VC2_REG_DID_ENTRY_PTR as usize] += 1;
        vc2.dirty = true;
    }
    display.refresh_resident();

    let after = display.screen.display_state().generations;
    assert!(after.vc2_ram > before.vc2_ram);
    assert!(after.vc2_regs > before.vc2_regs);
    assert_eq!(after.did, before.did);
    assert!(display.screen.did.iter().all(|&did| did == 0x5a));
}

#[test]
#[ignore = "requires the requested conformant hardware GPU backend"]
fn direct_ng1_scanout_gpu_reuses_planes_and_caches_display_state() {
    std::thread::Builder::new().name("direct-ng1-scanout".into()).stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let gpu = required_gpu();
            let (mut rgb, mut aux) = seed_planes();
            let mut resident = ResidentRaster::new(
                ResidentGpu::activate(Arc::clone(&gpu), &rgb, &aux).expect("activate resident planes"),
            );
            let initial_status = resident.status().unwrap();
            assert_eq!(initial_status.initialization_uploads, 2);
            assert_eq!(initial_status.initialization_bytes, 2 * PLANE_WORDS as u64 * 4);
            let mut scan = Ng1Scanout::new_resident(&gpu, &resident).expect("direct bindings");
            let mut legacy = Ng1Scanout::new(&gpu).expect("CPU-source GPU compositor");
            let mut display = Display::new();
            let mut sw = SwCompositor::new();
            let mut staging = Staging::default();

            let first = compare_frame(&mut scan, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            let traffic = scan.traffic();
            assert_eq!(traffic.frames_submitted, 1);
            assert_eq!(traffic.did_expansions.attempts, 1);
            assert_eq!(traffic.did_expansions.submitted, 1);
            assert_eq!(traffic.display.submitted, 4);
            assert_eq!(
                traffic.display.submitted_bytes,
                (2 * VC2_WORDS) as u64 + DID_PARAMS_BYTES + size_of::<Ng1Slot>() as u64 + 16,
            );
            assert!(traffic.display.submitted_bytes < 128 * 1024, "initial display state is not small");
            assert_no_plane_traffic(traffic);

            let again = compare_frame(&mut scan, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_eq!(again, first);
            assert_eq!(scan.traffic().display, traffic.display, "unchanged display state was uploaded");
            assert_eq!(scan.traffic().did, traffic.did);
            assert_eq!(scan.traffic().did_expansions, traffic.did_expansions);

            let mut gpu_context = fill_context();
            resident.enqueue(&mut gpu_context).expect("resident fill");
            let pending = scan.traffic();
            let state = display.screen.display_state();
            assert!(matches!(
                scan.compose(&state, &mut resident),
                Err(ResidentError::InvalidState("resident scanout requires all accepted work to be submitted"))
            ));
            assert_eq!(scan.traffic(), pending, "an unsubmitted prefix must not compose or upload");
            resident.submit().expect("submit resident fill");
            let mut cpu_context = fill_context();
            let fb = Framebuffers { rgb: rgb.as_mut_ptr(), aux: aux.as_mut_ptr() };
            let mode = rex3_shape::unpack(cpu_context.drawmode0.0, cpu_context.drawmode1.0, cpu_context.clipmode);
            draw_with_fb(&mut cpu_context, &fb, &mode);
            let drawn = compare_frame(&mut scan, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_ne!(drawn, first, "resident draw must become visible through the existing binding");
            assert_eq!(scan.traffic().display, traffic.display, "drawing must not re-upload display state");
            assert_ne!(resident.status().unwrap().submitted, initial_status.submitted);

            {
                let mut cmap = display.cmap.lock();
                for color in &mut cmap.palette { *color ^= 0x00ff_ffff; }
                cmap.dirty = true;
            }
            display.refresh();
            let palette = compare_frame(&mut scan, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_ne!(palette, drawn, "palette change must affect the composed frame");
            assert_eq!(scan.traffic().did_expansions.submitted, 1);

            {
                let mut dac = display.dac.lock();
                for color in &mut dac.palette { color[0] = 255 - color[0]; }
                dac.dirty = true;
            }
            display.refresh();
            let gamma = compare_frame(&mut scan, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_ne!(gamma, palette, "gamma change must affect the composed frame");
            assert_eq!(scan.traffic().did_expansions.submitted, 1);

            {
                let mut vc2 = display.vc2.lock();
                vc2.regs[VC2_REG_CURRENT_CURSOR_X as usize] += 13;
                vc2.regs[VC2_REG_DISPLAY_CONTROL as usize] |= VC2_CTRL_CURSOR_SIZE;
                vc2.ram[CURSOR] ^= 0xffff;
                vc2.dirty = true;
            }
            display.refresh();
            let cursor = compare_frame(&mut scan, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_ne!(cursor, gamma, "cursor changes must affect the composed frame");
            assert_eq!(scan.traffic().did_expansions.submitted, 2);

            display.screen.topscan = 1023;
            let scroll = compare_frame(&mut scan, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_ne!(scroll, cursor, "topscan must wrap through the resident plane");
            assert_eq!(scan.traffic().did_expansions.submitted, 2);

            {
                let mut xmap = display.xmap.lock();
                xmap.mode_table[1] |= 1 << 8;
                xmap.dirty = true;
            }
            display.refresh();
            let xmap = compare_frame(&mut scan, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_ne!(xmap, scroll, "XMAP change must affect the composed frame");
            assert_eq!(scan.traffic().did_expansions.submitted, 2);

            {
                let mut vc2 = display.vc2.lock();
                vc2.ram[DID_LINE] = 1;
                vc2.dirty = true;
            }
            display.refresh();
            let did = compare_frame(&mut scan, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_ne!(did, xmap, "decoded DID change must affect the composed frame");
            assert_eq!(scan.traffic().did_expansions.submitted, 3);
            let after_did = scan.traffic();
            compare_frame(&mut scan, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_eq!(scan.traffic().display, after_did.display);
            assert_eq!(scan.traffic().did, after_did.did);

            // Geometry changes re-expand on the GPU but never upload a
            // decoded map.
            for height in [HEIGHT / 2, HEIGHT] {
                {
                    let mut vc2 = display.vc2.lock();
                    vc2.ram[0x101] = height as u16;
                    vc2.dirty = true;
                }
                display.refresh();
                assert_eq!(display.screen.height, height);
                compare_frame(&mut scan, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            }
            assert_eq!(scan.traffic().did_expansions.submitted, 5);
            assert_no_plane_traffic(scan.traffic());
            let status = resident.status().unwrap();
            assert_eq!((status.initialization_uploads, status.initialization_bytes), (2, 2 * PLANE_WORDS as u64 * 4));
            assert_eq!((status.restore_uploads, status.restore_bytes), (0, 0));
            assert_eq!(status.observations, initial_status.observations, "scanout performed a raw-plane observation");
            assert_eq!(status.raw_plane_map_attempts, 0);
            assert!(!status.rgb.cpu_shadow_current(), "the resident draw must leave the CPU RGB shadow stale");
            assert!(status.aux.cpu_shadow_current());

            let mut other_display = Display::new();
            let before_other_source = scan.traffic();
            compare_frame(&mut scan, &mut resident, &mut other_display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_eq!(
                scan.traffic().did_expansions.submitted,
                before_other_source.did_expansions.submitted + 1,
            );
            compare_frame(&mut scan, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_eq!(
                scan.traffic().did_expansions.submitted,
                before_other_source.did_expansions.submitted + 2,
            );

            let stale_stamp = scan.compose(&display.screen.display_state(), &mut resident).unwrap().stamp();
            let mut current_stamp = scan.compose(&display.screen.display_state(), &mut resident).unwrap().stamp();
            let mut captured = vec![0xfeed_beef; WIDTH * HEIGHT];
            assert!(matches!(scan.capture(stale_stamp, &mut resident, &mut captured),
                Err(ResidentError::InvalidState("resident composed frame is stale"))));
            assert!(captured.iter().all(|&pixel| pixel == 0xfeed_beef));
            scan.capture(current_stamp, &mut resident, &mut captured).expect("capture current composed output");
            let composed = scan.traffic().composed_readbacks;
            assert_eq!((composed.attempts, composed.successes), (1, 1));
            assert_eq!(composed.successful_bytes, (WIDTH * HEIGHT * 4) as u64);

            let before_undecoded = scan.traffic();
            let mut undecoded = display.screen.display_state();
            undecoded.did = &[];
            let undecoded_stamp = scan.compose(&undecoded, &mut resident).expect("compose without CPU DID").stamp();
            let mut undecoded_frame = vec![0; WIDTH * HEIGHT];
            scan.capture(undecoded_stamp, &mut resident, &mut undecoded_frame)
                .expect("capture composition without CPU DID");
            assert_eq!(undecoded_frame, captured);
            current_stamp = undecoded_stamp;
            assert_eq!(scan.traffic().display, before_undecoded.display);
            assert_eq!(scan.traffic().did_expansions, before_undecoded.did_expansions);
            assert_no_plane_traffic(scan.traffic());

            let before_invalid = scan.traffic();
            let mut invalid = display.screen.display_state();
            invalid.cursor_x_adjust = i32::MAX;
            assert!(matches!(
                scan.compose(&invalid, &mut resident),
                Err(ResidentError::InvalidState("resident scanout display state is incomplete"))
            ));
            let mut invalid = display.screen.display_state();
            invalid.width = FB_STRIDE + 1;
            assert!(matches!(
                scan.compose(&invalid, &mut resident),
                Err(ResidentError::InvalidState("resident scanout dimensions exceed the Newport planes"))
            ));
            assert_eq!(scan.traffic(), before_invalid);

            let other_gpu = required_gpu();
            assert!(matches!(Ng1Scanout::new_resident(&other_gpu, &resident), Err(ResidentError::DeviceMismatch)));

            resident.reset(&rgb, &aux).expect("reset resident planes");
            assert!(matches!(scan.capture(current_stamp, &mut resident, &mut captured),
                Err(ResidentError::InvalidState("resident composed frame belongs to an old epoch"))));
            let before = scan.traffic();
            let state = display.screen.display_state();
            assert!(matches!(
                scan.compose(&state, &mut resident),
                Err(ResidentError::InvalidState("resident scanout bindings belong to an old or different epoch"))
            ));
            assert_eq!(scan.traffic(), before);
            let mut rebound = Ng1Scanout::new_resident(&gpu, &resident).expect("bind new epoch");
            let reset_frame = compare_frame(&mut rebound, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_no_plane_traffic(rebound.traffic());

            let restored_prefix = vec![0x00a5_5ac3; WIDTH];
            resident.restore(&restored_prefix, &[]).expect("restore RGB prefix");
            rgb[..WIDTH].copy_from_slice(&restored_prefix);
            let before_restore = rebound.traffic();
            let state = display.screen.display_state();
            assert!(matches!(
                rebound.compose(&state, &mut resident),
                Err(ResidentError::InvalidState("resident scanout bindings belong to an old or different epoch"))
            ));
            assert_eq!(rebound.traffic(), before_restore);
            let mut restored = Ng1Scanout::new_resident(&gpu, &resident).expect("bind restored epoch");
            let restored_frame = compare_frame(&mut restored, &mut resident, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_ne!(restored_frame, reset_frame, "restore must be visible through freshly bound planes");
            let status = resident.status().unwrap();
            assert_eq!((status.restore_uploads, status.restore_bytes), (1, (WIDTH * 4) as u64));
            assert_eq!(status.observations, initial_status.observations);
            assert_eq!(status.raw_plane_map_attempts, 0);
            assert_no_plane_traffic(restored.traffic());

            let before_fault = restored.traffic();
            resident.inject_fault_for_test(ResidentFaultInjection::DeviceLost);
            let state = display.screen.display_state();
            assert!(matches!(restored.compose(&state, &mut resident), Err(ResidentError::DeviceLost(_))));
            assert!(matches!(restored.compose(&state, &mut resident), Err(ResidentError::DeviceLost(_))));
            assert_eq!(restored.traffic(), before_fault, "a sticky resident fault cannot present or upload");
            gpu.check().unwrap();
        })
        .expect("spawn scanout test")
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}

#[test]
#[ignore = "requires the requested conformant hardware GPU backend"]
fn direct_ng1_scanout_gpu_did_expansion_matches_cpu_edge_cases() {
    std::thread::Builder::new().name("direct-ng1-did".into()).stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let gpu = required_gpu();
            let (rgb, aux) = seed_planes();
            let mut resident = ResidentRaster::new(
                ResidentGpu::activate(Arc::clone(&gpu), &rgb, &aux).expect("activate resident planes"),
            );
            let mut scan = Ng1Scanout::new_resident(&gpu, &resident).expect("direct bindings");
            let mut legacy = Ng1Scanout::new(&gpu).expect("CPU-source GPU compositor");
            let mut display = Display::new();
            let mut sw = SwCompositor::new();
            let mut staging = Staging::default();

            let initial = compare_frame(
                &mut scan,
                &mut resident,
                &mut display,
                &mut legacy,
                &mut sw,
                &mut staging,
                &rgb,
                &aux,
            );

            {
                let mut vc2 = display.vc2.lock();
                vc2.regs[VC2_REG_DISPLAY_CONTROL as usize] &= !VC2_CTRL_DID_EN;
                vc2.dirty = true;
            }
            display.refresh();
            let disabled = compare_frame(
                &mut scan,
                &mut resident,
                &mut display,
                &mut legacy,
                &mut sw,
                &mut staging,
                &rgb,
                &aux,
            );
            assert_ne!(disabled, initial);
            assert!(display.screen.did.iter().all(|&did| did == 0));

            {
                let mut vc2 = display.vc2.lock();
                vc2.regs[VC2_REG_DISPLAY_CONTROL as usize] |= VC2_CTRL_DID_EN;
                vc2.ram[DID_LINE] = 6;
                vc2.ram[DID_LINE + 1] = 0xffe0;
                vc2.dirty = true;
            }
            display.refresh();
            compare_frame(
                &mut scan,
                &mut resident,
                &mut display,
                &mut legacy,
                &mut sw,
                &mut staging,
                &rgb,
                &aux,
            );
            assert!(display.screen.did[..WIDTH].iter().all(|&did| did == 6));

            {
                let mut vc2 = display.vc2.lock();
                vc2.ram[DID_LINE] = 2;
                vc2.ram[0x300 + 3] = 0xffff;
                vc2.dirty = true;
            }
            display.refresh();
            compare_frame(
                &mut scan,
                &mut resident,
                &mut display,
                &mut legacy,
                &mut sw,
                &mut staging,
                &rgb,
                &aux,
            );
            for y in 0..3 {
                assert!(display.screen.did[y * FB_STRIDE..y * FB_STRIDE + WIDTH]
                    .iter()
                    .all(|&did| did == 2));
            }
            assert!(display.screen.did[3 * FB_STRIDE..3 * FB_STRIDE + WIDTH]
                .iter()
                .all(|&did| did == 6));

            {
                let mut vc2 = display.vc2.lock();
                vc2.ram[DID_LINE] = 4;
                vc2.ram[0x300 + 2] = 0xfffe;
                vc2.ram[0x300 + 3] = DID_LINE as u16;
                vc2.dirty = true;
            }
            display.refresh();
            compare_frame(
                &mut scan,
                &mut resident,
                &mut display,
                &mut legacy,
                &mut sw,
                &mut staging,
                &rgb,
                &aux,
            );
            assert!(display.screen.did[2 * FB_STRIDE..2 * FB_STRIDE + WIDTH]
                .iter()
                .all(|&did| did == 0));
            assert!(display.screen.did[3 * FB_STRIDE..3 * FB_STRIDE + WIDTH]
                .iter()
                .all(|&did| did == 4));

            {
                let mut vc2 = display.vc2.lock();
                vc2.ram[DID_LINE] = 5;
                vc2.ram[0x300 + 2] = DID_LINE as u16;
                vc2.regs[VC2_REG_SCANLINE_LEN as usize] = ((WIDTH / 2) << 5) as u16;
                vc2.dirty = true;
            }
            display.refresh();
            compare_frame(
                &mut scan,
                &mut resident,
                &mut display,
                &mut legacy,
                &mut sw,
                &mut staging,
                &rgb,
                &aux,
            );
            assert!(display.screen.did[..WIDTH / 2].iter().all(|&did| did == 5));
            assert!(display.screen.did[WIDTH / 2..WIDTH].iter().all(|&did| did == 4));

            {
                let mut vc2 = display.vc2.lock();
                vc2.ram[DID_LINE] = 7;
                vc2.regs[VC2_REG_SCANLINE_LEN as usize] = 0;
                vc2.dirty = true;
            }
            display.refresh();
            compare_frame(
                &mut scan,
                &mut resident,
                &mut display,
                &mut legacy,
                &mut sw,
                &mut staging,
                &rgb,
                &aux,
            );
            assert!(display.screen.did[..WIDTH].iter().all(|&did| did == 7));

            {
                let mut vc2 = display.vc2.lock();
                vc2.ram[0x101] = (HEIGHT / 2) as u16;
                vc2.ram[0x200] = 0x8000 | (((WIDTH - 16) as u16 / 2) << 8);
                vc2.dirty = true;
            }
            display.refresh();
            assert_eq!((display.screen.width, display.screen.height), (WIDTH - 16, HEIGHT / 2));
            compare_frame(
                &mut scan,
                &mut resident,
                &mut display,
                &mut legacy,
                &mut sw,
                &mut staging,
                &rgb,
                &aux,
            );

            {
                let mut vc2 = display.vc2.lock();
                vc2.ram[0x101] = HEIGHT as u16;
                vc2.ram[0x200] = 0x8000 | ((WIDTH as u16 / 2) << 8);
                vc2.dirty = true;
            }
            display.refresh();
            assert_eq!((display.screen.width, display.screen.height), (WIDTH, HEIGHT));
            compare_frame(
                &mut scan,
                &mut resident,
                &mut display,
                &mut legacy,
                &mut sw,
                &mut staging,
                &rgb,
                &aux,
            );

            assert_eq!(scan.traffic().did_expansions.submitted, 9);
            assert_no_plane_traffic(scan.traffic());
        })
        .expect("spawn DID expansion test")
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}

struct TestBoard(Box<Rex3>);

impl Drop for TestBoard {
    fn drop(&mut self) {
        self.0.stop();
    }
}

fn write_reg(rex: &Rex3, offset: u32, value: u32, go: bool) {
    let addr = (REX3_BASE | offset | if go { 0x800 } else { 0 }) as u64;
    loop {
        match rex.write32(addr, value) {
            BUS_OK => return,
            BUS_BUSY => std::thread::yield_now(),
            status => panic!("REX3 register write failed: {status:#x}"),
        }
    }
}

fn xy(x: i32, y: i32) -> u32 {
    (((x + REX3_COORD_BIAS) as u16 as u32) << 16) | (y + REX3_COORD_BIAS) as u16 as u32
}

#[test]
#[ignore = "requires the requested conformant hardware GPU backend"]
fn direct_ng1_scanout_gpu_uses_the_board_binding_seam() {
    std::thread::Builder::new().name("direct-ng1-board".into()).stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let gpu = required_gpu();
            let (mut rgb, mut aux) = seed_planes();
            let mut rex = Box::new(Rex3::new(crate::disp::StatusSource::detached()));
            rex.fb_rgb.get_mut().copy_from_slice(&rgb);
            rex.fb_aux.get_mut().copy_from_slice(&aux);
            rex.activate_resident_gpu(Arc::clone(&gpu)).expect("activate board");
            let board = TestBoard(rex);
            let mut source = &*board.0;
            let initial_status = source.scanout_status().unwrap();
            let mut scan = Ng1Scanout::new_resident(&gpu, &source).expect("board scanout binding");
            let mut legacy = Ng1Scanout::new(&gpu).expect("CPU-source GPU compositor");
            let mut display = Display::new();
            let mut sw = SwCompositor::new();
            let mut staging = Staging::default();
            let first = compare_frame(&mut scan, &mut source, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);

            board.0.start();
            let mut expected = fill_context();
            expected.drawmode0.0 |= 1 << 5;
            write_reg(&board.0, REX3_CLIPMODE, expected.clipmode, false);
            write_reg(&board.0, REX3_DRAWMODE1, expected.drawmode1.0, false);
            write_reg(&board.0, REX3_WRMASK, expected.wrmask, false);
            write_reg(&board.0, REX3_COLORRED, expected.colorred, false);
            write_reg(&board.0, REX3_COLORGRN, expected.colorgrn, false);
            write_reg(&board.0, REX3_COLORBLUE, expected.colorblue, false);
            write_reg(&board.0, REX3_COLORALPHA, expected.coloralpha, false);
            write_reg(&board.0, REX3_XYENDI, xy(58, 14), false);
            write_reg(&board.0, REX3_XYSTARTI, xy(40, 8), false);
            write_reg(&board.0, REX3_DRAWMODE0, expected.drawmode0.0, true);
            board.0.wait_resident_idle().expect("submit board fill");
            let fb = Framebuffers { rgb: rgb.as_mut_ptr(), aux: aux.as_mut_ptr() };
            let mode = rex3_shape::unpack(expected.drawmode0.0, expected.drawmode1.0, expected.clipmode);
            draw_with_fb(&mut expected, &fb, &mode);
            let drawn = compare_frame(&mut scan, &mut source, &mut display, &mut legacy, &mut sw, &mut staging, &rgb, &aux);
            assert_ne!(drawn, first, "board draw must appear through the original bindings");
            let status = source.scanout_status().unwrap();
            assert_ne!(status.submitted, initial_status.submitted);
            assert_eq!(status.observations, initial_status.observations);
            assert_eq!((status.initialization_uploads, status.restore_uploads), (2, 0));
            assert_eq!(status.raw_plane_map_attempts, 0);
            assert_no_plane_traffic(scan.traffic());
        })
        .expect("spawn board scanout test")
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}

fn configure_board_display(rex: &Rex3) {
    let display = Display::new();
    *rex.vc2.lock() = display.vc2.into_inner();
    *rex.xmap0.lock() = display.xmap.into_inner();
    *rex.cmap0.lock() = display.cmap.into_inner();
    *rex.bt445.lock() = display.dac.into_inner();
    {
        let mut vc2 = rex.vc2.lock();
        vc2.regs[VC2_REG_CURSOR_Y_LOC as usize] = vc2.regs[VC2_REG_WORKING_CURSOR_Y as usize];
        vc2.dirty = true;
    }
    rex.xmap0.lock().dirty = true;
    rex.cmap0.lock().dirty = true;
    rex.bt445.lock().dirty = true;
    let mut screen = rex.screen.lock();
    screen.fb_rgb.clear();
    screen.fb_aux.clear();
}

fn enqueue_fill(rex: &Rex3, context: &Rex3Context) {
    write_reg(rex, REX3_CLIPMODE, context.clipmode, false);
    write_reg(rex, REX3_DRAWMODE1, context.drawmode1.0, false);
    write_reg(rex, REX3_WRMASK, context.wrmask, false);
    write_reg(rex, REX3_COLORRED, context.colorred, false);
    write_reg(rex, REX3_COLORGRN, context.colorgrn, false);
    write_reg(rex, REX3_COLORBLUE, context.colorblue, false);
    write_reg(rex, REX3_COLORALPHA, context.coloralpha, false);
    write_reg(rex, REX3_XYENDI, xy(58, 14), false);
    write_reg(rex, REX3_XYSTARTI, xy(40, 8), false);
    write_reg(rex, REX3_DRAWMODE0, context.drawmode0.0, true);
}

fn draw_reference(mut context: Rex3Context, rgb: &mut [u32], aux: &mut [u32]) {
    let fb = Framebuffers { rgb: rgb.as_mut_ptr(), aux: aux.as_mut_ptr() };
    let mode = rex3_shape::unpack(context.drawmode0.0, context.drawmode1.0, context.clipmode);
    draw_with_fb(&mut context, &fb, &mode);
}

fn software_frame(rex: &Rex3, rgb: &[u32], aux: &[u32]) -> Vec<u32> {
    let mut screen = rex.screen.lock();
    let did = screen.decoded_did_oracle();
    let state = screen.display_state();
    let mut src = cpu_source(&state, rgb, aux);
    src.did = &did;
    let mut sw = SwCompositor::new();
    sw.compose_pixels(&src);
    let mut pixels = Vec::with_capacity(state.width * state.height);
    for y in 0..state.height {
        pixels.extend_from_slice(&sw.pixels()[y * FB_STRIDE..y * FB_STRIDE + state.width]);
    }
    pixels
}

struct QueuePresenter {
    gpu: Option<Arc<Gpu>>,
    accepts: bool,
    outputs: Arc<Mutex<Vec<wgpu::Buffer>>>,
    resizes: Arc<Mutex<Vec<(usize, usize)>>>,
}

impl crate::dev::ng1::rex3::Renderer for QueuePresenter {
    fn present(
        &mut self,
        _screen: &mut Rex3Screen,
        _overlay: &mut crate::debug_overlay::DebugOverlay,
        _status: &mut crate::disp::StatusBar,
        _sbtex: &mut crate::disp::StatusBarTexture,
        _stats: &crate::disp::BarStats,
        _need_readback: bool,
        _rgb: Option<&[u32]>,
        _aux: Option<&[u32]>,
    ) {
        panic!("resident frame fell through to the CPU-plane renderer");
    }

    fn takes_gpu_frames(&self) -> bool { self.accepts }
    fn resident_gpu_device(&self) -> Option<&Arc<Gpu>> { self.gpu.as_ref() }
    fn resize(&mut self, width: usize, height: usize) {
        self.resizes.lock().push((width, height));
    }

    fn present_resident_gpu_frame(
        &mut self,
        frame: &wgpu::Buffer,
        width: u32,
        height: u32,
        _screen: &mut Rex3Screen,
        _overlay: &mut crate::debug_overlay::DebugOverlay,
        _status: &mut crate::disp::StatusBar,
        _sbtex: &mut crate::disp::StatusBarTexture,
        _stats: &crate::disp::BarStats,
    ) -> Result<(), String> {
        let gpu = self.gpu.as_ref().unwrap();
        let bytes = u64::from(width) * u64::from(height) * 4;
        let output = gpu.buffer("resident test presenter output", bytes,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let mut encoder = gpu.encoder("resident test presenter");
        encoder.copy_buffer_to_buffer(frame, 0, &output, 0, bytes);
        gpu.submit(encoder);
        gpu.check()?;
        self.outputs.lock().push(output);
        Ok(())
    }
}

fn assert_no_cpu_plane_access(rex: &Rex3) {
    let (rgb, aux) = unsafe { ((*rex.fb_rgb.get()).access_counts(), (*rex.fb_aux.get()).access_counts()) };
    assert_eq!((rgb.reads, rgb.writes, aux.reads, aux.writes), (0, 0, 0, 0));
}

fn assert_pixels_eq(label: &str, actual: &[u32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    if let Some((index, (&actual, &expected))) = actual.iter().zip(expected).enumerate()
        .find(|(_, (actual, expected))| actual != expected)
    {
        panic!("{label}: pixel {index} is {actual:#010x}, expected {expected:#010x}");
    }
}

#[test]
#[ignore = "requires the requested conformant hardware GPU backend"]
fn resident_frame_gpu_marker_order_capture_and_display_changes() {
    std::thread::Builder::new().name("resident-frame-order".into()).stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let gpu = required_gpu();
            let (seed_rgb, seed_aux) = seed_planes();
            let mut first_rgb = seed_rgb.clone();
            let mut first_aux = seed_aux.clone();
            let mut rex = Box::new(Rex3::new(crate::disp::StatusSource::detached()));
            rex.fb_rgb.get_mut().copy_from_slice(&seed_rgb);
            rex.fb_aux.get_mut().copy_from_slice(&seed_aux);
            rex.fb_rgb.get_mut().isolate().unwrap();
            rex.fb_aux.get_mut().isolate().unwrap();
            configure_board_display(&rex);
            rex.activate_resident_gpu(Arc::clone(&gpu)).unwrap();
            unsafe {
                (*rex.fb_rgb.get()).reset_access_counts();
                (*rex.fb_aux.get()).reset_access_counts();
            }
            let rgb_guard = unsafe { (&*rex.fb_rgb.get()).protect() }.unwrap();
            let aux_guard = unsafe { (&*rex.fb_aux.get()).protect() }.unwrap();
            let board = TestBoard(rex);
            assert!(Arc::ptr_eq(&board.0.gpu().expect("resident display GPU"), &gpu));
            let outputs = Arc::new(Mutex::new(Vec::new()));
            let resizes = Arc::new(Mutex::new(Vec::new()));

            *board.0.renderer.lock() = Some(Box::new(QueuePresenter {
                gpu: None, accepts: false, outputs: Arc::clone(&outputs),
                resizes: Arc::clone(&resizes),
            }));
            assert!(matches!(ResidentDisplay::new(&board.0, &gpu),
                Err(ResidentError::Unsupported("resident frames require a wgpu presenter"))));
            let foreign = required_gpu();
            assert!(!Arc::ptr_eq(&foreign, &gpu));
            *board.0.renderer.lock() = Some(Box::new(QueuePresenter {
                gpu: Some(foreign), accepts: true, outputs: Arc::clone(&outputs),
                resizes: Arc::clone(&resizes),
            }));
            assert!(matches!(ResidentDisplay::new(&board.0, &gpu), Err(ResidentError::DeviceMismatch)));
            *board.0.renderer.lock() = Some(Box::new(QueuePresenter {
                gpu: Some(Arc::clone(&gpu)), accepts: true, outputs: Arc::clone(&outputs),
                resizes: Arc::clone(&resizes),
            }));
            board.0.start();
            let mut display = ResidentDisplay::new(&board.0, &gpu).unwrap();
            display.suppress_screenshot_files();
            write_reg(&board.0, REX3_TOPSCAN, 1023, false);
            let mut first = fill_context();
            first.drawmode0.0 |= 1 << 5;
            let mut later = fill_context();
            later.drawmode0.0 |= 1 << 5;
            later.colorred = 0xe7 << 11;
            later.colorgrn = 0x15 << 11;
            later.colorblue = 0x41 << 11;
            draw_reference(first.clone(), &mut first_rgb, &mut first_aux);
            let mut second_rgb = first_rgb.clone();
            let mut second_aux = first_aux.clone();
            draw_reference(later.clone(), &mut second_rgb, &mut second_aux);
            enqueue_fill(&board.0, &first);
            let first_stamp = display.render(&board.0, std::time::Duration::from_secs(5), |cut| {
                assert_eq!(board.0.resident_status().unwrap().accepted, cut);
                enqueue_fill(&board.0, &later);
                assert_eq!(board.0.resident_status().unwrap().accepted, cut,
                    "the later GO crossed the held frame marker");
            }).unwrap();
            board.0.wait_resident_idle().unwrap();
            let later_status = board.0.resident_status().unwrap();
            assert_ne!(later_status.accepted, first_stamp.position);
            let second_stamp = display.render(&board.0, std::time::Duration::from_secs(5), |_| {}).unwrap();
            assert_eq!(second_stamp.position, later_status.accepted);
            assert_eq!(board.0.screen.lock().frame_seq, 0, "ordinary presentation did not capture");
            assert_eq!(later_status.observations, ObservationCounters::default());
            assert_eq!(later_status.raw_plane_map_attempts, 0);
            assert_no_plane_traffic(display.traffic());
            assert_eq!(display.traffic().composed_readbacks.attempts, 0);
            assert_no_cpu_plane_access(&board.0);

            let expected_first = software_frame(&board.0, &first_rgb, &first_aux);
            let expected_second = software_frame(&board.0, &second_rgb, &second_aux);
            assert!(expected_first != expected_second);
            let buffers = outputs.lock().clone();
            assert_eq!(buffers.len(), 2);
            let mut staging = Staging::default();
            let read_presented = |staging: &mut Staging, buffer: &wgpu::Buffer| {
                let bytes = staging.read(&gpu, buffer, &[(0, (WIDTH * HEIGHT * 4) as u64)]).unwrap();
                bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
            };
            let first_pixels = read_presented(&mut staging, &buffers[0]);
            let second_pixels = read_presented(&mut staging, &buffers[1]);
            let to_gpu = |pixels: &[u32]| pixels.iter().map(|pixel| pixel.swap_bytes() >> 8).collect::<Vec<_>>();
            assert_pixels_eq("first marked prefix", &first_pixels, &to_gpu(&expected_first));
            assert_pixels_eq("later draw", &second_pixels, &to_gpu(&expected_second));

            board.0.request_frame();
            let before_fence_race = board.0.screen.lock().frame_seq;
            let resize_count = resizes.lock().len();
            let mode = (5 << DCBMODE_DCBADDR_SHIFT)
                | ((crate::dev::ng1::xmap9::XMAP9_REG_MODE_TABLE_WRITE as u32) << DCBMODE_DCBCRS_SHIFT)
                | DCBMODE_DATAWIDTH_4;
            assert!(matches!(
                display.render(&board.0, std::time::Duration::from_secs(2), |_| {
                    let mut vc2 = board.0.vc2.lock();
                    vc2.ram[0x101] = (HEIGHT / 2) as u16;
                    vc2.dirty = true;
                    drop(vc2);
                    write_reg(&board.0, REX3_DCBMODE, mode, false);
                    write_reg(&board.0, REX3_DCBDATA0, 0, false);
                }),
                Err(ResidentError::Timeout("display fence after frame marker"))
            ));
            assert_eq!(board.0.screen.lock().frame_seq, before_fence_race);
            assert!(board.0.frame_pending());
            assert_eq!(resizes.lock().len(), resize_count, "a deferred frame must not resize the presenter");
            let target = board.0.xmap_fence.load(Ordering::Acquire);
            let deadline = Instant::now() + std::time::Duration::from_secs(5);
            while board.0.gfifo_fence.load(Ordering::Acquire) != target && Instant::now() < deadline {
                std::thread::yield_now();
            }
            assert_eq!(board.0.gfifo_fence.load(Ordering::Acquire), target);
            display.render(&board.0, std::time::Duration::from_secs(2), |_| {}).unwrap();
            assert!(!board.0.frame_pending(), "the deferred request completes after the real fence");
            assert_eq!(resizes.lock().last(), Some(&(WIDTH, HEIGHT / 2)), "deferred resize must be replayed");
            let expected_after_fence = software_frame(&board.0, &second_rgb, &second_aux);

            board.0.request_frame();
            display.render(&board.0, std::time::Duration::from_secs(5), |_| {
                board.0.request_frame();
            }).unwrap();
            assert!(board.0.frame_pending(), "a request after the cut must remain pending");
            assert!(!board.0.frame_available());
            display.render(&board.0, std::time::Duration::from_secs(5), |_| {}).unwrap();
            assert!(!board.0.frame_pending() && board.0.frame_available());
            let mut captured = Vec::new();
            let (_, _, seq) = crate::gfx_display::copy_frame(&*board.0, &mut captured).unwrap();
            assert_pixels_eq("requested composed frame", &captured, &expected_after_fence);

            board.0.request_screenshot();
            display.fail_next_capture(CaptureFault::AfterMap);
            assert!(matches!(display.render(&board.0, std::time::Duration::from_secs(5), |_| {}),
                Err(ResidentError::Failed(_))));
            assert!(board.0.screenshot_pending());
            assert_eq!(board.0.screen.lock().frame_seq, seq);
            let mut stale = vec![0xfeed_beef];
            assert_eq!(crate::gfx_display::fresh_frame(&*board.0, &mut stale, std::time::Duration::ZERO), None);
            assert_eq!(stale, [0xfeed_beef]);
            display.render(&board.0, std::time::Duration::from_secs(5), |_| {}).unwrap();
            assert!(!board.0.screenshot_pending() && !board.0.frame_pending());
            assert_eq!(board.0.screen.lock().frame_seq, seq + 1);
            assert_eq!(crate::gfx_display::copy_frame(&*board.0, &mut captured).unwrap().2, seq + 1);
            assert_pixels_eq("retried screenshot", &captured, &expected_after_fence);

            board.0.request_frame();
            display.fail_next_capture(CaptureFault::BeforeCopy);
            assert!(display.render(&board.0, std::time::Duration::from_secs(5), |_| {}).is_err());
            assert!(board.0.frame_pending());
            display.render(&board.0, std::time::Duration::from_secs(5), |_| {}).unwrap();

            board.0.set_capture(true);
            display.render(&board.0, std::time::Duration::from_secs(5), |_| {}).unwrap();
            assert!(board.0.frame_available() && !board.0.frame_pending());
            let continuous_seq = board.0.screen.lock().frame_seq;
            display.fail_next_capture(CaptureFault::AfterMap);
            assert!(display.render(&board.0, std::time::Duration::from_secs(5), |_| {}).is_err());
            assert!(!board.0.frame_pending(), "continuous capture has no outstanding ticket");
            assert!(!board.0.frame_available(), "failed continuous capture cannot expose stale pixels");
            assert_eq!(board.0.screen.lock().frame_seq, continuous_seq);
            let mut unavailable = vec![0xfeed_beef];
            assert_eq!(crate::gfx_display::copy_frame(&*board.0, &mut unavailable), None);
            assert_eq!(unavailable, [0xfeed_beef]);
            display.render(&board.0, std::time::Duration::from_secs(5), |_| {}).unwrap();
            assert!(board.0.frame_available());
            board.0.set_capture(false);

            {
                let mut cmap = board.0.cmap0.lock();
                for color in &mut cmap.palette { *color ^= 0x00ff_ffff; }
                cmap.dirty = true;
            }
            {
                let mut dac = board.0.bt445.lock();
                for color in &mut dac.palette { color[0] = 255 - color[0]; }
                dac.dirty = true;
            }
            {
                let mut vc2 = board.0.vc2.lock();
                vc2.regs[VC2_REG_CURRENT_CURSOR_X as usize] += 13;
                vc2.ram[CURSOR] ^= 0xffff;
                vc2.dirty = true;
            }
            write_reg(&board.0, REX3_TOPSCAN, 0, false);
            board.0.request_frame();
            display.render(&board.0, std::time::Duration::from_secs(5), |_| {}).unwrap();
            let changed = software_frame(&board.0, &second_rgb, &second_aux);
            assert!(changed != expected_after_fence, "palette, gamma, cursor and topscan changes must be visible");
            assert_eq!(crate::gfx_display::copy_frame(&*board.0, &mut captured).unwrap().0, WIDTH);
            assert_pixels_eq("changed display state", &captured, &changed);
            let traffic = display.traffic();
            assert!(traffic.display.submitted > 4, "changed display generations must be uploaded");
            assert_eq!(traffic.composed_readbacks.attempts, traffic.composed_readbacks.successes + 3);
            assert_no_plane_traffic(traffic);
            let status = board.0.resident_status().unwrap();
            assert_eq!(status.observations, ObservationCounters::default());
            assert_eq!(status.raw_plane_map_attempts, 0);
            assert_no_cpu_plane_access(&board.0);
            board.0.stop();
            assert_no_cpu_plane_access(&board.0);
            drop(aux_guard);
            drop(rgb_guard);
        }).expect("spawn resident frame integration")
        .join().unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}

struct SharedTestBoard(Arc<Rex3>);

impl Drop for SharedTestBoard {
    fn drop(&mut self) { self.0.stop(); }
}

#[test]
#[ignore = "requires the requested conformant hardware GPU backend"]
fn resident_frame_gpu_headless_screenshot_png_matches_cpu() {
    const CHILD: &str = "IRIS_RESIDENT_SCREENSHOT_CHILD";
    if std::env::var_os(CHILD).is_some() {
        std::thread::Builder::new().name("resident-screenshot".into()).stack_size(64 * 1024 * 1024)
            .spawn(|| {
                use std::time::Duration;

                let gpu = required_gpu();
                let (rgb, aux) = seed_planes();
                let mut rex = Box::new(Rex3::new(crate::disp::StatusSource::detached()));
                rex.fb_rgb.get_mut().copy_from_slice(&rgb);
                rex.fb_aux.get_mut().copy_from_slice(&aux);
                rex.fb_rgb.get_mut().isolate().unwrap();
                rex.fb_aux.get_mut().isolate().unwrap();
                configure_board_display(&rex);
                rex.activate_resident_gpu(Arc::clone(&gpu)).unwrap();
                rex.fb_rgb.get_mut().reset_access_counts();
                rex.fb_aux.get_mut().reset_access_counts();
                let rgb_guard = unsafe { (&*rex.fb_rgb.get()).protect() }.unwrap();
                let aux_guard = unsafe { (&*rex.fb_aux.get()).protect() }.unwrap();
                let board = TestBoard(rex);
                board.0.start();
                assert!(board.0.renderer.lock().is_none());
                let mut display = ResidentDisplay::new(&board.0, &gpu).unwrap();
                let initial_seq = board.0.screen.lock().frame_seq;
                board.0.request_screenshot();
                display.fail_next_capture(CaptureFault::BeforeCopy);
                assert!(display.render(&board.0, Duration::from_secs(10), |_| {}).is_err());
                assert!(board.0.screenshot_pending());
                assert_eq!(board.0.screen.lock().frame_seq, initial_seq);
                assert!(!std::path::Path::new("screenshot_0000.png").exists());
                display.render(&board.0, Duration::from_secs(10), |_| {}).unwrap();
                assert!(!board.0.screenshot_pending());
                assert_eq!(board.0.screen.lock().frame_seq, initial_seq + 1);
                let expected = software_frame(&board.0, &rgb, &aux);
                let deadline = Instant::now() + Duration::from_secs(10);
                let (info, pixels) = loop {
                    let decoded = (|| {
                        let file = std::fs::File::open("screenshot_0000.png").ok()?;
                        let mut reader = png::Decoder::new(std::io::BufReader::new(file)).read_info().ok()?;
                        let mut pixels = vec![0; reader.output_buffer_size()?];
                        let info = reader.next_frame(&mut pixels).ok()?;
                        pixels.truncate(info.buffer_size());
                        Some((info, pixels))
                    })();
                    if let Some(decoded) = decoded { break decoded; }
                    assert!(Instant::now() < deadline, "composed screenshot PNG was not written");
                    std::thread::sleep(Duration::from_millis(10));
                };
                assert_eq!((info.width as usize, info.height as usize, info.color_type), (WIDTH, HEIGHT, png::ColorType::Rgb));
                let expected: Vec<u8> = expected.iter()
                    .flat_map(|pixel| [*pixel as u8, (*pixel >> 8) as u8, (*pixel >> 16) as u8])
                    .collect();
                assert_eq!(pixels, expected, "encoded screenshot differs from CPU composition");
                let traffic = display.traffic();
                assert_eq!((traffic.composed_readbacks.attempts, traffic.composed_readbacks.successes), (2, 1));
                assert_no_plane_traffic(traffic);
                let status = board.0.resident_status().unwrap();
                assert_eq!(status.observations, ObservationCounters::default());
                assert_eq!(status.raw_plane_map_attempts, 0);
                assert_no_cpu_plane_access(&board.0);
                board.0.stop();
                drop(aux_guard);
                drop(rgb_guard);
            }).unwrap().join().unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        return;
    }

    let name = std::thread::current().name().unwrap().to_owned();
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let directory = std::env::current_dir().unwrap().join("target")
        .join(format!("resident-screenshot-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(directory.parent().unwrap()).unwrap();
    std::fs::create_dir(&directory).unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &name, "--ignored", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .current_dir(&directory)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn().expect("spawn resident screenshot test");
    let deadline = Instant::now() + std::time::Duration::from_secs(60);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            std::fs::remove_dir_all(&directory).unwrap();
            panic!("resident screenshot test timed out: {}", String::from_utf8_lossy(&output.stderr));
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let output = child.wait_with_output().unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success() && stdout.contains("1 passed; 0 failed"), "{stdout}\n{stderr}");
}

#[test]
#[ignore = "requires the requested conformant hardware GPU backend"]
fn resident_frame_gpu_headless_requests_and_vnc_capture() {
    std::thread::Builder::new().name("resident-frame-headless".into()).stack_size(64 * 1024 * 1024)
        .spawn(|| {
            use std::io::{Read, Write};
            use std::time::{Duration, Instant};

            let gpu = required_gpu();
            let (rgb, aux) = seed_planes();
            let mut rex = Box::new(Rex3::new(crate::disp::StatusSource::detached()));
            rex.fb_rgb.get_mut().copy_from_slice(&rgb);
            rex.fb_aux.get_mut().copy_from_slice(&aux);
            rex.fb_rgb.get_mut().isolate().unwrap();
            rex.fb_aux.get_mut().isolate().unwrap();
            configure_board_display(&rex);
            rex.activate_resident_gpu(Arc::clone(&gpu)).unwrap();
            unsafe {
                (*rex.fb_rgb.get()).reset_access_counts();
                (*rex.fb_aux.get()).reset_access_counts();
            }
            let rgb_guard = unsafe { (&*rex.fb_rgb.get()).protect() }.unwrap();
            let aux_guard = unsafe { (&*rex.fb_aux.get()).protect() }.unwrap();
            let board = SharedTestBoard(Arc::from(rex));
            board.0.start();
            write_reg(&board.0, REX3_TOPSCAN, 1023, false);
            assert!(board.0.renderer.lock().is_none());
            let mut headless = ResidentDisplay::new(&board.0, &gpu).unwrap();
            headless.suppress_screenshot_files();
            board.0.request_screenshot();
            headless.render(&board.0, Duration::from_secs(10), |_| {}).unwrap();
            assert!(!board.0.screenshot_pending());
            let mut screenshot = Vec::new();
            assert!(crate::gfx_display::copy_frame(&*board.0, &mut screenshot).is_some());
            assert_pixels_eq("headless screenshot", &screenshot, &software_frame(&board.0, &rgb, &aux));
            drop(headless);
            board.0.start_resident_display(Arc::clone(&gpu)).unwrap();

            let mut captured = Vec::new();
            let (width, height, seq) = crate::gfx_display::fresh_frame(
                &*board.0, &mut captured, Duration::from_secs(10),
            ).expect("headless composed frame");
            assert_eq!((width, height), (WIDTH, HEIGHT));
            assert_pixels_eq("headless composed frame", &captured, &software_frame(&board.0, &rgb, &aux));
            let initial = captured.clone();
            let cursor_y = {
                let mut vc2 = board.0.vc2.lock();
                vc2.regs[VC2_REG_CURSOR_Y_LOC as usize] += 7;
                vc2.dirty = true;
                vc2.regs[VC2_REG_CURSOR_Y_LOC as usize]
            };
            let deadline = Instant::now() + Duration::from_secs(5);
            while board.0.vc2.lock().regs[VC2_REG_WORKING_CURSOR_Y as usize] != cursor_y
                && Instant::now() < deadline
            {
                std::thread::sleep(Duration::from_millis(2));
            }
            assert_eq!(board.0.vc2.lock().regs[VC2_REG_WORKING_CURSOR_Y as usize], cursor_y);
            let (_, _, updated_seq) = crate::gfx_display::fresh_frame(
                &*board.0, &mut captured, Duration::from_secs(10),
            ).expect("headless cursor refresh");
            assert!(updated_seq > seq);
            assert!(captured != initial, "vblank cursor changes must reach the composed frame");
            assert_pixels_eq("headless cursor update", &captured, &software_frame(&board.0, &rgb, &aux));
            let seq = updated_seq;
            std::thread::sleep(Duration::from_millis(100));
            assert_eq!(board.0.screen.lock().frame_seq, seq,
                "headless idle must not perform an unrequested readback");

            let display: Arc<dyn GfxDisplay> = board.0.clone();
            let ps2 = Arc::new(crate::dev::ps2::Ps2Controller::new(None));
            let input = Arc::new(crate::hostinput::HostInput::new(Some(display), ps2));
            let server = crate::vnc::VncServer::start(input, "127.0.0.1:0", Some(30), "resident").unwrap();
            let mut stream = std::net::TcpStream::connect(server.addr()).unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            let mut version = [0; 12];
            stream.read_exact(&mut version).unwrap();
            assert_eq!(&version, b"RFB 003.008\n");
            stream.write_all(&version).unwrap();
            let mut security = [0; 2];
            stream.read_exact(&mut security).unwrap();
            assert_eq!(security, [1, 1]);
            stream.write_all(&[1]).unwrap();
            let mut security_result = [0; 4];
            stream.read_exact(&mut security_result).unwrap();
            assert_eq!(security_result, [0; 4]);
            stream.write_all(&[1]).unwrap();
            let mut init = [0; 24];
            stream.read_exact(&mut init).unwrap();
            assert_eq!(u16::from_be_bytes([init[0], init[1]]) as usize, WIDTH);
            assert_eq!(u16::from_be_bytes([init[2], init[3]]) as usize, HEIGHT);
            let name_len = u32::from_be_bytes(init[20..24].try_into().unwrap()) as usize;
            let mut name = vec![0; name_len];
            stream.read_exact(&mut name).unwrap();
            assert_eq!(&name, b"resident");

            let mut request = vec![3, 0, 0, 0, 0, 0];
            request.extend_from_slice(&(WIDTH as u16).to_be_bytes());
            request.extend_from_slice(&(HEIGHT as u16).to_be_bytes());
            stream.write_all(&request).unwrap();
            let mut update = [0; 4];
            stream.read_exact(&mut update).unwrap();
            assert_eq!(update, [0, 0, 0, 1]);
            let mut rect = [0; 12];
            stream.read_exact(&mut rect).unwrap();
            assert_eq!(&rect[..4], &[0; 4]);
            assert_eq!(u16::from_be_bytes([rect[4], rect[5]]) as usize, WIDTH);
            assert_eq!(u16::from_be_bytes([rect[6], rect[7]]) as usize, HEIGHT);
            assert_eq!(i32::from_be_bytes(rect[8..12].try_into().unwrap()), 0);
            let mut pixels = vec![0; WIDTH * HEIGHT * 4];
            stream.read_exact(&mut pixels).unwrap();
            let vnc: Vec<u32> = pixels.chunks_exact(4)
                .map(|pixel| u32::from_le_bytes(pixel.try_into().unwrap())).collect();
            let expected: Vec<u32> = captured.iter().map(|pixel| pixel & 0x00ff_ffff).collect();
            assert_pixels_eq("VNC composed frame", &vnc, &expected);
            assert!(board.0.screen.lock().frame_seq > seq, "VNC enabled composed capture");
            drop(stream);
            let deadline = Instant::now() + Duration::from_secs(5);
            while server.clients() != 0 && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            assert_eq!(server.clients(), 0, "VNC client must disconnect");
            drop(server);
            board.0.stop();
            let status = board.0.resident_status().unwrap();
            assert_eq!(status.observations, ObservationCounters::default());
            assert_eq!(status.raw_plane_map_attempts, 0);
            assert_no_cpu_plane_access(&board.0);
            drop(aux_guard);
            drop(rgb_guard);
        }).expect("spawn resident headless integration")
        .join().unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}
