use super::plane::AccessCounts;
use super::*;
use crate::compositor::{CompositorSource, SwCompositor};
use crate::dev::ng1::rex3_generic::{draw_with_fb, Framebuffers};
use crate::dev::ng1::rex3_shape;
use crate::dev::ng1::wgpu::resident::{ObservationCounters, ObservationReason, PlaneMask};
use crate::dev::ng1::wgpu::Ng1Scanout;
use crate::traits::{BusDevice, Device};
use crate::wgpu::{Gpu, GpuOptions, Staging};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const PLANE_WORDS: usize = 2048 * 1024;
const PLANE_BYTES: u64 = (PLANE_WORDS * size_of::<u32>()) as u64;
const PROBE_ENV: &str = "IRIS_REX3_PLANE_FAULT_PROBE";

fn isolated_plane(value: u32) -> CpuPlane {
    let mut plane: CpuPlane = vec![value; PLANE_WORDS].into_boxed_slice().into();
    plane.isolate().expect("allocate an isolated CPU plane");
    plane
}

fn counts(rex: &Rex3) -> (AccessCounts, AccessCounts) {
    // SAFETY: this reads only the plane metadata; its counters are atomic.
    unsafe { ((*rex.fb_rgb.get()).access_counts(), (*rex.fb_aux.get()).access_counts()) }
}

fn reset_counts(rex: &Rex3) {
    // SAFETY: the board is stopped, or the resident worker is quiescent.
    unsafe {
        (*rex.fb_rgb.get()).reset_access_counts();
        (*rex.fb_aux.get()).reset_access_counts();
    }
}

#[test]
fn resident_plane_guard_is_opt_in_counted_and_unwinds() {
    let mut plane: CpuPlane = vec![10, 20, 30].into_boxed_slice().into();
    assert!(plane.mapping_range().is_none(), "ordinary test planes remain boxed");
    assert!(unsafe { plane.protect() }.is_err(), "allocator pages must never be protected");
    plane.reset_access_counts();
    assert_eq!(plane[1], 20);
    plane[2] = 31;
    assert_eq!(plane.access_counts(), AccessCounts { reads: 1, writes: 1 });
    let copy = plane.clone();
    assert_eq!(copy, plane);
    assert!(copy.mapping_range().is_none(), "clone is an ordinary independent box");

    plane.isolate().unwrap();
    assert!(plane.isolate().is_err(), "isolation must not replace a live mapping");
    let (address, bytes, page) = plane.mapping_range().unwrap();
    assert_eq!(address % page, 0);
    assert_eq!(bytes % page, 0);
    assert!(bytes >= 3 * size_of::<u32>());
    assert_eq!(&*plane, &[10, 20, 31]);

    let mut unrelated = vec![0x1234_5678u32; 100].into_boxed_slice();
    plane.reset_access_counts();
    let guard = unsafe { plane.protect() }.unwrap();
    assert!(unsafe { plane.protect() }.is_err(), "nested guards must not unprotect one another");
    unrelated[0] = 0x8765_4321;
    assert_eq!(unrelated[0], 0x8765_4321, "an unrelated allocation stayed accessible");
    let blocked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        std::hint::black_box(plane[0]);
    }));
    assert!(blocked.is_err(), "forming a CPU slice while protected must be counted and rejected");
    let blocked_write = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        plane[0] = 99;
    }));
    assert!(blocked_write.is_err(), "forming a mutable CPU slice while protected must be rejected");
    assert_eq!(plane.access_counts(), AccessCounts { reads: 1, writes: 1 });
    drop(guard);
    plane.reset_access_counts();
    assert_eq!(plane[0], 10);
    plane[0] = 11;
    assert_eq!(plane[0], 11);
    assert_eq!(plane.access_counts(), AccessCounts { reads: 2, writes: 1 });

    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = unsafe { plane.protect() }.unwrap();
        panic!("exercise protection guard during unwind");
    }));
    assert!(unwind.is_err());
    assert_eq!(plane[0], 11, "unwinding restored read access");
    plane[0] = 12;
    let guard = unsafe { plane.protect() }.unwrap();
    drop(plane);
    drop(guard); // The guard keeps the allocation alive and restores it before release.
}

#[test]
fn resident_plane_fault_probe_child() {
    let Ok(probe) = std::env::var(PROBE_ENV) else { return };
    #[cfg(unix)]
    {
        // A deliberate fault must not leave a core file in the checkout.
        let limit = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_CORE, &limit) }, 0);
    }
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Diagnostics::Debug::{
            SetErrorMode, SEM_FAILCRITICALERRORS, SEM_NOGPFAULTERRORBOX,
        };
        SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    }

    let mut rgb = isolated_plane(0x1020_3040);
    let mut aux = isolated_plane(0x5060_7080);
    let rgb_ptr = rgb.as_mut_ptr();
    let aux_ptr = aux.as_mut_ptr();
    let mut parts = probe.split('-');
    let ptr = match parts.next() {
        Some("rgb") => rgb_ptr,
        Some("aux") => aux_ptr,
        _ => panic!("invalid probe plane: {probe}"),
    };
    let operation = parts.next().expect("probe operation");
    let index = match parts.next() {
        Some("first") => 0,
        Some("last") => PLANE_WORDS - 1,
        _ => panic!("invalid probe edge: {probe}"),
    };
    assert!(parts.next().is_none());
    let _rgb_guard = unsafe { rgb.protect() }.unwrap();
    let _aux_guard = unsafe { aux.protect() }.unwrap();
    println!("resident-plane-probe-armed:{probe}");
    std::io::stdout().flush().unwrap();
    // SAFETY: intentionally violate OS page protection in this child only.
    // The pointer was cached before protection, bypassing the slice wrapper.
    unsafe {
        let ptr = ptr.add(index);
        match operation {
            "read" => { std::hint::black_box(ptr.read_volatile()); }
            "write" => ptr.write_volatile(0xaabb_ccdd),
            _ => panic!("invalid probe operation: {probe}"),
        }
    }
    std::process::exit(97); // A protected access must never reach here.
}

#[test]
fn resident_plane_raw_pointer_faults_in_child_processes() {
    let exe = std::env::current_exe().expect("locate current test binary");
    let module = module_path!().split_once("::").unwrap().1;
    let child_test = format!("{module}::resident_plane_fault_probe_child");
    for plane in ["rgb", "aux"] {
        for operation in ["read", "write"] {
            for edge in ["first", "last"] {
                let probe = format!("{plane}-{operation}-{edge}");
                let mut child = Command::new(&exe)
                    .args(["--exact", &child_test, "--nocapture", "--test-threads=1"])
                    .env(PROBE_ENV, &probe)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .expect("spawn protected access probe");
                let deadline = Instant::now() + Duration::from_secs(30);
                loop {
                    if child.try_wait().expect("poll protected access probe").is_some() { break; }
                    if Instant::now() >= deadline {
                        child.kill().expect("terminate timed-out probe");
                        let output = child.wait_with_output().unwrap();
                        panic!("protected {probe} probe timed out: {}", String::from_utf8_lossy(&output.stderr));
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                let output = child.wait_with_output().expect("collect protected access probe");
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(stdout.contains(&format!("resident-plane-probe-armed:{probe}")),
                    "{probe} failed before protection was armed: {stdout}\n{stderr}");
                #[cfg(windows)]
                assert_eq!(output.status.code().map(|code| code as u32), Some(0xc000_0005),
                    "{probe} did not cause STATUS_ACCESS_VIOLATION: {stdout}\n{stderr}");
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    assert_eq!(output.status.signal(), Some(libc::SIGSEGV),
                        "{probe} did not cause SIGSEGV: {stdout}\n{stderr}");
                }
                eprintln!("resident plane {probe}: protected raw pointer faulted as expected ({})", output.status);
            }
        }
    }
}

struct TestRex(Box<Rex3>);

impl Drop for TestRex {
    fn drop(&mut self) { self.0.stop(); }
}

fn required_gpu() -> Arc<Gpu> {
    let backend = std::env::var("REX3_RESIDENT_BACKEND").unwrap_or_else(|_| "auto".into());
    let expected = match backend.as_str() {
        "vulkan" => Some(wgpu::Backend::Vulkan),
        "dx12" => Some(wgpu::Backend::Dx12),
        "metal" => Some(wgpu::Backend::Metal),
        "gl" => Some(wgpu::Backend::Gl),
        "auto" => None,
        _ => panic!("unknown required resident backend {backend}"),
    };
    let gpu = Gpu::new(&GpuOptions { backend: backend.clone(), ..GpuOptions::default() })
        .unwrap_or_else(|error| panic!("required resident backend {backend}: {error}"));
    if let Some(expected) = expected {
        assert_eq!(gpu.caps.backend, expected, "the requested backend must actually be used");
    }
    assert!(
        matches!(gpu.caps.device_type, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu),
        "resident no-touch gate requires a hardware adapter: {}", gpu.caps,
    );
    eprintln!("resident no-touch adapter: {}", gpu.caps);
    gpu
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

fn context(plane: u32, rgbmode: bool, start: (i32, i32), end: (i32, i32), mask: u32, color: u32) -> Rex3Context {
    let mut ctx = Rex3Context::power_on_default();
    ctx.drawmode0 = DrawMode0(DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK_SH
        | (1 << 8) | (1 << 9) | (1 << 5));
    ctx.drawmode1 = DrawMode1(plane | (DRAWMODE1_DRAWDEPTH_24 << 3) | ((rgbmode as u32) << 15)
        | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH);
    ctx.clipmode = 0xf << CLIPMODE_CIDMATCH_SHIFT;
    ctx.wrmask = mask;
    if rgbmode {
        ctx.colorred = (color & 0xff) << 11;
        ctx.colorgrn = ((color >> 8) & 0xff) << 11;
        ctx.colorblue = ((color >> 16) & 0xff) << 11;
    } else {
        ctx.colorred = (color & 0x000f_ffff) << 11;
    }
    ctx.coloralpha = (color >> 24) << 11;
    ctx.xstart = (start.0 + REX3_COORD_BIAS) << 11;
    ctx.xsave = ctx.xstart;
    ctx.ystart = (start.1 + REX3_COORD_BIAS) << 11;
    ctx.xend = (end.0 + REX3_COORD_BIAS) << 11;
    ctx.yend = (end.1 + REX3_COORD_BIAS) << 11;
    ctx
}

fn enqueue(rex: &Rex3, ctx: &Rex3Context, start: (i32, i32), end: (i32, i32)) {
    write_reg(rex, REX3_CLIPMODE, ctx.clipmode, false);
    write_reg(rex, REX3_DRAWMODE1, ctx.drawmode1.0, false);
    write_reg(rex, REX3_WRMASK, ctx.wrmask, false);
    write_reg(rex, REX3_COLORRED, ctx.colorred, false);
    write_reg(rex, REX3_COLORGRN, ctx.colorgrn, false);
    write_reg(rex, REX3_COLORBLUE, ctx.colorblue, false);
    write_reg(rex, REX3_COLORALPHA, ctx.coloralpha, false);
    write_reg(rex, REX3_XYENDI, xy(end.0, end.1), false);
    write_reg(rex, REX3_XYSTARTI, xy(start.0, start.1), false);
    write_reg(rex, REX3_DRAWMODE0, ctx.drawmode0.0, true);
}

fn cpu_draw(ctx: &mut Rex3Context, rgb: &mut [u32], aux: &mut [u32]) {
    let fb = Framebuffers { rgb: rgb.as_mut_ptr(), aux: aux.as_mut_ptr() };
    let mode = rex3_shape::unpack(ctx.drawmode0.0, ctx.drawmode1.0, ctx.clipmode);
    draw_with_fb(ctx, &fb, &mode);
    ctx.hostrw_arm_single();
}

fn assert_plane_bytes(actual: &[u8], expected: &[u32], plane: &str) {
    let expected = bytemuck::cast_slice::<u32, u8>(expected);
    assert_eq!(actual.len(), expected.len(), "{plane} observation size");
    if let Some(byte) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        panic!("{plane} observation differs at word {}: GPU byte {:#04x}, CPU byte {:#04x}",
            byte / 4, actual[byte], expected[byte]);
    }
}

fn assert_plane_words(actual: &[u32], expected: &[u32], plane: &str) {
    assert_eq!(actual.len(), expected.len(), "{plane} shadow size");
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        panic!("{plane} shadow differs at word {index}: {:#010x} instead of {:#010x}",
            actual[index], expected[index]);
    }
}

#[test]
#[ignore = "requires the requested conformant resident GPU backend"]
fn resident_gpu_no_touch_gfifo_fence_and_direct_scanout() {
    std::thread::Builder::new().name("resident-plane-no-touch".into()).stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let gpu = required_gpu();
            let seed_rgb: Vec<u32> = (0..PLANE_WORDS as u32)
                .map(|i| i.wrapping_mul(0x9e37_79b9) ^ 0x1234_5678).collect();
            let seed_aux: Vec<u32> = (0..PLANE_WORDS as u32)
                .map(|i| (i.wrapping_mul(0x85eb_ca6b) ^ 0x8765_4321) & !0xff).collect();
            let mut expected_rgb = seed_rgb.clone();
            let mut expected_aux = seed_aux.clone();
            let mut rex = Box::new(Rex3::new(crate::disp::StatusSource::detached()));
            rex.fb_rgb.get_mut().copy_from_slice(&seed_rgb);
            rex.fb_aux.get_mut().copy_from_slice(&seed_aux);
            rex.fb_rgb.get_mut().isolate().unwrap();
            rex.fb_aux.get_mut().isolate().unwrap();
            let (rgb_base, rgb_bytes, rgb_page) = rex.fb_rgb.get_mut().mapping_range().unwrap();
            let (aux_base, aux_bytes, aux_page) = rex.fb_aux.get_mut().mapping_range().unwrap();
            assert_eq!(rgb_base % rgb_page, 0);
            assert_eq!(aux_base % aux_page, 0);
            assert!(rgb_bytes >= PLANE_BYTES as usize && aux_bytes >= PLANE_BYTES as usize);
            assert!(rgb_base + rgb_bytes <= aux_base || aux_base + aux_bytes <= rgb_base,
                "RGB and auxiliary pages must not overlap");

            reset_counts(&rex);
            rex.activate_resident_gpu(Arc::clone(&gpu)).expect("seed resident planes");
            let seed_accesses = counts(&rex);
            assert!(seed_accesses.0.reads + seed_accesses.0.writes > 0);
            assert!(seed_accesses.1.reads + seed_accesses.1.writes > 0);
            let initial = rex.resident_status().unwrap();
            assert_eq!((initial.initialization_uploads, initial.initialization_bytes), (2, 2 * PLANE_BYTES));
            assert_eq!(initial.observations, ObservationCounters::default());
            assert_eq!(initial.raw_plane_map_attempts, 0);
            reset_counts(&rex);
            let board = TestRex(rex);
            #[cfg(feature = "rex-jit")]
            assert!(board.0.rex_jit.is_some() && board.0.jit_enabled.load(Ordering::Relaxed),
                "the no-fallback gate requires an enabled REX3 JIT");
            // SAFETY: activation copied the seed synchronously, no CPU slice
            // remains live, and the stopped resident board has no refresh thread.
            let rgb_guard = unsafe { (&*board.0.fb_rgb.get()).protect() }.unwrap();
            let aux_guard = unsafe { (&*board.0.fb_aux.get()).protect() }.unwrap();
            board.0.start();
            assert!(board.0.refresh_thread.lock().is_none());

            let cases = [
                (DRAWMODE1_PLANES_RGB, true, (5, 8), (45, 14), 0x00ff_ffff, 0x0033_2211),
                (DRAWMODE1_PLANES_RGB, true, (31, 12), (51, 22), 0x0000_ff00, 0x00cc_bb99),
                (DRAWMODE1_PLANES_CID, false, (9, 10), (18, 19), 0x0000_00ff, 3),
                (DRAWMODE1_PLANES_OLAY, true, (30, 25), (39, 31), 0xffff_ffff, 0x0044_3322),
                (DRAWMODE1_PLANES_RGB, true, (1344, 3), (1350, 5), u32::MAX, 0x0044_3322),
            ];
            for (plane, rgbmode, start, end, mask, color) in cases.into_iter().take(3) {
                let mut expected = context(plane, rgbmode, start, end, mask, color);
                enqueue(&board.0, &expected, start, end);
                cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
            }
            board.0.gfifo_push(GFIFO_DISP_SYNC, 73);
            let deadline = Instant::now() + Duration::from_secs(30);
            while board.0.gfifo_fence.load(Ordering::Acquire) != 73 {
                assert!(Instant::now() < deadline, "resident display fence was not published");
                std::thread::yield_now();
            }
            let fence_status = board.0.resident_status().unwrap();
            assert_ne!(fence_status.accepted, initial.accepted, "real GFIFO fills were accepted");
            assert_eq!(fence_status.submitted, fence_status.accepted, "fence submits its prefix");
            assert_eq!(fence_status.observations, ObservationCounters::default());
            assert_eq!(fence_status.raw_plane_map_attempts, 0);
            assert_eq!(counts(&board.0), (AccessCounts::default(), AccessCounts::default()));

            for (plane, rgbmode, start, end, mask, color) in cases.into_iter().skip(3) {
                let mut expected = context(plane, rgbmode, start, end, mask, color);
                enqueue(&board.0, &expected, start, end);
                cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
            }
            board.0.wait_resident_idle().expect("idle and submit resident GFIFO");
            gpu.wait(None).expect("complete resident work");
            let idle = board.0.resident_status().unwrap();
            assert_ne!(idle.accepted, fence_status.accepted, "later GFIFO work reached idle submission");
            assert_eq!(idle.submitted, idle.accepted);
            assert_eq!(idle.completed, Some(idle.submitted));
            assert_eq!(idle.rgb.last_write, fence_status.rgb.last_write, "offscreen RGB fill is a no-op");
            assert_ne!(idle.aux.last_write, fence_status.aux.last_write, "later auxiliary fill ran");
            assert!(!idle.rgb.cpu_shadow_current() && !idle.aux.cpu_shadow_current());
            assert_eq!(idle.observations, ObservationCounters::default());
            assert_eq!(idle.raw_plane_map_attempts, 0);
            assert_eq!((idle.initialization_uploads, idle.initialization_bytes), (2, 2 * PLANE_BYTES));
            assert_eq!((idle.restore_uploads, idle.restore_bytes), (0, 0));
            assert_eq!(board.0.jit_go_count.load(Ordering::Relaxed), 0);
            assert_eq!(board.0.interp_go_count.load(Ordering::Relaxed), 0);
            assert_eq!(counts(&board.0), (AccessCounts::default(), AccessCounts::default()));

            let mut resident = &*board.0;
            let mut scan = Ng1Scanout::new_resident(&gpu, &resident).expect("bind board planes");
            let mut screen = board.0.screen.lock();
            screen.fb_rgb.clear();
            screen.fb_aux.clear();
            screen.refresh_display_state(&board.0.vc2, &board.0.xmap0, &board.0.cmap0, &board.0.bt445, &board.0.diag);
            screen.width = 64;
            screen.height = 48;
            screen.topscan = 1023;
            screen.did.fill(0);
            screen.xmap_mode[0] = (1 << 8) | (3 << 10);
            for (i, entry) in screen.ramdac_palette.iter_mut().enumerate() {
                *entry = (i as u32) * 0x0001_0101;
            }
            let state = screen.display_state();
            let frame = scan.compose(&state, &mut resident).expect("compose directly from resident planes");
            assert_eq!(frame.position, idle.submitted);
            let frame_bytes = u64::from(frame.width) * u64::from(frame.height) * 4;
            let consumer = gpu.buffer("resident test composed frame consumer", frame_bytes,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
            let mut encoder = gpu.encoder("resident test composed frame consumption");
            encoder.copy_buffer_to_buffer(frame.buffer, 0, &consumer, 0, frame_bytes);
            gpu.submit(encoder);
            let traffic = scan.traffic();
            assert_eq!(traffic.frames_submitted, 1);
            assert_eq!(traffic.raw_plane_uploads, Default::default());
            assert_eq!(traffic.raw_plane_readbacks, Default::default());
            let direct_status = board.0.resident_status().unwrap();
            assert_eq!(direct_status.observations, ObservationCounters::default());
            assert_eq!(direct_status.raw_plane_map_attempts, 0);
            assert_eq!(counts(&board.0), (AccessCounts::default(), AccessCounts::default()));
            board.0.stop();
            assert_eq!(counts(&board.0), (AccessCounts::default(), AccessCounts::default()));
            let stopped = board.0.resident_status().unwrap();
            assert_eq!(stopped.observations, ObservationCounters::default());
            assert_eq!(stopped.raw_plane_map_attempts, 0);
            assert_eq!(unsafe { (&*board.0.fb_rgb.get()).mapping_range() }, Some((rgb_base, rgb_bytes, rgb_page)));
            assert_eq!(unsafe { (&*board.0.fb_aux.get()).mapping_range() }, Some((aux_base, aux_bytes, aux_page)));
            drop(aux_guard);
            drop(rgb_guard);

            let shadow_rgb = unsafe { &*board.0.fb_rgb.get() };
            let shadow_aux = unsafe { &*board.0.fb_aux.get() };
            assert_plane_words(shadow_rgb, &seed_rgb, "CPU RGB");
            assert_plane_words(shadow_aux, &seed_aux, "CPU aux");
            assert!(shadow_rgb.as_ref() != expected_rgb.as_slice(), "RGB shadow must be stale");
            assert!(shadow_aux.as_ref() != expected_aux.as_slice(), "aux shadow must be stale");
            let before_observe = counts(&board.0);
            let observation = match &mut *board.0.raster.lock() {
                RasterMode::Resident(raster) => raster.observe(ObservationReason::Validation, PlaneMask::Both).unwrap(),
                _ => panic!("resident mode disappeared"),
            };
            assert_eq!(counts(&board.0), before_observe, "observation does not install a CPU shadow");
            assert_eq!(observation.position(), idle.accepted);
            assert_plane_bytes(observation.rgb().unwrap(), &expected_rgb, "RGB");
            assert_plane_bytes(observation.aux().unwrap(), &expected_aux, "aux");
            let observed = board.0.resident_status().unwrap();
            assert_eq!(observed.raw_plane_map_attempts, 1);
            let readback = observed.observations.for_reason(ObservationReason::Validation);
            assert_eq!((readback.attempts, readback.successes), (1, 1));
            assert_eq!((readback.attempted_transfers, readback.successful_transfers), (2, 2));
            assert_eq!((readback.attempted_bytes, readback.successful_bytes), (2 * PLANE_BYTES, 2 * PLANE_BYTES));

            let cpu_source = CompositorSource {
                fb_rgb: &expected_rgb, fb_aux: &expected_aux, fb_rgb_damage: &[], fb_aux_damage: &[],
                did: state.did, xmap_mode: state.xmap_mode, xmap_config: state.xmap_config,
                xmap_cursor_cmap: state.xmap_cursor_cmap, xmap_popup_cmap: state.xmap_popup_cmap,
                cmap: state.cmap, ramdac: state.ramdac, vc2_ram: state.vc2_ram, vc2_regs: state.vc2_regs,
                topscan: state.topscan, cursor_x_adjust: state.cursor_x_adjust,
                width: state.width, height: state.height, status_bar_only: false, prebuilt: None,
            };
            let mut software = SwCompositor::new();
            software.compose_pixels(&cpu_source);
            let bytes = Staging::default().read(&gpu, &consumer, &[(0, frame_bytes)]).unwrap();
            for y in 0..state.height {
                for x in 0..state.width {
                    let pixel = software.pixels()[y * crate::disp::FB_STRIDE + x];
                    let expected = ((pixel & 0xff) << 16) | (pixel & 0xff00) | ((pixel >> 16) & 0xff);
                    let at = (y * state.width + x) * 4;
                    let actual = u32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap());
                    assert_eq!(actual, expected, "direct scanout pixel ({x}, {y})");
                }
            }
            eprintln!("resident no-touch: accepted fills, fence, idle, direct composition and exit; \
                CPU accesses=0/0, raw observations=0 before explicit 16 MiB observation");
        })
        .expect("spawn resident no-touch test")
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}
