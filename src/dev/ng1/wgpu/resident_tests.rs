use super::*;
use crate::dev::ng1::rex3::{
    DrawMode0, DrawMode1, Rex3, DRAWMODE0_ADRMODE_A_LINE_SH, DRAWMODE0_ADRMODE_BLOCK_SH,
    DRAWMODE0_ADRMODE_F_LINE_SH, DRAWMODE0_ADRMODE_I_LINE_SH, DRAWMODE0_ADRMODE_SPAN_SH,
    DRAWMODE0_OPCODE_DRAW, DRAWMODE0_OPCODE_NOOP, DRAWMODE0_OPCODE_READ,
    DRAWMODE0_OPCODE_SCR2SCR,
    DRAWMODE1_COMPARE_DISABLE_SH,
    DRAWMODE1_COMPARE_GT, DRAWMODE1_COMPARE_SHIFT, DRAWMODE1_DRAWDEPTH_4,
    DRAWMODE1_DRAWDEPTH_8, DRAWMODE1_DRAWDEPTH_12, DRAWMODE1_DRAWDEPTH_24,
    DRAWMODE1_LOGICOP_AND_SH, DRAWMODE1_LOGICOP_MASK, DRAWMODE1_LOGICOP_NOR_SH,
    DRAWMODE1_LOGICOP_NSRC_SH, DRAWMODE1_LOGICOP_OR_SH, DRAWMODE1_LOGICOP_SRC_SH,
    DRAWMODE1_LOGICOP_XNOR_SH, DRAWMODE1_LOGICOP_XOR_SH, DRAWMODE1_PLANES_CID,
    DRAWMODE1_PLANES_NONE, DRAWMODE1_PLANES_OLAY, DRAWMODE1_PLANES_PUP,
    DRAWMODE1_PLANES_RGB, DRAWMODE1_PLANES_RGBA,
    CLIPMODE_CIDMATCH_SHIFT, CLIPMODE_ENSMASK_SMASK0, GFIFO_DISP_SYNC, HOSTRW_BUF_QWORDS, REX3_BASE, REX3_CLIPMODE, REX3_COLORALPHA,
    REX3_COLORBLUE, REX3_COLORGRN, REX3_COLORRED, REX3_COORD_BIAS, REX3_DRAWMODE0,
    REX3_DRAWMODE1, REX3_WRMASK, REX3_XYENDI, REX3_XYSTARTI,
};
use crate::dev::ng1::rex3_generic::{draw_with_fb, fline_setup, Framebuffers};
use crate::dev::ng1::rex3_shape;
use crate::traits::{BusDevice, Device, Resettable, BUS_BUSY, BUS_OK};
use crate::wgpu::GpuOptions;

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("resident-rex3-test".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(f)
        .expect("spawn resident test")
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}

fn required_gpu() -> Arc<Gpu> {
    let backend = std::env::var("REX3_RESIDENT_BACKEND").unwrap_or_else(|_| "auto".into());
    let gpu = Gpu::new(&GpuOptions { backend: backend.clone(), ..GpuOptions::default() })
        .unwrap_or_else(|error| panic!("required resident backend {backend}: {error}"));
    let expected = match backend.as_str() {
        "vulkan" => Some(wgpu::Backend::Vulkan),
        "dx12" => Some(wgpu::Backend::Dx12),
        "metal" => Some(wgpu::Backend::Metal),
        "gl" => Some(wgpu::Backend::Gl),
        "auto" => None,
        _ => panic!("unknown required backend {backend}"),
    };
    if let Some(expected) = expected {
        assert_eq!(gpu.caps.backend, expected, "the requested backend must actually be used");
    }
    assert!(
        matches!(gpu.caps.device_type, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu),
        "resident gate requires a hardware adapter: {}", gpu.caps,
    );
    eprintln!("resident test adapter: {}", gpu.caps);
    gpu
}

fn seeded_plane(salt: u32) -> Vec<u32> {
    (0..PLANE_WORDS as u32)
        .map(|index| index.wrapping_mul(0x9e37_79b9) ^ salt)
        .collect()
}

fn compare_observed(actual: &[u8], expected: &[u32], label: &str) {
    let expected = bytemuck::cast_slice::<u32, u8>(expected);
    assert_eq!(actual.len(), expected.len(), "{label} size");
    if let Some(byte) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        let index = byte / 4;
        let actual = u32::from_ne_bytes(actual[index * 4..index * 4 + 4].try_into().unwrap());
        let expected = u32::from_ne_bytes(expected[index * 4..index * 4 + 4].try_into().unwrap());
        panic!(
            "{label} at ({}, {}): GPU {actual:#010x}, CPU {expected:#010x}",
            index % STRIDE as usize,
            index / STRIDE as usize,
        );
    }
}

fn compare_plane(actual: &[u32], expected: &[u32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        panic!(
            "{label} at ({}, {}): GPU {:#010x}, CPU {:#010x}",
            index % STRIDE as usize,
            index / STRIDE as usize,
            actual[index],
            expected[index],
        );
    }
}

fn install_plane(shadow: &mut [u32], bytes: &[u8]) {
    bytemuck::cast_slice_mut::<u32, u8>(shadow).copy_from_slice(bytes);
}

fn context(
    plane: u32,
    rgbmode: bool,
    span: bool,
    start: (i32, i32),
    end: (i32, i32),
    mask: u32,
    color: u32,
) -> Rex3Context {
    let mut ctx = Rex3Context::power_on_default();
    let shape = if span { DRAWMODE0_ADRMODE_SPAN_SH | (1 << 8) } else {
        DRAWMODE0_ADRMODE_BLOCK_SH | (1 << 8) | (1 << 9)
    };
    ctx.drawmode0 = DrawMode0(DRAWMODE0_OPCODE_DRAW | shape);
    ctx.drawmode1 = DrawMode1(
        plane | (DRAWMODE1_DRAWDEPTH_24 << 3) | ((rgbmode as u32) << 15)
            | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH,
    );
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
    ctx.pat_bit = 7;
    ctx.zpat_bit = 9;
    ctx.host_cursor = 1;
    ctx.host_shifter = 0x1234_5678_9abc_def0;
    ctx.hostrw[0] = 0x0123_4567_89ab_cdef;
    ctx.hostrw[HOSTRW_BUF_QWORDS / 2] = 0xa5a5_5a5a_f0f0_0f0f;
    ctx.hostrw[HOSTRW_BUF_QWORDS - 1] = 0xfedc_ba98_7654_3210;
    ctx
}

fn line_context(
    shape: u32,
    plane: u32,
    rgbmode: bool,
    start: (i32, i32),
    end: (i32, i32),
    mask: u32,
    color: u32,
) -> Rex3Context {
    let mut ctx = context(plane, rgbmode, true, (0, 0), (0, 0), mask, color);
    ctx.drawmode0 = DrawMode0(DRAWMODE0_OPCODE_DRAW | shape | (1 << 8) | (1 << 9));
    let fixed = |sixteenths: i32| (REX3_COORD_BIAS << 11) + sixteenths * 128;
    ctx.xstart = fixed(start.0);
    ctx.xsave = ctx.xstart;
    ctx.ystart = fixed(start.1);
    ctx.xend = fixed(end.0);
    ctx.yend = fixed(end.1);
    if shape == DRAWMODE0_ADRMODE_I_LINE_SH {
        let dx = ctx.xend - ctx.xstart;
        let dy = ctx.yend - ctx.ystart;
        let adx = dx.abs() >> 11;
        let ady = dy.abs() >> 11;
        let mut octant = 0;
        if dy < 0 { octant |= crate::dev::ng1::rex3::OCTANT_YDEC; }
        if dx < 0 { octant |= crate::dev::ng1::rex3::OCTANT_XDEC; }
        if adx > ady { octant |= crate::dev::ng1::rex3::OCTANT_XMAJOR; }
        let (major, minor) = if adx > ady { (adx, ady) } else { (ady, adx) };
        let incr1 = 2 * minor;
        let incr2 = 2 * (minor - major);
        ctx.bresoctinc1.set_octant(octant);
        ctx.bresoctinc1.set_incr1(incr1 as u32 & 0x000f_ffff);
        ctx.bresrndinc2.set_incr2(incr2 as u32 & 0x001f_ffff);
        ctx.bresd = ((incr1 - major) as u32) & 0x07ff_ffff;
    } else {
        fline_setup(&mut ctx);
    }
    ctx
}

fn assert_context_eq(actual: &Rex3Context, expected: &Rex3Context, label: &str) {
    macro_rules! fields {
        ($($field:ident),+ $(,)?) => {
            $(assert_eq!(actual.$field, expected.$field, "{label}: {}", stringify!($field));)+
        };
    }
    macro_rules! registers {
        ($($field:ident),+ $(,)?) => {
            $(assert_eq!(actual.$field.0, expected.$field.0, "{label}: {}", stringify!($field));)+
        };
    }
    registers!(drawmode1, drawmode0, lsmode, bresoctinc1, bresrndinc2);
    fields!(
        lspattern, lspatsave, zpattern, colorback, colorvram, alpharef, smask0x, smask0y,
        xstart, ystart, xend, yend, xsave, xymove, bresd, bress1, brese1, bress2,
        aweight0, aweight1, colorred, coloralpha, colorgrn, colorblue, slopered, slopealpha,
        slopegrn, slopeblue, wrmask, colorx, smask1x, smask1y, smask2x, smask2y, smask3x,
        smask3y, smask4x, smask4y, topscan, xywin, clipmode, host_shifter, hostcnt,
        host_cursor, host_len, pat_bit, zpat_bit, mid_primitive, lssave, lsrestore, stepz,
        stall0, stall1,
    );
    if let Some(index) = actual.hostrw.0.iter().zip(&expected.hostrw.0).position(|(a, b)| a != b) {
        panic!(
            "{label}: hostrw[{index}] is {:#018x}, expected {:#018x}",
            actual.hostrw[index], expected.hostrw[index],
        );
    }
    // `host` is a nonarchitectural owner pointer and differs between boards.
}

fn context_state(ctx: &Rex3Context) -> (i32, i32, u8, u8, bool, u32, u32, u32, u32, u32, u32) {
    (
        ctx.xstart, ctx.ystart, ctx.pat_bit, ctx.zpat_bit, ctx.mid_primitive,
        ctx.host_len, ctx.host_cursor, ctx.colorred, ctx.colorgrn, ctx.colorblue, ctx.coloralpha,
    )
}

fn cpu_draw(ctx: &mut Rex3Context, rgb: &mut [u32], aux: &mut [u32]) {
    let fb = Framebuffers { rgb: rgb.as_mut_ptr(), aux: aux.as_mut_ptr() };
    let mode = rex3_shape::unpack(ctx.drawmode0.0, ctx.drawmode1.0, ctx.clipmode);
    draw_with_fb(ctx, &fb, &mode);
    ctx.hostrw_arm_single();
}

fn cpu_host_draw(ctx: &mut Rex3Context, rgb: &mut [u32], aux: &mut [u32]) {
    let fb = Framebuffers { rgb: rgb.as_mut_ptr(), aux: aux.as_mut_ptr() };
    let mode = rex3_shape::unpack(ctx.drawmode0.0, ctx.drawmode1.0, ctx.clipmode);
    draw_with_fb(ctx, &fb, &mode);
    while ctx.host_len > 1 && ctx.host_cursor < ctx.host_len {
        let before = ctx.host_cursor;
        ctx.hostcnt = 0;
        draw_with_fb(ctx, &fb, &mode);
        if ctx.host_cursor == before {
            break;
        }
    }
    ctx.host_len = 1;
    ctx.host_cursor = 0;
}

#[test]
fn resident_limits_cover_plane_bindings_and_batch_dispatch() {
    let good = wgpu::Limits::default();
    validate_limits(&good).expect("default limits cover resident raster");
    let checks: [(&str, Box<dyn Fn(&mut wgpu::Limits)>); 9] = [
        ("max_buffer_size", Box::new(|l| l.max_buffer_size = READBACK_BYTES - 1)),
        ("max_storage_buffer_binding_size", Box::new(|l| l.max_storage_buffer_binding_size = PLANE_BYTES - 1)),
        ("max_storage_buffers_per_shader_stage", Box::new(|l| l.max_storage_buffers_per_shader_stage = 7)),
        ("max_bind_groups", Box::new(|l| l.max_bind_groups = 2)),
        ("max_bindings_per_bind_group", Box::new(|l| l.max_bindings_per_bind_group = 5)),
        ("max_compute_invocations_per_workgroup", Box::new(|l| l.max_compute_invocations_per_workgroup = 63)),
        ("max_compute_workgroup_size_x", Box::new(|l| l.max_compute_workgroup_size_x = 63)),
        ("max_compute_workgroup_storage_size", Box::new(|l| l.max_compute_workgroup_storage_size = 100)),
        ("max_compute_workgroups_per_dimension", Box::new(|l| l.max_compute_workgroups_per_dimension = 100)),
    ];
    for (name, reduce) in checks {
        let mut limits = good.clone();
        reduce(&mut limits);
        assert!(matches!(validate_limits(&limits), Err(ResidentError::Limit { name: actual, .. }) if actual == name));
    }
}

#[test]
fn destination_hazards_follow_logic_and_blend_sources() {
    on_large_stack(|| {
        let mut ctx = context(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (0, 0),
            (1, 1),
            u32::MAX,
            0x7f12_3456,
        );
        assert!(!reads_destination(&ctx));
        ctx.drawmode1.0 = (ctx.drawmode1.0 & !DRAWMODE1_LOGICOP_MASK)
            | DRAWMODE1_LOGICOP_XOR_SH;
        assert!(reads_destination(&ctx));
        ctx.drawmode1.0 |= 1 << 18;
        assert!(reads_destination(&ctx));
        ctx.drawmode1.0 |= 1 << 25;
        assert!(!reads_destination(&ctx));
    });
}

#[test]
fn resident_positions_confirm_only_submitted_current_epoch_work() {
    let mut status = ResidentStatus::seeded(Epoch(41));
    let seed = status.accepted;
    assert_eq!(status.completed, None);
    assert!(status.rgb.cpu_shadow_current() && status.aux.cpu_shadow_current());
    let no_op = status.next_position().unwrap();
    status.accept(no_op, None);
    assert_eq!((status.rgb.last_write, status.aux.last_write), (seed, seed));
    status.confirm(no_op);
    assert_eq!(status.completed, None, "unsubmitted work is not complete");
    status.submit(no_op);
    status.confirm(no_op);
    assert_eq!(status.completed, Some(no_op));
    let rgb = status.next_position().unwrap();
    status.accept(rgb, Some(FillPlane::Rgb));
    assert!(!status.rgb.cpu_shadow_current());
    assert!(status.aux.cpu_shadow_current());
    status.submit(rgb);
    status.confirm(rgb);
    status.confirm(no_op);
    assert_eq!(status.completed, Some(rgb), "out-of-order callbacks cannot regress completion");

    let mut reset = ResidentStatus::seeded(Epoch(42));
    reset.confirm(no_op);
    assert_eq!(reset.completed, None, "an old callback cannot complete a new epoch");
    reset.confirm(reset.submitted);
    assert_eq!(reset.completed, Some(reset.submitted));
    reset.accepted.sequence = Sequence(u64::MAX);
    assert!(matches!(reset.next_position(), Err(ResidentError::InvalidState(_))));
}

#[test]
fn resident_observation_accounting_is_typed_and_survives_epoch_changes() {
    let mut status = ResidentStatus::seeded(Epoch(51));
    let seed = status.accepted;
    status.observations.for_reason_mut(ObservationReason::Snapshot).attempt(PlaneMask::Both, seed).unwrap();
    let next = status.next_position().unwrap();
    status.accept(next, Some(FillPlane::Rgb));
    status.observations.for_reason_mut(ObservationReason::Snapshot).attempt(PlaneMask::Rgb, next).unwrap();
    status.observations.for_reason_mut(ObservationReason::Snapshot).success(PlaneMask::Rgb, next).unwrap();
    let snapshot = status.observations.for_reason(ObservationReason::Snapshot);
    assert_eq!((snapshot.attempts, snapshot.successes), (2, 1));
    assert_eq!((snapshot.attempted_transfers, snapshot.successful_transfers), (3, 1));
    assert_eq!((snapshot.attempted_bytes, snapshot.successful_bytes), (3 * PLANE_BYTES, PLANE_BYTES));
    assert_eq!((snapshot.last_attempted, snapshot.last_successful), (Some(next), Some(next)));
    assert_eq!(*status.observations.for_reason(ObservationReason::HostRead), ObservationCounts::default());
    status.raw_plane_map_attempts = 7;

    let restored = status.restored(Epoch(52), 3, 0).unwrap();
    assert_eq!(restored.observations, status.observations);
    assert_eq!(restored.raw_plane_map_attempts, 7);
    assert_eq!((restored.restore_uploads, restored.restore_bytes), (1, 12));
    assert_eq!(restored.rgb.cpu_shadow, None);
    assert!(restored.aux.cpu_shadow_current());
    let reset = ResidentStatus::reseeded(Epoch(53), &restored).unwrap();
    assert_eq!(reset.observations, status.observations);
    assert_eq!(reset.raw_plane_map_attempts, 7);
    assert_eq!((reset.initialization_uploads, reset.initialization_bytes), (4, 4 * PLANE_BYTES));
    assert_eq!((reset.restore_uploads, reset.restore_bytes), (1, 12));
    assert!(reset.rgb.cpu_shadow_current() && reset.aux.cpu_shadow_current());
}

#[test]
fn resident_defaults_fail_closed_and_injected_faults_are_sticky() {
    struct Backend {
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ResidentBackend for Backend {
        fn enqueue(&mut self, _: &mut Rex3Context) -> Result<(), ResidentError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn recorded_status(&self) -> Result<ResidentStatus, ResidentError> {
            Ok(ResidentStatus::seeded(Epoch(99)))
        }

        fn reset(&mut self, _: &[u32], _: &[u32]) -> Result<(), ResidentError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    on_large_stack(|| {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut backend = Backend { calls: Arc::clone(&calls) };
        assert_eq!(backend.submit(), Err(ResidentError::Unsupported("submit")));
        assert_eq!(
            backend.observe(ObservationReason::Monitor, PlaneMask::Rgb).unwrap_err(),
            ResidentError::Unsupported("observation"),
        );
        assert_eq!(backend.restore(&[], &[]), Err(ResidentError::Unsupported("restore")));
        assert_eq!(backend.status(), Err(ResidentError::Unsupported("status")));

        for (injection, expected) in [
            (ResidentFaultInjection::DeviceLost, ResidentError::DeviceLost("injected device loss".into())),
            (ResidentFaultInjection::DeviceError, ResidentError::Failed("injected device error".into())),
        ] {
            let mut raster = ResidentRaster::new(Backend { calls: Arc::clone(&calls) });
            raster.inject_fault_for_test(injection);
            assert_eq!(raster.submit(), Err(expected.clone()));
            assert_eq!(raster.observe(ObservationReason::Snapshot, PlaneMask::Both).unwrap_err(), expected);
            assert_eq!(raster.restore(&[], &[]), Err(expected.clone()));
            assert_eq!(raster.reset(&[], &[]), Err(expected.clone()));
            assert_eq!(raster.status(), Err(expected.clone()));
            assert_eq!(raster.enqueue(&mut Rex3Context::power_on_default()), Err(expected.clone()));
            raster.fail(ResidentError::Failed("later error".into()));
            let diagnostics = raster.diagnostics().unwrap();
            assert_eq!(diagnostics.fault, Some(expected));
            assert_eq!(diagnostics.status.accepted.epoch, Epoch(99));
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_observation_selection_positions_and_shadow_acknowledgement() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let seed_rgb = seeded_plane(0x1020_3040);
        let seed_aux = seeded_plane(0xf0e0_d0c0);
        let mut shadow_rgb = seed_rgb.clone();
        let mut shadow_aux = seed_aux.clone();
        let resident = Arc::new(Mutex::new(ResidentGpu::activate(
            Arc::clone(&gpu), &shadow_rgb, &shadow_aux,
        ).unwrap()));
        let staging = resident.lock().readback.clone();
        assert_eq!(staging.size(), READBACK_BYTES);
        assert_eq!(staging.usage(), wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let mut raster = ResidentRaster::new(SharedResident(Arc::clone(&resident)));
        let mut expected_rgb = seed_rgb.clone();
        let mut expected_aux = seed_aux.clone();
        let draws = [
            context(DRAWMODE1_PLANES_RGB, true, false, (0, 0), (2, 2), u32::MAX, 0x1234_5678),
            context(DRAWMODE1_PLANES_CID, false, false, (4, 4), (5, 5), u32::MAX, 0x8765_4321),
            context(DRAWMODE1_PLANES_RGB, true, false, (7, 7), (9, 9), 0, 0x1122_3344),
        ];
        for mut draw in draws {
            let mut reference = draw;
            raster.enqueue(&mut draw).unwrap();
            cpu_draw(&mut reference, &mut expected_rgb, &mut expected_aux);
        }
        let pending = raster.status().unwrap();
        assert_eq!(pending.accepted.sequence, Sequence(3));
        assert_eq!(pending.submitted.sequence, Sequence(0));
        assert_eq!((pending.rgb.last_write.sequence, pending.aux.last_write.sequence), (Sequence(1), Sequence(2)));
        assert_eq!(pending.observations, ObservationCounters::default());
        assert_eq!(pending.raw_plane_map_attempts, 0);

        let both = raster.observe(ObservationReason::Snapshot, PlaneMask::Both).unwrap();
        assert_eq!((both.position(), both.reason(), both.planes()), (pending.accepted, ObservationReason::Snapshot, PlaneMask::Both));
        assert_eq!(both.rgb_generation(), Some(pending.rgb.last_write));
        assert_eq!(both.aux_generation(), Some(pending.aux.last_write));
        compare_observed(both.rgb().unwrap(), &expected_rgb, "snapshot RGB");
        compare_observed(both.aux().unwrap(), &expected_aux, "snapshot aux");
        let observed = raster.status().unwrap();
        assert_eq!(observed.raw_plane_map_attempts, 1, "both planes share one explicit staging map");
        assert_eq!((observed.submitted, observed.completed), (pending.accepted, Some(pending.accepted)));
        assert_eq!((observed.rgb.last_observed, observed.aux.last_observed), (Some(both.position()), Some(both.position())));
        assert!(!observed.rgb.cpu_shadow_current() && !observed.aux.cpu_shadow_current(), "returning bytes does not install a shadow");
        let counts = observed.observations.for_reason(ObservationReason::Snapshot);
        assert_eq!((counts.attempts, counts.successes), (1, 1));
        assert_eq!((counts.attempted_transfers, counts.successful_transfers), (2, 2));
        assert_eq!((counts.attempted_bytes, counts.successful_bytes), (READBACK_BYTES, READBACK_BYTES));
        assert_eq!((counts.last_attempted, counts.last_successful), (Some(both.position()), Some(both.position())));

        assert!(matches!(
            raster.acknowledge_cpu_shadow(&both, Some(&shadow_rgb[..100]), None),
            Err(ResidentError::InvalidShadow { plane: "RGB", .. })
        ));
        assert!(matches!(
            raster.acknowledge_cpu_shadow(&both, Some(&shadow_rgb), None),
            Err(ResidentError::InvalidShadow { plane: "RGB", .. })
        ));
        install_plane(&mut shadow_rgb, both.rgb().unwrap());
        let impostor = shadow_rgb.clone();
        assert!(matches!(
            raster.acknowledge_cpu_shadow(&both, Some(&impostor), None),
            Err(ResidentError::InvalidShadow { plane: "RGB", reason: "not the activated board shadow" })
        ));
        assert!(matches!(
            raster.acknowledge_cpu_shadow(&both, Some(&shadow_rgb), Some(&shadow_aux)),
            Err(ResidentError::InvalidShadow { plane: "auxiliary", .. })
        ));
        assert!(!raster.status().unwrap().rgb.cpu_shadow_current(), "two-plane acknowledgement is atomic");
        install_plane(&mut shadow_aux, both.aux().unwrap());
        raster.acknowledge_cpu_shadow(&both, Some(&shadow_rgb), Some(&shadow_aux)).unwrap();
        let installed = raster.status().unwrap();
        assert!(installed.rgb.cpu_shadow_current() && installed.aux.cpu_shadow_current());
        assert_eq!(installed.rgb.cpu_shadow, both.rgb_generation());
        assert_eq!(installed.aux.cpu_shadow, both.aux_generation());

        let mut draw = context(DRAWMODE1_PLANES_RGB, true, false, (3, 1), (4, 1), u32::MAX, 0x0099_8877);
        let mut reference = draw;
        raster.enqueue(&mut draw).unwrap();
        cpu_draw(&mut reference, &mut expected_rgb, &mut expected_aux);
        let rgb = raster.observe(ObservationReason::HostRead, PlaneMask::Rgb).unwrap();
        assert_eq!(rgb.position().sequence, Sequence(4));
        assert_eq!(rgb.rgb_generation(), Some(rgb.position()));
        assert_eq!(rgb.aux_generation(), None);
        assert!(rgb.aux().is_none());
        compare_observed(rgb.rgb().unwrap(), &expected_rgb, "selected RGB");

        let mut later = context(DRAWMODE1_PLANES_RGB, true, false, (8, 2), (9, 2), u32::MAX, 0x0077_6655);
        let mut reference = later;
        raster.enqueue(&mut later).unwrap();
        cpu_draw(&mut reference, &mut expected_rgb, &mut expected_aux);
        raster.acknowledge_cpu_shadow(&both, Some(&shadow_rgb), None).unwrap();
        let older = raster.status().unwrap();
        assert_eq!(older.rgb.cpu_shadow, both.rgb_generation());
        assert!(!older.rgb.cpu_shadow_current(), "an old full copy cannot acknowledge a newer write");

        let byte = (STRIDE as usize + 3) * size_of::<u32>();
        let shadow_bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut shadow_rgb);
        shadow_bytes[byte..byte + 4].copy_from_slice(&rgb.rgb().unwrap()[byte..byte + 4]);
        assert!(matches!(
            raster.acknowledge_cpu_shadow(&rgb, Some(&shadow_rgb), None),
            Err(ResidentError::InvalidShadow { plane: "RGB", .. })
        ));
        assert_eq!(raster.status().unwrap().rgb.cpu_shadow, both.rgb_generation());
        install_plane(&mut shadow_rgb, rgb.rgb().unwrap());
        raster.acknowledge_cpu_shadow(&rgb, Some(&shadow_rgb), None).unwrap();
        let out_of_order = raster.status().unwrap();
        assert_eq!(out_of_order.rgb.cpu_shadow, rgb.rgb_generation());
        assert!(!out_of_order.rgb.cpu_shadow_current());

        let latest = raster.observe(ObservationReason::Validation, PlaneMask::Rgb).unwrap();
        assert_eq!(latest.position().sequence, Sequence(5));
        compare_observed(latest.rgb().unwrap(), &expected_rgb, "latest RGB");
        install_plane(&mut shadow_rgb, latest.rgb().unwrap());
        raster.acknowledge_cpu_shadow(&latest, Some(&shadow_rgb), None).unwrap();
        assert!(raster.status().unwrap().rgb.cpu_shadow_current());
        let aux = raster.observe(ObservationReason::Monitor, PlaneMask::Aux).unwrap();
        assert_eq!(aux.position(), latest.position());
        assert_eq!(aux.rgb_generation(), None);
        assert_eq!(aux.aux_generation(), both.aux_generation());
        assert!(aux.rgb().is_none());
        compare_observed(aux.aux().unwrap(), &expected_aux, "selected aux");
        let counts = raster.status().unwrap();
        assert_eq!(counts.raw_plane_map_attempts, 4);
        for reason in [ObservationReason::HostRead, ObservationReason::Monitor, ObservationReason::Validation] {
            let count = counts.observations.for_reason(reason);
            assert_eq!((count.attempts, count.successes), (1, 1));
            assert_eq!((count.attempted_transfers, count.successful_transfers), (1, 1));
            assert_eq!((count.attempted_bytes, count.successful_bytes), (PLANE_BYTES, PLANE_BYTES));
        }
        assert_eq!(resident.lock().readback, staging, "one staging allocation serves every selection");

        shadow_rgb.copy_from_slice(&seed_rgb);
        shadow_aux.copy_from_slice(&seed_aux);
        raster.reset(&shadow_rgb, &shadow_aux).unwrap();
        let reset = raster.status().unwrap();
        assert_ne!(reset.accepted.epoch, both.position().epoch);
        assert_eq!(reset.initialization_uploads, 4);
        assert_eq!(reset.observations, counts.observations, "observation totals survive reset");
        assert_eq!(reset.raw_plane_map_attempts, counts.raw_plane_map_attempts);
        assert_eq!(resident.lock().readback, staging, "reset retains the staging allocation");
        assert!(matches!(
            raster.acknowledge_cpu_shadow(&both, Some(&shadow_rgb), None),
            Err(ResidentError::InvalidState("observation belongs to an old epoch"))
        ));
        assert!(raster.status().unwrap().rgb.cpu_shadow_current());
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_partial_restore_keeps_suffix_and_rebases_stream() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let shadow_rgb = seeded_plane(0x7634_1209);
        let shadow_aux = seeded_plane(0x9087_6543);
        let mut expected_rgb = shadow_rgb.clone();
        let mut expected_aux = shadow_aux.clone();
        let resident = Arc::new(Mutex::new(ResidentGpu::activate(
            Arc::clone(&gpu), &shadow_rgb, &shadow_aux,
        ).unwrap()));
        let staging = resident.lock().readback.clone();
        let mut raster = ResidentRaster::new(SharedResident(Arc::clone(&resident)));
        let old_capture = raster.observe(ObservationReason::Validation, PlaneMask::Rgb).unwrap();
        let mut draw = context(DRAWMODE1_PLANES_RGB, true, false, (40, 50), (43, 52), u32::MAX, 0x1256_789a);
        let mut reference = draw;
        raster.enqueue(&mut draw).unwrap();
        cpu_draw(&mut reference, &mut expected_rgb, &mut expected_aux);
        let pending = raster.status().unwrap();
        assert_eq!(pending.accepted.sequence, Sequence(1));
        assert_eq!(pending.submitted.sequence, Sequence(0));
        let prefix_rgb = [0x1111_2222, 0x3333_4444, 0x5555_6666];
        raster.restore(&prefix_rgb, &[]).unwrap();
        assert_context_eq(&draw, &reference, "context after partial-prefix restore");
        expected_rgb[..prefix_rgb.len()].copy_from_slice(&prefix_rgb);
        let restored = raster.status().unwrap();
        assert_ne!(restored.accepted.epoch, pending.accepted.epoch);
        assert_eq!(restored.accepted.sequence, Sequence(0));
        assert_eq!(restored.submitted, restored.accepted);
        assert_eq!((restored.restore_uploads, restored.restore_bytes), (1, 12));
        assert_eq!(restored.raw_plane_map_attempts, 1, "restore adds no raw-plane map");
        assert_eq!(restored.rgb.cpu_shadow, None);
        assert!(restored.aux.cpu_shadow_current(), "unchanged current shadow survives rebase");
        assert_eq!(
            restored.observations.for_reason(ObservationReason::Validation).attempts,
            1,
            "restore submits but never reads back",
        );
        {
            let guard = resident.lock();
            let mut progress = guard.progress.lock();
            let before = progress.completed;
            progress.confirm(pending.accepted);
            assert_eq!(progress.completed, before, "old submission callback cannot confirm a restored epoch");
        }
        assert!(matches!(
            raster.acknowledge_cpu_shadow(&old_capture, Some(&shadow_rgb), None),
            Err(ResidentError::InvalidState("observation belongs to an old epoch"))
        ));
        let restored_capture = raster.observe(ObservationReason::Snapshot, PlaneMask::Both).unwrap();
        compare_observed(restored_capture.rgb().unwrap(), &expected_rgb, "RGB restored prefix and drawn suffix");
        compare_observed(restored_capture.aux().unwrap(), &expected_aux, "unchanged auxiliary plane");
        assert_eq!(restored_capture.rgb_generation(), Some(restored.accepted));
        assert_eq!(restored_capture.aux_generation(), Some(restored.accepted));

        let prefix_aux = [0x0123_4567, 0x89ab_cdef, 0xaaaa_5555, 0x7777_9999, 0x2222_4444];
        raster.restore(&[], &prefix_aux).unwrap();
        expected_aux[..prefix_aux.len()].copy_from_slice(&prefix_aux);
        let second = raster.status().unwrap();
        assert_ne!(second.accepted.epoch, restored.accepted.epoch);
        assert_eq!((second.restore_uploads, second.restore_bytes), (2, 32));
        assert_eq!(second.raw_plane_map_attempts, 2);
        assert_eq!(second.rgb.cpu_shadow, None, "unknown unchanged shadow stays unknown across epochs");
        assert_eq!(second.aux.cpu_shadow, None, "partial upload does not claim a current shadow");
        let second_capture = raster.observe(ObservationReason::Snapshot, PlaneMask::Both).unwrap();
        compare_observed(second_capture.rgb().unwrap(), &expected_rgb, "retained RGB plane");
        compare_observed(second_capture.aux().unwrap(), &expected_aux, "aux restored prefix and suffix");

        let observation_counts = raster.status().unwrap().observations;
        raster.restore(&[], &[]).unwrap();
        let empty = raster.status().unwrap();
        assert_ne!(empty.accepted.epoch, second.accepted.epoch);
        assert_eq!((empty.restore_uploads, empty.restore_bytes), (2, 32));
        assert_eq!(empty.observations, observation_counts);
        assert_eq!(empty.raw_plane_map_attempts, 3);
        let oversized = vec![0x4242_4242; PLANE_WORDS + 3];
        raster.restore(&oversized, &[]).unwrap();
        expected_rgb.fill(0x4242_4242);
        let clamped = raster.status().unwrap();
        assert_eq!((clamped.restore_uploads, clamped.restore_bytes), (3, 32 + PLANE_BYTES));
        assert_eq!(clamped.raw_plane_map_attempts, 3);
        let full = raster.observe(ObservationReason::Validation, PlaneMask::Rgb).unwrap();
        compare_observed(full.rgb().unwrap(), &expected_rgb, "overlong restore clamped to one plane");
        assert_eq!(resident.lock().readback, staging);
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_failed_observation_is_counted_and_latched() {
    struct FailingObservation(Arc<Mutex<ResidentGpu>>);

    impl ResidentBackend for FailingObservation {
        fn enqueue(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError> {
            self.0.lock().enqueue(context)
        }

        fn submit(&mut self) -> Result<(), ResidentError> {
            self.0.lock().submit()
        }

        fn observe(&mut self, reason: ObservationReason, planes: PlaneMask) -> Result<Observation, ResidentError> {
            self.0.lock().observe_using(reason, planes, |backend, planes| {
                let _ = backend.read_planes(planes)?;
                Err(ResidentError::Failed("injected observation failure after map".into()))
            })
        }

        fn recorded_status(&self) -> Result<ResidentStatus, ResidentError> {
            self.0.lock().recorded_status()
        }
    }

    on_large_stack(|| {
        let gpu = required_gpu();
        let rgb = seeded_plane(0x0011_2233);
        let aux = seeded_plane(0xaabb_ccdd);
        let resident = Arc::new(Mutex::new(ResidentGpu::activate(Arc::clone(&gpu), &rgb, &aux).unwrap()));
        let mut raster = ResidentRaster::new(FailingObservation(Arc::clone(&resident)));
        let failure = ResidentError::Failed("injected observation failure after map".into());
        assert_eq!(raster.observe(ObservationReason::Snapshot, PlaneMask::Both).unwrap_err(), failure);
        let diagnostics = raster.diagnostics().unwrap();
        assert_eq!(diagnostics.fault, Some(failure.clone()));
        let counts = diagnostics.status.observations.for_reason(ObservationReason::Snapshot);
        assert_eq!((counts.attempts, counts.successes), (1, 0));
        assert_eq!((counts.attempted_transfers, counts.successful_transfers), (2, 0));
        assert_eq!((counts.attempted_bytes, counts.successful_bytes), (READBACK_BYTES, 0));
        assert_eq!(counts.last_attempted, Some(diagnostics.status.accepted));
        assert_eq!(counts.last_successful, None);
        assert_eq!((diagnostics.status.rgb.last_observed, diagnostics.status.aux.last_observed), (None, None));
        assert_eq!(raster.submit(), Err(failure.clone()));
        assert_eq!(raster.observe(ObservationReason::HostRead, PlaneMask::Rgb).unwrap_err(), failure);
        assert_eq!(raster.reset(&rgb, &aux), Err(failure.clone()));
        assert_eq!(raster.restore(&[], &[]), Err(failure.clone()));
        assert_eq!(
            *raster.diagnostics().unwrap().status.observations.for_reason(ObservationReason::HostRead),
            ObservationCounts::default(),
            "a request refused by the sticky fault is not a transfer attempt",
        );
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_injected_device_loss_rejects_later_work() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let rgb = seeded_plane(0x9876_5432);
        let aux = seeded_plane(0x1234_5678);
        let resident = Arc::new(Mutex::new(ResidentGpu::activate(Arc::clone(&gpu), &rgb, &aux).unwrap()));
        let mut raster = ResidentRaster::new(SharedResident(resident));
        let mut draw = context(DRAWMODE1_PLANES_RGB, true, false, (1, 1), (2, 2), u32::MAX, 0x1234_5678);
        raster.enqueue(&mut draw).unwrap();
        raster.submit().unwrap();
        let before = raster.status().unwrap();
        assert_eq!(before.submitted, before.accepted);
        raster.inject_fault_for_test(ResidentFaultInjection::DeviceLost);
        let error = ResidentError::DeviceLost("injected device loss".into());
        assert_eq!(raster.submit(), Err(error.clone()));
        assert_eq!(raster.observe(ObservationReason::Monitor, PlaneMask::Both).unwrap_err(), error);
        assert_eq!(raster.restore(&[], &[]), Err(error.clone()));
        assert_eq!(raster.reset(&rgb, &aux), Err(error.clone()));
        let diagnostics = raster.diagnostics().unwrap();
        assert_eq!(diagnostics.fault, Some(error));
        assert_eq!(diagnostics.status.accepted, before.accepted);
        assert_eq!(diagnostics.status.observations, ObservationCounters::default());
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_linear_fills_match_complete_cpu_planes() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let seed_rgb = seeded_plane(0x1256_34ef);
        let seed_aux = seeded_plane(0xfedc_ba98);
        assert!(matches!(
            ResidentGpu::activate(Arc::clone(&gpu), &[], &seed_aux),
            Err(ResidentError::InvalidSeed { plane: "RGB", .. })
        ));
        let mut resident = ResidentGpu::activate_with_limits(
            Arc::clone(&gpu), &seed_rgb, &seed_aux, 1, NonZeroUsize::new(1).unwrap(), None,
        ).expect("activate linear resident planes");
        assert!(resident.batch.timing.is_none(), "resident submits never enable the batch timer's readback map");
        assert_eq!(resident.sets.len(), 1);
        assert_eq!(resident.sets[0].payload.size(), MAX_PAYLOAD_BYTES);
        assert!(resident.payload.is_empty());
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        for plane in [&resident.planes.rgb, &resident.planes.aux] {
            assert_eq!(plane.size(), PLANE_BYTES);
            assert_eq!(plane.usage(), usage);
        }
        let initial = resident.status().unwrap();
        assert_eq!(initial.accepted.sequence, Sequence(0));
        assert_eq!(initial.submitted, initial.accepted);
        assert_eq!((initial.initialization_uploads, initial.initialization_bytes), (2, 2 * PLANE_BYTES));
        assert!(initial.rgb.cpu_shadow_current() && initial.aux.cpu_shadow_current());
        let seed = resident.observe(ObservationReason::Validation, PlaneMask::Both).unwrap();
        compare_observed(seed.rgb().unwrap(), &seed_rgb, "RGB seed");
        compare_observed(seed.aux().unwrap(), &seed_aux, "aux seed");

        let mut expected_rgb = seed_rgb.clone();
        let mut expected_aux = seed_aux.clone();
        let cases = [
            (DRAWMODE1_PLANES_RGB, true, false, (1338, 1021), (1353, 1026), 0xaa00_00ff, 0x7a33_2211),
            (DRAWMODE1_PLANES_RGB, true, false, (-3, -2), (10, 9), 0x00ff_ffff, 0x9933_77aa),
            (DRAWMODE1_PLANES_RGB, true, true, (6, 8), (20, 200), 0xff00_ff00, 0x6a11_44bb),
            (DRAWMODE1_PLANES_OLAY, false, false, (5, 6), (19, 17), u32::MAX, 0x7a05_1234),
            (DRAWMODE1_PLANES_PUP, false, true, (7, 8), (24, 50), 0x0000_00cc, 0x3300_0013),
            (DRAWMODE1_PLANES_CID, false, false, (1339, 1022), (1355, 1025), 0x0f0f_00ff, 0x5500_0237),
            (DRAWMODE1_PLANES_OLAY, true, false, (3, 3), (11, 12), 0xffff_ffff, 0x1245_6789),
            (DRAWMODE1_PLANES_RGB, true, false, (1344, 3), (1350, 5), u32::MAX, 0xff22_3344),
            (DRAWMODE1_PLANES_PUP, false, false, (1, 1), (5, 5), 0, 0x1234_5678),
            (DRAWMODE1_PLANES_RGB, true, false, (7, 7), (33, 33), 0xffff_ffff, 0x1324_3546),
            (DRAWMODE1_PLANES_RGB, true, false, (31, 31), (34, 34), 0xff00_ff00, 0xaabb_ccdd),
            (DRAWMODE1_PLANES_CID, false, false, (30, 30), (33, 33), 0x00ff_00ff, 0x55aa_1234),
            (DRAWMODE1_PLANES_RGB, true, true, (0, 0), (33, 0), 0x0000_ffff, 0x9876_5432),
            (DRAWMODE1_PLANES_RGB, true, false, (1339, 1023), (1345, 1024), 0xffff_0000, 0x4152_6374),
        ];
        let mut last_rgb = initial.rgb.last_write;
        let mut last_aux = initial.aux.last_write;
        let mut submitted_fills = 0;
        for (index, (plane, rgbmode, span, start, end, mask, color)) in cases.into_iter().enumerate() {
            let mut actual = context(plane, rgbmode, span, start, end, mask, color);
            if index == 1 {
                actual.xywin = ((2u32) << 16) | 1;
            }
            let mut expected = actual;
            let fill = prepare_resident_flat_fill(&actual).unwrap().fill();
            resident.enqueue(&mut actual).expect("enqueue supported fill");
            cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
            assert_context_eq(&actual, &expected, &format!("context after draw {index}"));
            let status = resident.status().unwrap();
            assert_eq!(status.accepted.sequence, Sequence(index as u64 + 1));
            if let Some(fill) = fill {
                submitted_fills += 1;
                match fill.plane {
                    FillPlane::Rgb => last_rgb = status.accepted,
                    FillPlane::Aux => last_aux = status.accepted,
                }
            }
            assert_eq!(status.rgb.last_write, last_rgb);
            assert_eq!(status.aux.last_write, last_aux);
            if let Some(completed) = status.completed {
                assert_eq!(completed.epoch, status.accepted.epoch);
                assert!(completed.sequence <= status.submitted.sequence);
            }
        }
        let mut actual = context(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (52, 32),
            (44, 24),
            0x00ff_0fff,
            0,
        );
        actual
            .bresoctinc1
            .set_octant(crate::dev::ng1::rex3::OCTANT_XDEC | crate::dev::ng1::rex3::OCTANT_YDEC);
        actual.drawmode1.0 |= (1 << 16) | (1 << 17) | (1 << 18);
        actual.colorvram = 0x00ab_cdef;
        let mut expected = actual;
        let fill = prepare_resident_flat_fill(&actual).unwrap().fill();
        resident.enqueue(&mut actual).expect("enqueue reverse fast clear");
        cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
        assert_context_eq(&actual, &expected, "context after reverse fast clear");
        let status = resident.status().unwrap();
        assert_eq!(status.accepted.sequence, Sequence(cases.len() as u64 + 1));
        submitted_fills += 1;
        last_rgb = status.accepted;
        assert_eq!(status.rgb.last_write, last_rgb);
        assert_eq!(status.aux.last_write, last_aux);
        assert!(fill.is_some());

        assert_eq!(resident.batch.sets(), 1);
        assert!(resident.batch.flushes >= submitted_fills - 1, "one-slot batches must submit and retry");
        resident.submit().unwrap();
        assert_eq!(resident.batch.prims_total, submitted_fills, "each retried fill is recorded once");
        gpu.wait(None).unwrap();
        let status = resident.status().unwrap();
        assert_eq!(status.submitted, status.accepted);
        assert_eq!(status.completed, Some(status.submitted));
        assert!(!status.rgb.cpu_shadow_current());
        assert!(!status.aux.cpu_shadow_current());
        assert_eq!((status.initialization_uploads, status.initialization_bytes), (2, 2 * PLANE_BYTES));
        let fills = resident.observe(ObservationReason::Validation, PlaneMask::Both).unwrap();
        compare_observed(fills.rgb().unwrap(), &expected_rgb, "RGB fills");
        compare_observed(fills.aux().unwrap(), &expected_aux, "aux fills");
        for y in 0..HEIGHT as usize {
            let offscreen = y * STRIDE as usize + REX3_SCREEN_WIDTH as usize;
            let end = (y + 1) * STRIDE as usize;
            assert_eq!(&expected_rgb[offscreen..end], &seed_rgb[offscreen..end]);
            assert_eq!(&expected_aux[offscreen..end], &seed_aux[offscreen..end]);
        }

        let old_position = status.submitted;
        let old_bindings = resident.scanout_bindings(&gpu).unwrap();
        assert_eq!(old_bindings.epoch, old_position.epoch);
        let progress = Arc::clone(&resident.progress);
        resident.reset(&seed_rgb, &seed_aux).expect("reset resident planes");
        let after_reset = resident.status().unwrap();
        assert_ne!(after_reset.accepted.epoch, old_position.epoch);
        assert_eq!(after_reset.accepted.sequence, Sequence(0));
        assert!(after_reset.rgb.cpu_shadow_current() && after_reset.aux.cpu_shadow_current());
        let completed = after_reset.completed;
        progress.lock().confirm(old_position);
        assert_eq!(progress.lock().completed, completed, "stale completion changed reset status");
        let reset_seed = resident.observe(ObservationReason::Validation, PlaneMask::Both).unwrap();
        compare_observed(reset_seed.rgb().unwrap(), &seed_rgb, "reset RGB seed");
        compare_observed(reset_seed.aux().unwrap(), &seed_aux, "reset aux seed");

        let mut no_op = context(
            DRAWMODE1_PLANES_RGB, true, false, (1344, 0), (1350, 5), u32::MAX, 0x1234_5678,
        );
        resident.enqueue(&mut no_op).unwrap();
        let accepted = resident.status().unwrap();
        assert_eq!(accepted.accepted.sequence, Sequence(1));
        assert_eq!(accepted.submitted.sequence, Sequence(0));
        assert_eq!(accepted.rgb.last_write, after_reset.rgb.last_write);
        assert_eq!(accepted.aux.last_write, after_reset.aux.last_write);
        resident.submit().unwrap();
        gpu.wait(None).unwrap();
        let submitted = resident.status().unwrap();
        assert_eq!(submitted.submitted, submitted.accepted);
        assert_eq!(submitted.completed, Some(submitted.submitted));
        assert!(submitted.rgb.cpu_shadow_current() && submitted.aux.cpu_shadow_current());

        let bindings = resident.scanout_bindings(&gpu).expect("same-device scanout bindings");
        assert_eq!(bindings.epoch, submitted.accepted.epoch);
        assert_ne!(bindings.epoch, old_bindings.epoch);
        let _group = &bindings.group;
        let writable = Fragment::new("writable.wgsl", r#"
            @group(0) @binding(0) var<storage, read_write> plane: array<u32>;
            @compute @workgroup_size(1) fn main() { plane[0] = 1u; }
        "#);
        assert!(
            crate::wgpu::shader::Source::new("invalid scanout write")
                .body(writable)
                .compute_pipeline(&gpu.device, "main", &[&bindings.layout])
                .is_err(),
            "scanout bindings must not allow writable shaders",
        );
        let other_gpu = required_gpu();
        assert!(matches!(
            resident.scanout_bindings(&other_gpu),
            Err(ResidentError::DeviceMismatch)
        ));
        let before_rejection = resident.status().unwrap();
        let mut unsupported = context(
            DRAWMODE1_PLANES_RGB, true, false, (1, 2), (10, 12), u32::MAX, 0x1234_5678,
        );
        unsupported.drawmode0.0 |= (1 << 6) | (1 << 12);
        let unchanged = unsupported;
        assert_eq!(
            resident.enqueue(&mut unsupported),
            Err(ResidentError::Unsupported("host pattern"))
        );
        assert_context_eq(&unsupported, &unchanged, "rejected patterned host mode");
        let status = resident.status().unwrap();
        assert_eq!(status.accepted, before_rejection.accepted);
        assert_eq!(status.observations, before_rejection.observations);
        assert_eq!(status.raw_plane_map_attempts, before_rejection.raw_plane_map_attempts);
        gpu.check().unwrap();
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_pixel_formats_logic_and_yflip_match_complete_cpu_planes() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let seed_rgb = seeded_plane(0x1d63_b8a4);
        let seed_aux = seeded_plane(0xe29c_475b);
        let mut expected_rgb = seed_rgb.clone();
        let mut expected_aux = seed_aux.clone();
        let mut resident = ResidentGpu::activate_with_limits(
            Arc::clone(&gpu),
            &seed_rgb,
            &seed_aux,
            1,
            NonZeroUsize::new(1).unwrap(),
            None,
        ).expect("activate pixel-format resident planes");

        let set_depth = |ctx: &mut Rex3Context, depth: u32| {
            ctx.drawmode1.0 = (ctx.drawmode1.0 & !(0x3 << 3)) | (depth << 3);
        };
        let set_logic = |ctx: &mut Rex3Context, operation: u32| {
            ctx.drawmode1.0 = (ctx.drawmode1.0 & !DRAWMODE1_LOGICOP_MASK) | operation;
        };

        let mut cases = Vec::new();

        let mut rgb4 = context(
            DRAWMODE1_PLANES_RGB, true, false, (12, 14), (30, 24), 0x0000_00ff, 0x6a31_72b4,
        );
        set_depth(&mut rgb4, DRAWMODE1_DRAWDEPTH_4);
        rgb4.drawmode1.0 |= 1 << 16;
        rgb4.drawmode0.0 |= 1 << 20;
        rgb4.xymove = ((3i16 as u16 as u32) << 16) | (-2i16 as u16 as u32);
        rgb4.xywin = ((-1i16 as u16 as u32) << 16) | (4i16 as u16 as u32);
        cases.push(("dithered RGB4 source", rgb4));

        let mut rgb8_double = context(
            DRAWMODE1_PLANES_RGB, true, false, (42, 18), (63, 31), 0x0000_ff00, 0x9574_2db1,
        );
        set_depth(&mut rgb8_double, DRAWMODE1_DRAWDEPTH_8);
        rgb8_double.drawmode1.0 |= (1 << 5) | (1 << 16);
        set_logic(&mut rgb8_double, DRAWMODE1_LOGICOP_XOR_SH);
        cases.push(("dithered RGB8 double-source XOR", rgb8_double));

        let mut rgba12 = context(
            DRAWMODE1_PLANES_RGBA, true, false, (70, 26), (89, 39), 0x00ff_f000, 0xc346_8a1f,
        );
        set_depth(&mut rgba12, DRAWMODE1_DRAWDEPTH_12);
        rgba12.drawmode1.0 |= 1 << 5;
        set_logic(&mut rgba12, DRAWMODE1_LOGICOP_OR_SH);
        cases.push(("RGBA12 double-source OR", rgba12));

        let mut color_index = context(
            DRAWMODE1_PLANES_RGB, false, false, (98, 34), (117, 47), 0x00ff_ffff, 0x720a_c35d,
        );
        set_logic(&mut color_index, DRAWMODE1_LOGICOP_NSRC_SH);
        cases.push(("RGB24 color-index inverse source", color_index));

        let mut overlay = context(
            DRAWMODE1_PLANES_OLAY, true, false, (126, 42), (145, 55), 0x00ff_0000, 0x4d83_16e9,
        );
        set_depth(&mut overlay, DRAWMODE1_DRAWDEPTH_8);
        overlay.drawmode1.0 |= 1 << 5;
        set_logic(&mut overlay, DRAWMODE1_LOGICOP_AND_SH);
        cases.push(("overlay double-source AND", overlay));

        let mut popup = context(
            DRAWMODE1_PLANES_PUP, false, false, (154, 50), (173, 63), 0x0000_00cc, 0x5e00_0002,
        );
        set_logic(&mut popup, DRAWMODE1_LOGICOP_NOR_SH);
        cases.push(("popup NOR", popup));

        let mut cid = context(
            DRAWMODE1_PLANES_CID, false, false, (182, 58), (201, 71), 0x0000_0030, 0xa700_0003,
        );
        cid.drawmode1.0 |= 1 << 5;
        set_logic(&mut cid, DRAWMODE1_LOGICOP_XNOR_SH);
        cases.push(("CID double-source XNOR", cid));

        let mut alpha_reject = context(
            DRAWMODE1_PLANES_RGB, true, false, (210, 66), (229, 79), u32::MAX, 0x2036_9bc5,
        );
        alpha_reject.drawmode1.0 = (alpha_reject.drawmode1.0 & !(0x7 << DRAWMODE1_COMPARE_SHIFT))
            | (DRAWMODE1_COMPARE_GT << DRAWMODE1_COMPARE_SHIFT);
        alpha_reject.alpharef = 0x40;
        cases.push(("alpha reject", alpha_reject));

        let mut alpha_pass = context(
            DRAWMODE1_PLANES_RGB, true, false, (238, 74), (257, 87), 0xff0f_ffff, 0x8036_9bc5,
        );
        alpha_pass.drawmode1.0 = (alpha_pass.drawmode1.0 & !(0x7 << DRAWMODE1_COMPARE_SHIFT))
            | (DRAWMODE1_COMPARE_GT << DRAWMODE1_COMPARE_SHIFT);
        alpha_pass.alpharef = 0x40;
        set_logic(&mut alpha_pass, DRAWMODE1_LOGICOP_OR_SH);
        cases.push(("alpha pass", alpha_pass));

        let mut flipped_fill = context(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (266, 0x13ff - 55),
            (285, 0x13ff - 45),
            0x00ff_ffff,
            0xb46e_21d8,
        );
        flipped_fill.drawmode1.0 |= (1 << 6) | (1 << 16);
        flipped_fill.drawmode0.0 |= 1 << 20;
        set_depth(&mut flipped_fill, DRAWMODE1_DRAWDEPTH_12);
        flipped_fill.xymove = ((2i16 as u16 as u32) << 16) | (-3i16 as u16 as u32);
        flipped_fill.xywin = ((-4i16 as u16 as u32) << 16) | (1i16 as u16 as u32);
        cases.push(("dithered YFLIP block", flipped_fill));

        let mut flipped_line = line_context(
            DRAWMODE0_ADRMODE_I_LINE_SH,
            DRAWMODE1_PLANES_RGB,
            true,
            (300 * 16, (0x13ff - 70) * 16),
            (305 * 16, (0x13ff - 91) * 16),
            0x0000_ffff,
            0xd257_9b31,
        );
        flipped_line.drawmode1.0 |= (1 << 6) | (1 << 16);
        flipped_line.drawmode0.0 |= 1 << 20;
        set_depth(&mut flipped_line, DRAWMODE1_DRAWDEPTH_8);
        set_logic(&mut flipped_line, DRAWMODE1_LOGICOP_XOR_SH);
        flipped_line.xymove = ((-3i16 as u16 as u32) << 16) | (2i16 as u16 as u32);
        flipped_line.xywin = ((1i16 as u16 as u32) << 16) | (-4i16 as u16 as u32);
        cases.push(("dithered YFLIP y-major line", flipped_line));

        for (label, mut actual) in cases {
            let mut expected = actual;
            resident.enqueue(&mut actual).unwrap_or_else(|error| panic!("{label}: {error}"));
            cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
            assert_context_eq(&actual, &expected, label);
        }

        let observed = resident.observe(ObservationReason::Validation, PlaneMask::Both)
            .expect("observe pixel-format planes");
        compare_observed(
            observed.rgb().unwrap(),
            &expected_rgb,
            "pixel-format RGB",
        );
        compare_observed(
            observed.aux().unwrap(),
            &expected_aux,
            "pixel-format aux",
        );
        assert!(resident.payload.is_empty(), "fill and line submission drained payload");
        gpu.check().unwrap();
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_blending_matches_complete_cpu_planes() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let seed_rgb = seeded_plane(0x73c1_4ea9);
        let seed_aux = seeded_plane(0x8c3e_b156);
        let mut expected_rgb = seed_rgb.clone();
        let mut expected_aux = seed_aux.clone();
        let mut resident = ResidentGpu::activate_with_limits(
            Arc::clone(&gpu),
            &seed_rgb,
            &seed_aux,
            1,
            NonZeroUsize::new(1).unwrap(),
            None,
        ).expect("activate blending resident planes");

        let set_depth = |ctx: &mut Rex3Context, depth: u32| {
            ctx.drawmode1.0 = (ctx.drawmode1.0 & !(0x3 << 3)) | (depth << 3);
        };
        let set_blend = |
            ctx: &mut Rex3Context,
            source_factor: u32,
            destination_factor: u32,
            backblend: bool,
            blend_alpha: bool,
        | {
            ctx.drawmode1.0 &= !(
                (1 << 18) | (0x7 << 19) | (0x7 << 22) | (1 << 25) | (1 << 27)
            );
            ctx.drawmode1.0 |= (1 << 18)
                | (source_factor << 19)
                | (destination_factor << 22)
                | ((backblend as u32) << 25)
                | ((blend_alpha as u32) << 27);
        };

        let mut cases = Vec::new();
        for source_factor in 0..=5 {
            for destination_factor in 0..=5 {
                let x = 20 + destination_factor as i32 * 7;
                let y = 130 + source_factor as i32 * 7;
                let alpha = 0x20 + (source_factor * 29 + destination_factor * 17) % 0xc0;
                let color = (alpha << 24)
                    | ((0x31 + source_factor * 19) << 16)
                    | ((0x57 + destination_factor * 13) << 8)
                    | (0xa3 ^ (source_factor * 11 + destination_factor * 7));
                let mut ctx = context(
                    DRAWMODE1_PLANES_RGB,
                    true,
                    false,
                    (x, y),
                    (x + 3, y + 3),
                    u32::MAX,
                    color,
                );
                set_blend(
                    &mut ctx,
                    source_factor,
                    destination_factor,
                    false,
                    (source_factor + destination_factor) & 1 != 0,
                );
                cases.push(("RGB24 blend factor matrix", ctx));
            }
        }

        let mut backblend = context(
            DRAWMODE1_PLANES_RGB, true, false, (80, 130), (91, 141), u32::MAX, 0x6d28_b4e1,
        );
        backblend.colorback = 0xb739_5ac2;
        set_blend(&mut backblend, 2, 3, true, false);
        cases.push(("RGB24 background blend", backblend));

        let mut rgb4 = context(
            DRAWMODE1_PLANES_RGB, true, false, (100, 148), (115, 159), 0x0000_00ff, 0x4a37_8bd5,
        );
        set_depth(&mut rgb4, DRAWMODE1_DRAWDEPTH_4);
        rgb4.drawmode1.0 |= 1 << 16;
        set_blend(&mut rgb4, 4, 5, false, false);
        cases.push(("dithered RGB4 alpha blend", rgb4));

        let mut rgb8 = context(
            DRAWMODE1_PLANES_RGB, true, false, (124, 156), (139, 167), 0x0000_ff00, 0x9c61_27eb,
        );
        set_depth(&mut rgb8, DRAWMODE1_DRAWDEPTH_8);
        rgb8.drawmode1.0 |= (1 << 5) | (1 << 16);
        set_blend(&mut rgb8, 3, 4, false, true);
        cases.push(("dithered RGB8 double-source blend", rgb8));

        let mut rgba12 = context(
            DRAWMODE1_PLANES_RGBA, true, false, (148, 164), (163, 175), 0x00ff_f000, 0xc583_4a16,
        );
        set_depth(&mut rgba12, DRAWMODE1_DRAWDEPTH_12);
        rgba12.drawmode1.0 |= 1 << 16;
        set_blend(&mut rgba12, 2, 5, false, true);
        cases.push(("dithered RGBA12 blend", rgba12));

        let mut overlay = context(
            DRAWMODE1_PLANES_OLAY, true, false, (172, 172), (187, 183), 0x00ff_0000, 0x7f42_9dc3,
        );
        set_depth(&mut overlay, DRAWMODE1_DRAWDEPTH_8);
        overlay.drawmode1.0 |= 1 << 5;
        set_blend(&mut overlay, 4, 5, false, true);
        cases.push(("overlay blend", overlay));

        let mut popup = context(
            DRAWMODE1_PLANES_PUP, false, false, (196, 180), (211, 191), 0x0000_00cc, 0x5300_0002,
        );
        popup.colorback = 0xa739_61d4;
        set_blend(&mut popup, 3, 2, true, false);
        cases.push(("popup background blend", popup));

        let mut cid = context(
            DRAWMODE1_PLANES_CID, false, false, (220, 188), (235, 199), 0x0000_0030, 0xb100_0003,
        );
        cid.drawmode1.0 |= 1 << 5;
        set_blend(&mut cid, 5, 1, false, true);
        cases.push(("CID double-source blend", cid));

        let mut line = line_context(
            DRAWMODE0_ADRMODE_I_LINE_SH,
            DRAWMODE1_PLANES_RGB,
            true,
            (260 * 16, (0x13ff - 210) * 16),
            (285 * 16, (0x13ff - 218) * 16),
            0x00ff_ffff,
            0x723e_95c1,
        );
        line.drawmode1.0 |= (1 << 6) | (1 << 16);
        set_depth(&mut line, DRAWMODE1_DRAWDEPTH_12);
        set_blend(&mut line, 4, 5, false, true);
        cases.push(("dithered YFLIP line blend", line));

        for (label, mut actual) in cases {
            let mut expected = actual;
            resident.enqueue(&mut actual).unwrap_or_else(|error| panic!("{label}: {error}"));
            cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
            assert_context_eq(&actual, &expected, label);
        }

        let observed = resident.observe(ObservationReason::Validation, PlaneMask::Both)
            .expect("observe blended planes");
        compare_observed(observed.rgb().unwrap(), &expected_rgb, "blended RGB");
        compare_observed(observed.aux().unwrap(), &expected_aux, "blended aux");
        gpu.check().unwrap();
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_patterns_match_complete_cpu_planes() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let seed_rgb = seeded_plane(0x2ad7_61b4);
        let seed_aux = seeded_plane(0xd528_9e4b);
        let mut expected_rgb = seed_rgb.clone();
        let mut expected_aux = seed_aux.clone();
        let mut resident = ResidentGpu::activate_with_limits(
            Arc::clone(&gpu),
            &seed_rgb,
            &seed_aux,
            1,
            NonZeroUsize::new(1).unwrap(),
            None,
        ).expect("activate pattern resident planes");

        let mut cases = Vec::new();

        let mut z_transparent = context(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (-5, 240),
            (29, 247),
            0x00ff_ffff,
            0x7942_b6d3,
        );
        z_transparent.drawmode0.0 |= (1 << 12) | (1 << 13);
        z_transparent.zpattern = 0xa5c3_69f0;
        z_transparent.zpat_bit = 27;
        z_transparent.lspattern = 0xd79a_c000;
        z_transparent.pat_bit = 26;
        z_transparent.lsmode.set_lsrcount(2);
        z_transparent.lsmode.set_lsrepeat(4);
        z_transparent.lsmode.set_lslength(6);
        cases.push(("offscreen transparent combined-pattern block", z_transparent));

        let mut z_opaque = context(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (40, 250),
            (71, 258),
            u32::MAX,
            0x9071_3ec5,
        );
        z_opaque.drawmode0.0 |= (1 << 12) | (1 << 16);
        z_opaque.zpattern = 0xc639_5aa5;
        z_opaque.zpat_bit = 23;
        z_opaque.colorback = 0x304d_a287;
        z_opaque.drawmode1.0 = (z_opaque.drawmode1.0 & !(0x7 << DRAWMODE1_COMPARE_SHIFT))
            | (DRAWMODE1_COMPARE_GT << DRAWMODE1_COMPARE_SHIFT);
        z_opaque.alpharef = 0x50;
        cases.push(("opaque Z pattern alpha compare", z_opaque));

        let mut line_transparent = line_context(
            DRAWMODE0_ADRMODE_I_LINE_SH,
            DRAWMODE1_PLANES_RGB,
            true,
            (90 * 16, 270 * 16),
            (137 * 16, 284 * 16),
            0x00ff_ffff,
            0x6835_c291,
        );
        line_transparent.drawmode0.0 |= 1 << 13;
        line_transparent.lspattern = 0xd6b5_a000;
        line_transparent.pat_bit = 26;
        line_transparent.lsmode.set_lsrcount(2);
        line_transparent.lsmode.set_lsrepeat(3);
        line_transparent.lsmode.set_lslength(5);
        cases.push(("transparent repeated line pattern", line_transparent));

        let mut line_opaque = line_context(
            DRAWMODE0_ADRMODE_F_LINE_SH,
            DRAWMODE1_PLANES_OLAY,
            true,
            (150 * 16 + 3, 292 * 16 + 11),
            (192 * 16 + 9, 306 * 16 + 2),
            0x00ff_0000,
            0xa861_4fd2,
        );
        line_opaque.drawmode0.0 |= (1 << 13) | (1 << 14) | (1 << 17);
        line_opaque.lspattern = 0x9b6d_4000;
        line_opaque.pat_bit = 29;
        line_opaque.lsmode.set_lsrcount(1);
        line_opaque.lsmode.set_lsrepeat(4);
        line_opaque.lsmode.set_lslength(7);
        line_opaque.colorback = 0x5c27_b391;
        cases.push(("opaque repeated line pattern", line_opaque));

        let mut combined_reverse = context(
            DRAWMODE1_PLANES_CID,
            false,
            false,
            (236, 326),
            (204, 318),
            0x0000_0033,
            0x7a00_0002,
        );
        combined_reverse.drawmode0.0 |= (1 << 12) | (1 << 13) | (1 << 17);
        combined_reverse.bresoctinc1.set_octant(
            crate::dev::ng1::rex3::OCTANT_XDEC | crate::dev::ng1::rex3::OCTANT_YDEC,
        );
        combined_reverse.zpattern = 0xe7a5_39c6;
        combined_reverse.zpat_bit = 25;
        combined_reverse.lspattern = 0xb5ca_d000;
        combined_reverse.pat_bit = 24;
        combined_reverse.lsmode.set_lsrcount(5);
        combined_reverse.lsmode.set_lsrepeat(6);
        combined_reverse.lsmode.set_lslength(9);
        combined_reverse.colorback = 0x9300_0001;
        cases.push(("combined reverse block patterns", combined_reverse));

        let mut flipped = context(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (252, 0x13ff - 340),
            (283, 0x13ff - 331),
            0x00ff_ffff,
            0xc357_8d21,
        );
        flipped.drawmode0.0 |= (1 << 12) | (1 << 13) | (1 << 16);
        flipped.drawmode1.0 |= (1 << 6) | (1 << 16);
        flipped.zpattern = 0x96e1_5ab4;
        flipped.zpat_bit = 28;
        flipped.lspattern = 0xe6b9_8000;
        flipped.pat_bit = 27;
        flipped.lsmode.set_lsrcount(3);
        flipped.lsmode.set_lsrepeat(5);
        flipped.lsmode.set_lslength(4);
        flipped.colorback = 0x4138_a6d9;
        cases.push(("dithered YFLIP combined patterns", flipped));

        let mut flipped_line = line_context(
            DRAWMODE0_ADRMODE_A_LINE_SH,
            DRAWMODE1_PLANES_RGB,
            true,
            (310 * 16 + 5, (0x13ff - 360) * 16 + 7),
            (327 * 16 + 11, (0x13ff - 399) * 16 + 3),
            u32::MAX,
            0x0172_4eb9,
        );
        flipped_line.drawmode0.0 |= (1 << 12) | (1 << 13) | (1 << 14) | (1 << 17);
        flipped_line.drawmode1.0 |= 1 << 6;
        flipped_line.zpattern = 0xdb69_a5c3;
        flipped_line.zpat_bit = 30;
        flipped_line.lspattern = 0xacd3_6000;
        flipped_line.pat_bit = 28;
        flipped_line.lsmode.set_lsrcount(4);
        flipped_line.lsmode.set_lsrepeat(3);
        flipped_line.lsmode.set_lslength(8);
        flipped_line.colorback = 0x6d28_91c4;
        flipped_line.aweight0 = u32::MAX;
        flipped_line.aweight1 = u32::MAX;
        cases.push(("YFLIP antialiased combined patterns", flipped_line));

        for (label, mut actual) in cases {
            let mut expected = actual;
            resident.enqueue(&mut actual).unwrap_or_else(|error| panic!("{label}: {error}"));
            cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
            assert_context_eq(&actual, &expected, label);
        }

        let observed = resident.observe(ObservationReason::Validation, PlaneMask::Both)
            .expect("observe patterned planes");
        compare_observed(observed.rgb().unwrap(), &expected_rgb, "patterned RGB");
        compare_observed(observed.aux().unwrap(), &expected_aux, "patterned aux");
        gpu.check().unwrap();
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_shade_dda_matches_complete_cpu_planes() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let seed_rgb = seeded_plane(0x6eb2_41c9);
        let seed_aux = seeded_plane(0x914d_be36);
        let mut expected_rgb = seed_rgb.clone();
        let mut expected_aux = seed_aux.clone();
        let mut resident = ResidentGpu::activate_with_limits(
            Arc::clone(&gpu),
            &seed_rgb,
            &seed_aux,
            1,
            NonZeroUsize::new(1).unwrap(),
            None,
        ).expect("activate shade resident planes");

        let mut cases = Vec::new();

        let mut checkpointed_rgb = context(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (12, 18),
            (711, 417),
            0x00ff_ffff,
            0,
        );
        checkpointed_rgb.drawmode0.0 |= 1 << 18;
        checkpointed_rgb.colorred = 0x0007_e7f1;
        checkpointed_rgb.colorgrn = 0x0001_2a95;
        checkpointed_rgb.colorblue = 0x0038_05a3;
        checkpointed_rgb.coloralpha = 0x0006_f3c7;
        checkpointed_rgb.slopered = 0x0000_1801;
        checkpointed_rgb.slopegrn = -0x0000_0937;
        checkpointed_rgb.slopeblue = 0x0000_27d1;
        checkpointed_rgb.slopealpha = 0x0000_0c53;
        cases.push(("checkpointed RGB block shade", checkpointed_rgb));

        let mut clipped_ci = context(
            DRAWMODE1_PLANES_RGB,
            false,
            false,
            (86, 456),
            (-18, 431),
            0x0000_ffff,
            0,
        );
        clipped_ci.drawmode0.0 |= (1 << 18) | (1 << 21);
        clipped_ci.drawmode1.0 =
            (clipped_ci.drawmode1.0 & !(0x3 << 3)) | (DRAWMODE1_DRAWDEPTH_8 << 3);
        clipped_ci
            .bresoctinc1
            .set_octant(crate::dev::ng1::rex3::OCTANT_XDEC | crate::dev::ng1::rex3::OCTANT_YDEC);
        clipped_ci.colorred = 0x0007_d5a1;
        clipped_ci.coloralpha = 0x0002_9715;
        clipped_ci.slopered = 0x0000_071f;
        clipped_ci.slopealpha = -0x0000_0a31;
        cases.push(("clipped reverse CI8 shade and clamp", clipped_ci));

        let mut patterned = context(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (104, 438),
            (169, 449),
            u32::MAX,
            0,
        );
        patterned.drawmode0.0 |= (1 << 12) | (1 << 16) | (1 << 18);
        patterned.zpattern = 0xa5c3_69f0;
        patterned.zpat_bit = 27;
        patterned.colorback = 0x4d83_16e9;
        patterned.colorred = 0x0001_8b21;
        patterned.colorgrn = 0x0003_3d47;
        patterned.colorblue = 0x0005_a913;
        patterned.coloralpha = 0x0006_17d5;
        patterned.slopered = 0x0000_0457;
        patterned.slopegrn = 0x0000_091d;
        patterned.slopeblue = -0x0000_052b;
        patterned.slopealpha = 0x0000_02e1;
        cases.push(("opaque pattern with advancing shade", patterned));

        let mut fractional = line_context(
            DRAWMODE0_ADRMODE_F_LINE_SH,
            DRAWMODE1_PLANES_RGB,
            true,
            (220 * 16 + 3, 474 * 16 + 9),
            (307 * 16 + 11, 503 * 16 + 2),
            0x00ff_ffff,
            0,
        );
        fractional.drawmode0.0 |= 1 << 18;
        fractional.colorred = 0x0007_f219;
        fractional.colorgrn = 0x0002_76b5;
        fractional.colorblue = 0x0004_c14f;
        fractional.coloralpha = 0x0038_0ad3;
        fractional.slopered = 0x0000_1127;
        fractional.slopegrn = -0x0000_07d3;
        fractional.slopeblue = 0x0000_03a9;
        fractional.slopealpha = 0x0000_2551;
        cases.push(("fractional line shade", fractional));

        let mut antialiased = line_context(
            DRAWMODE0_ADRMODE_A_LINE_SH,
            DRAWMODE1_PLANES_OLAY,
            true,
            (344 * 16 + 5, 526 * 16 + 13),
            (369 * 16 + 7, 589 * 16 + 3),
            0x00ff_0000,
            0,
        );
        antialiased.drawmode0.0 |= 1 << 18;
        antialiased.colorred = 0x0001_5b37;
        antialiased.colorgrn = 0x0003_a2c1;
        antialiased.colorblue = 0x0006_1379;
        antialiased.coloralpha = 0x8000_0000;
        antialiased.slopered = 0x0000_0631;
        antialiased.slopegrn = 0x0000_0a17;
        antialiased.slopeblue = -0x0000_04a3;
        antialiased.slopealpha = 0x0038_0011;
        cases.push(("antialiased overlay line shade", antialiased));

        let mut no_write = context(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (-80, 612),
            (-17, 623),
            0,
            0,
        );
        no_write.drawmode0.0 |= 1 << 18;
        no_write.colorred = 0x0002_1937;
        no_write.colorgrn = 0x0004_2a51;
        no_write.colorblue = 0x0006_3b73;
        no_write.coloralpha = 0x0001_4c95;
        no_write.slopered = 0x0000_0103;
        no_write.slopegrn = -0x0000_0205;
        no_write.slopeblue = 0x0000_0307;
        no_write.slopealpha = -0x0000_0409;
        cases.push(("no-write shade context completion", no_write));

        for (label, mut actual) in cases {
            let mut expected = actual;
            resident.enqueue(&mut actual).unwrap_or_else(|error| panic!("{label}: {error}"));
            cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
            assert_context_eq(&actual, &expected, label);
        }

        let observed = resident.observe(ObservationReason::Validation, PlaneMask::Both)
            .expect("observe shaded planes");
        compare_observed(observed.rgb().unwrap(), &expected_rgb, "shaded RGB");
        compare_observed(observed.aux().unwrap(), &expected_aux, "shaded aux");
        assert!(resident.payload.is_empty(), "shade payload submission drained");
        gpu.check().unwrap();
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_screen_copies_match_complete_cpu_planes() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let seed_rgb = seeded_plane(0x35a7_c19e);
        let seed_aux = seeded_plane(0xca58_3e61);
        let mut expected_rgb = seed_rgb.clone();
        let mut expected_aux = seed_aux.clone();
        let mut resident = ResidentGpu::activate_with_limits(
            Arc::clone(&gpu),
            &seed_rgb,
            &seed_aux,
            1,
            NonZeroUsize::new(1).unwrap(),
            None,
        ).expect("activate copy resident planes");

        let copy = |
            plane: u32,
            rgbmode: bool,
            span: bool,
            start: (i32, i32),
            end: (i32, i32),
            movement: (i16, i16),
            mask: u32,
        | {
            let mut context = context(plane, rgbmode, span, start, end, mask, 0);
            context.drawmode0.0 =
                (context.drawmode0.0 & !0x3) | DRAWMODE0_OPCODE_SCR2SCR;
            context.xymove =
                ((movement.0 as u16 as u32) << 16) | movement.1 as u16 as u32;
            context
        };
        let set_depth = |ctx: &mut Rex3Context, depth: u32| {
            ctx.drawmode1.0 = (ctx.drawmode1.0 & !(0x3 << 3)) | (depth << 3);
        };
        let set_logic = |ctx: &mut Rex3Context, operation: u32| {
            ctx.drawmode1.0 =
                (ctx.drawmode1.0 & !DRAWMODE1_LOGICOP_MASK) | operation;
        };
        let mut cases = Vec::new();

        let mut right_smear = copy(
            DRAWMODE1_PLANES_RGB,
            false,
            false,
            (20, 700),
            (43, 700),
            (1, 0),
            0x0000_00ff,
        );
        set_depth(&mut right_smear, DRAWMODE1_DRAWDEPTH_8);
        cases.push(("right-moving overlap feedback", right_smear));

        let mut right_reverse = copy(
            DRAWMODE1_PLANES_RGB,
            false,
            false,
            (83, 704),
            (60, 704),
            (1, 0),
            0x0000_00ff,
        );
        set_depth(&mut right_reverse, DRAWMODE1_DRAWDEPTH_8);
        right_reverse
            .bresoctinc1
            .set_octant(crate::dev::ng1::rex3::OCTANT_XDEC);
        cases.push(("right-moving reverse-safe overlap", right_reverse));

        let mut left_forward = copy(
            DRAWMODE1_PLANES_RGB,
            false,
            false,
            (101, 708),
            (124, 708),
            (-1, 0),
            0x0000_00ff,
        );
        set_depth(&mut left_forward, DRAWMODE1_DRAWDEPTH_8);
        cases.push(("left-moving forward-safe overlap", left_forward));

        let mut left_reverse = copy(
            DRAWMODE1_PLANES_RGB,
            false,
            false,
            (164, 712),
            (141, 712),
            (-1, 0),
            0x0000_00ff,
        );
        set_depth(&mut left_reverse, DRAWMODE1_DRAWDEPTH_8);
        left_reverse
            .bresoctinc1
            .set_octant(crate::dev::ng1::rex3::OCTANT_XDEC);
        cases.push(("left-moving reverse overlap feedback", left_reverse));

        let mut down_smear = copy(
            DRAWMODE1_PLANES_RGB,
            false,
            false,
            (190, 720),
            (197, 743),
            (0, 1),
            0x0000_00ff,
        );
        set_depth(&mut down_smear, DRAWMODE1_DRAWDEPTH_8);
        cases.push(("down-moving overlap feedback", down_smear));

        let mut down_reverse = copy(
            DRAWMODE1_PLANES_RGB,
            false,
            false,
            (214, 783),
            (221, 760),
            (0, 1),
            0x0000_00ff,
        );
        set_depth(&mut down_reverse, DRAWMODE1_DRAWDEPTH_8);
        down_reverse
            .bresoctinc1
            .set_octant(crate::dev::ng1::rex3::OCTANT_YDEC);
        cases.push(("down-moving reverse-safe overlap", down_reverse));

        let mut low_to_high = copy(
            DRAWMODE1_PLANES_RGB,
            false,
            false,
            (250, 724),
            (273, 724),
            (1, 0),
            0x0000_ff00,
        );
        set_depth(&mut low_to_high, DRAWMODE1_DRAWDEPTH_8);
        cases.push(("low-to-high packed slot copy", low_to_high));

        let mut high_to_low = copy(
            DRAWMODE1_PLANES_RGB,
            false,
            false,
            (290, 728),
            (313, 728),
            (1, 0),
            0x0000_00ff,
        );
        set_depth(&mut high_to_low, DRAWMODE1_DRAWDEPTH_8);
        high_to_low.drawmode1.0 |= 1 << 5;
        cases.push(("high-to-low packed slot copy", high_to_low));

        let mut clipped = copy(
            DRAWMODE1_PLANES_RGB,
            false,
            false,
            (330, 732),
            (361, 732),
            (1, 0),
            0x0000_00ff,
        );
        set_depth(&mut clipped, DRAWMODE1_DRAWDEPTH_8);
        clipped.clipmode |= CLIPMODE_ENSMASK_SMASK0;
        clipped.smask0x =
            (((REX3_COORD_BIAS + 345) as u16 as u32) << 16)
                | (REX3_COORD_BIAS + 400) as u16 as u32;
        clipped.smask0y =
            (((REX3_COORD_BIAS + 732) as u16 as u32) << 16)
                | (REX3_COORD_BIAS + 732) as u16 as u32;
        cases.push(("clipped overlap feedback", clipped));

        let mut logic_feedback = copy(
            DRAWMODE1_PLANES_RGB,
            false,
            false,
            (380, 736),
            (411, 736),
            (1, 0),
            0x0000_00f3,
        );
        set_depth(&mut logic_feedback, DRAWMODE1_DRAWDEPTH_8);
        set_logic(&mut logic_feedback, DRAWMODE1_LOGICOP_XOR_SH);
        cases.push(("logic and WRMASK overlap feedback", logic_feedback));

        let mut ignored_effects = copy(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (430, 740),
            (461, 747),
            (3, 2),
            0x00ff_ffff,
        );
        ignored_effects.drawmode0.0 |= (1 << 12) | (1 << 13) | (1 << 18);
        ignored_effects.zpattern = 0;
        ignored_effects.lspattern = 0;
        ignored_effects.slopered = 0x0000_1235;
        ignored_effects.slopegrn = -0x0000_0711;
        ignored_effects.slopeblue = 0x0000_0953;
        ignored_effects.slopealpha = 0x0000_03a7;
        ignored_effects.drawmode1.0 =
            (ignored_effects.drawmode1.0 & !(0x7 << DRAWMODE1_COMPARE_SHIFT))
                | (DRAWMODE1_COMPARE_GT << DRAWMODE1_COMPARE_SHIFT);
        ignored_effects.alpharef = 0xff;
        cases.push(("copy ignores compare and patterns but advances state", ignored_effects));

        let mut skipped = copy(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (490, 752),
            (521, 759),
            (2, 1),
            u32::MAX,
        );
        skipped.drawmode0.0 |= (1 << 10) | (1 << 12) | (1 << 18);
        skipped.slopered = 0x0000_0101;
        skipped.slopegrn = 0x0000_0203;
        skipped.slopeblue = -0x0000_0305;
        skipped.slopealpha = 0x0000_0407;
        cases.push(("block skip-first suppresses writes but advances state", skipped));

        let mut overlay = copy(
            DRAWMODE1_PLANES_OLAY,
            true,
            false,
            (550, 764),
            (581, 771),
            (1, 1),
            0x00ff_ff00,
        );
        set_depth(&mut overlay, DRAWMODE1_DRAWDEPTH_8);
        cases.push(("overlay overlap feedback", overlay));

        let span = copy(
            DRAWMODE1_PLANES_CID,
            false,
            true,
            (610, 776),
            (641, 776),
            (1, 0),
            0x0000_0033,
        );
        cases.push(("CID span overlap feedback", span));

        let mut integer_line = line_context(
            DRAWMODE0_ADRMODE_I_LINE_SH,
            DRAWMODE1_PLANES_RGB,
            false,
            (680 * 16, 784 * 16),
            (727 * 16, 795 * 16),
            0x0000_00ff,
            0,
        );
        integer_line.drawmode0.0 =
            (integer_line.drawmode0.0 & !0x3) | DRAWMODE0_OPCODE_SCR2SCR;
        integer_line.xymove = (1u32 << 16) | 1;
        set_depth(&mut integer_line, DRAWMODE1_DRAWDEPTH_8);
        cases.push(("integer line copy", integer_line));

        let mut fractional_line = line_context(
            DRAWMODE0_ADRMODE_F_LINE_SH,
            DRAWMODE1_PLANES_RGB,
            true,
            (760 * 16 + 3, 804 * 16 + 11),
            (803 * 16 + 9, 829 * 16 + 2),
            0x00ff_ffff,
            0,
        );
        fractional_line.drawmode0.0 =
            (fractional_line.drawmode0.0 & !0x3) | DRAWMODE0_OPCODE_SCR2SCR;
        fractional_line.xymove =
            ((-2i16 as u16 as u32) << 16) | 3i16 as u16 as u32;
        cases.push(("fractional line copy", fractional_line));

        let mut antialiased_line = line_context(
            DRAWMODE0_ADRMODE_A_LINE_SH,
            DRAWMODE1_PLANES_PUP,
            false,
            (840 * 16 + 5, (0x13ff - 850) * 16 + 7),
            (867 * 16 + 11, (0x13ff - 899) * 16 + 3),
            0x0000_00cc,
            0,
        );
        antialiased_line.drawmode0.0 =
            (antialiased_line.drawmode0.0 & !0x3) | DRAWMODE0_OPCODE_SCR2SCR;
        antialiased_line.drawmode1.0 |= 1 << 6;
        antialiased_line.xymove =
            ((2i16 as u16 as u32) << 16) | (-1i16 as u16 as u32);
        cases.push(("YFLIP antialiased popup line copy", antialiased_line));

        for (label, mut actual) in cases {
            let mut expected = actual;
            resident.enqueue(&mut actual).unwrap_or_else(|error| panic!("{label}: {error}"));
            cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
            assert_context_eq(&actual, &expected, label);
        }

        let observed = resident.observe(ObservationReason::Validation, PlaneMask::Both)
            .expect("observe copied planes");
        compare_observed(observed.rgb().unwrap(), &expected_rgb, "copied RGB");
        compare_observed(observed.aux().unwrap(), &expected_aux, "copied aux");
        assert!(resident.payload.is_empty(), "copy submission drained line payload");
        gpu.check().unwrap();
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_continuation_and_noop_match_complete_cpu_planes() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let seed_rgb = seeded_plane(0x6d31_a7c5);
        let seed_aux = seeded_plane(0x92ce_583a);
        let mut expected_rgb = seed_rgb.clone();
        let mut expected_aux = seed_aux.clone();
        let mut resident = ResidentGpu::activate_with_limits(
            Arc::clone(&gpu),
            &seed_rgb,
            &seed_aux,
            1,
            NonZeroUsize::new(1).unwrap(),
            None,
        )
        .expect("activate continuation resident planes");

        macro_rules! run_steps {
            ($label:expr, $context:expr, $steps:expr) => {{
                let mut actual = $context;
                let mut expected = actual;
                for step in 0..$steps {
                    resident
                        .enqueue(&mut actual)
                        .unwrap_or_else(|error| panic!("{} step {step}: {error}", $label));
                    cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
                    assert_context_eq(
                        &actual,
                        &expected,
                        &format!("{} step {step}", $label),
                    );
                }
            }};
        }

        let advancing_block = |x: i32, y: i32, stops: u32| {
            let mut draw = context(
                DRAWMODE1_PLANES_RGB,
                true,
                false,
                (x, y),
                (x + 1, y + 1),
                0x00ff_ffff,
                0,
            );
            draw.drawmode0.0 &= !((1 << 8) | (1 << 9));
            draw.drawmode0.0 |= stops | (1 << 12) | (1 << 13) | (1 << 18);
            draw.zpattern = u32::MAX;
            draw.lspattern = u32::MAX;
            draw.lsmode.set_lsrcount(2);
            draw.lsmode.set_lsrepeat(3);
            draw.lsmode.set_lslength(5);
            draw.colorred = 0x0001_8123;
            draw.colorgrn = 0x0002_9345;
            draw.colorblue = 0x0003_a567;
            draw.coloralpha = 0x0004_b789;
            draw.slopered = 0x0000_0101;
            draw.slopegrn = 0x0000_0203;
            draw.slopeblue = -0x0000_0305;
            draw.slopealpha = 0x0000_0407;
            draw
        };
        run_steps!("block single-step rows", advancing_block(20, 40, 0), 4);
        run_steps!("block row-at-a-time", advancing_block(30, 50, 1 << 8), 2);
        run_steps!("block stop-on-y single-step", advancing_block(40, 60, 1 << 9), 4);
        run_steps!(
            "block complete",
            advancing_block(50, 70, (1 << 8) | (1 << 9)),
            1
        );

        let mut partial = advancing_block(270, 300, (1 << 8) | (1 << 9));
        partial.xsave = (260 + REX3_COORD_BIAS) << 11;
        partial.yend = (302 + REX3_COORD_BIAS) << 11;
        partial.mid_primitive = true;
        run_steps!("partial first block row", partial, 1);

        let mut reverse_stride = context(
            DRAWMODE1_PLANES_OLAY,
            true,
            false,
            (360, 340),
            (356, 334),
            0x00ff_ff00,
            0x8a4c_2e17,
        );
        reverse_stride.bresoctinc1.set_octant(
            crate::dev::ng1::rex3::OCTANT_XDEC
                | crate::dev::ng1::rex3::OCTANT_YDEC,
        );
        reverse_stride.drawmode0.0 |= 1 << 23;
        run_steps!("reverse YSTRIDE block", reverse_stride, 1);

        let mut length_span = context(
            DRAWMODE1_PLANES_RGB,
            true,
            true,
            (410, 370),
            (455, 370),
            0xffff_00ff,
            0x8f63_4127,
        );
        length_span.drawmode0.0 |= 1 << 15;
        run_steps!("LENGTH32 span", length_span, 2);

        let mut fractional_span = context(
            DRAWMODE1_PLANES_RGB,
            true,
            true,
            (470, 390),
            (475, 390),
            0x00ff_ffff,
            0x9a74_5213,
        );
        fractional_span.xstart += 7 * 128;
        fractional_span.xsave = fractional_span.xstart;
        fractional_span.xend += 3 * 128;
        fractional_span.ystart += 5 * 128;
        run_steps!("fractional span endpoints", fractional_span, 1);

        let mut no_plane = advancing_block(500, 400, (1 << 8) | (1 << 9));
        no_plane.drawmode1.0 =
            (no_plane.drawmode1.0 & !7) | DRAWMODE1_PLANES_NONE;
        run_steps!("no-plane state advancement", no_plane, 1);

        for (index, shape) in [
            DRAWMODE0_ADRMODE_I_LINE_SH,
            DRAWMODE0_ADRMODE_F_LINE_SH,
            DRAWMODE0_ADRMODE_A_LINE_SH,
        ]
        .into_iter()
        .enumerate()
        {
            let x = 560 + index as i32 * 24;
            let mut line = line_context(
                shape,
                DRAWMODE1_PLANES_RGB,
                true,
                (x * 16 + 3, 440 * 16 + 5),
                ((x + 5) * 16 + 9, 440 * 16 + 11),
                0x00ff_ffff,
                0x9b75_4312 ^ index as u32,
            );
            line.drawmode0.0 &= !((1 << 8) | (1 << 9));
            line.drawmode0.0 |= (1 << 10) | (1 << 11);
            run_steps!(&format!("single-step line {index}"), line, 6);
        }

        let mut length_line = line_context(
            DRAWMODE0_ADRMODE_I_LINE_SH,
            DRAWMODE1_PLANES_CID,
            false,
            (640 * 16, 480 * 16),
            (685 * 16, 480 * 16),
            0x0000_0033,
            3,
        );
        length_line.drawmode0.0 |= 1 << 15;
        run_steps!("LENGTH32 integer line", length_line, 2);

        let mut copy_span = context(
            DRAWMODE1_PLANES_RGB,
            true,
            true,
            (720, 520),
            (765, 520),
            0x00ff_ffff,
            0,
        );
        copy_span.drawmode0.0 =
            (copy_span.drawmode0.0 & !3) | DRAWMODE0_OPCODE_SCR2SCR | (1 << 15);
        copy_span.xymove = 1 << 16;
        run_steps!("overlapping LENGTH32 copy span", copy_span, 2);

        let mut fractional_copy = context(
            DRAWMODE1_PLANES_RGB,
            true,
            true,
            (780, 540),
            (785, 540),
            0x00ff_ffff,
            0,
        );
        fractional_copy.drawmode0.0 =
            (fractional_copy.drawmode0.0 & !3) | DRAWMODE0_OPCODE_SCR2SCR;
        fractional_copy.xstart += 11 * 128;
        fractional_copy.xsave = fractional_copy.xstart;
        fractional_copy.xend += 2 * 128;
        fractional_copy.xymove = (-1i16 as u16 as u32) << 16;
        run_steps!("fractional copy span endpoints", fractional_copy, 1);

        let mut copy_rows = context(
            DRAWMODE1_PLANES_PUP,
            false,
            false,
            (820, 560),
            (827, 561),
            0x0000_00cc,
            0,
        );
        copy_rows.drawmode0.0 =
            (copy_rows.drawmode0.0 & !3 & !(1 << 9)) | DRAWMODE0_OPCODE_SCR2SCR;
        copy_rows.xymove = 1;
        run_steps!("row-at-a-time copy block", copy_rows, 2);

        let mut noop = advancing_block(900, 600, (1 << 8) | (1 << 9));
        noop.drawmode0.0 = (noop.drawmode0.0 & !3) | DRAWMODE0_OPCODE_NOOP;
        let mut expected_noop = noop;
        resident.enqueue(&mut noop).expect("accept NOOP");
        cpu_draw(
            &mut expected_noop,
            &mut expected_rgb,
            &mut expected_aux,
        );
        assert_context_eq(&noop, &expected_noop, "direct NOOP");

        let observed = resident
            .observe(ObservationReason::Validation, PlaneMask::Both)
            .expect("observe continuation planes");
        compare_observed(
            observed.rgb().unwrap(),
            &expected_rgb,
            "continuation RGB",
        );
        compare_observed(
            observed.aux().unwrap(),
            &expected_aux,
            "continuation aux",
        );

        let outer_rgb = seeded_plane(0x14b8_6c2f);
        let outer_aux = seeded_plane(0xeb47_93d0);
        let mut resident_board =
            TestRex(Box::new(Rex3::new(crate::disp::StatusSource::detached())));
        let mut cpu_board =
            TestRex(Box::new(Rex3::new(crate::disp::StatusSource::detached())));
        for board in [&mut resident_board, &mut cpu_board] {
            board.0.fb_rgb.get_mut().copy_from_slice(&outer_rgb);
            board.0.fb_aux.get_mut().copy_from_slice(&outer_aux);
        }
        #[cfg(feature = "rex-jit")]
        cpu_board.0.jit_enabled.store(false, Ordering::Relaxed);
        let outer_resident = Arc::new(Mutex::new(
            ResidentGpu::activate_with_limits(
                Arc::clone(&gpu),
                &outer_rgb,
                &outer_aux,
                1,
                NonZeroUsize::new(1).unwrap(),
                None,
            )
            .unwrap(),
        ));
        resident_board
            .0
            .attach_resident_raster(SharedResident(Arc::clone(&outer_resident)))
            .unwrap();

        let mut armed = context(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (1000, 700),
            (1001, 701),
            0x00ff_ffff,
            0xb583_6142,
        );
        armed.host_len = 2;
        armed.host_cursor = 0;
        armed.hostcnt = 7;
        for board in [&mut resident_board, &mut cpu_board] {
            let host = &*board.0 as *const Rex3;
            *board.0.context.get_mut() = armed;
            board.0.context.get_mut().host = host;
        }
        resident_board.0.execute_go();
        cpu_board.0.execute_go();
        assert_context_eq(
            unsafe { &*resident_board.0.context.get() },
            unsafe { &*cpu_board.0.context.get() },
            "armed non-host retry",
        );

        let mut armed_noop = line_context(
            DRAWMODE0_ADRMODE_F_LINE_SH,
            DRAWMODE1_PLANES_RGB,
            true,
            (1040 * 16 + 3, 740 * 16 + 5),
            (1047 * 16 + 11, 744 * 16 + 7),
            0x00ff_ffff,
            0,
        );
        armed_noop.drawmode0.0 =
            (armed_noop.drawmode0.0 & !3) | DRAWMODE0_OPCODE_NOOP | (1 << 5);
        armed_noop.host_len = 2;
        armed_noop.host_cursor = 0;
        armed_noop.hostcnt = 9;
        for board in [&mut resident_board, &mut cpu_board] {
            let host = &*board.0 as *const Rex3;
            *board.0.context.get_mut() = armed_noop;
            board.0.context.get_mut().host = host;
        }
        resident_board.0.execute_go();
        cpu_board.0.execute_go();
        assert_context_eq(
            unsafe { &*resident_board.0.context.get() },
            unsafe { &*cpu_board.0.context.get() },
            "armed NOOP retry",
        );
        let mut backend = outer_resident.lock();
        let outer_observed = backend
            .observe(ObservationReason::Validation, PlaneMask::Both)
            .expect("observe armed retry planes");
        compare_observed(
            outer_observed.rgb().unwrap(),
            unsafe { &*cpu_board.0.fb_rgb.get() },
            "armed retry RGB",
        );
        compare_observed(
            outer_observed.aux().unwrap(),
            unsafe { &*cpu_board.0.fb_aux.get() },
            "armed retry aux",
        );
        assert_eq!(backend.status().unwrap().accepted.sequence, Sequence(4));
        gpu.check().unwrap();
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_host_fills_match_complete_cpu_planes() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let seed_rgb = seeded_plane(0x5ca1_37e9);
        let seed_aux = seeded_plane(0xa35e_c816);
        let mut expected_rgb = seed_rgb.clone();
        let mut expected_aux = seed_aux.clone();
        let mut resident = ResidentGpu::activate_with_limits(
            Arc::clone(&gpu),
            &seed_rgb,
            &seed_aux,
            1,
            NonZeroUsize::new(1).unwrap(),
            None,
        )
        .expect("activate HostW resident planes");

        struct Case {
            label: &'static str,
            plane: u32,
            rgb_mode: bool,
            span: bool,
            start: (i32, i32),
            end: (i32, i32),
            draw_mode: u32,
            host_mode: u32,
            words: &'static [u64],
            initial_count: u32,
            initial_shifter: u64,
        }
        let cases = [
            Case {
                label: "packed 8-bit block",
                plane: DRAWMODE1_PLANES_RGB,
                rgb_mode: true,
                span: false,
                start: (31, 41),
                end: (38, 42),
                draw_mode: 1 << 6,
                host_mode: (1 << 7) | (1 << 8) | (1 << 10) | (1 << 11),
                words: &[0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210],
                initial_count: 0,
                initial_shifter: 0,
            },
            Case {
                label: "swapped 32-bit span",
                plane: DRAWMODE1_PLANES_RGB,
                rgb_mode: true,
                span: true,
                start: (47, 43),
                end: (49, 43),
                draw_mode: 1 << 6,
                host_mode: (3 << 8) | (1 << 11),
                words: &[0x1122_3344_aabb_ccdd, 0x5566_7788_0102_0304],
                initial_count: 0,
                initial_shifter: 0,
            },
            Case {
                label: "packed 12-bit color and alpha",
                plane: DRAWMODE1_PLANES_RGBA,
                rgb_mode: true,
                span: true,
                start: (53, 45),
                end: (56, 45),
                draw_mode: (1 << 6) | (1 << 7),
                host_mode: (1 << 7) | (2 << 8) | (1 << 10),
                words: &[0x0abc_0123_0456_0789],
                initial_count: 0,
                initial_shifter: 0,
            },
            Case {
                label: "packed alpha-only span",
                plane: DRAWMODE1_PLANES_RGB,
                rgb_mode: true,
                span: true,
                start: (61, 47),
                end: (66, 47),
                draw_mode: 1 << 7,
                host_mode: (1 << 7) | (1 << 8) | (1 << 10),
                words: &[0x20_40_60_80_a0_c0_e0_ff],
                initial_count: 0,
                initial_shifter: 0,
            },
            Case {
                label: "packed CI4 auxiliary span",
                plane: DRAWMODE1_PLANES_OLAY,
                rgb_mode: false,
                span: true,
                start: (71, 49),
                end: (78, 49),
                draw_mode: 1 << 6,
                host_mode: (1 << 7) | (1 << 10),
                words: &[0x0102_0304_0506_0708],
                initial_count: 0,
                initial_shifter: 0,
            },
            Case {
                label: "row-boundary batch continuation",
                plane: DRAWMODE1_PLANES_RGB,
                rgb_mode: true,
                span: false,
                start: (83, 51),
                end: (85, 53),
                draw_mode: (1 << 6) | (1 << 8),
                host_mode: (1 << 7) | (1 << 8),
                words: &[
                    0x1020_3000_0000_0000,
                    0x4050_6000_0000_0000,
                    0x7080_9000_0000_0000,
                ],
                initial_count: 0,
                initial_shifter: 0,
            },
            Case {
                label: "PIO replication",
                plane: DRAWMODE1_PLANES_RGB,
                rgb_mode: true,
                span: false,
                start: (91, 55),
                end: (94, 57),
                draw_mode: (1 << 6) | (1 << 8) | (1 << 9),
                host_mode: 3 << 8,
                words: &[0x3567_89ab_0000_0000],
                initial_count: 0,
                initial_shifter: 0,
            },
            Case {
                label: "residual packed span",
                plane: DRAWMODE1_PLANES_RGB,
                rgb_mode: true,
                span: true,
                start: (101, 59),
                end: (103, 59),
                draw_mode: 1 << 7,
                host_mode: (1 << 7) | (1 << 8),
                words: &[0x90a0_b0c0_d0e0_f000],
                initial_count: 3,
                initial_shifter: 0x1939_5979_99b9_d9f9,
            },
        ];

        for (index, case) in cases.iter().enumerate() {
            let mut actual = context(
                case.plane,
                case.rgb_mode,
                case.span,
                case.start,
                case.end,
                0xffff_ffff,
                0xb46a_3517,
            );
            actual.drawmode0.0 |= case.draw_mode;
            if case.label == "row-boundary batch continuation" {
                actual.drawmode0.0 &= !(1 << 9);
            }
            actual.drawmode1.0 &= !((1 << 7) | (3 << 8) | (1 << 10) | (1 << 11));
            actual.drawmode1.0 |= case.host_mode;
            actual.host_len = case.words.len() as u32;
            actual.host_cursor = 0;
            actual.hostcnt = case.initial_count;
            actual.host_shifter = case.initial_shifter;
            for (slot, word) in case.words.iter().copied().enumerate() {
                actual.hostrw[slot] = word;
            }
            if index == 1 {
                actual.wrmask = 0x00ff_ffff;
                actual.drawmode1.0 =
                    (actual.drawmode1.0 & !DRAWMODE1_LOGICOP_MASK) | DRAWMODE1_LOGICOP_XOR_SH;
            }
            if index == 3 {
                actual.drawmode1.0 |= 1 << 18;
                actual.drawmode1.0 |= 1 << 19;
                actual.drawmode1.0 |= 5 << 22;
            }
            if index == 5 {
                actual.clipmode |= CLIPMODE_ENSMASK_SMASK0;
                actual.smask0x =
                    ((REX3_COORD_BIAS + 83) as u32) << 16 | (REX3_COORD_BIAS + 84) as u32;
                actual.smask0y =
                    ((REX3_COORD_BIAS + 51) as u32) << 16 | (REX3_COORD_BIAS + 53) as u32;
            }

            let mut expected = actual;
            resident
                .enqueue(&mut actual)
                .unwrap_or_else(|error| panic!("enqueue {}: {error}", case.label));
            cpu_host_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
            assert_context_eq(&actual, &expected, case.label);
            let status = resident.status().unwrap();
            assert_eq!(status.accepted.sequence, Sequence(index as u64 + 1));
            assert_eq!(status.submitted, status.accepted);
            assert_eq!(status.host_write_transfers, index as u64 + 1);
        }

        gpu.wait(None).unwrap();
        assert!(resident.payload.is_empty());
        assert_eq!(resident.batch.prims_total, cases.len() as u64);
        assert_eq!(
            resident.status().unwrap().host_write_qwords,
            cases
                .iter()
                .map(|case| case.words.len() as u64)
                .sum::<u64>(),
        );
        let observed = resident
            .observe(ObservationReason::Validation, PlaneMask::Both)
            .expect("observe HostW planes");
        compare_observed(
            observed.rgb().unwrap(),
            &expected_rgb,
            "HostW complete RGB plane",
        );
        compare_observed(
            observed.aux().unwrap(),
            &expected_aux,
            "HostW complete auxiliary plane",
        );

        let board_rgb = seeded_plane(0x3e91_7ac4);
        let board_aux = seeded_plane(0xc16e_853b);
        let mut resident_board =
            TestRex(Box::new(Rex3::new(crate::disp::StatusSource::detached())));
        let mut cpu_board = TestRex(Box::new(Rex3::new(crate::disp::StatusSource::detached())));
        for board in [&mut resident_board, &mut cpu_board] {
            board.0.fb_rgb.get_mut().copy_from_slice(&board_rgb);
            board.0.fb_aux.get_mut().copy_from_slice(&board_aux);
        }
        #[cfg(feature = "rex-jit")]
        cpu_board.0.jit_enabled.store(false, Ordering::Relaxed);
        let board_resident = Arc::new(Mutex::new(
            ResidentGpu::activate_with_limits(
                Arc::clone(&gpu),
                &board_rgb,
                &board_aux,
                1,
                NonZeroUsize::new(1).unwrap(),
                None,
            )
            .unwrap(),
        ));
        resident_board
            .0
            .attach_resident_raster(SharedResident(Arc::clone(&board_resident)))
            .unwrap();
        let mut board_host = context(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (117, 63),
            (120, 64),
            0x00ff_ffff,
            0xd47b_290e,
        );
        board_host.drawmode0.0 |= 1 << 6;
        board_host.drawmode1.0 |= (1 << 7) | (1 << 8);
        board_host.host_len = 2;
        board_host.host_cursor = 0;
        board_host.hostcnt = 0;
        board_host.hostrw[0] = 0x1020_3040_5060_7080;
        board_host.hostrw[1] = 0x90a0_b0c0_d0e0_f000;
        for board in [&mut resident_board, &mut cpu_board] {
            let owner = &*board.0 as *const Rex3;
            *board.0.context.get_mut() = board_host;
            board.0.context.get_mut().host = owner;
        }
        resident_board.0.execute_go();
        cpu_board.0.execute_go();
        assert_context_eq(
            unsafe { &*resident_board.0.context.get() },
            unsafe { &*cpu_board.0.context.get() },
            "board HostW batch",
        );
        let mut board_backend = board_resident.lock();
        let board_observed = board_backend
            .observe(ObservationReason::Validation, PlaneMask::Both)
            .expect("observe board HostW planes");
        compare_observed(
            board_observed.rgb().unwrap(),
            unsafe { &*cpu_board.0.fb_rgb.get() },
            "board HostW RGB",
        );
        compare_observed(
            board_observed.aux().unwrap(),
            unsafe { &*cpu_board.0.fb_aux.get() },
            "board HostW auxiliary",
        );
        let board_status = board_backend.status().unwrap();
        assert_eq!(board_status.accepted.sequence, Sequence(1));
        assert_eq!(
            (
                board_status.host_write_transfers,
                board_status.host_write_qwords
            ),
            (1, 2)
        );
        gpu.check().unwrap();
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_lines_match_complete_cpu_planes() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let seed_rgb = seeded_plane(0x49a2_8c17);
        let seed_aux = seeded_plane(0xb65d_73e8);
        let mut expected_rgb = seed_rgb.clone();
        let mut expected_aux = seed_aux.clone();
        let mut resident = ResidentGpu::activate_with_limits(
            Arc::clone(&gpu),
            &seed_rgb,
            &seed_aux,
            1,
            NonZeroUsize::new(1).unwrap(),
            None,
        ).expect("activate line resident planes");

        let deltas = [
            (3, 7), (3, -7), (-3, 7), (-3, -7),
            (7, 3), (7, -3), (-7, 3), (-7, -3),
        ];
        let mut draws = 0u64;
        for (index, delta) in deltas.into_iter().enumerate() {
            let x = 40 + index as i32 * 12;
            let y = 80 + index as i32 * 10;
            let mut actual = line_context(
                DRAWMODE0_ADRMODE_I_LINE_SH,
                DRAWMODE1_PLANES_RGB,
                true,
                (x * 16, y * 16),
                ((x + delta.0) * 16, (y + delta.1) * 16),
                0xff0f_ffff,
                0x5a31_72b4 ^ index as u32,
            );
            actual.bresrndinc2.set_rnd(if index & 1 == 0 { 0 } else { 0xff });
            if index == 2 { actual.drawmode0.0 |= 1 << 10; }
            if index == 5 { actual.drawmode0.0 |= 1 << 11; }
            let mut expected = actual;
            resident.enqueue(&mut actual).expect("enqueue integer line");
            cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
            assert_context_eq(&actual, &expected, &format!("integer line context {index}"));
            draws += 1;
        }

        let mut fractional = line_context(
            DRAWMODE0_ADRMODE_F_LINE_SH,
            DRAWMODE1_PLANES_RGB,
            true,
            (220 * 16 + 3, 180 * 16 + 11),
            (246 * 16 + 13, 193 * 16 + 2),
            0xffff_00ff,
            0x7d26_a159,
        );
        fractional.xywin = ((2i16 as u16 as u32) << 16) | (-3i16 as u16 as u32);
        let mut expected = fractional;
        resident.enqueue(&mut fractional).expect("enqueue fractional line");
        cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
        assert_context_eq(&fractional, &expected, "fractional line context");
        draws += 1;

        let mut antialiased = line_context(
            DRAWMODE0_ADRMODE_A_LINE_SH,
            DRAWMODE1_PLANES_RGB,
            true,
            (300 * 16 + 7, 210 * 16 + 5),
            (316 * 16 + 9, 239 * 16 + 12),
            u32::MAX,
            0x0135_79bd,
        );
        antialiased.drawmode0.0 |= (1 << 10) | (1 << 22);
        antialiased.aweight0 = u32::MAX;
        antialiased.aweight1 = 0;
        let mut expected = antialiased;
        resident.enqueue(&mut antialiased).expect("enqueue antialiased line");
        cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
        assert_context_eq(&antialiased, &expected, "antialiased line context");
        draws += 1;

        let mut auxiliary = line_context(
            DRAWMODE0_ADRMODE_I_LINE_SH,
            DRAWMODE1_PLANES_CID,
            false,
            (360 * 16, 280 * 16),
            (389 * 16, 291 * 16),
            0x00ff_0f0f,
            0x2a00_0237,
        );
        let mut expected = auxiliary;
        resident.enqueue(&mut auxiliary).expect("enqueue auxiliary line");
        cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
        assert_context_eq(&auxiliary, &expected, "auxiliary line context");
        draws += 1;

        resident.submit().unwrap();
        assert!(resident.payload.is_empty());
        assert_eq!(resident.batch.prims_total, draws);
        gpu.wait(None).unwrap();
        let lines = resident.observe(ObservationReason::Validation, PlaneMask::Both).unwrap();
        compare_observed(lines.rgb().unwrap(), &expected_rgb, "RGB lines");
        compare_observed(lines.aux().unwrap(), &expected_aux, "auxiliary lines");
        let status = resident.status().unwrap();
        assert_eq!(status.accepted.sequence, Sequence(draws));
        assert_eq!(status.submitted, status.accepted);
        assert_eq!(status.completed, Some(status.submitted));
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_clipping_matches_complete_cpu_planes() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let seed_rgb = seeded_plane(0x5ca1_37e9);
        let seed_aux = seeded_plane(0xa35e_c816);
        let mut expected_rgb = seed_rgb.clone();
        let mut expected_aux = seed_aux.clone();
        let mut resident = ResidentGpu::activate_with_limits(
            Arc::clone(&gpu),
            &seed_rgb,
            &seed_aux,
            8,
            NonZeroUsize::new(1).unwrap(),
            None,
        ).expect("activate clipped resident planes");
        let range = |min: i32, max: i32| {
            ((min as i16 as u16 as u32) << 16) | max as i16 as u16 as u32
        };

        let mut block = context(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (20, 20),
            (45, 40),
            0xff0f_ffff,
            0x6a31_72b4,
        );
        block.drawmode0.0 |= 1 << 20;
        block.xymove = ((3i16 as u16 as u32) << 16) | (-2i16 as u16 as u32);
        block.xywin = ((5i16 as u16 as u32) << 16) | 4;
        block.smask0x = range(REX3_COORD_BIAS + 25, REX3_COORD_BIAS + 43);
        block.smask0y = range(REX3_COORD_BIAS + 20, REX3_COORD_BIAS + 35);
        block.smask1x = range(REX3_COORD_BIAS + 30, REX3_COORD_BIAS + 36);
        block.smask1y = range(REX3_COORD_BIAS + 22, REX3_COORD_BIAS + 42);
        block.smask2x = range(REX3_COORD_BIAS + 40, REX3_COORD_BIAS + 50);
        block.smask2y = range(REX3_COORD_BIAS + 25, REX3_COORD_BIAS + 32);
        block.clipmode = CLIPMODE_ENSMASK_SMASK0 | (1 << 1) | (1 << 2) | (0b1010 << CLIPMODE_CIDMATCH_SHIFT);
        let mut expected = block;
        resident.enqueue(&mut block).expect("enqueue clipped block");
        cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
        assert_context_eq(&block, &expected, "clipped block context");

        let mut line = line_context(
            DRAWMODE0_ADRMODE_I_LINE_SH,
            DRAWMODE1_PLANES_RGB,
            true,
            (60 * 16, 50 * 16),
            (100 * 16, 70 * 16),
            0x00ff_ffff,
            0x5531_97c2,
        );
        line.drawmode0.0 |= 1 << 20;
        line.xymove = ((-4i16 as u16 as u32) << 16) | 3;
        line.xywin = ((2i16 as u16 as u32) << 16) | (-1i16 as u16 as u32);
        line.smask0x = range(REX3_COORD_BIAS + 64, REX3_COORD_BIAS + 92);
        line.smask0y = range(REX3_COORD_BIAS + 53, REX3_COORD_BIAS + 72);
        line.smask1x = range(REX3_COORD_BIAS + 68, REX3_COORD_BIAS + 82);
        line.smask1y = range(REX3_COORD_BIAS + 55, REX3_COORD_BIAS + 65);
        line.smask2x = range(REX3_COORD_BIAS + 83, REX3_COORD_BIAS + 96);
        line.smask2y = range(REX3_COORD_BIAS + 61, REX3_COORD_BIAS + 72);
        line.clipmode = CLIPMODE_ENSMASK_SMASK0 | (1 << 1) | (1 << 2) | (0b0101 << CLIPMODE_CIDMATCH_SHIFT);
        let mut expected = line;
        resident.enqueue(&mut line).expect("enqueue clipped line");
        cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
        assert_context_eq(&line, &expected, "clipped line context");

        let mut cid = context(
            DRAWMODE1_PLANES_CID,
            false,
            false,
            (120, 80),
            (135, 90),
            0x0000_0003,
            0x0000_0003,
        );
        let mut expected = cid;
        resident.enqueue(&mut cid).expect("enqueue CID update");
        cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
        assert_context_eq(&cid, &expected, "CID update context");

        let mut ordered = context(
            DRAWMODE1_PLANES_RGB,
            true,
            false,
            (120, 80),
            (135, 90),
            u32::MAX,
            0x7f42_68ac,
        );
        ordered.clipmode = 1 << (CLIPMODE_CIDMATCH_SHIFT + 3);
        let mut expected = ordered;
        resident.enqueue(&mut ordered).expect("enqueue ordered CID-tested fill");
        cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
        assert_context_eq(&ordered, &expected, "ordered CID fill context");

        resident.submit().unwrap();
        assert_eq!(resident.batch.prims_total, 4);
        gpu.wait(None).unwrap();
        let clipped = resident.observe(ObservationReason::Validation, PlaneMask::Both).unwrap();
        compare_observed(clipped.rgb().unwrap(), &expected_rgb, "clipped RGB");
        compare_observed(clipped.aux().unwrap(), &expected_aux, "clipped auxiliary");
        let status = resident.status().unwrap();
        assert_eq!(status.accepted.sequence, Sequence(4));
        assert_eq!(status.submitted, status.accepted);
        assert_eq!(status.completed, Some(status.submitted));
    });
}

struct TestRex(Box<Rex3>);

impl Drop for TestRex {
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

fn enqueue_context(rex: &Rex3, context: &Rex3Context, start: (i32, i32), end: (i32, i32)) {
    write_reg(rex, REX3_DRAWMODE1, context.drawmode1.0, false);
    write_reg(rex, REX3_WRMASK, context.wrmask, false);
    write_reg(rex, REX3_COLORRED, context.colorred, false);
    write_reg(rex, REX3_COLORGRN, context.colorgrn, false);
    write_reg(rex, REX3_COLORBLUE, context.colorblue, false);
    write_reg(rex, REX3_COLORALPHA, context.coloralpha, false);
    write_reg(rex, REX3_XYENDI, xy(end.0, end.1), false);
    write_reg(rex, REX3_XYSTARTI, xy(start.0, start.1), false);
    write_reg(rex, REX3_DRAWMODE0, context.drawmode0.0, true);
}

struct SharedResident(Arc<Mutex<ResidentGpu>>);

impl ResidentBackend for SharedResident {
    fn poll_device_health(&mut self) -> Result<(), ResidentError> {
        self.0.lock().poll_device_health()
    }

    fn enqueue(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError> {
        self.0.lock().enqueue(context)
    }

    fn submit(&mut self) -> Result<(), ResidentError> {
        self.0.lock().submit()
    }

    fn observe(&mut self, reason: ObservationReason, planes: PlaneMask) -> Result<Observation, ResidentError> {
        self.0.lock().observe(reason, planes)
    }

    fn acknowledge_cpu_shadow(
        &mut self,
        observation: &Observation,
        rgb: Option<&[u32]>,
        aux: Option<&[u32]>,
    ) -> Result<(), ResidentError> {
        self.0.lock().acknowledge_cpu_shadow(observation, rgb, aux)
    }

    fn status(&mut self) -> Result<ResidentStatus, ResidentError> {
        self.0.lock().status()
    }

    fn recorded_status(&self) -> Result<ResidentStatus, ResidentError> {
        self.0.lock().recorded_status()
    }

    fn reset(&mut self, rgb: &[u32], aux: &[u32]) -> Result<(), ResidentError> {
        self.0.lock().reset(rgb, aux)
    }

    fn restore(&mut self, rgb: &[u32], aux: &[u32]) -> Result<(), ResidentError> {
        self.0.lock().restore(rgb, aux)
    }
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_gfifo_draws_match_complete_cpu_planes() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let seed_rgb = seeded_plane(0x3487_5612);
        let seed_aux = seeded_plane(0xabcd_9876);
        let mut rex = TestRex(Box::new(Rex3::new(crate::disp::StatusSource::detached())));
        rex.0.fb_rgb.get_mut().copy_from_slice(&seed_rgb);
        rex.0.fb_aux.get_mut().copy_from_slice(&seed_aux);
        let resident = Arc::new(Mutex::new(ResidentGpu::activate_with_limits(
            Arc::clone(&gpu), &seed_rgb, &seed_aux, 1, NonZeroUsize::new(1).unwrap(), None,
        ).unwrap()));
        rex.0.attach_resident_raster(SharedResident(Arc::clone(&resident))).unwrap();
        rex.0.start();

        let cases = [
            (DRAWMODE1_PLANES_RGB, true, false, (1338, 1020), (1355, 1023), 0xaa00_ffff, 0x7a33_2211),
            (DRAWMODE1_PLANES_RGB, true, true, (1339, 1022), (1347, 1022), 0x00ff_00ff, 0x1234_5678),
            (DRAWMODE1_PLANES_OLAY, true, false, (3, 5), (18, 19), 0xff0f_ffff, 0x4433_2211),
            (DRAWMODE1_PLANES_PUP, false, true, (5, 7), (15, 7), 0x00ff_ffff, 0x3300_0013),
            (DRAWMODE1_PLANES_CID, false, false, (1340, 1021), (1351, 1025), 0x0fff_ffff, 0x5500_0237),
            (DRAWMODE1_PLANES_RGB, true, false, (1344, 3), (1350, 5), u32::MAX, 0xff22_3344),
        ];
        let mut expected_rgb = seed_rgb.clone();
        let mut expected_aux = seed_aux.clone();
        let mut expected_context = None;
        write_reg(&rex.0, REX3_CLIPMODE, 0xf << CLIPMODE_CIDMATCH_SHIFT, false);
        for (plane, rgbmode, span, start, end, mask, color) in cases {
            let mut expected = context(plane, rgbmode, span, start, end, mask, color);
            expected.drawmode0.0 |= 1 << 5; // Exercise the real GO setup path.
            enqueue_context(&rex.0, &expected, start, end);
            cpu_draw(&mut expected, &mut expected_rgb, &mut expected_aux);
            expected_context = Some(expected);
        }
        rex.0.wait_resident_idle().expect("drain and submit resident GFIFO");
        gpu.wait(None).unwrap();
        let status = rex.0.resident_status().unwrap();
        assert_eq!(status.accepted.sequence, Sequence(cases.len() as u64));
        assert_eq!(status.completed, Some(status.accepted));
        assert_eq!(status.rgb.last_write.sequence, Sequence(2), "offscreen GO is a no-op");
        assert_eq!(status.aux.last_write.sequence, Sequence(5));
        assert_eq!(
            context_state(unsafe { &*rex.0.context.get() }),
            context_state(expected_context.as_ref().unwrap()),
        );
        compare_plane(unsafe { &*rex.0.fb_rgb.get() }, &seed_rgb, "CPU RGB shadow");
        compare_plane(unsafe { &*rex.0.fb_aux.get() }, &seed_aux, "CPU aux shadow");
        assert_eq!(status.observations, ObservationCounters::default(), "idle/batch-full must not read back");
        let mut guard = resident.lock();
        let capture = guard.observe(ObservationReason::Validation, PlaneMask::Both).unwrap();
        compare_observed(capture.rgb().unwrap(), &expected_rgb, "GFIFO RGB");
        compare_observed(capture.aux().unwrap(), &expected_aux, "GFIFO aux");
        drop(guard);

        let start = (8, 20);
        let end = (22, 24);
        let mut prefix = context(
            DRAWMODE1_PLANES_RGB, true, false, start, end, 0x00ff_ffff, 0x1244_8866,
        );
        prefix.drawmode0.0 |= 1 << 5;
        enqueue_context(&rex.0, &prefix, start, end);
        cpu_draw(&mut prefix, &mut expected_rgb, &mut expected_aux);
        write_reg(
            &rex.0, REX3_DRAWMODE0,
            DRAWMODE0_OPCODE_READ | DRAWMODE0_ADRMODE_BLOCK_SH | (1 << 8) | (1 << 9), true,
        );
        assert_eq!(rex.0.wait_resident_idle(), Err(ResidentError::Unsupported("opcode")));
        gpu.wait(None).unwrap();
        let mut guard = resident.lock();
        let prefix_status = guard.status().unwrap();
        assert_eq!(prefix_status.accepted.sequence, Sequence(cases.len() as u64 + 1));
        assert_eq!(prefix_status.completed, Some(prefix_status.accepted));
        let capture = guard.observe(ObservationReason::Validation, PlaneMask::Both).unwrap();
        compare_observed(capture.rgb().unwrap(), &expected_rgb, "GFIFO accepted prefix");
        compare_observed(capture.aux().unwrap(), &expected_aux, "GFIFO unchanged aux");
        drop(guard);
        rex.0.stop();
    });
}

#[test]
#[ignore = "requires the requested conformant hardware GPU backend"]
fn resident_gpu_gfifo_full_context_matches_cpu_across_boundaries_and_retries() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let seed_rgb = seeded_plane(0x1357_9bdf);
        let seed_aux = seeded_plane(0x2468_ace0);
        let mut resident_board = TestRex(Box::new(Rex3::new(crate::disp::StatusSource::detached())));
        let mut cpu_board = TestRex(Box::new(Rex3::new(crate::disp::StatusSource::detached())));
        for board in [&mut resident_board, &mut cpu_board] {
            board.0.fb_rgb.get_mut().copy_from_slice(&seed_rgb);
            board.0.fb_aux.get_mut().copy_from_slice(&seed_aux);
            let context = board.0.context.get_mut();
            context.host_shifter = 0x1234_5678_9abc_def0;
            context.hostrw[0] = 0x0123_4567_89ab_cdef;
            context.hostrw[HOSTRW_BUF_QWORDS / 2] = 0xa5a5_5a5a_f0f0_0f0f;
            context.hostrw[HOSTRW_BUF_QWORDS - 1] = 0xfedc_ba98_7654_3210;
        }
        #[cfg(feature = "rex-jit")]
        cpu_board.0.jit_enabled.store(false, Ordering::Relaxed);
        let resident = Arc::new(Mutex::new(ResidentGpu::activate_with_limits(
            Arc::clone(&gpu), &seed_rgb, &seed_aux, 1, NonZeroUsize::new(1).unwrap(), None,
        ).unwrap()));
        resident_board.0.attach_resident_raster(SharedResident(Arc::clone(&resident))).unwrap();
        resident_board.0.start();
        cpu_board.0.start();
        for board in [&resident_board, &cpu_board] {
            write_reg(&board.0, REX3_CLIPMODE, 0xf << CLIPMODE_CIDMATCH_SHIFT, false);
        }
        let cases = [
            (DRAWMODE1_PLANES_RGB, true, false, (7, 7), (33, 33), u32::MAX, 0x1234_5678),
            (DRAWMODE1_PLANES_RGB, true, false, (31, 31), (35, 34), 0xff00_ff00, 0xaabb_ccdd),
            (DRAWMODE1_PLANES_CID, false, false, (30, 30), (33, 33), 0x00ff_00ff, 0x0012_3456),
            (DRAWMODE1_PLANES_RGB, true, true, (-4, 0), (9, 0), 0x00ff_ffff, 0x0345_6789),
            (DRAWMODE1_PLANES_OLAY, true, false, (1338, 1022), (1350, 1025), 0x0fff_ffff, 0x89ab_cdef),
            (DRAWMODE1_PLANES_RGB, true, false, (1344, 20), (1350, 22), u32::MAX, 0x0bad_f00d),
            (DRAWMODE1_PLANES_PUP, false, false, (0, 0), (8, 8), 0, 0x0012_3456),
            (DRAWMODE1_PLANES_RGB, true, false, (0, 1023), (33, 1025), 0xffff_0000, 0x7654_3210),
        ];
        for (plane, rgbmode, span, start, end, mask, color) in cases {
            let mut draw = context(plane, rgbmode, span, start, end, mask, color);
            draw.drawmode0.0 |= 1 << 5;
            enqueue_context(&resident_board.0, &draw, start, end);
            enqueue_context(&cpu_board.0, &draw, start, end);
        }
        resident_board.0.wait_resident_idle().expect("submit all resident GOs");
        cpu_board.0.wait_idle();
        gpu.wait(None).unwrap();
        let expected_rgb = unsafe { &*cpu_board.0.fb_rgb.get() };
        let expected_aux = unsafe { &*cpu_board.0.fb_aux.get() };
        assert_context_eq(
            unsafe { &*resident_board.0.context.get() },
            unsafe { &*cpu_board.0.context.get() },
            "final GFIFO context",
        );
        #[cfg(feature = "rexdiag")]
        {
            assert_eq!(cpu_board.0.interp_go_count.load(Ordering::Relaxed), cases.len() as u64);
            assert_eq!(resident_board.0.interp_go_count.load(Ordering::Relaxed), 0);
            assert_eq!(resident_board.0.jit_go_count.load(Ordering::Relaxed), 0);
        }
        let mut backend = resident.lock();
        let before = backend.status().unwrap();
        assert_eq!(before.accepted.sequence, Sequence(cases.len() as u64));
        assert_eq!(before.submitted, before.accepted);
        assert_eq!(before.observations, ObservationCounters::default());
        assert_eq!(backend.batch.sets(), 1);
        assert_eq!(backend.batch.prims_total, 6, "two no-op GOs create no GPU fill");
        assert!(backend.batch.flushes >= 5, "one-slot batches repeatedly submit and retry");
        let actual = backend.observe(ObservationReason::Validation, PlaneMask::Both).unwrap();
        compare_observed(actual.rgb().unwrap(), expected_rgb, "GFIFO full-context RGB");
        compare_observed(actual.aux().unwrap(), expected_aux, "GFIFO full-context aux");
        for y in 0..HEIGHT as usize {
            let offscreen = y * STRIDE as usize + REX3_SCREEN_WIDTH as usize;
            let end = (y + 1) * STRIDE as usize;
            assert_eq!(&expected_rgb[offscreen..end], &seed_rgb[offscreen..end]);
            assert_eq!(&expected_aux[offscreen..end], &seed_aux[offscreen..end]);
        }
        drop(backend);
        resident_board.0.stop();
        cpu_board.0.stop();
        cpu_board.0.power_on();
        resident_board.0.reset_resident_gpu().unwrap();
        assert_context_eq(
            unsafe { &*resident_board.0.context.get() },
            unsafe { &*cpu_board.0.context.get() },
            "reset GFIFO context",
        );
        let reset = resident_board.0.resident_status().unwrap();
        assert_ne!(reset.accepted.epoch, before.accepted.epoch);
        assert_eq!((reset.initialization_uploads, reset.raw_plane_map_attempts), (4, 1));
        let mut backend = resident.lock();
        let reset_planes = backend.observe(ObservationReason::Validation, PlaneMask::Both).unwrap();
        compare_observed(reset_planes.rgb().unwrap(), unsafe { &*cpu_board.0.fb_rgb.get() }, "reset RGB");
        compare_observed(reset_planes.aux().unwrap(), unsafe { &*cpu_board.0.fb_aux.get() }, "reset aux");
        assert_eq!(backend.status().unwrap().raw_plane_map_attempts, 2);
    });
}

#[test]
#[ignore = "requires the requested conformant host GPU"]
fn resident_gpu_gfifo_activation_reset_and_fault() {
    on_large_stack(|| {
        let gpu = required_gpu();
        let mut rex = TestRex(Box::new(Rex3::new(crate::disp::StatusSource::detached())));
        let before_rgb = unsafe { (*rex.0.fb_rgb.get())[20 * STRIDE as usize + 10] };
        let before_aux = unsafe { (*rex.0.fb_aux.get())[20 * STRIDE as usize + 10] };
        rex.0.activate_resident_gpu(Arc::clone(&gpu)).expect("activate through board gate");
        let initial = rex.0.resident_status().unwrap();
        let initial_bindings = rex.0.resident_scanout_bindings(&gpu).unwrap();
        assert_eq!(initial_bindings.epoch, initial.accepted.epoch);
        rex.0.start();
        assert!(rex.0.is_running());
        assert!(rex.0.refresh_thread.lock().is_none());
        let dm1_rgb = DRAWMODE1_PLANES_RGB | (DRAWMODE1_DRAWDEPTH_24 << 3)
            | (1 << 15) | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH;
        let dm1_aux = DRAWMODE1_PLANES_CID | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH;
        let draw = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK_SH | (1 << 8) | (1 << 9);
        write_reg(&rex.0, REX3_DRAWMODE1, dm1_rgb, false);
        write_reg(&rex.0, REX3_WRMASK, 0x00ff_ffff, false);
        write_reg(&rex.0, REX3_CLIPMODE, 0xf << CLIPMODE_CIDMATCH_SHIFT, false);
        write_reg(&rex.0, REX3_COLORRED, 0x11 << 11, false);
        write_reg(&rex.0, REX3_COLORGRN, 0x22 << 11, false);
        write_reg(&rex.0, REX3_COLORBLUE, 0x33 << 11, false);
        write_reg(&rex.0, REX3_XYENDI, xy(12, 22), false);
        write_reg(&rex.0, REX3_XYSTARTI, xy(10, 20), false);
        write_reg(&rex.0, REX3_DRAWMODE0, draw, true);
        write_reg(&rex.0, REX3_DRAWMODE1, dm1_aux, false);
        write_reg(&rex.0, REX3_WRMASK, 0x0000_00ff, false);
        write_reg(&rex.0, REX3_COLORRED, 3 << 11, false);
        write_reg(&rex.0, REX3_XYENDI, xy(12, 22), false);
        write_reg(&rex.0, REX3_XYSTARTI, xy(10, 20), false);
        write_reg(&rex.0, REX3_DRAWMODE0, draw, true);
        write_reg(&rex.0, REX3_WRMASK, 0, false);
        write_reg(&rex.0, REX3_XYENDI, xy(12, 22), false);
        write_reg(&rex.0, REX3_XYSTARTI, xy(10, 20), false);
        write_reg(&rex.0, REX3_DRAWMODE0, draw, true);
        rex.0.gfifo.push(GFIFO_DISP_SYNC, 9);
        rex.0.wait_resident_idle().expect("drain and submit resident GFIFO");
        assert_eq!(rex.0.gfifo_fence.load(Ordering::Acquire), 9);
        let status = rex.0.resident_status().unwrap();
        assert_eq!(status.accepted.sequence, Sequence(3));
        assert_eq!(status.submitted, status.accepted);
        assert_eq!(status.rgb.last_write.sequence, Sequence(1));
        assert_eq!(status.aux.last_write.sequence, Sequence(2));
        assert!(!status.rgb.cpu_shadow_current() && !status.aux.cpu_shadow_current());
        assert_eq!(status.observations, ObservationCounters::default(), "idle and display fence must not read back");
        assert_eq!(unsafe { (*rex.0.fb_rgb.get())[20 * STRIDE as usize + 10] }, before_rgb);
        assert_eq!(unsafe { (*rex.0.fb_aux.get())[20 * STRIDE as usize + 10] }, before_aux);
        gpu.wait(None).unwrap();
        assert_eq!(rex.0.resident_status().unwrap().completed, Some(status.submitted));

        rex.0.stop();
        rex.0.reset_resident_gpu().expect("quiescent resident reset");
        let reset = rex.0.resident_status().unwrap();
        assert_ne!(reset.accepted.epoch, initial.accepted.epoch);
        assert_ne!(rex.0.resident_scanout_bindings(&gpu).unwrap().epoch, initial_bindings.epoch);
        assert_eq!(reset.accepted.sequence, Sequence(0));
        assert!(reset.rgb.cpu_shadow_current() && reset.aux.cpu_shadow_current());
        rex.0.start();
        write_reg(&rex.0, REX3_DRAWMODE0, DRAWMODE0_OPCODE_READ | DRAWMODE0_ADRMODE_BLOCK_SH | (1 << 8) | (1 << 9), true);
        assert_eq!(rex.0.wait_resident_idle(), Err(ResidentError::Unsupported("opcode")));
        assert_eq!(rex.0.resident_status(), Err(ResidentError::Unsupported("opcode")));
        rex.0.stop();
        assert!(!rex.0.is_running());
        assert!(rex.0.processor_thread.lock().is_none());
        rex.0.reset_resident_gpu().expect("reset recovers an unsupported-operation fault");
        let recovered = rex.0.resident_status().unwrap();
        assert_ne!(recovered.accepted.epoch, reset.accepted.epoch);
        assert_eq!(recovered.accepted.sequence, Sequence(0));
        assert!(recovered.rgb.cpu_shadow_current() && recovered.aux.cpu_shadow_current());
    });
}
