use std::sync::Arc;
use std::time::Instant;

use super::direct::DID_EXPAND;
use super::VC2_WORDS;
use crate::disp::{FB_MAX_H, FB_STRIDE};
use crate::wgpu::{Gpu, GpuOptions, Source};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 1024;
const DID_PTR: usize = 0x100;
const RUNS_PTR: usize = 0x1000;
const SAMPLES: u32 = 64;
const ROUNDS: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DidParams {
    did_ptr: u32,
    scan_len: u32,
    width: u32,
    height: u32,
    enabled: u32,
    display_on: u32,
    pad: [u32; 2],
}

#[derive(Default)]
struct Measurements {
    timestamp_ns: Vec<f64>,
    wall_ns: Vec<f64>,
}

#[derive(Clone, Copy)]
struct Variant {
    name: &'static str,
    entry: &'static str,
    threads_per_workgroup: u32,
    workgroups: u32,
    production: bool,
}

const VARIANTS: [Variant; 2] = [
    Variant {
        name: "serial",
        entry: "expand_did_serial",
        threads_per_workgroup: 1,
        workgroups: HEIGHT,
        production: false,
    },
    Variant {
        name: "packed",
        entry: "expand_did",
        threads_per_workgroup: 64,
        workgroups: HEIGHT,
        production: true,
    },
];

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
    let gpu = Gpu::new(&GpuOptions {
        backend: backend.clone(),
        ..GpuOptions::default()
    })
    .unwrap_or_else(|error| panic!("required DID benchmark backend {backend}: {error}"));
    if let Some(expected) = expected {
        assert_eq!(
            gpu.caps.backend, expected,
            "the requested backend must actually be used"
        );
    }
    assert!(
        matches!(
            gpu.caps.device_type,
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
        ),
        "DID benchmark requires a hardware adapter: {}",
        gpu.caps,
    );
    assert!(
        gpu.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY),
        "DID benchmark requires timestamp queries: {}",
        gpu.caps,
    );
    eprintln!("DID expansion benchmark adapter: {}", gpu.caps);
    gpu
}

fn did_ram(divergent: bool) -> Vec<u16> {
    let mut ram = vec![0; VC2_WORDS];
    let run_counts = [1u32, 2, 4, 8];
    let mut pointers = [0u16; 4];
    let mut ptr = RUNS_PTR;

    for (pattern, &runs) in run_counts.iter().enumerate() {
        pointers[pattern] = ptr as u16;
        ram[ptr] = pattern as u16;
        ptr += 1;
        for run in 1..runs {
            let x = WIDTH * run / runs;
            ram[ptr] = ((x << 5) | ((pattern as u32 + run) & 0x1f)) as u16;
            ptr += 1;
        }
        ram[ptr] = 0xffe0;
        ptr += 1;
    }

    for y in 0..HEIGHT as usize {
        ram[DID_PTR + y] = pointers[if divergent { y % pointers.len() } else { 0 }];
    }
    ram
}

fn measure(
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    label: &'static str,
    variant: Variant,
) -> (Vec<f64>, f64) {
    let mut warmup = gpu.encoder("DID expansion benchmark warmup");
    {
        let mut pass = warmup.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("DID expansion benchmark warmup"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(variant.workgroups, 1, 1);
    }
    let submission = gpu.submit(warmup);
    gpu.wait(Some(submission))
        .expect("wait for DID benchmark warmup");

    let count = 2 * SAMPLES;
    let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("DID expansion benchmark timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count,
    });
    let bytes = u64::from(count) * size_of::<u64>() as u64;
    let resolve = gpu.buffer(
        "DID expansion benchmark resolve",
        bytes,
        wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
    );
    let read = gpu.buffer(
        "DID expansion benchmark readback",
        bytes,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let mut encoder = gpu.encoder("DID expansion benchmark");
    for sample in 0..SAMPLES {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &queries,
                beginning_of_pass_write_index: Some(2 * sample),
                end_of_pass_write_index: Some(2 * sample + 1),
            }),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(variant.workgroups, 1, 1);
    }
    encoder.resolve_query_set(&queries, 0..count, &resolve, 0);
    encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, bytes);
    let started = Instant::now();
    let submission = gpu.submit(encoder);
    let slice = read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.wait(Some(submission)).expect("wait for DID benchmark");
    gpu.wait(None).expect("finish DID benchmark mapping");
    let mapped = slice
        .get_mapped_range()
        .expect("map DID benchmark timestamps");
    let period_ns = f64::from(gpu.queue.get_timestamp_period());
    let mut samples = mapped
        .chunks_exact(2 * size_of::<u64>())
        .map(|pair| {
            let begin = u64::from_le_bytes(pair[..8].try_into().unwrap());
            let end = u64::from_le_bytes(pair[8..16].try_into().unwrap());
            end.saturating_sub(begin) as f64 * period_ns
        })
        .collect::<Vec<_>>();
    drop(mapped);
    read.unmap();
    samples.sort_by(f64::total_cmp);
    let wall_ns = started.elapsed().as_secs_f64() * 1_000_000_000.0 / f64::from(SAMPLES);
    (samples, wall_ns)
}

fn report(label: &str, variant: Variant, measurements: &Measurements) {
    let mut samples = measurements.timestamp_ns.clone();
    samples.sort_by(f64::total_cmp);
    let mut wall = measurements.wall_ns.clone();
    wall.sort_by(f64::total_cmp);
    let percentile = |percent: usize| samples[(samples.len() - 1) * percent / 100] / 1_000.0;
    eprintln!(
        "DID expansion {label} {}: threads/workgroup={} workgroups={} \
         production={} samples={} timestamp \
         min={:.3}us p50={:.3}us p95={:.3}us max={:.3}us wall-p50={:.3}us",
        variant.name,
        variant.threads_per_workgroup,
        variant.workgroups,
        variant.production,
        samples.len(),
        samples[0] / 1_000.0,
        percentile(50),
        percentile(95),
        samples[samples.len() - 1] / 1_000.0,
        wall[wall.len() / 2] / 1_000.0,
    );
}

#[test]
#[ignore = "manual hardware latency benchmark"]
fn did_expansion_dispatch_latency() {
    let gpu = required_gpu();
    let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DID expansion benchmark"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(2, wgpu::BufferBindingType::Uniform),
            ],
        });
    let pipelines: [wgpu::ComputePipeline; 2] = std::array::from_fn(|index| {
        let variant = VARIANTS[index];
        Source::new(&format!("DID expansion benchmark {}", variant.name))
            .body(DID_EXPAND)
            .compute_pipeline(&gpu.device, variant.entry, &[&layout])
            .expect("create DID expansion benchmark pipeline")
    });
    let vc2 = gpu.buffer(
        "DID expansion benchmark VC2 RAM",
        (2 * VC2_WORDS) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let did = gpu.buffer(
        "DID expansion benchmark map",
        (FB_STRIDE * FB_MAX_H) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let params = gpu.buffer(
        "DID expansion benchmark parameters",
        size_of::<DidParams>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("DID expansion benchmark"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vc2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: did.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
            },
        ],
    });
    gpu.write(
        &params,
        0,
        bytemuck::bytes_of(&DidParams {
            did_ptr: DID_PTR as u32,
            scan_len: WIDTH,
            width: WIDTH,
            height: HEIGHT,
            enabled: 1,
            display_on: 1,
            pad: [0; 2],
        }),
    );

    let workloads = [("uniform", false), ("divergent", true)];
    let mut results: [[Measurements; 2]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Measurements::default()));
    for round in 0..ROUNDS {
        let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
        for workload in order {
            let (label, divergent) = workloads[workload];
            gpu.write(&vc2, 0, bytemuck::cast_slice(&did_ram(divergent)));
            for variant in order {
                let (samples, wall_ns) =
                    measure(&gpu, &pipelines[variant], &group, label, VARIANTS[variant]);
                results[workload][variant].timestamp_ns.extend(samples);
                results[workload][variant].wall_ns.push(wall_ns);
            }
        }
    }
    for (workload, (label, _)) in workloads.iter().enumerate() {
        for (variant, config) in VARIANTS.iter().copied().enumerate() {
            report(label, config, &results[workload][variant]);
        }
    }
}
