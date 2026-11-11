//! The integer libraries a kernel computes models' wide arithmetic with,
//! against the host: `f64.wgsl`'s double add, subtract, multiply, floor
//! and ceil bit for bit against Rust's f64, and `wide.wgsl`'s 128-bit
//! products and shifts against i128. Needs a GPU (or lavapipe); skips
//! without one.

use std::sync::Arc;

use super::super::device::Gpu;
use super::super::library;
use super::super::memory::Staging;
use super::super::shader::{Fragment, Source};

const KERNEL: Fragment = Fragment::new(
    "softfloat_test.wgsl",
    r#"
#include "f64.wgsl"

@group(0) @binding(0) var<storage, read> ins: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> outs: array<vec4<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if i >= arrayLength(&ins) {
        return;
    }
    let v = ins[i];
    let a = I64(v.x, v.y);
    let b = I64(v.z, v.w);
    let add = f64_add(a, b);
    let sub = f64_sub(a, b);
    let mul = f64_mul(a, b);
    outs[4u * i] = vec4<u32>(add.lo, add.hi, sub.lo, sub.hi);
    outs[4u * i + 1u] = vec4<u32>(mul.lo, mul.hi, bitcast<u32>(f64_floor_i32(a)), bitcast<u32>(f64_ceil_i32(a)));
    let conv = f64_from_i32(bitcast<i32>(v.x));
    outs[4u * i + 2u] = vec4<u32>(select(0u, 1u, f64_ge(a, b)), conv.lo, conv.hi, 0u);
    let ms = mul_shr_s(a, b, i % 128u);
    let ml = i64_mul(a, b);
    outs[4u * i + 3u] = vec4<u32>(ms.lo, ms.hi, ml.lo, ml.hi);
}
"#,
);

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn f(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next() >> 11) as f64 / (1u64 << 53) as f64)
    }

    /// A double with a random significand and an exponent within `span`
    /// of 0, either sign.
    fn wide(&mut self, span: i64) -> f64 {
        let e = (1023 + (self.next() % (2 * span as u64 + 1)) as i64 - span) as u64;
        f64::from_bits((self.next() & 0x800F_FFFF_FFFF_FFFF) | e << 52)
    }
}

fn run(gpu: &Arc<Gpu>, ins: &[[u32; 4]]) -> Vec<[u32; 4]> {
    let src = Source::new("softfloat test").library(library::I64).library(library::WIDE).library(library::F64).body(KERNEL);
    let entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("softfloat"), entries: &[entry(0, true), entry(1, false)] });
    let pipeline = src.compute_pipeline(&gpu.device, "main", &[&layout]).unwrap_or_else(|e| panic!("{e}"));
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let inb = gpu.buffer("softfloat in", (ins.len() * 16) as u64, storage);
    let outb = gpu.buffer("softfloat out", (ins.len() * 64) as u64, storage);
    gpu.write(&inb, 0, bytemuck::cast_slice(ins));
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("softfloat"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: outb.as_entire_binding() },
        ],
    });
    let mut enc = gpu.encoder("softfloat");
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("softfloat"), timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((ins.len() as u32).div_ceil(64), 1, 1);
    }
    gpu.submit(enc);
    let bytes = Staging::default().read(gpu, &outb, &[(0, outb.size())]).unwrap_or_else(|e| panic!("{e}"));
    gpu.check().unwrap_or_else(|e| panic!("{e}"));
    bytemuck::cast_slice(&bytes).to_vec()
}

fn bits(lo: u32, hi: u32) -> u64 {
    lo as u64 | (hi as u64) << 32
}

fn subnormal(v: f64) -> bool {
    v != 0.0 && v.abs() < f64::MIN_POSITIVE
}

/// Equal as the kernel promises: the same bits, or both NaN.
fn same(got: u64, want: f64) -> bool {
    let g = f64::from_bits(got);
    (g.is_nan() && want.is_nan()) || got == want.to_bits()
}

#[test]
#[ignore = "requires a conformant host GPU"]
fn softfloat_and_wide_products_match_the_host() {
    let gpu = super::gpu();
    let mut rng = Rng(0x5EED_F64);
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    // Magnitudes over a wide range, then the inputs the boards make:
    // positions on the 2^-8 grid, s.24 slopes, half-row offsets.
    for _ in 0..20000 {
        pairs.push((rng.wide(60), rng.wide(60)));
    }
    for _ in 0..20000 {
        let x = (rng.f(-20000.0, 20000.0) * 256.0).round() / 256.0;
        let s = (rng.f(-40.0, 40.0) * (1u64 << 24) as f64).round() / (1u64 << 24) as f64;
        let h = rng.f(-3000.0, 3000.0).round() + 0.5;
        pairs.push((x, s * h));
        pairs.push((s, h));
        pairs.push((x, 0.5));
    }
    // Near cancellation, ties, exact integers and halves, specials.
    for _ in 0..5000 {
        let a = rng.wide(30);
        let ulps = (rng.next() % 9) as i64 - 4;
        let b = -f64::from_bits((a.to_bits() as i64 + ulps) as u64);
        pairs.push((a, b));
        let k = (rng.next() % 4000) as f64 - 2000.0;
        pairs.push((k, 0.5));
        pairs.push((k + 0.5, -0.5));
        pairs.push((k, f64::EPSILON * k));
    }
    for a in [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2147483647.0, 2147483648.0, -2147483648.0, -2147483649.0, 1e300, -1e300, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        for b in [0.0, -0.0, 1.0, 2.0f64.powi(-53), -(2.0f64.powi(-53)), 1e300, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            pairs.push((a, b));
        }
    }
    // Keep only pairs whose results stay normal (the library flushes).
    pairs.retain(|&(a, b)| !subnormal(a) && !subnormal(b) && !subnormal(a + b) && !subnormal(a - b) && !subnormal(a * b));
    let ins: Vec<[u32; 4]> = pairs.iter().map(|&(a, b)| [a.to_bits() as u32, (a.to_bits() >> 32) as u32, b.to_bits() as u32, (b.to_bits() >> 32) as u32]).collect();
    let outs = run(&gpu, &ins);
    let mut bad = Vec::new();
    for (i, &(a, b)) in pairs.iter().enumerate() {
        let o = &outs[4 * i..4 * i + 4];
        let mut fail = |what: &str, got: String, want: String| {
            if bad.len() < 20 {
                bad.push(format!("{what}({a:e} [{:#018x}], {b:e} [{:#018x}]): gpu {got}, host {want}", a.to_bits(), b.to_bits()));
            }
        };
        if !same(bits(o[0][0], o[0][1]), a + b) {
            fail("add", format!("{:e}", f64::from_bits(bits(o[0][0], o[0][1]))), format!("{:e}", a + b));
        }
        if !same(bits(o[0][2], o[0][3]), a - b) {
            fail("sub", format!("{:e}", f64::from_bits(bits(o[0][2], o[0][3]))), format!("{:e}", a - b));
        }
        if !same(bits(o[1][0], o[1][1]), a * b) {
            fail("mul", format!("{:e}", f64::from_bits(bits(o[1][0], o[1][1]))), format!("{:e}", a * b));
        }
        if o[1][2] as i32 != a.floor() as i32 {
            fail("floor", format!("{}", o[1][2] as i32), format!("{}", a.floor() as i32));
        }
        if o[1][3] as i32 != a.ceil() as i32 {
            fail("ceil", format!("{}", o[1][3] as i32), format!("{}", a.ceil() as i32));
        }
        if (o[2][0] != 0) != (a >= b) {
            fail("ge", format!("{}", o[2][0]), format!("{}", a >= b));
        }
        let conv = a.to_bits() as u32 as i32 as f64;
        if bits(o[2][1], o[2][2]) != conv.to_bits() {
            fail("from_i32", format!("{:e}", f64::from_bits(bits(o[2][1], o[2][2]))), format!("{conv:e}"));
        }
        let (ia, ib) = (a.to_bits() as i64, b.to_bits() as i64);
        let ms = ((ia as i128 * ib as i128) >> (i % 128)) as i64;
        if bits(o[3][0], o[3][1]) != ms as u64 {
            fail("mul_shr_s", format!("{:#x}", bits(o[3][0], o[3][1])), format!("{ms:#x}"));
        }
        if bits(o[3][2], o[3][3]) != ia.wrapping_mul(ib) as u64 {
            fail("i64_mul", format!("{:#x}", bits(o[3][2], o[3][3])), format!("{:#x}", ia.wrapping_mul(ib)));
        }
    }
    assert!(bad.is_empty(), "{} pairs, mismatches:\n{}", pairs.len(), bad.join("\n"));
}
