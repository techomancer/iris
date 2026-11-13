// The quad exchanges without subgroup operations: through workgroup
// memory, between two barriers. The caller is in workgroup-uniform
// control flow (raster.wgsl).

var<workgroup> quad_values: array<vec4<u32>, 64>;

fn quad_exchange(v: vec4<u32>, flip: u32) -> vec4<u32> {
    quad_values[raster_lane] = v;
    workgroupBarrier();
    let r = quad_values[raster_lane ^ flip];
    workgroupBarrier();
    return r;
}

// The value of the quad's lane across in x (lane ^ 1).
fn quad_swap_x(v: vec4<u32>) -> vec4<u32> {
    return quad_exchange(v, 1u);
}

// The value of the quad's lane across the diagonal (lane ^ 3).
fn quad_swap_diagonal(v: vec4<u32>) -> vec4<u32> {
    return quad_exchange(v, 3u);
}
