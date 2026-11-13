// The quad exchanges on a device with subgroup operations: lanes 4k to
// 4k + 3 of a subgroup are a quad (raster_entry_subgroup.wgsl).

// The value of the quad's lane across in x (lane ^ 1).
fn quad_swap_x(v: vec4<u32>) -> vec4<u32> {
    return quadSwapX(v);
}

// The value of the quad's lane across the diagonal (lane ^ 3).
fn quad_swap_diagonal(v: vec4<u32>) -> vec4<u32> {
    return quadSwapDiagonal(v);
}
