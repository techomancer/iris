// The scanout pass driver: one lane per screen pixel, the board's
// `scanout_pixel(x, y) -> u32` (0x00RRGGBB) into a frame of `stride`
// pixels a row. Group 0 is the frame; group 1 binding 0 is the board's
// display slot, declared by the board's fragment; groups 2 and up are the
// board's memory, bound as its raster kernels bind it.

struct ScanoutHeader {
    width: u32,
    height: u32,
    stride: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> scan: ScanoutHeader;
@group(0) @binding(1) var<storage, read_write> scan_out: array<u32>;

@compute @workgroup_size(8, 8)
fn scanout(@builtin(global_invocation_id) g: vec3<u32>) {
    if g.x >= scan.width || g.y >= scan.height {
        return;
    }
    scan_out[g.y * scan.stride + g.x] = scanout_pixel(g.x, g.y);
}
