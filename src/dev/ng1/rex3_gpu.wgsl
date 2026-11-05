struct FlatFill {
    min_x: u32,
    min_y: u32,
    width: u32,
    height: u32,
    color: u32,
    write_mask: u32,
    stride: u32,
    plane_len: u32,
}

@group(0) @binding(0)
var<storage, read_write> rgb: array<u32>;

@group(0) @binding(1)
var<uniform> fill: FlatFill;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel_count = fill.width * fill.height;
    if id.x >= pixel_count {
        return;
    }

    let x = fill.min_x + id.x % fill.width;
    let y = fill.min_y + id.x / fill.width;
    let address = y * fill.stride + x;
    if address >= fill.plane_len {
        return;
    }

    let old = rgb[address];
    rgb[address] = (old & ~fill.write_mask) | (fill.color & fill.write_mask);
}
