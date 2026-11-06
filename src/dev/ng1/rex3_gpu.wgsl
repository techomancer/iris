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
var<storage, read> fills: array<FlatFill>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) id: vec3<u32>) {
    let fill_count = arrayLength(&fills);
    for (var draw = 0u; draw < fill_count; draw += 1u) {
        let fill = fills[draw];
        let pixel_count = fill.width * fill.height;
        for (var pixel = id.x; pixel < pixel_count; pixel += 256u) {
            let x = fill.min_x + pixel % fill.width;
            let y = fill.min_y + pixel / fill.width;
            let address = y * fill.stride + x;
            if address < fill.plane_len {
                let old = rgb[address];
                rgb[address] = (old & ~fill.write_mask) | (fill.color & fill.write_mask);
            }
        }
        storageBarrier();
    }
}
