// A frame of 0x00RRGGBB words already on the device (a board's scan-out),
// rows of `size.x`, into the presenter's frame texture: no readback.

@group(0) @binding(0) var<storage, read> frame_words: array<u32>;
@group(0) @binding(1) var frame_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> size: vec4<u32>;

@compute @workgroup_size(8, 8)
fn unpack(@builtin(global_invocation_id) g: vec3<u32>) {
    if g.x >= size.x || g.y >= size.y {
        return;
    }
    let w = frame_words[g.y * size.x + g.x];
    let c = vec4<f32>(f32((w >> 16u) & 0xffu), f32((w >> 8u) & 0xffu), f32(w & 0xffu), 255.0) / 255.0;
    textureStore(frame_out, vec2<i32>(g.xy), c);
}
