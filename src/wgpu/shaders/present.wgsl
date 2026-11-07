// Presenting into a window: textured quads (the frame letterboxed, the
// debug overlay over it, the status bar under it).

// Where a quad goes, in normalised device coordinates (x0, top, x1,
// bottom), and the texture coordinates it shows (u0, v0 at the top left,
// u1, v1 at the bottom right).
struct Quad {
    rect: vec4<f32>,
    uv: vec4<f32>,
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> quad: Quad;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn quad_vs(@builtin(vertex_index) i: u32) -> VOut {
    var corner = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let k = corner[i];
    var o: VOut;
    o.pos = vec4<f32>(mix(quad.rect.x, quad.rect.z, k.x), mix(quad.rect.y, quad.rect.w, k.y), 0.0, 1.0);
    o.uv = vec2<f32>(mix(quad.uv.x, quad.uv.z, k.x), mix(quad.uv.y, quad.uv.w, k.y));
    return o;
}

@fragment
fn quad_fs(v: VOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, v.uv);
}
