//! Vertex goldens: matrix and light setups through the GL path, no raster.

use super::*;
use crate::dev::kona::back::raster::fog::Curve;
use crate::dev::kona::back::raster::{DrawState, Smooth, SmoothPoint, SmoothPoly};
use vertex::grid;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

struct Rig {
    gl: Gl,
    path: GlPath,
    /// Every triangle made so far, and the draw state each ran under.
    tris: Vec<[WVertex; 3]>,
    states: Vec<DrawState>,
    clears: Vec<ClearReq>,
    lines: Vec<[WVertex; 3]>,
    discs: Vec<SmoothPoint>,
    aa: Vec<SmoothPoly>,
    aa_verts: Vec<WVertex>,
}

impl Rig {
    /// A 100 x 100 window and viewport, an orthographic projection of -1..1.
    fn new() -> Rig {
        let mut r = Rig { gl: Gl::default(), path: GlPath::new((100, 100)), tris: Vec::new(), states: Vec::new(), clears: Vec::new(), lines: Vec::new(), discs: Vec::new(), aa: Vec::new(), aa_verts: Vec::new() };
        r.gl.place([0, 0], Rect { x0: 0, y0: 0, x1: 100, y1: 100 }, 0, 0, false);
        r.cmd(SET_VIEWPORT, &bits(&[0.0, 0.0, 100.0, 100.0]));
        r
    }

    fn cmd(&mut self, selector: u32, a: &[u32]) {
        let mut prims = Prims::default();
        self.path.command(&mut self.gl, selector, a, 0, &mut prims);
        self.states.extend(prims.tris.iter().map(|_| prims.state.expect("triangles come with a state")));
        self.tris.append(&mut prims.tris);
        self.clears.append(&mut prims.clears);
        self.lines.append(&mut prims.lines);
        self.discs.append(&mut prims.discs);
        // Indices into the vertices are rebased as the runs are appended.
        let base = self.aa_verts.len() as u32;
        self.aa.extend(prims.aa.iter().map(|p| SmoothPoly { start: p.start + base, ..*p }));
        self.aa_verts.append(&mut prims.aa_verts);
    }

    fn begin(&mut self, primitive: u32) {
        self.cmd(primitive, &[0]);
    }

    fn v(&mut self, x: f32, y: f32) {
        self.cmd(SUBMIT_VERTEX_2_COMPONENTS_SINGLE_FLOAT, &bits(&[x, y]));
    }

    fn v3(&mut self, x: f32, y: f32, z: f32) {
        self.cmd(SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, &bits(&[x, y, z]));
    }

    fn end(&mut self) {
        self.cmd(END, &[0]);
    }

    fn colour(&mut self, r: f32, g: f32, b: f32) {
        self.cmd(SET_CURRENT_COLOUR_3_COMPONENTS_SINGLE_FLOAT, &bits(&[r, g, b]));
    }

    fn triangle(&mut self, p: [(f32, f32); 3]) {
        self.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
        for (x, y) in p {
            self.v(x, y);
        }
        self.end();
    }
}

fn xy(d: &[WVertex; 3]) -> Vec<(i32, i32)> {
    d.iter().map(|v| (v.x, v.y)).collect()
}

/// A triangle in the unit view volume lands on the viewport: ndc -1 and 1 are
/// the viewport's edges, in 1/256 pixels.
#[test]
fn a_triangle_is_mapped_to_the_viewport_on_the_quarter_pixel_grid() {
    let mut r = Rig::new();
    r.triangle([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]);
    assert_eq!(xy(&r.tris[0]), [(0, 0), (100 * 256, 0), (50 * 256, 100 * 256)]);
    // Depth: ndc z 0 is the middle of the depth range.
    assert!(r.tris[0].iter().all(|v| v.z == depth32(0.5)));
}

/// Matrices compose, push and pop, and the model-view applies before the
/// projection.
#[test]
fn matrices_move_vertices_before_the_viewport() {
    let mut r = Rig::new();
    r.cmd(TRANSLATE_FLOATS, &bits(&[0.5, 0.0, 0.0]));
    r.cmd(PUSH_MATRIX_STACK, &[0]);
    r.cmd(SCALE_FLOATS, &bits(&[0.5, 0.5, 1.0]));
    r.triangle([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]);
    r.cmd(POP_MATRIX_STACK, &[0]);
    r.triangle([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]);
    // Scaled by a half and moved 0.5: x from 0.5 to 1.0 (ndc), 75 to 100 pixels.
    assert_eq!(xy(&r.tris[0]), [(75 * 256, 50 * 256), (100 * 256, 50 * 256), (75 * 256, 75 * 256)]);
    assert_eq!((r.gl.mv.get()[0], r.gl.mv.get()[12]), (1.0, 0.5), "the pop restored the translation alone");
}

/// Nothing cuts a triangle in window space: one that crosses the window edge
/// keeps its vertices where the projection puts them, with their own colours.
/// Set-up cuts the job's rectangle instead.
#[test]
fn a_triangle_crossing_the_window_is_not_cut_in_window_space() {
    let mut r = Rig::new();
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.colour(1.0, 0.0, 0.0);
    r.v(0.0, -0.5);
    r.colour(0.0, 1.0, 0.0);
    r.v(3.0, -0.5);
    r.colour(0.0, 0.0, 1.0);
    r.v(0.0, 0.5);
    r.end();
    assert_eq!(r.tris.len(), 1);
    // x = 3 is ndc 3: 50 * (3 + 1) = 200 pixels.
    assert_eq!(xy(&r.tris[0]), [(50 * 256, 25 * 256), (200 * 256, 25 * 256), (50 * 256, 75 * 256)]);
    assert_eq!(r.tris[0].map(|v| v.c), [[8191, 0, 0, 8191], [0, 8191, 0, 8191], [0, 0, 8191, 8191]]);
    assert_eq!(r.path.counts.clipped, 0);
}

/// A primitive wholly outside the volume is clipped away.
#[test]
fn a_primitive_outside_the_volume_draws_nothing() {
    let mut r = Rig::new();
    // Beyond the far plane.
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.v3(0.0, 0.0, 2.0);
    r.v3(0.5, 0.0, 2.0);
    r.v3(0.0, 0.5, 2.0);
    r.end();
    assert!(r.tris.is_empty());
    assert_eq!(r.path.counts.clipped_away, 1);
    // In front of the near plane.
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.v3(0.0, 0.0, -2.0);
    r.v3(0.5, 0.0, -2.0);
    r.v3(0.0, 0.5, -2.0);
    r.end();
    assert!(r.tris.is_empty());
    assert_eq!(r.path.counts.clipped_away, 2);
}

/// The far plane is clipped geometrically: a triangle with one vertex beyond it
/// is cut where it meets the plane, at the depth range's far end. The cut
/// vertices are those of the unit square's far edge, found by hand: the
/// vertex at z = 3 against the others at z = 0 meets z = 1 a third of the way
/// along each edge.
#[test]
fn a_triangle_straddling_the_far_plane_is_cut_at_it() {
    let mut r = Rig::new();
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.v3(-0.5, -0.5, 0.0);
    r.v3(0.5, -0.5, 0.0);
    r.v3(0.0, 0.5, 3.0);
    r.end();
    assert_eq!(r.tris.len(), 2, "a quadrilateral, as a fan of two");
    assert_eq!(r.path.counts.clipped, 1);
    let far = depth32(1.0);
    let verts: Vec<_> = r.tris.iter().flatten().collect();
    assert!(verts.iter().all(|v| v.z <= far));
    // Cut vertices: from (-0.5, -0.5, 0) to (0, 0.5, 3) at t = 1/3, and from (0.5, -0.5, 0) the same.
    let cut = |x: f32, y: f32| (grid(50.0 * (x + 1.0)), grid(50.0 * (y + 1.0)));
    for want in [cut(-0.5 + 0.5 / 3.0, -0.5 + 1.0 / 3.0), cut(0.5 - 0.5 / 3.0, -0.5 + 1.0 / 3.0)] {
        let hit = verts.iter().find(|v| (v.x - want.0).abs() <= 2 && (v.y - want.1).abs() <= 2);
        let hit = hit.unwrap_or_else(|| panic!("no vertex near {want:?} in {verts:?}"));
        assert!(hit.z >= far - 0x2000, "on the far plane: {:#x} of {far:#x}", hit.z);
    }
}

/// The near plane likewise: the vertex at z = -3 is cut where the edges meet
/// z = -1, two thirds of the way along each, and the cut vertices sit at the
/// near end of the depth range.
#[test]
fn a_triangle_straddling_the_near_plane_is_cut_at_it() {
    let mut r = Rig::new();
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.v3(-0.5, -0.5, -3.0);
    r.v3(0.5, -0.5, 0.0);
    r.v3(0.0, 0.5, 0.0);
    r.end();
    assert_eq!(r.tris.len(), 2);
    let verts: Vec<_> = r.tris.iter().flatten().collect();
    let cut = |x: f32, y: f32| (grid(50.0 * (x + 1.0)), grid(50.0 * (y + 1.0)));
    for want in [cut(-0.5 + 1.0 * 2.0 / 3.0, -0.5), cut(-0.5 + 0.5 * 2.0 / 3.0, -0.5 + 1.0 * 2.0 / 3.0)] {
        let hit = verts.iter().find(|v| (v.x - want.0).abs() <= 2 && (v.y - want.1).abs() <= 2);
        let hit = hit.unwrap_or_else(|| panic!("no vertex near {want:?} in {verts:?}"));
        assert!(hit.z <= 0x2000, "on the near plane: {:#x}", hit.z);
    }
}

/// Strips, fans, quads and polygons come out as triangles.
#[test]
fn strips_fans_quads_and_polygons_are_assembled() {
    let sq = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)];
    let mut r = Rig::new();
    r.begin(BEGIN_PRIMITIVE_TRIANGLE_STRIP);
    for (x, y) in [(-0.5, -0.5), (-0.5, 0.0), (0.0, -0.5), (0.0, 0.0), (0.5, -0.5)] {
        r.v(x, y);
    }
    r.end();
    assert_eq!(r.tris.len(), 3, "a strip of five vertices is three triangles");
    r.tris.clear();
    r.begin(BEGIN_PRIMITIVE_TRIANGLE_FAN);
    for (x, y) in [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)] {
        r.v(x, y);
    }
    r.end();
    assert_eq!(r.tris.len(), 2, "a fan of four is two triangles");
    r.tris.clear();
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_QUADS);
    sq.iter().for_each(|&(x, y)| r.v(x, y));
    r.end();
    assert_eq!(r.tris.len(), 2, "a quad is a fan of two triangles");
    assert!(r.tris.iter().all(|t| t[0] == r.tris[0][0]), "from its first vertex");
    r.tris.clear();
    r.begin(BEGIN_PRIMITIVE_POLYGON);
    for (x, y) in [(0.0, 0.0), (0.5, 0.0), (0.7, 0.3), (0.5, 0.6), (0.0, 0.5)] {
        r.v(x, y);
    }
    r.end();
    assert_eq!(r.tris.len(), 3, "a polygon of five is one fan of three");
    r.tris.clear();
    r.begin(BEGIN_PRIMITIVE_QUAD_STRIP);
    for (x, y) in [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.0), (0.5, 0.0), (-0.5, 0.5), (0.5, 0.5)] {
        r.v(x, y);
    }
    r.end();
    assert_eq!(r.tris.len(), 4, "a quad strip of six is two quads");
}

/// A line is the rectangle of half-width w / 2 about the segment, two triangles whose
/// vertices carry the distance along it; a strip and a loop are their segments; a
/// segment is clipped to the volume; a segment of no length draws nothing.
#[test]
fn lines_are_rectangles_about_their_segments() {
    let mut r = Rig::new();
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_LINES);
    r.v(-0.5, 0.0);
    r.v(0.5, 0.0);
    r.end();
    assert_eq!((r.path.counts.lines, r.lines.len()), (1, 2));
    // Width 1 about a horizontal segment from x = 25 to 75 at y = 50: 49.5..50.5 each side.
    let ys: Vec<_> = r.lines.iter().flatten().map(|v| v.y).collect();
    assert_eq!((ys.iter().min().copied(), ys.iter().max().copied()), (Some(grid(49.5)), Some(grid(50.5))));
    let ds: Vec<_> = r.lines.iter().flatten().map(|v| v.d).collect();
    assert_eq!((ds.iter().cloned().fold(f32::MAX, f32::min), ds.iter().cloned().fold(0.0, f32::max)), (0.0, 50.0), "d runs along the segment in pixels");
    r.cmd(SET_LINE_WIDTH, &[4.0f32.to_bits(), 0]);
    r.lines.clear();
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_LINES);
    r.v(-0.5, 0.0);
    r.v(0.5, 0.0);
    r.end();
    let ys: Vec<_> = r.lines.iter().flatten().map(|v| v.y).collect();
    assert_eq!((ys.iter().min().copied(), ys.iter().max().copied()), (Some(grid(48.0)), Some(grid(52.0))), "width 4");
    r.lines.clear();
    r.begin(BEGIN_PRIMITIVE_LINE_LOOP);
    for (x, y) in [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)] {
        r.v(x, y);
    }
    r.end();
    assert_eq!(r.lines.len(), 6, "a loop of three is three segments");
    r.lines.clear();
    r.begin(BEGIN_PRIMITIVE_LINE_STRIP);
    for (x, y) in [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)] {
        r.v(x, y);
    }
    r.end();
    assert_eq!(r.lines.len(), 4, "a strip of three is two");
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_LINES);
    r.v(0.0, 0.0);
    r.v(0.0, 0.0);
    r.end();
    assert_eq!(r.path.counts.empty_lines, 1);
    let before = r.lines.len();
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_LINES);
    r.v3(-3.0, 0.0, 5.0);
    r.v3(3.0, 0.0, 5.0);
    r.end();
    assert_eq!((r.lines.len(), r.path.counts.clipped_away), (before, 1), "beyond the far plane: clipped away");
}

/// A point of a size to one pixel is a unit square about the vertex holding one
/// sample; a larger one is a sixteen-triangle disc.
#[test]
fn points_are_a_square_or_a_disc() {
    let mut r = Rig::new();
    r.begin(BEGIN_PRIMITIVE_POINTS);
    r.v(0.0, 0.0);
    r.end();
    assert_eq!((r.path.counts.points, r.tris.len()), (1, 2));
    let xs: Vec<_> = r.tris.iter().flatten().map(|v| v.x).collect();
    assert_eq!((xs.iter().min().copied(), xs.iter().max().copied()), (Some(grid(49.5)), Some(grid(50.5))));
    r.cmd(SET_POINT_SIZE, &[POINT_PIXEL_MAX.to_bits(), 0]);
    r.tris.clear();
    r.begin(BEGIN_PRIMITIVE_POINTS);
    r.v(0.0, 0.0);
    r.end();
    assert_eq!(r.tris.len(), 2, "3.5 is still one pixel");
    r.cmd(SET_POINT_SIZE, &[8.0f32.to_bits(), 0]);
    r.tris.clear();
    r.begin(BEGIN_PRIMITIVE_POINTS);
    r.v(0.0, 0.0);
    r.end();
    assert_eq!(r.tris.len(), 16, "a disc of sixteen triangles");
    let xs: Vec<_> = r.tris.iter().flatten().map(|v| v.x).collect();
    assert_eq!((xs.iter().min().copied(), xs.iter().max().copied()), (Some(grid(46.0)), Some(grid(54.0))), "size 8: radius 4");
    r.begin(BEGIN_PRIMITIVE_POINTS);
    r.v(5.0, 0.0);
    r.end();
    assert_eq!(r.path.counts.points, 4, "a point outside the window is still handed over: set-up cuts it");
}

/// Culling and two-sided orientation follow the front face.
#[test]
fn faces_are_culled_by_their_orientation() {
    let mut r = Rig::new();
    r.cmd(ENABLE_CAPABILITY, &[gl::CULL_FACE]);
    r.cmd(SET_FACE_CULLING, &[gl::BACK]);
    r.triangle([(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)]);
    assert_eq!(r.tris.len(), 1, "counter-clockwise is front, back is culled");
    r.triangle([(-0.5, -0.5), (0.0, 0.5), (0.5, -0.5)]);
    assert_eq!((r.tris.len(), r.path.counts.culled), (1, 1));
    r.cmd(SET_FRONT_FACE_WINDING, &[gl::CW]);
    r.triangle([(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)]);
    assert_eq!(r.path.counts.culled, 2, "clockwise front flips it");
    r.cmd(SET_FACE_CULLING, &[gl::FRONT_AND_BACK]);
    r.triangle([(-0.5, -0.5), (0.0, 0.5), (0.5, -0.5)]);
    assert_eq!(r.path.counts.culled, 3, "front and back culls both");
}

/// Flat shading takes the provoking vertex's colour; smooth keeps each.
#[test]
fn flat_shading_takes_the_colour_of_the_vertex_gl_names() {
    let mut r = Rig::new();
    r.cmd(SET_FLAT_OR_SMOOTH_SHADING, &[gl::FLAT]);
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.colour(1.0, 0.0, 0.0);
    r.v(-0.5, -0.5);
    r.colour(0.0, 1.0, 0.0);
    r.v(0.5, -0.5);
    r.colour(0.0, 0.0, 1.0);
    r.v(0.0, 0.5);
    r.end();
    assert!(r.tris[0].iter().all(|v| v.c == [0, 0, 8191, 8191]), "the last of a triangle");
    r.cmd(SET_FLAT_OR_SMOOTH_SHADING, &[0x1d01]);
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.colour(1.0, 0.0, 0.0);
    r.v(-0.5, -0.5);
    r.colour(0.0, 1.0, 0.0);
    r.v(0.5, -0.5);
    r.colour(0.0, 0.0, 1.0);
    r.v(0.0, 0.5);
    r.end();
    assert_eq!(r.tris[1].map(|v| v.c), [[8191, 0, 0, 8191], [0, 8191, 0, 8191], [0, 0, 8191, 8191]]);
}

/// A light at the eye, facing a surface head on, gives the diffuse colour; the
/// scene ambient and a light off give the ambient term.
#[test]
fn a_light_lights_the_vertex() {
    let mut r = Rig::new();
    r.cmd(ENABLE_CAPABILITY, &[gl::LIGHTING]);
    r.cmd(ENABLE_CAPABILITY, &[gl::LIGHT0]);
    // Light 0 directional, along +z, white diffuse (the initial value).
    let mut a = vec![gl::LIGHT0, gl::POSITION];
    a.extend(bits(&[0.0, 0.0, 1.0, 0.0]));
    r.cmd(SET_LIGHT_PARAMETER_FLOAT_FORM, &a);
    // Material diffuse red.
    let mut m = vec![gl::FRONT_AND_BACK, gl::DIFFUSE];
    m.extend(bits(&[1.0, 0.0, 0.0, 1.0]));
    r.cmd(SET_MATERIAL_PARAMETER_FLOAT_FORM, &m);
    r.cmd(SET_CURRENT_NORMAL_3_COMPONENTS_SINGLE_FLOAT, &bits(&[0.0, 0.0, 1.0]));
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.v(-0.5, -0.5);
    r.v(0.5, -0.5);
    r.v(0.0, 0.5);
    r.end();
    let c = r.tris[0][0].c;
    // Diffuse 1.0 plus ambient 0.2 * 0.2 on the red channel, clamped; blue gets the ambient only.
    assert_eq!(c[0], 8191);
    assert_eq!(c[2], colour13(0.2 * 0.2));
    r.cmd(DISABLE_CAPABILITY, &[gl::LIGHT0]);
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.v(-0.5, -0.5);
    r.v(0.5, -0.5);
    r.v(0.0, 0.5);
    r.end();
    assert_eq!(r.tris[1][0].c[0], colour13(0.2 * 0.2), "no light: ambient only");
}

/// Texture words are `s/w, t/w, r/w, q/w` of a textured vertex and zero
/// otherwise; the form of each coordinate token is counted.
#[test]
fn textured_vertices_carry_the_perspective_words() {
    let mut r = Rig::new();
    // A perspective projection: w = -z_eye.
    let mut p = math::IDENT;
    p[11] = -1.0;
    p[15] = 0.0;
    r.cmd(SELECT_CURRENT_MATRIX_STACK, &[gl::PROJECTION]);
    r.cmd(LOAD_MATRIX_16_FLOATS, &bits(&p));
    r.cmd(SELECT_CURRENT_MATRIX_STACK, &[gl::MODELVIEW]);
    r.cmd(ENABLE_CAPABILITY, &[gl::TEXTURE_2D]);
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    for (s, t, x) in [(0.0f32, 0.0f32, -0.5f32), (1.0, 0.0, 0.5), (0.5, 1.0, 0.0)] {
        r.cmd(SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT, &bits(&[s, t]));
        r.v3(x, 0.0 + if x == 0.0 { 0.5 } else { -0.5 }, -2.0);
    }
    r.end();
    let d = &r.tris[0];
    // w is 2 and q is 1, so s/w = s / 2 and the fourth word is q/w = 1/w.
    assert_eq!(d[1].t[0], [0.5, 0.0, 0.0, 0.5]);
    assert_eq!(d[2].t[0], [0.25, 0.5, 0.0, 0.5]);
    assert_eq!(r.path.counts.texcoord, [0, 3, 0, 0, 0]);
    // With q = 2 the fourth word is q/w, not 1/w.
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    for x in [-0.5f32, 0.5, 0.0] {
        r.cmd(SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT, &bits(&[1.0, 1.0, 0.0, 2.0]));
        r.v3(x, if x == 0.0 { 0.5 } else { -0.5 }, -2.0);
    }
    r.end();
    assert_eq!(r.tris[1][0].t[0], [0.5, 0.5, 0.0, 1.0], "s/w, t/w, r/w and q/w with w 2 and q 2: the fourth is 1, not 1/w = 0.5");
    r.tris.remove(1);
    // Texturing off: the words are zero.
    r.cmd(DISABLE_CAPABILITY, &[gl::TEXTURE_2D]);
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.v3(-0.5, -0.5, -2.0);
    r.v3(0.5, -0.5, -2.0);
    r.v3(0.0, 0.5, -2.0);
    r.end();
    assert_eq!(r.tris[1][0].t[0], [0.0; 4]);
}

/// The fog setter takes a parameter name and four values, in float or integer form; the enable is the
/// capability; the draw state carries the curve once fog is on, and a mode outside the three changes
/// nothing and is counted.
#[test]
fn fog_parameters_reach_the_draw_state() {
    let mut r = Rig::new();
    assert!(r.gl.draw_state(r.path.screen()).fog.is_none(), "off at start-up");
    let fog = |r: &mut Rig, name: u32, v: [f32; 4]| r.cmd(SET_FOG_PARAMETER_FLOAT_FORM, &[name, v[0].to_bits(), v[1].to_bits(), v[2].to_bits(), v[3].to_bits()]);
    fog(&mut r, gl::FOG_MODE, [gl::LINEAR as f32, 0.0, 0.0, 0.0]);
    fog(&mut r, gl::FOG_START, [2.0, 0.0, 0.0, 0.0]);
    fog(&mut r, gl::FOG_END, [6.0, 0.0, 0.0, 0.0]);
    fog(&mut r, gl::FOG_COLOR, [1.0, 0.5, 2.0, 1.0]);
    assert!(r.gl.draw_state(r.path.screen()).fog.is_none(), "set but not enabled");
    r.cmd(ENABLE_CAPABILITY, &[gl::FOG]);
    let f = r.gl.draw_state(r.path.screen()).fog.expect("on");
    assert_eq!((f.curve, f.a, f.b), (Curve::Linear, 6.0, 0.25));
    assert_eq!(f.colour, [8191.0, 4095.5, 8191.0], "colour clamped to 0..1, in 13-bit units");
    // The exponential curves take the density.
    fog(&mut r, gl::FOG_MODE, [gl::EXP2 as f32, 0.0, 0.0, 0.0]);
    fog(&mut r, gl::FOG_DENSITY, [0.3, 0.0, 0.0, 0.0]);
    let f = r.gl.draw_state(r.path.screen()).fog.expect("on");
    assert_eq!((f.curve, f.a), (Curve::Exp2, 0.3));
    // The integer form: the mode is its number, a colour component is the integer times 2^-31 clamped to 0..1.
    r.cmd(SET_FOG_PARAMETER_INTEGER_FORM, &[gl::FOG_MODE, gl::EXP, 0, 0, 0]);
    r.cmd(SET_FOG_PARAMETER_INTEGER_FORM, &[gl::FOG_COLOR, 0x4000_0000, 0x8000_0000, 0x7fff_ffff, 0]);
    let s = &r.gl.raster.fog;
    assert_eq!(s.mode, gl::EXP);
    assert_eq!(s.colour, [0.5, 0.0, 1.0], "negative integers give 0; 2^31 - 1 is within a single's step of 1");
    // An unknown mode or name is an invalid enumerant and changes nothing; the index, the offset and the
    // function-table mode are accepted and not modelled; start, end and density are not range-checked.
    r.cmd(SET_FOG_PARAMETER_INTEGER_FORM, &[gl::FOG_MODE, 0x1234, 0, 0, 0]);
    r.cmd(SET_FOG_PARAMETER_FLOAT_FORM, &[0xb67, 0, 0, 0, 0]);
    r.cmd(SET_FOG_PARAMETER_FLOAT_FORM, &[gl::FOG_MODE, (gl::LINEAR as f32 + 1.0).to_bits(), 0, 0, 0]);
    assert_eq!((r.gl.raster.fog.mode, r.path.counts.errors, r.gl.error), (gl::EXP, 3, gl::INVALID_ENUM));
    r.cmd(SET_FOG_PARAMETER_FLOAT_FORM, &[gl::FOG_INDEX, 0, 0, 0, 0]);
    r.cmd(SET_FOG_PARAMETER_FLOAT_FORM, &[gl::FOG_OFFSET, 0, 0, 0, 0]);
    r.cmd(SET_FOG_PARAMETER_FLOAT_FORM, &[gl::FOG_MODE, (gl::FOG_FUNC as f32).to_bits(), 0, 0, 0]);
    assert_eq!((r.gl.raster.fog.mode, r.path.counts.unmodelled[&SET_FOG_PARAMETER_FLOAT_FORM]), (gl::EXP, 3));
    r.cmd(SET_FOG_PARAMETER_FLOAT_FORM, &[gl::FOG_START, (-5.0f32).to_bits(), 0, 0, 0]);
    r.cmd(SET_FOG_PARAMETER_FLOAT_FORM, &[gl::FOG_DENSITY, (-1.0f32).to_bits(), 0, 0, 0]);
    assert_eq!((r.gl.raster.fog.start, r.gl.raster.fog.density, r.path.counts.errors), (-5.0, -1.0, 3));
    r.cmd(DISABLE_CAPABILITY, &[gl::FOG]);
    assert!(r.gl.draw_state(r.path.screen()).fog.is_none());
}

/// With fog on a vertex carries its eye-space distance over w, and 1 / w; with fog off it carries
/// nothing.
#[test]
fn fogged_vertices_carry_the_coordinate_words() {
    let mut r = Rig::new();
    let mut p = math::IDENT;
    p[11] = -1.0;
    p[15] = 0.0;
    r.cmd(SELECT_CURRENT_MATRIX_STACK, &[gl::PROJECTION]);
    r.cmd(LOAD_MATRIX_16_FLOATS, &bits(&p));
    r.cmd(SELECT_CURRENT_MATRIX_STACK, &[gl::MODELVIEW]);
    let tri = |r: &mut Rig, z: f32| {
        r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
        r.v3(-0.5, -0.5, z);
        r.v3(0.5, -0.5, z);
        r.v3(0.0, 0.5, z);
        r.end();
    };
    tri(&mut r, -4.0);
    assert_eq!(r.tris[0][0].fog, [0.0; 2]);
    r.cmd(ENABLE_CAPABILITY, &[gl::FOG]);
    tri(&mut r, -4.0);
    // Eye z is -4 and w is 4: the coordinate is 4, over w is 1, and 1 / w is 0.25.
    assert_eq!(r.tris[1][0].fog, [1.0, 0.25]);
    // A line and a point carry them too.
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_LINES);
    r.v3(-0.5, 0.0, -2.0);
    r.v3(0.5, 0.0, -2.0);
    r.end();
    assert_eq!(r.lines[0][0].fog, [1.0, 0.5]);
}

/// The polygon mode: front and back are set apart, an unmatched face is ignored with no error and a
/// mode is kept as sent (the pipe checks neither), and the query answers the pair.
#[test]
fn the_polygon_mode_setter_takes_a_face_and_a_mode() {
    let mut r = Rig::new();
    assert_eq!(r.gl.raster.polygon_mode, [gl::FILL; 2]);
    r.cmd(SET_POLYGON_FILL_MODE, &[gl::BACK, gl::LINE]);
    assert_eq!(r.gl.raster.polygon_mode, [gl::FILL, gl::LINE]);
    r.cmd(SET_POLYGON_FILL_MODE, &[gl::FRONT, gl::POINT]);
    assert_eq!(r.gl.raster.polygon_mode, [gl::POINT, gl::LINE]);
    r.cmd(SET_POLYGON_FILL_MODE, &[gl::FRONT_AND_BACK, gl::FILL]);
    assert_eq!(r.gl.raster.polygon_mode, [gl::FILL; 2]);
    r.cmd(SET_POLYGON_FILL_MODE, &[0x1234, gl::LINE]);
    assert_eq!(r.gl.raster.polygon_mode, [gl::FILL; 2], "an unmatched face is ignored");
    r.cmd(SET_POLYGON_FILL_MODE, &[gl::FRONT, 0x1234]);
    assert_eq!(r.gl.raster.polygon_mode, [0x1234, gl::FILL], "the mode is kept as sent");
    assert!(r.path.counts.unmodelled.get(&SET_POLYGON_FILL_MODE).is_none());
    assert_eq!((r.gl.error, r.path.counts.errors), (0, 0), "no error for either");
    // Below the line mode reads as point and above it as fill, whatever the value.
    for (mode, points) in [(0x1234, 3), (0x1b00, 3), (0x2000, 0)] {
        r.cmd(SET_POLYGON_FILL_MODE, &[gl::FRONT_AND_BACK, mode]);
        let before = (r.path.counts.points, r.path.counts.lines);
        r.triangle([(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)]);
        let after = (r.path.counts.points, r.path.counts.lines);
        assert_eq!((after.0 - before.0, after.1 - before.1), (points, 0), "mode {mode:#x}");
    }
}

/// A face in the line mode draws its boundary edges as lines and no fill; the other face still fills.
#[test]
fn a_polygon_in_the_line_mode_is_drawn_as_its_edges() {
    let mut r = Rig::new();
    r.cmd(SET_POLYGON_FILL_MODE, &[gl::FRONT, gl::LINE]);
    r.triangle([(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)]);
    assert_eq!((r.tris.len(), r.lines.len()), (0, 6), "three edges, two triangles each");
    assert_eq!(r.path.counts.lines, 3);
    // The same triangle wound the other way is a back face, still filled.
    r.lines.clear();
    r.triangle([(-0.5, -0.5), (0.0, 0.5), (0.5, -0.5)]);
    assert_eq!((r.tris.len(), r.lines.len()), (1, 0));
    // Width follows the line width, and each edge starts and ends at the polygon's vertices.
    r.cmd(SET_POLYGON_FILL_MODE, &[gl::FRONT_AND_BACK, gl::LINE]);
    r.cmd(SET_LINE_WIDTH, &[2.0f32.to_bits(), 0]);
    r.lines.clear();
    r.triangle([(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)]);
    let first = &r.lines[0];
    assert_eq!((first[0].y, first[1].y), (grid(25.0 + 1.0), grid(25.0 - 1.0)), "a rectangle two pixels wide about the bottom edge");
}

/// A face in the point mode draws a point at each vertex that starts a boundary edge.
#[test]
fn a_polygon_in_the_point_mode_is_drawn_as_its_vertices() {
    let mut r = Rig::new();
    r.cmd(SET_POLYGON_FILL_MODE, &[gl::FRONT_AND_BACK, gl::POINT]);
    r.triangle([(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)]);
    assert_eq!((r.tris.len(), r.path.counts.points), (6, 3), "three one-pixel squares");
    let xs: Vec<_> = r.tris.iter().flatten().map(|v| v.x).collect();
    assert_eq!((xs.iter().min().copied(), xs.iter().max().copied()), (Some(grid(24.5)), Some(grid(75.5))));
}

/// The edge flag of a vertex decides whether the edge from it is drawn; a strip's, a fan's and a
/// quad strip's edges are all drawn.
#[test]
fn edge_flags_choose_which_edges_the_line_mode_draws() {
    let mut r = Rig::new();
    r.cmd(SET_POLYGON_FILL_MODE, &[gl::FRONT_AND_BACK, gl::LINE]);
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.v(-0.5, -0.5);
    r.cmd(SET_CURRENT_EDGE_FLAG, &[0]);
    r.v(0.5, -0.5);
    r.cmd(SET_CURRENT_EDGE_FLAG, &[1]);
    r.v(0.0, 0.5);
    r.end();
    assert_eq!(r.path.counts.lines, 2, "the edge that starts at the second vertex is not drawn");
    for (prim, verts, lines) in [(BEGIN_PRIMITIVE_TRIANGLE_STRIP, 4, 6), (BEGIN_PRIMITIVE_TRIANGLE_FAN, 4, 6), (BEGIN_PRIMITIVE_QUAD_STRIP, 4, 4)] {
        let before = r.path.counts.lines;
        r.cmd(SET_CURRENT_EDGE_FLAG, &[0]);
        r.begin(prim);
        for k in 0..verts {
            r.v(-0.5 + 0.25 * k as f32, if k % 2 == 0 { -0.5 } else { 0.5 });
        }
        r.end();
        assert_eq!(r.path.counts.lines - before, lines, "primitive {prim:#x}");
    }
}

/// Clipping keeps the flags of the edges it keeps and draws the edge it adds along the plane.
#[test]
fn clipping_leaves_the_outline_of_what_is_inside() {
    let mut r = Rig::new();
    r.cmd(SET_POLYGON_FILL_MODE, &[gl::FRONT_AND_BACK, gl::LINE]);
    // Past the right guard band (x > 8): a triangle with one vertex far outside is cut to a quad.
    r.triangle([(-0.5, -0.5), (20.0, -0.5), (-0.5, 0.5)]);
    assert_eq!(r.path.counts.lines, 4, "the clipped triangle is a four-sided polygon");
}

/// The smoothing enables reach the draw state; a multisampled window smooths lines only (see the board test).
#[test]
fn the_smoothing_enables_reach_the_draw_state() {
    let mut r = Rig::new();
    let smooth = |r: &mut Rig| r.gl.draw_state(r.path.screen()).smooth;
    assert_eq!(smooth(&mut r), Smooth::default());
    r.cmd(ENABLE_CAPABILITY, &[gl::POINT_SMOOTH]);
    r.cmd(ENABLE_CAPABILITY, &[gl::LINE_SMOOTH]);
    assert_eq!(smooth(&mut r), Smooth { points: true, lines: true, polygons: false });
    r.cmd(ENABLE_CAPABILITY, &[gl::POLYGON_SMOOTH]);
    r.cmd(DISABLE_CAPABILITY, &[gl::POINT_SMOOTH]);
    assert_eq!(smooth(&mut r), Smooth { points: false, lines: true, polygons: true });
    assert!(r.path.counts.caps.is_empty(), "the three are taken");
}

/// A smoothed point is the square about its disc, with the disc's centre and radius, and no triangles
/// of the unsmoothed forms; a size of nothing draws nothing.
#[test]
fn a_smoothed_point_is_a_square_about_a_disc() {
    let mut r = Rig::new();
    r.cmd(ENABLE_CAPABILITY, &[gl::POINT_SMOOTH]);
    r.cmd(SET_POINT_SIZE, &[3.0f32.to_bits(), 0]);
    r.begin(BEGIN_PRIMITIVE_POINTS);
    r.v(0.0, 0.0);
    r.end();
    assert_eq!((r.tris.len(), r.discs.len(), r.path.counts.points), (0, 1, 1));
    let d = r.discs[0];
    assert_eq!((d.x, d.y, d.r), (50.0, 50.0, 1.5));
    let xs: Vec<_> = d.quad.iter().map(|v| v.x).collect();
    assert_eq!((xs.iter().min().copied(), xs.iter().max().copied()), (Some(grid(48.0)), Some(grid(52.0))), "the radius and half a pixel about the centre");
    r.cmd(SET_POINT_SIZE, &[0.0f32.to_bits(), 0]);
    r.begin(BEGIN_PRIMITIVE_POINTS);
    r.v(0.0, 0.0);
    r.end();
    assert_eq!((r.tris.len(), r.discs.len()), (2, 1), "no size: the unsmoothed rule's one pixel square, as before");
}

/// A smoothed line is its rectangle whole, four vertices from the first end's left, for set-up to weigh;
/// an unsmoothed line is two triangles.
#[test]
fn a_smoothed_line_is_one_rectangle() {
    let mut r = Rig::new();
    let line = |r: &mut Rig| {
        r.begin(BEGIN_PRIMITIVE_INDEPENDENT_LINES);
        r.v(-0.5, 0.0);
        r.v(0.5, 0.0);
        r.end();
    };
    line(&mut r);
    assert_eq!((r.lines.len(), r.aa.len()), (2, 0));
    r.cmd(ENABLE_CAPABILITY, &[gl::LINE_SMOOTH]);
    line(&mut r);
    assert_eq!((r.lines.len(), r.aa.len()), (2, 1));
    let (poly, verts) = (r.aa[0], &r.aa_verts);
    assert_eq!((poly.n, poly.line), (4, true));
    let ys: Vec<_> = verts.iter().map(|v| v.y).collect();
    assert_eq!(ys, [grid(50.5), grid(49.5), grid(49.5), grid(50.5)]);
    assert_eq!(verts.iter().map(|v| v.d).collect::<Vec<_>>(), [0.0, 0.0, 50.0, 50.0], "the distance along the line, at each end");
}

/// A smoothed polygon goes whole, a fan's worth of vertices in the order given; a smoothed polygon in a
/// line mode is drawn as lines, which are smoothed by the polygon switch (inferred), not the line one.
#[test]
fn a_smoothed_polygon_goes_whole() {
    let mut r = Rig::new();
    r.cmd(ENABLE_CAPABILITY, &[gl::POLYGON_SMOOTH]);
    r.begin(BEGIN_PRIMITIVE_POLYGON);
    for (x, y) in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.6, 0.0)] {
        r.v(x, y);
    }
    r.end();
    assert_eq!((r.tris.len(), r.aa.len(), r.aa[0].n, r.aa[0].line), (0, 1, 5, false));
    r.cmd(SET_POLYGON_FILL_MODE, &[gl::FRONT_AND_BACK, gl::LINE]);
    r.begin(BEGIN_PRIMITIVE_POLYGON);
    for (x, y) in [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)] {
        r.v(x, y);
    }
    r.end();
    assert_eq!((r.aa.len(), r.lines.len(), r.path.counts.lines), (4, 0, 3), "the outline's lines follow the polygon switch, not the line one");
    r.cmd(DISABLE_CAPABILITY, &[gl::POLYGON_SMOOTH]);
    r.cmd(ENABLE_CAPABILITY, &[gl::LINE_SMOOTH]);
    r.begin(BEGIN_PRIMITIVE_POLYGON);
    for (x, y) in [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)] {
        r.v(x, y);
    }
    r.end();
    assert_eq!((r.aa.len(), r.lines.len()), (4, 6), "the line switch alone leaves them plain");
}

/// The draw state is built again only when a state command changes it, and
/// it holds the window, the clip rectangle, the viewport and the depth switches.
#[test]
fn the_draw_state_changes_only_with_state_commands() {
    let mut r = Rig::new();
    r.triangle([(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)]);
    r.triangle([(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)]);
    assert_eq!(r.states[0], r.states[1]);
    let s = r.states[0];
    let full = Rect { x0: 0, y0: 0, x1: 100, y1: 100 };
    assert_eq!((s.origin, s.clip, s.viewport, s.depth_test), ([0, 0], full, full, false));
    r.cmd(ENABLE_CAPABILITY, &[gl::DEPTH_TEST]);
    r.triangle([(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)]);
    assert_ne!(r.states[1], r.states[2]);
    assert!(r.states[2].depth_test);
}

/// A window away from the origin moves the origin and the clip rectangle, and
/// a scissor narrows the clip rectangle but not the viewport; a smaller
/// viewport inside a larger window is its own rectangle, in window-relative
/// coordinates.
#[test]
fn the_clip_and_viewport_rectangles_follow_the_window_scissor_and_viewport() {
    let mut r = Rig::new();
    r.path = GlPath::new((1000, 1000));
    r.gl.place([300, 200], Rect { x0: 300, y0: 200, x1: 400, y1: 300 }, 0, 0, false);
    r.cmd(SET_VIEWPORT, &bits(&[10.0, 20.0, 30.0, 40.0]));
    r.cmd(ENABLE_CAPABILITY, &[gl::SCISSOR_TEST]);
    r.cmd(SET_SCISSOR_BOX, &[5, 5, 50, 50]);
    r.triangle([(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)]);
    let s = r.states[0];
    assert_eq!(s.origin, [300, 200]);
    assert_eq!(s.clip, Rect { x0: 305, y0: 205, x1: 355, y1: 255 });
    assert_eq!(s.viewport, Rect { x0: 310, y0: 220, x1: 340, y1: 260 });
    // The clip rectangle is also cut to the screen.
    r.gl.place([950, 960], Rect { x0: 950, y0: 960, x1: 1050, y1: 1060 }, 0, 0, false);
    r.cmd(DISABLE_CAPABILITY, &[gl::SCISSOR_TEST]);
    r.triangle([(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)]);
    assert_eq!(r.states[1].clip, Rect { x0: 950, y0: 960, x1: 1000, y1: 1000 });
}

/// A clear carries the draw state's clip rectangle and the values to clear to.
#[test]
fn a_clear_is_a_request_over_the_clip_rectangle() {
    let mut r = Rig::new();
    r.cmd(SET_COLOUR_CLEAR_VALUE, &bits(&[0.0, 0.0, 1.0, 1.0]));
    r.cmd(CLEAR_SELECTED_BUFFERS, &[gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT]);
    assert_eq!(r.clears.len(), 1);
    assert_eq!(r.clears[0].rect, Rect { x0: 0, y0: 0, x1: 100, y1: 100 });
    assert_eq!(r.clears[0].colour, Some(0xff00_00ff));
    assert!(r.clears[0].depth.is_some());
}

/// Forms and tokens the path does not take are counted by selector, a vertex
/// outside a begin draws nothing, and a matrix command inside one is an error.
#[test]
fn what_is_not_taken_is_counted() {
    let mut r = Rig::new();
    r.cmd(SET_CURRENT_COLOUR_INDEX_DOUBLE, &[0; 2]);
    r.cmd(SET_LIGHT_PARAMETER_FLOAT_FORM, &[gl::LIGHT0, 0x9999, 0, 0, 0, 0]);
    assert_eq!(r.path.counts.unmodelled.get(&0x72), Some(&1));
    assert_eq!(r.path.counts.unmodelled.get(&0x7), Some(&1));
    r.v(0.0, 0.0);
    assert_eq!((r.path.counts.stray, r.tris.len()), (1, 0));
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.cmd(LOAD_IDENTITY_MATRIX, &[0]);
    assert_eq!((r.path.counts.errors, r.gl.error), (1, gl::INVALID_OPERATION));
}

/// The census: with it on, each vertex is told apart by where it fell.
#[test]
fn the_census_tells_vertices_apart_by_the_window() {
    let mut r = Rig::new();
    r.path.census = true;
    r.begin(BEGIN_PRIMITIVE_POINTS);
    r.v(0.0, 0.0);
    r.v(5.0, 0.0);
    r.v3(0.0, 0.0, 9.0);
    r.cmd(SUBMIT_VERTEX_4_COMPONENTS_SINGLE_FLOAT, &bits(&[0.0, 0.0, 0.0, -1.0]));
    r.end();
    let c = &r.path.counts;
    assert_eq!((c.inside, c.outside_window, c.outside_volume, c.behind), (1, 1, 1, 1));
}

/// A double is two words, the high one first.
fn dbl(v: &[f64]) -> Vec<u32> {
    v.iter().flat_map(|x| [(x.to_bits() >> 32) as u32, x.to_bits() as u32]).collect()
}

/// The double forms of the vertex, colour, normal and texture-coordinate tokens leave the
/// context where the float forms do.
#[test]
fn the_double_forms_equal_the_float_forms() {
    let mut a = Rig::new();
    let mut b = Rig::new();
    for r in [&mut a, &mut b] {
        r.cmd(ENABLE_CAPABILITY, &[gl::TEXTURE_2D]);
    }
    a.cmd(SET_CURRENT_COLOUR_3_COMPONENTS_DOUBLE, &dbl(&[0.25, 0.5, 0.75]));
    b.cmd(SET_CURRENT_COLOUR_3_COMPONENTS_SINGLE_FLOAT, &bits(&[0.25, 0.5, 0.75]));
    a.cmd(SET_CURRENT_NORMAL_3_COMPONENTS_DOUBLE, &dbl(&[0.0, 0.5, 1.0]));
    b.cmd(SET_CURRENT_NORMAL_3_COMPONENTS_SINGLE_FLOAT, &bits(&[0.0, 0.5, 1.0]));
    assert_eq!((a.gl.color, a.gl.normal), (b.gl.color, b.gl.normal));
    assert_eq!(a.gl.color, [0.25, 0.5, 0.75, 1.0]);
    a.cmd(SET_CURRENT_COLOUR_4_COMPONENTS_DOUBLE, &dbl(&[0.125, 0.25, 0.5, 0.75]));
    b.cmd(SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, &bits(&[0.125, 0.25, 0.5, 0.75]));
    assert_eq!(a.gl.color, [0.125, 0.25, 0.5, 0.75]);
    for (n, sel, fsel) in [
        (1, SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_DOUBLE, SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SINGLE_FLOAT),
        (2, SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_DOUBLE, SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT),
        (3, SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_DOUBLE, SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT),
        (4, SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_DOUBLE, SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT),
    ] {
        let v = [0.5f64, 0.25, 2.0, 4.0];
        a.cmd(sel, &dbl(&v[..n]));
        b.cmd(fsel, &bits(&v[..n].iter().map(|&x| x as f32).collect::<Vec<_>>()));
        assert_eq!(a.gl.texcoord, b.gl.texcoord, "{n} components");
    }
    assert_eq!(a.gl.texcoord, [0.5, 0.25, 2.0, 4.0]);
    // Vertices of every width: the same triangles, and a value a float cannot hold rounds as `as f32` does.
    for (r, double) in [(&mut a, true), (&mut b, false)] {
        r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
        let p: [&[f64]; 3] = [&[-1.0, -1.0], &[1.0, -1.0, 0.0], &[0.0, 1.0, 0.0, 1.0]];
        let (dsel, fsel) = [
            (SUBMIT_VERTEX_2_COMPONENTS_DOUBLE, SUBMIT_VERTEX_2_COMPONENTS_SINGLE_FLOAT),
            (SUBMIT_VERTEX_3_COMPONENTS_DOUBLE, SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT),
            (SUBMIT_VERTEX_4_COMPONENTS_DOUBLE, SUBMIT_VERTEX_4_COMPONENTS_SINGLE_FLOAT),
        ].into_iter().unzip::<_, _, Vec<u32>, Vec<u32>>();
        for (k, v) in p.iter().enumerate() {
            if double {
                r.cmd(dsel[k], &dbl(v));
            } else {
                r.cmd(fsel[k], &bits(&v.iter().map(|&x| x as f32).collect::<Vec<_>>()));
            }
        }
        r.end();
    }
    assert_eq!(a.tris.len(), 1);
    assert_eq!(a.tris, b.tris);
    assert_eq!(a.path.counts.unmodelled.len(), 0);
    // A count that does not match the form is not narrowed, and is counted under the float form.
    a.cmd(SUBMIT_VERTEX_3_COMPONENTS_DOUBLE, &dbl(&[1.0, 2.0]));
    assert_eq!(a.path.counts.unmodelled.get(&SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT), Some(&1));
    let c = 1.0 + 2f64.powi(-40);
    a.cmd(SET_CURRENT_COLOUR_3_COMPONENTS_DOUBLE, &dbl(&[c, 0.0, 0.0]));
    assert_eq!(a.gl.color[0], c as f32);
    // A colour double is narrowed and nothing more: it is not scaled or clamped at entry (the clamp to 0..1
    // comes later, in the colour-clamp pass).
    a.cmd(SET_CURRENT_COLOUR_4_COMPONENTS_DOUBLE, &dbl(&[2.0, -0.5, 0.25, 8.0]));
    assert_eq!(a.gl.color, [2.0, -0.5, 0.25, 8.0]);
}

fn plane(r: &mut Rig, n: u32, p: [f64; 4]) {
    let mut a = vec![gl::CLIP_PLANE0 + n];
    a.extend(dbl(&p));
    r.cmd(SET_USER_CLIP_PLANE, &a);
}

fn xs(r: &Rig) -> (i32, i32) {
    let all = r.tris.iter().flatten().map(|v| v.x);
    (all.clone().min().expect("a triangle"), all.max().expect("a triangle"))
}

/// An enabled user plane cuts a triangle in eye space, a disabled one does nothing, and
/// the cut lands on the plane.
#[test]
fn an_enabled_user_plane_cuts_a_triangle_and_a_disabled_one_does_not() {
    let mut r = Rig::new();
    plane(&mut r, 0, [1.0, 0.0, 0.0, 0.0]);
    r.triangle([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]);
    assert_eq!(xs(&r), (0, 100 * 256), "set but not enabled");
    r.tris.clear();
    r.cmd(ENABLE_CAPABILITY, &[gl::CLIP_PLANE0]);
    r.triangle([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]);
    assert_eq!(xs(&r), (50 * 256, 100 * 256), "the left half is gone");
    r.tris.clear();
    r.cmd(DISABLE_CAPABILITY, &[gl::CLIP_PLANE0]);
    r.triangle([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]);
    assert_eq!(xs(&r), (0, 100 * 256));
    // A triangle wholly behind an enabled plane is gone.
    r.tris.clear();
    r.cmd(ENABLE_CAPABILITY, &[gl::CLIP_PLANE0]);
    r.triangle([(-0.9, -0.5), (-0.1, -0.5), (-0.5, 0.5)]);
    assert!(r.tris.is_empty());
    assert_eq!(r.path.counts.clipped_away, 1);
}

/// A plane is kept in eye space: the coefficients times the inverse of the model-view matrix at
/// the call, so later model-view changes do not move it. A matrix with no inverse leaves it.
#[test]
fn a_plane_is_kept_in_eye_space_at_the_model_view_of_the_call() {
    let mut r = Rig::new();
    r.cmd(TRANSLATE_FLOATS, &bits(&[-0.5, 0.0, 0.0]));
    plane(&mut r, 2, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(r.gl.user_plane[2], [1.0, 0.0, 0.0, 0.5]);
    // Singular: unchanged.
    r.cmd(SCALE_FLOATS, &bits(&[0.0, 1.0, 1.0]));
    plane(&mut r, 2, [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(r.gl.user_plane[2], [1.0, 0.0, 0.0, 0.5]);
    // The model-view changes afterwards; the plane stays where it was put (eye x >= -0.5).
    r.cmd(LOAD_IDENTITY_MATRIX, &[0]);
    r.cmd(ENABLE_CAPABILITY, &[gl::CLIP_PLANE0 + 2]);
    r.triangle([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]);
    assert_eq!(xs(&r), (25 * 256, 100 * 256));
}

/// A name out of range is an error, and the planes answer their query in eye space.
#[test]
fn a_plane_name_out_of_range_is_an_error_and_the_query_returns_the_plane() {
    let mut r = Rig::new();
    let mut a = vec![gl::CLIP_PLANE0 + 6];
    a.extend(dbl(&[1.0, 0.0, 0.0, 0.0]));
    r.cmd(SET_USER_CLIP_PLANE, &a);
    assert_eq!((r.path.counts.errors, r.gl.error), (1, gl::INVALID_ENUM));
    r.gl.error = 0;
    plane(&mut r, 5, [0.0, 1.0, 0.0, -0.5]);
    let mut c = query::Counts::default();
    assert_eq!(query::answer(&mut r.gl, &mut c, QUERY_CLIP_PLANE, &[gl::CLIP_PLANE0 + 5]), bits(&[0.0, 1.0, 0.0, -0.5]));
    assert_eq!(query::answer(&mut r.gl, &mut c, QUERY_CLIP_PLANE, &[gl::CLIP_PLANE0 + 9]), Vec::<u32>::new());
    assert_eq!(r.gl.error, gl::INVALID_ENUM);
}

/// Lines and points are cut by a user plane too, and a user plane and the far plane cut together.
#[test]
fn lines_points_and_the_volume_planes_work_with_a_user_plane() {
    let mut r = Rig::new();
    r.cmd(ENABLE_CAPABILITY, &[gl::CLIP_PLANE0]);
    plane(&mut r, 0, [1.0, 0.0, 0.0, 0.0]);
    r.begin(BEGIN_PRIMITIVE_POINTS);
    r.v(-0.5, 0.0);
    r.v(0.5, 0.0);
    r.end();
    assert_eq!((r.path.counts.points, r.path.counts.clipped_away), (1, 1));
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_LINES);
    r.v(-0.5, 0.0);
    r.v(0.5, 0.0);
    r.end();
    assert_eq!(r.path.counts.lines, 1);
    let x0 = r.lines.iter().flatten().map(|v| v.x).min().expect("a line");
    assert!(x0 >= 50 * 256 - 256, "{x0}");
    // A triangle across both the user plane and the far plane.
    r.tris.clear();
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.v3(-1.0, -1.0, 0.0);
    r.v3(1.0, -1.0, 0.0);
    r.v3(0.0, 1.0, 3.0);
    r.end();
    assert!(!r.tris.is_empty());
    let (lo, _) = xs(&r);
    assert!(lo >= 50 * 256);
    assert!(r.tris.iter().flatten().all(|v| v.z <= 4_294_967_040));
}

/// The clip plane command is framed at nine words: a name and four doubles, high word first.
#[test]
fn the_clip_plane_command_frames_nine_words() {
    use crate::dev::kona::tables::GL;
    assert_eq!(GL.iter().find(|s| s.selector == SET_USER_CLIP_PLANE).map(|s| s.argc), Some(9));
}

/// The enable tokens take the six clip planes in 0x3000 to 0x3fff; the rest of the range is an
/// invalid enumerant that changes nothing.
#[test]
fn the_enable_tokens_take_only_six_clip_planes() {
    let mut r = Rig::new();
    for n in 0..6 {
        r.cmd(ENABLE_CAPABILITY, &[gl::CLIP_PLANE0 + n]);
    }
    assert_eq!((r.gl.user_on, r.path.counts.errors), (0b11_1111, 0));
    for name in [0x3006, 0x3007, 0x3100, 0x3fff] {
        r.cmd(ENABLE_CAPABILITY, &[name]);
        r.cmd(DISABLE_CAPABILITY, &[name]);
    }
    assert_eq!((r.gl.user_on, r.path.counts.errors, r.gl.error), (0b11_1111, 8, gl::INVALID_ENUM));
}

fn tex_gen(r: &mut Rig, coord: u32, name: u32, v: [u32; 7]) {
    let mut a = vec![gl::TEXTURE_COORD_S + coord, name];
    a.extend(v);
    r.cmd(TEXTURE_COORDINATE_GENERATION_FLOAT_FORM, &a);
}

fn tex_mode(r: &mut Rig, coord: u32, mode: u32) {
    tex_gen(r, coord, gl::TEXTURE_GEN_MODE, [mode, 0, 0, 0, 0, 0, 0]);
}

fn tex_plane(r: &mut Rig, coord: u32, name: u32, p: [f32; 4]) {
    let w = bits(&p);
    tex_gen(r, coord, name, [w[0], w[1], w[2], w[3], 0, 0, 0]);
}

/// The texture words (s, t, r, q, over w) of the first triangle's vertices.
fn tex_of(r: &Rig) -> Vec<[f32; 4]> {
    r.tris[0].iter().map(|v| v.t[0]).collect()
}

fn textured() -> Rig {
    let mut r = Rig::new();
    r.cmd(ENABLE_CAPABILITY, &[gl::TEXTURE_2D]);
    r
}

/// Object linear: the plane's product with the vertex as given, for the coordinates enabled; the
/// others keep the current coordinate.
#[test]
fn object_linear_generation_replaces_only_the_enabled_coordinates() {
    let mut r = textured();
    r.cmd(SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT, &bits(&[0.0, 0.75, 0.0, 1.0]));
    tex_mode(&mut r, 0, 0x2401);
    tex_plane(&mut r, 0, gl::OBJECT_PLANE, [0.5, 0.0, 0.0, 0.5]);
    // Set up but not enabled: no effect.
    r.triangle([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]);
    assert_eq!(tex_of(&r), vec![[0.0, 0.75, 0.0, 1.0]; 3]);
    r.tris.clear();
    r.cmd(ENABLE_CAPABILITY, &[gl::TEXTURE_GEN_S]);
    r.triangle([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]);
    assert_eq!(tex_of(&r), [[0.0, 0.75, 0.0, 1.0], [1.0, 0.75, 0.0, 1.0], [0.5, 0.75, 0.0, 1.0]]);
    // The texture matrix runs after generation.
    r.tris.clear();
    r.cmd(SELECT_CURRENT_MATRIX_STACK, &[gl::TEXTURE]);
    r.cmd(SCALE_FLOATS, &bits(&[2.0, 1.0, 1.0]));
    r.triangle([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]);
    assert_eq!(tex_of(&r)[1], [2.0, 0.75, 0.0, 1.0]);
    // Disabled again: back to the current coordinate.
    r.tris.clear();
    r.cmd(DISABLE_CAPABILITY, &[gl::TEXTURE_GEN_S]);
    r.triangle([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]);
    assert_eq!(tex_of(&r)[1][0], 0.0);
}

/// Generation runs under the 3D enable as it does under the 2D one: a volume with generated s, t and r
/// is the usual way to draw one.
#[test]
fn generation_runs_under_the_3d_enable_too() {
    let mut r = Rig::new();
    r.cmd(ENABLE_CAPABILITY, &[gl::TEXTURE_3D]);
    r.cmd(SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT, &bits(&[0.0, 0.75, 0.0, 1.0]));
    tex_mode(&mut r, 0, 0x2401);
    tex_plane(&mut r, 0, gl::OBJECT_PLANE, [0.5, 0.0, 0.0, 0.5]);
    r.cmd(ENABLE_CAPABILITY, &[gl::TEXTURE_GEN_S]);
    r.triangle([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]);
    assert_eq!(tex_of(&r), [[0.0, 0.75, 0.0, 1.0], [1.0, 0.75, 0.0, 1.0], [0.5, 0.75, 0.0, 1.0]]);
}

/// Eye linear: the eye plane is kept in eye space as of the call, and the eye position moves
/// with the model-view matrix. Eye linear is the start-up mode, s from x and t from y.
#[test]
fn eye_linear_generation_uses_the_eye_position_and_an_eye_plane_set_in_eye_space() {
    let mut r = textured();
    r.cmd(ENABLE_CAPABILITY, &[gl::TEXTURE_GEN_S]);
    r.cmd(ENABLE_CAPABILITY, &[gl::TEXTURE_GEN_S + 1]);
    r.cmd(TRANSLATE_FLOATS, &bits(&[0.25, 0.0, 0.0]));
    r.triangle([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]);
    // Start-up planes, set at the identity: s = eye x = x + 0.25, t = eye y.
    assert_eq!(tex_of(&r), [[-0.75, -1.0, 0.0, 1.0], [1.25, -1.0, 0.0, 1.0], [0.25, 1.0, 0.0, 1.0]]);
    // A plane given under the translation is kept in eye space: x + 0 in object space is eye x - 0.25.
    r.tris.clear();
    tex_plane(&mut r, 0, gl::EYE_PLANE, [1.0, 0.0, 0.0, 0.0]);
    r.triangle([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]);
    assert_eq!(tex_of(&r)[0][0], -1.0);
}

/// Sphere map reflects the view direction about the eye-space normal.
#[test]
fn sphere_map_generation_reflects_the_view_about_the_normal() {
    let mut r = textured();
    for c in 0..2 {
        tex_mode(&mut r, c, 0x2402);
        r.cmd(ENABLE_CAPABILITY, &[gl::TEXTURE_GEN_S + c]);
    }
    r.cmd(SET_CURRENT_NORMAL_3_COMPONENTS_SINGLE_FLOAT, &bits(&[0.0, 0.0, 1.0]));
    r.begin(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES);
    r.v3(0.0, 0.0, -0.5);
    r.v3(0.5, 0.0, -0.5);
    r.v3(0.0, 0.5, -0.5);
    r.end();
    let t = tex_of(&r);
    assert_eq!(t[0][..2], [0.5, 0.5], "straight at the viewer");
    // u = (1, 0, -1) / sqrt 2 reflects to (1, 0, 1) / sqrt 2: s = rx / (2 sqrt(rx^2 + (rz + 1)^2)) + 1/2.
    let r2 = 0.5f32.sqrt();
    let s = r2 / (2.0 * (r2 * r2 + (r2 + 1.0) * (r2 + 1.0)).sqrt()) + 0.5;
    assert!((t[1][0] - s).abs() < 1e-5 && (t[1][1] - 0.5).abs() < 1e-5, "{:?}", t[1]);
    assert!((t[2][1] - s).abs() < 1e-5 && (t[2][0] - 0.5).abs() < 1e-5, "{:?}", t[2]);
}

/// Generation shows only when texturing is on, and the parameters are checked.
#[test]
fn generation_needs_texturing_and_the_parameters_are_checked() {
    let mut r = Rig::new();
    tex_plane(&mut r, 0, gl::OBJECT_PLANE, [1.0, 0.0, 0.0, 0.0]);
    tex_mode(&mut r, 0, 0x2401);
    r.cmd(ENABLE_CAPABILITY, &[gl::TEXTURE_GEN_S]);
    r.triangle([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]);
    assert_eq!(tex_of(&r), vec![[0.0; 4]; 3], "texturing is off");
    assert_eq!(r.path.counts.errors, 0);
    // A coordinate or name the pipe does not accept is an invalid enumerant; changes nothing.
    tex_mode(&mut r, 4, 0x2401);
    assert_eq!((r.path.counts.errors, r.gl.error), (1, gl::INVALID_ENUM));
    tex_gen(&mut r, 0, 0x1234, [0; 7]);
    assert_eq!(r.path.counts.errors, 2);
    // The distance planes and modes are accepted and counted, and change nothing.
    tex_plane(&mut r, 0, 0x81f4, [1.0; 4]);
    tex_mode(&mut r, 0, 0x81f0);
    tex_mode(&mut r, 0, 0x81f3);
    assert_eq!(r.path.counts.unmodelled.get(&TEXTURE_COORDINATE_GENERATION_FLOAT_FORM), Some(&3));
    assert_eq!((r.path.counts.errors, r.gl.texgen.mode[0]), (2, 0x2401));
    // Names and modes outside the accepted ranges are invalid enumerants, whatever their neighbours.
    for name in [0x24ff, 0x2503, 0x81f3, 0x81f8] {
        tex_gen(&mut r, 0, name, [0; 7]);
    }
    for mode in [0x23ff, 0x2403, 0x81ef, 0x81f4] {
        tex_mode(&mut r, 0, mode);
    }
    assert_eq!((r.path.counts.errors, r.gl.texgen.mode[0]), (10, 0x2401));
    // Sphere map on r is not refused: the mode is kept and the coordinate is left as it was (inferred).
    tex_mode(&mut r, 2, 0x2402);
    assert_eq!((r.path.counts.errors, r.gl.texgen.mode[2]), (10, 0x2402));
    // The framing: the command is nine words.
    let argc = crate::dev::kona::tables::GL.iter().find(|s| s.selector == TEXTURE_COORDINATE_GENERATION_FLOAT_FORM).map(|s| s.argc);
    assert_eq!(argc, Some(9));
}

/// Start-up is OpenGL's: s = (1,0,0,0), t = (0,1,0,0), r and q zero, eye linear. An eye plane is
/// multiplied by the inverse model-view at the call; an object plane is stored as sent.
#[test]
fn texgen_starts_as_gl_does_and_planes_are_stored_as_the_pipe_stores_them() {
    let mut r = Rig::new();
    let g = r.gl.texgen;
    assert_eq!((g.eye[0], g.eye[1], g.eye[2], g.eye[3]), ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0; 4], [0.0; 4]));
    assert_eq!((g.object[0], g.object[1], g.object[2], g.object[3]), (g.eye[0], g.eye[1], [0.0; 4], [0.0; 4]));
    assert_eq!(g.mode, [0x2400; 4]);
    r.cmd(SELECT_CURRENT_MATRIX_STACK, &[gl::MODELVIEW]);
    r.cmd(TRANSLATE_FLOATS, &bits(&[0.0, 0.0, -5.0]));
    tex_plane(&mut r, 0, gl::EYE_PLANE, [0.0, 0.0, 1.0, 0.0]);
    tex_plane(&mut r, 0, gl::OBJECT_PLANE, [0.0, 0.0, 1.0, 0.0]);
    assert_eq!(r.gl.texgen.object[0], [0.0, 0.0, 1.0, 0.0], "stored unchanged");
    assert_eq!(r.gl.texgen.eye[0], [0.0, 0.0, 1.0, 5.0], "times the inverse model-view");
}

/// The generation and clip-plane state is part of a context's saved state.
#[test]
fn texgen_and_clip_plane_state_round_trips_through_a_stream() {
    use crate::dev::kona::stream::{Reader, Writer};
    let mut r = Rig::new();
    tex_mode(&mut r, 1, 0x2402);
    tex_plane(&mut r, 3, gl::OBJECT_PLANE, [1.0, 2.0, 3.0, 4.0]);
    tex_plane(&mut r, 2, gl::EYE_PLANE, [5.0, 6.0, 7.0, 8.0]);
    r.cmd(ENABLE_CAPABILITY, &[gl::TEXTURE_GEN_S + 3]);
    plane(&mut r, 4, [1.0, 0.5, 0.25, 0.125]);
    r.cmd(ENABLE_CAPABILITY, &[gl::CLIP_PLANE0 + 4]);
    let mut w = Writer::default();
    r.gl.visit(&mut w);
    let mut g = Gl::default();
    let mut rd = Reader::new(&w.out);
    g.visit(&mut rd);
    assert!(rd.finish().is_ok());
    assert_eq!(g.texgen, r.gl.texgen);
    assert_eq!((g.user_plane, g.user_on), (r.gl.user_plane, r.gl.user_on));
}
