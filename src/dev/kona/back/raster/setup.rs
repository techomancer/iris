//! Primitive set-up: a triangle of window-relative vertices into a `TriJob`.
//!
//! This is the one place the raster stage uses `f64`, once per triangle, to
//! solve the attribute planes; what it hands the per-pixel module (`pixel`) is
//! integers and `f32` (a test holds `pixel.rs` to that).
//!
//! Vertices arrive on the 1/256-pixel grid, rounded once by the GL path as the engine's subtract of
//! 49152.0 rounds them (`vertex::grid`).
//! Set-up takes the GL half-pixel offset off x and y (so a GL
//! unit square lights the pixel GL specifies), then snaps to the
//! 1/8 grid, nearest with ties up: every primitive is computed to 8 x 8
//! sub-pixel accuracy (established, published). The window's origin is added
//! after, in whole pixels.
//!
//! Edges are exact integers on that grid. A sample sits at the pixel's integer
//! corner (sample 0 at offset (0, 0), inferred; a multisampled window's samples are
//! at their own offsets from that corner, see `multisample`); one exactly on a left or
//! bottom edge is out, one on a right or top edge is in (inferred, from the
//! whole-pixel rectangle the textport sends). Every sample falls on one side of
//! an edge two triangles share, so shared edges are watertight. A triangle of
//! no area draws nothing.
//!
//! There is no clip in window space: a triangle is cut only by the job's
//! rectangle, the box of its sample points cut to the clip rectangle and the
//! viewport. The attribute planes are valued at that rectangle's corner, not at
//! the triangle's own box, so a triangle far outside the screen keeps its
//! slopes and its colour where it crosses the screen's edge.

use super::smooth::{Cover, MAX_EDGES};
use super::{DrawState, Rect, WVertex};

/// A triangle after set-up: three edge functions on the 1/8-pixel grid, exact
/// in integers, and a plane for each attribute in single precision, valued
/// at the pixel `origin`, the corner of the job's rectangle. All the `f64` of
/// set-up is spent making these.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tri {
    /// `e(x, y) = a * x + b * y + c` at the sample of pixel (x, y); a sample
    /// is inside when `e > 0`, or `e == 0` and `tie` (a right or top edge).
    pub a: [i64; 3],
    pub b: [i64; 3],
    pub c: [i64; 3],
    pub tie: [bool; 3],
    pub origin: [i32; 2],
    /// Red, green, blue and alpha in 0..=8191:
    /// `value = at_origin + dx * (x - origin.x) + dy * (y - origin.y)`.
    pub at_origin: [f32; 4],
    pub dx: [f32; 4],
    pub dy: [f32; 4],
    /// Depth, a fraction of `u32::MAX` in 32.16 fixed point, on the same plane form
    /// in integers: the compare is exact whatever the host.
    pub z: Plane,
    /// For a line's triangle, the distance along the line in pixels, as the colour
    /// planes are: `at`, `dx`, `dy`.
    pub line_d: [f32; 3],
    /// The texture words `s/w, t/w, r/w, q/w`, each `at`, `dx`, `dy` as the colour planes are.
    pub tex: [[f32; 3]; 4],
    /// The fog coordinate over w, then 1 / w, as planes the same way.
    pub fog: [[f32; 3]; 2],
    /// What covers the triangle's pixels besides the triangle, for a smoothed primitive.
    pub cover: Cover,
}

/// An integer attribute plane: `at + dx * i + dy * j` at the pixel `i, j` from
/// the job's origin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Plane {
    pub at: i64,
    pub dx: i64,
    pub dy: i64,
}

/// A triangle to draw inside `rect`, under draw state `state` (an entry of the
/// scheduler's table).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TriJob {
    pub tri: Tri,
    /// A triangle of a line: the line stipple applies to it, and the polygon stipple does not.
    pub line: bool,
    /// Where the triangle can write: its box of sample points cut to the clip
    /// rectangle and the viewport.
    pub rect: Rect,
    pub state: u16,
}

/// Why a triangle has no set-up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    /// Its three vertices are on a line (after snapping to the 1/8 grid).
    NoArea,
    /// It has area, but no pixel's sample falls inside its box.
    NoSample,
    /// It has samples, but none inside the clip rectangle and the viewport.
    Outside,
}

/// One vertex on the 1/8 grid with its attributes.
#[derive(Clone, Copy, Debug)]
struct V {
    x: i64,
    y: i64,
    attr: [f64; 12],
}

/// How a triangle's pixels are chosen and weighted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape<'a> {
    /// One sample at each pixel's corner against the three edges: the plain rule.
    Plain,
    /// A triangle of the fan a smoothed polygon or line is drawn as (`smooth`): every pixel the
    /// polygon reaches, weighted by the share of it the polygon covers. `inner[k]` says the edge from
    /// the triangle's vertex `k` is an inner edge of the fan.
    Smooth { inner: [bool; 3], outline: &'a Outline },
    /// A triangle of the square about a smoothed point of radius `r` pixels at (`x`, `y`) in the
    /// window's pixel coordinates, GL's: the pixel centres are half a pixel on.
    Disc { x: f32, y: f32, r: f32 },
}

/// A smoothed polygon's edges, each as the point it starts at and its unit normal pointing inside,
/// in pixels of the sample plane, from the vertices as set-up snaps them. An edge of no length (clipping
/// can leave two vertices alike) is left out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outline {
    n: usize,
    e: [[f64; 4]; MAX_EDGES],
}

impl Outline {
    /// The outline of the polygon `v`, or None for one smoothing does not take: under three vertices,
    /// over `MAX_EDGES` edges or of no area.
    pub fn new(v: &[WVertex], ds: &DrawState) -> Option<Outline> {
        if v.len() < 3 || v.len() > MAX_EDGES {
            return None;
        }
        let p: Vec<V> = v.iter().map(|w| vertex(w, ds.origin)).collect();
        let area2: i64 = (0..p.len()).map(|k| p[k].x * p[(k + 1) % p.len()].y - p[(k + 1) % p.len()].x * p[k].y).sum();
        if area2 == 0 {
            return None;
        }
        // The normal to the left of an edge points inside a counter-clockwise polygon.
        let turn = if area2 > 0 { 1.0 } else { -1.0 };
        let mut o = Outline { n: 0, e: [[0.0; 4]; MAX_EDGES] };
        for k in 0..p.len() {
            let (s, e) = (p[k], p[(k + 1) % p.len()]);
            let (dx, dy) = ((e.x - s.x) as f64, (e.y - s.y) as f64);
            let len = dx.hypot(dy);
            if len > 0.0 {
                o.e[o.n] = [s.x as f64 / 8.0, s.y as f64 / 8.0, -dy / len * turn, dx / len * turn];
                o.n += 1;
            }
        }
        Some(o)
    }
}

fn vertex(v: &WVertex, origin: [i32; 2]) -> V {
    // 1/256 to 1/8: take the half pixel off (128 of 256), then round to the
    // nearest eighth, ties up (adding half of 32 and shifting floors).
    let snap = |c: i32, o: i32| ((c as i64 - 128 + 16) >> 5) + o as i64 * 8;
    V {
        x: snap(v.x, origin[0]),
        y: snap(v.y, origin[1]),
        attr: [v.c[0] as f64, v.c[1] as f64, v.c[2] as f64, v.c[3] as f64, v.z as f64, v.d as f64, v.t[0][0] as f64, v.t[0][1] as f64, v.t[0][2] as f64, v.t[0][3] as f64, v.fog[0] as f64, v.fog[1] as f64],
    }
}

/// The set-up of triangle `v` under draw state `ds` (entry `state` of the
/// scheduler's table). `Skip` says why there is none: no area, no pixel
/// sample in its box, or none inside the clip rectangle and the viewport.
#[cfg(test)]
pub fn triangle(v: &[WVertex; 3], ds: &DrawState, state: u16) -> Result<TriJob, Skip> {
    shaped(v, ds, state, Shape::Plain)
}

/// `triangle`, chosen and weighted by `shape`.
pub fn shaped(v: &[WVertex; 3], ds: &DrawState, state: u16, shape: Shape<'_>) -> Result<TriJob, Skip> {
    let mut p = [vertex(&v[0], ds.origin), vertex(&v[1], ds.origin), vertex(&v[2], ds.origin)];
    let area2 = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[2].x - p[0].x) * (p[1].y - p[0].y);
    if area2 == 0 {
        return Err(Skip::NoArea);
    }
    // Whether the edge that starts at each vertex is an inner edge, in the order the edges are taken.
    let mut inner = match shape {
        Shape::Smooth { inner, .. } => inner,
        _ => [false; 3],
    };
    // Counter-clockwise, so the inside is on the left of every edge.
    if area2 < 0 {
        p.swap(1, 2);
        // The edges are the same three, run the other way: the one that started at the second vertex
        // now starts at the third.
        inner = [inner[2], inner[1], inner[0]];
    }
    // The box of integer sample points (multiples of 8) the triangle spans.
    let lo = |f: fn(&V) -> i64| p.iter().map(f).min().expect("three");
    let hi = |f: fn(&V) -> i64| p.iter().map(f).max().expect("three");
    let ceil8 = |v: i64| (v + 7) >> 3;
    let floor8 = |v: i64| v >> 3;
    // With samples inside the pixel, a pixel whose corner is left of the triangle can still be covered.
    let first = |v: i64| if ds.samples.is_some() { floor8(v) } else { ceil8(v) };
    // A smoothed triangle reaches a pixel whose sample is outside it: by less than a pixel.
    let pad = matches!(shape, Shape::Smooth { .. }) as i64;
    let (x0, x1) = (first(lo(|v| v.x)) - pad, floor8(hi(|v| v.x)) + 1 + pad);
    let (y0, y1) = (first(lo(|v| v.y)) - pad, floor8(hi(|v| v.y)) + 1 + pad);
    if x0 >= x1 || y0 >= y1 {
        return Err(Skip::NoSample);
    }
    // The guard band keeps positions inside `i32` (nine viewport widths), so
    // the clamp only guards an absurd viewport.
    let clamp = |v: i64| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let bounds = Rect { x0: clamp(x0), y0: clamp(y0), x1: clamp(x1), y1: clamp(y1) };
    let rect = bounds.intersect(&ds.clip).intersect(&ds.viewport);
    if rect.is_empty() {
        return Err(Skip::Outside);
    }
    let mut t = Tri { origin: [rect.x0, rect.y0], ..Tri::default() };
    for i in 0..3 {
        let (s, e) = (p[i], p[(i + 1) % 3]);
        let (dx, dy) = (e.x - s.x, e.y - s.y);
        // e(sx, sy) = dx * (sy - s.y) - dy * (sx - s.x) at sx = 8x, sy = 8y.
        t.a[i] = -dy * 8;
        t.b[i] = dx * 8;
        t.c[i] = dy * s.x - dx * s.y;
        t.tie[i] = dy > 0 || (dy == 0 && dx < 0);
    }
    t.cover = match shape {
        Shape::Plain => Cover::None,
        Shape::Smooth { outline, .. } => {
            // Each edge as a plane from the job's first pixel: its normal and the distance there.
            let (ox, oy) = (rect.x0 as f64, rect.y0 as f64);
            let mut plane = [[0.0f32; 3]; MAX_EDGES];
            for (pl, [sx, sy, nx, ny]) in plane.iter_mut().zip(&outline.e[..outline.n]) {
                *pl = [*nx as f32, *ny as f32, (nx * (ox - sx) + ny * (oy - sy)) as f32];
            }
            Cover::Edges { inner, n: outline.n as u8, plane }
        }
        // Sample-plane coordinates: GL's pixel centres are half a pixel on, and the window's origin is added.
        Shape::Disc { x, y, r } => Cover::Disc { x: x - 0.5 + ds.origin[0] as f32, y: y - 0.5 + ds.origin[1] as f32, r },
    };
    // Attribute planes, in pixel units, from the vertices' own positions,
    // valued at the rectangle's corner.
    let px = |v: &V| (v.x as f64 / 8.0, v.y as f64 / 8.0);
    let ((ax, ay), (bx, by), (cx, cy)) = (px(&p[0]), px(&p[1]), px(&p[2]));
    let det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    for k in 0..12 {
        let (db, dc) = (p[1].attr[k] - p[0].attr[k], p[2].attr[k] - p[0].attr[k]);
        let gx = (db * (cy - ay) - dc * (by - ay)) / det;
        let gy = (dc * (bx - ax) - db * (cx - ax)) / det;
        let at = p[0].attr[k] + gx * (rect.x0 as f64 - ax) + gy * (rect.y0 as f64 - ay);
        if k < 4 {
            t.dx[k] = gx as f32;
            t.dy[k] = gy as f32;
            t.at_origin[k] = at as f32;
        } else if k == 5 {
            t.line_d = [at as f32, gx as f32, gy as f32];
        } else if k > 9 {
            t.fog[k - 10] = [at as f32, gx as f32, gy as f32];
        } else if k > 5 {
            t.tex[k - 6] = [at as f32, gx as f32, gy as f32];
        } else {
            // 16 fraction bits; the depth fraction is under 2^32, so the plane's value fits `i64`; its
            // slope, times a pixel offset, can exceed it for a sliver at the clamp, which the
            // per-pixel module's wrapping arithmetic tolerates.
            let fx = |v: f64| (v * 65536.0).round().clamp(-(1i64 << 62) as f64, (1i64 << 62) as f64) as i64;
            // Polygon offset: factor times the larger depth slope, plus units times the
            // smallest step of the depth the planes hold (a 24-bit depth in the top of
            // the 32-bit fraction, so 256). The formula is Iris's own.
            let off = ds.offset.map_or(0.0, |[factor, units]| factor as f64 * gx.abs().max(gy.abs()) + units as f64 * 256.0);
            t.z = Plane { at: fx(at + off), dx: fx(gx), dy: fx(gy) };
        }
    }
    Ok(TriJob { tri: t, line: false, rect, state })
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn vtx(x: f32, y: f32, c: [u16; 4], z: u32) -> WVertex {
        WVertex { x: (x * 256.0) as i32, y: (y * 256.0) as i32, z, c, t: [[0.0; 4]; 2], d: 0.0, fog: [0.0; 2] }
    }

    const BIG: Rect = Rect { x0: -4096, y0: -4096, x1: 4096, y1: 4096 };

    fn ds(origin: [i32; 2], clip: Rect, viewport: Rect) -> DrawState {
        DrawState { origin, clip, viewport, ..DrawState::default() }
    }

    fn tri3(a: WVertex, b: WVertex, c: WVertex, d: &DrawState) -> Result<TriJob, Skip> {
        triangle(&[a, b, c], d, 7)
    }

    #[test]
    fn a_triangle_of_no_area_has_no_setup() {
        let (c, d) = ([0; 4], ds([0, 0], BIG, BIG));
        assert_eq!(tri3(vtx(1.0, 1.0, c, 0), vtx(2.0, 2.0, c, 0), vtx(3.0, 3.0, c, 0), &d).err(), Some(Skip::NoArea));
        // A sliver between two sample columns has area and no sample.
        assert_eq!(tri3(vtx(1.625, 1.5, c, 0), vtx(1.875, 1.5, c, 0), vtx(1.625, 5.5, c, 0), &d).err(), Some(Skip::NoSample));
        // A triangle with samples wholly outside the clip rectangle or the viewport.
        let off = ds([0, 0], Rect { x0: 20, y0: 20, x1: 30, y1: 30 }, BIG);
        assert_eq!(tri3(vtx(1.5, 1.5, c, 0), vtx(5.5, 1.5, c, 0), vtx(1.5, 5.5, c, 0), &off).err(), Some(Skip::Outside));
        let off = ds([0, 0], BIG, Rect { x0: 20, y0: 20, x1: 30, y1: 30 });
        assert_eq!(tri3(vtx(1.5, 1.5, c, 0), vtx(5.5, 1.5, c, 0), vtx(1.5, 5.5, c, 0), &off).err(), Some(Skip::Outside));
    }

    /// The half pixel comes off, then the grid is eighths with ties up: a vertex
    /// at 1.5 in GL window coordinates is pixel 1.0, 8 on the grid.
    #[test]
    fn vertices_lose_half_a_pixel_and_snap_to_eighths_ties_up() {
        let c = [0; 4];
        let t = tri3(vtx(1.5, 1.5, c, 0), vtx(5.5, 1.5, c, 0), vtx(1.5, 5.5, c, 0), &ds([0, 0], BIG, BIG)).unwrap();
        assert_eq!(t.rect, Rect { x0: 1, y0: 1, x1: 6, y1: 6 }, "samples at 1..=5 in each axis");
        assert_eq!(t.tri.origin, [1, 1]);
        assert_eq!(t.state, 7);
        let v = vertex(&vtx(0.5 + 1.0 / 16.0, 0.0, c, 0), [0, 0]);
        assert_eq!(v.x, 1, "a sixteenth is half an eighth: ties go up");
        assert_eq!(vertex(&vtx(0.5 + 1.0 / 16.0 - 1.0 / 256.0, 0.0, c, 0), [0, 0]).x, 0);
    }

    #[test]
    fn the_planes_are_valued_at_the_origin_and_slope_across_the_triangle() {
        let (r, g) = ([8191, 0, 0, 8191], [0, 8191, 0, 8191]);
        let t = tri3(vtx(0.5, 0.5, r, 0), vtx(8.5, 0.5, g, 0), vtx(0.5, 8.5, [0, 0, 8191, 8191], 0x8000_0000), &ds([0, 0], BIG, BIG)).unwrap().tri;
        assert_eq!(t.origin, [0, 0]);
        assert_eq!(t.at_origin[0], 8191.0);
        assert!((t.dx[0] + 8191.0 / 8.0).abs() < 1e-2 && (t.dy[0] + 8191.0 / 8.0).abs() < 1e-2, "red falls to zero along both edges");
        assert!((t.dx[1] - 8191.0 / 8.0).abs() < 1e-2);
        assert!((t.z.dy as f64 / 65536.0 - 2147483648.0 / 8.0).abs() < 1.0);
    }

    /// Polygon offset moves the depth plane's value, not its slope: factor times the
    /// larger slope plus units steps of 256.
    #[test]
    fn polygon_offset_adds_slope_and_units_to_the_depth_plane() {
        let c = [0; 4];
        let v = |x: f32, y: f32, z: u32| vtx(x, y, c, z);
        let tri = [v(0.5, 0.5, 0), v(8.5, 0.5, 0x0800_0000), v(0.5, 8.5, 0)];
        let plain = triangle(&tri, &ds([0, 0], BIG, BIG), 0).unwrap().tri.z;
        let off = triangle(&tri, &DrawState { offset: Some([2.0, 3.0]), ..ds([0, 0], BIG, BIG) }, 0).unwrap().tri.z;
        assert_eq!((off.dx, off.dy), (plain.dx, plain.dy), "the slope is unchanged");
        let slope = plain.dx.abs().max(plain.dy.abs()) as f64 / 65536.0;
        let want = ((2.0 * slope + 3.0 * 256.0) * 65536.0).round() as i64;
        assert!((off.at - plain.at - want).abs() <= 2, "{} against {want}", off.at - plain.at);
    }

    #[test]
    fn the_window_origin_moves_the_triangle() {
        let c = [0; 4];
        let t = tri3(vtx(0.5, 0.5, c, 0), vtx(4.5, 0.5, c, 0), vtx(0.5, 4.5, c, 0), &ds([100, 50], BIG, BIG)).unwrap();
        assert_eq!(t.rect, Rect { x0: 100, y0: 50, x1: 105, y1: 55 });
    }

    /// The job's rectangle is the triangle's box cut to the clip rectangle and
    /// the viewport, and the planes are valued at its corner.
    #[test]
    fn a_triangle_far_off_screen_keeps_its_colour_at_the_screen_edge() {
        let (r, g, b) = ([8191, 0, 0, 8191], [0, 8191, 0, 8191], [0, 0, 8191, 8191]);
        // Red at (-4000, -4000), green at (4000, -4000), blue at (-4000, 4000): at (x, y) the red
        // weight is 1 - (x + 4000) / 8000 - (y + 4000) / 8000, and green and blue the two terms.
        let screen = Rect { x0: 0, y0: 0, x1: 64, y1: 64 };
        let t = tri3(vtx(-4000.0, -4000.0, r, 0), vtx(4000.0, -4000.0, g, 0), vtx(-4000.0, 4000.0, b, 0), &ds([0, 0], screen, screen)).unwrap();
        assert_eq!(t.rect, screen);
        assert_eq!(t.tri.origin, [0, 0]);
        // The half pixel comes off the vertices, so pixel (0, 0) sits at 4000.5 / 8000 along each axis.
        let (wg, wb) = (4000.5 / 8000.0, 4000.5 / 8000.0);
        let want = |w: f64| (w * 8191.0) as f32;
        let at = t.tri.at_origin;
        assert!((at[1] - want(wg)).abs() < 2.0 && (at[2] - want(wb)).abs() < 2.0 && (at[0] - want(1.0 - wg - wb)).abs() < 2.0, "{at:?}");
    }
}
