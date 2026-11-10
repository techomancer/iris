//! Smoothed (antialiased) primitives: the coverage of a pixel by a point, a line or a polygon.
//!
//! What is known of the pipe: it has a polygon-smoothing class and a line-smoothing class, with the
//! sampling-word values it writes for each (established), and the capabilities are the public smooth
//! enables, whose state rows the generic state query lists (established as rows). Multisampling wins
//! over polygon smoothing, and smoothing wins over multisampling for lines. What is unknown is the
//! rule itself: the rasterisation rules for multisampled lines, points and polygons beyond the
//! three sampling selects (tie rules, multisampled line and polygon coverage), and how the coverage
//! reaches the fragment.
//!
//! Everything below is therefore an approximation of the model's own, kept in this one file so a rule
//! the hardware turns out to follow replaces it here and nowhere else:
//!
//! - **Coverage scales alpha.** A smoothed fragment's alpha is multiplied by the fraction of the
//!   pixel's square the primitive covers, after texture and fog and before the alpha test, as OpenGL's
//!   published definition has it. Whether the board does that, or feeds a coverage word to its image
//!   engines as it does for sample masks, is unknown.
//! - **A polygon edge, a line side or a line end is a half-plane.** The pixel's share of one is the
//!   exact area of the unit square on that side of a straight line (`half_plane`); a pixel at a
//!   corner takes the product of the shares of all the primitive's edges, which is exact for an
//!   axis-aligned corner and close otherwise. A line thinner than a pixel is overstated by the
//!   product. A polygon of more than `MAX_EDGES` edges (after clipping) is not smoothed.
//! - **A point is a disc,** its coverage the share of an 8 x 8 grid of sample positions in the pixel
//!   that lie inside it (`DISC_GRID`), so it takes 65 levels.
//! - **A pixel is its unit square.** Coverage is measured over the square centred on the pixel's
//!   position in GL's window coordinates (the plain rule's sample sits at the square's corner in
//!   the board's own frame; the set-up half-pixel conversion makes the two agree for a GL unit
//!   square). Every pixel with any coverage is drawn, so a smoothed primitive reaches up to one pixel
//!   past where the plain rule stops.
//! - **Inner edges are not smoothed.** A polygon is drawn as a fan of triangles; only the edges of the
//!   polygon are ramps, and the fan's inner edges keep the unsmoothed centre test, so a pixel on one
//!   belongs to exactly one triangle and the seam does not blend twice. Each triangle weighs its
//!   pixels by every edge of the polygon, so a pixel at a corner the fan splits is covered as the
//!   corner covers it.
//! - **A multisampled window:** smoothing wins for lines and points there and multisampling for
//!   polygons (the sampling word's values and the point handler's order of tests, established), so a
//!   smoothed line or point still takes coverage, written to every sample of the pixel with alpha
//!   scaled, and a polygon is multisampled (`DrawState::smooth`).
//! - **Lines and points deliver coverage as alpha in 8 bits** (the pipe loads its line coverage
//!   ramps with 8-bit values scaled by 255.0: established for the ramps, inferred for their use as
//!   alpha), so their coverage is rounded to a multiple of 1/255 (`quantise`). The ramps are indexed
//!   by distance in 32 steps; the exact area is the model's, not that. A smoothed polygon's
//!   coverage is computed in the pixel generator and its rule is unknown, so it stays the exact
//!   area (approximated). The alpha-to-mask fade table (nine entries, entry n with n bits set)
//!   is written at start-up; how alpha indexes it is unknown and the model does not use it.
//!
//! What would settle them: a capture of a smoothed line, point and polygon on the board (or the
//! coverage tables and the alpha-to-mask fade table, which are unread)
//! giving the per-pixel result for a known primitive.
//!
//! This file is held to the per-pixel module's rule: single-precision multiplies and adds, no doubles,
//! so a result is the same on every host.

use super::setup::Tri;

/// What covers a triangle's pixels besides the triangle itself.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Cover {
    /// Nothing: the plain rule.
    #[default]
    None,
    /// A triangle of a smoothed polygon or line. `inner[k]` says edge `k` of the triangle is an inner
    /// edge of the fan, which keeps the centre test; the triangle's other edges are the polygon's
    /// and ramp. The primitive's `n` edges are each a plane `[nx, ny, d]`, its unit normal pointing
    /// inside and the distance of the job's first pixel, so a pixel `(i, j)` from there is `nx i + ny j + d`
    /// inside the edge.
    Edges { inner: [bool; 3], n: u8, plane: [[f32; 3]; MAX_EDGES] },
    /// A triangle of the square about a smoothed point: the disc's centre and radius in pixels, in
    /// the plane of the sample positions.
    Disc { x: f32, y: f32, r: f32 },
}

/// The most edges a smoothed polygon can have and be smoothed.
pub const MAX_EDGES: usize = 8;

/// Sample positions a side of a disc's pixel is cut into.
const DISC_GRID: u32 = 8;

/// A line's or point's coverage as the 8-bit alpha the ramps deliver.
#[inline(always)]
pub fn quantise(cov: f32) -> f32 {
    ((cov * 255.0f32 + 0.5f32) as u32) as f32 / 255.0f32
}

/// The share of the unit square about the origin on the inside of a line, as an exact area. The
/// line's unit normal points inside, `(nx, ny)`, and the origin is `s` pixels inside it (negative:
/// outside). The projection of the square on the normal is a trapezoid of half-width
/// `(|nx| + |ny|) / 2`, flat over `||nx| - |ny|| / 2` and rising linearly outside that, whose
/// integral is the answer.
#[inline(always)]
pub fn half_plane(nx: f32, ny: f32, s: f32) -> f32 {
    let (ax, ay) = (nx.abs(), ny.abs());
    let (a, b) = if ax >= ay { (ax, ay) } else { (ay, ax) };
    let (w, h) = ((a + b) * 0.5f32, (a - b) * 0.5f32);
    let t = -s;
    let below = if t <= -w {
        0.0f32
    } else if t >= w {
        1.0f32
    } else if t < -h {
        // The rising part: only when b > 0, since otherwise -w == -h.
        let u = t + w;
        u * u / (2.0f32 * a * b)
    } else if t <= h {
        b / (2.0f32 * a) + (t + h) / a
    } else {
        let u = w - t;
        1.0f32 - u * u / (2.0f32 * a * b)
    };
    1.0f32 - below
}

/// The share of the pixel (`x`, `y`) a disc of radius `r` about (`cx`, `cy`) covers. Whole and empty
/// pixels are told by distance without sampling.
#[inline(always)]
fn disc(cx: f32, cy: f32, r: f32, x: i32, y: i32) -> f32 {
    // The farthest a pixel's corner is from its centre.
    const REACH: f32 = 0.7072f32;
    let (dx, dy) = (x as f32 - cx, y as f32 - cy);
    let d2 = dx * dx + dy * dy;
    if r > REACH && d2 <= (r - REACH) * (r - REACH) {
        return 1.0f32;
    }
    if d2 >= (r + REACH) * (r + REACH) {
        return 0.0f32;
    }
    let step = 1.0f32 / DISC_GRID as f32;
    let (x0, y0) = (dx - 0.5f32 + step * 0.5f32, dy - 0.5f32 + step * 0.5f32);
    let mut n = 0u32;
    for j in 0..DISC_GRID {
        let py = y0 + step * j as f32;
        for i in 0..DISC_GRID {
            let px = x0 + step * i as f32;
            n += (px * px + py * py <= r * r) as u32;
        }
    }
    n as f32 / (DISC_GRID * DISC_GRID) as f32
}

/// The share of pixel (`x`, `y`) the triangle `t` covers, 0 when the pixel is not drawn at all. With
/// `Cover::None` it is the plain rule's one or none.
#[inline(always)]
pub fn coverage(t: &Tri, x: i32, y: i32) -> f32 {
    match t.cover {
        Cover::None => (0..3).all(|k| inside(t, k, x, y)) as u32 as f32,
        Cover::Disc { x: cx, y: cy, r } => {
            // The square about the disc holds the pixels the disc can touch; the triangle's own
            // centre test shares them between its two halves.
            if !(0..3).all(|k| inside(t, k, x, y)) {
                return 0.0f32;
            }
            disc(cx, cy, r, x, y)
        }
        Cover::Edges { inner, n, plane } => {
            for k in 0..3 {
                if inner[k] && !inside(t, k, x, y) {
                    return 0.0f32;
                }
            }
            let (i, j) = ((x - t.origin[0]) as f32, (y - t.origin[1]) as f32);
            let mut c = 1.0f32;
            for [nx, ny, d] in &plane[..n as usize] {
                let s = nx * i + ny * j + d;
                // Farther inside than a pixel's corner is from its centre: whole.
                if s >= 0.7072f32 {
                    continue;
                }
                c *= half_plane(*nx, *ny, s);
                if c <= 0.0f32 {
                    return 0.0f32;
                }
            }
            c
        }
    }
}

/// Whether the sample of pixel (`x`, `y`) is on the inside of edge `k` of `t`, a tie going to a
/// right or top edge.
#[inline(always)]
fn inside(t: &Tri, k: usize, x: i32, y: i32) -> bool {
    let e = t.a[k] * x as i64 + t.b[k] * y as i64 + t.c[k];
    e > 0 || (e == 0 && t.tie[k])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_axis_aligned_edge_covers_the_distance_it_cuts() {
        for (s, want) in [(-1.0f32, 0.0f32), (-0.5, 0.0), (-0.25, 0.25), (0.0, 0.5), (0.25, 0.75), (0.5, 1.0), (3.0, 1.0)] {
            assert_eq!(half_plane(1.0, 0.0, s), want, "x normal at {s}");
            assert_eq!(half_plane(0.0, -1.0, s), want, "y normal at {s}");
        }
    }

    #[test]
    fn a_slanted_edge_through_the_centre_covers_half_and_the_two_sides_make_a_whole() {
        let n = std::f32::consts::FRAC_1_SQRT_2;
        assert!((half_plane(n, n, 0.0) - 0.5).abs() < 1e-6);
        for (nx, ny) in [(0.6f32, 0.8f32), (n, -n), (0.2, 0.9798), (-0.9, 0.4359)] {
            for s in [-0.6f32, -0.3, -0.1, 0.0, 0.05, 0.2, 0.45, 0.7] {
                let (p, q) = (half_plane(nx, ny, s), half_plane(-nx, -ny, -s));
                assert!((p + q - 1.0).abs() < 1e-5, "n ({nx}, {ny}) s {s}: {p} + {q}");
                assert!((0.0..=1.0).contains(&p));
            }
        }
        // Against the area counted on a fine grid of the square.
        let area = |s: f32| {
            // Brute force over a fine grid of the square.
            let n = 400;
            let mut c = 0;
            for j in 0..n {
                for i in 0..n {
                    let (x, y) = ((i as f32 + 0.5) / n as f32 - 0.5, (j as f32 + 0.5) / n as f32 - 0.5);
                    c += (0.6 * x + 0.8 * y + s >= 0.0) as u32;
                }
            }
            c as f32 / (n * n) as f32
        };
        for s in [-0.6f32, -0.35, -0.1, 0.0, 0.2, 0.5] {
            assert!((half_plane(0.6, 0.8, s) - area(s)).abs() < 2e-3, "s {s}: {} against {}", half_plane(0.6, 0.8, s), area(s));
        }
    }

    #[test]
    fn a_disc_covers_by_its_share_of_the_pixel() {
        // Centred on the pixel, radius one half: the disc inscribed in the square, pi / 4.
        let c = disc(5.0, 5.0, 0.5, 5, 5);
        assert!((c - std::f32::consts::FRAC_PI_4).abs() < 0.04, "{c}");
        // Far inside is whole, far outside is nothing, and a pixel whose square touches the rim is partial.
        assert_eq!(disc(5.0, 5.0, 6.0, 5, 6), 1.0);
        assert_eq!(disc(5.0, 5.0, 2.0, 9, 5), 0.0);
        let rim = disc(5.0, 5.0, 2.0, 7, 5);
        assert!(rim > 0.0 && rim < 1.0, "{rim}");
        // By symmetry the four pixels about a centre on a pixel corner share it equally.
        let q = [disc(5.5, 5.5, 1.0, 5, 5), disc(5.5, 5.5, 1.0, 6, 5), disc(5.5, 5.5, 1.0, 5, 6), disc(5.5, 5.5, 1.0, 6, 6)];
        assert!(q.iter().all(|&v| v == q[0]) && (q[0] * 4.0 - std::f32::consts::PI).abs() < 0.15, "{q:?}");
    }
}
