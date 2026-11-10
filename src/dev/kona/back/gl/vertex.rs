//! KONA's vertex: what the GL core clips, and the policy that clips it.
//!
//! `KVertex` is the clip-space vertex with the attributes the pipe carries
//! (front and back colour, one texture coordinate of four words). It
//! implements the core's `ClipVertex`, so `Clip::clip_polygon_by` cuts it as
//! it cuts VPro's and IMPACT's. `KonaGeometry` is the core's `Geometry` for
//! this board: near and far are clipped geometrically against the planes
//! z + w >= 0 and w - z >= 0, which cut where the board's z' = (z - w) / 2 planes do
//! (the half cancels in the cut position) (inferred: the vertex bus carries z already
//! scaled, and the engines clip the volume's third axis; which exact range is
//! cut is unconfirmed); x and y are clipped only at a guard band
//! of eight clip widths, which keeps window positions inside the `i32` grid,
//! and the job's rectangle takes the rest. The
//! divide is a single-precision reciprocal and a multiply, and window
//! positions truncate toward minus infinity, as VPro's do (unknown for this
//! board).

use crate::dev::gl::vertex::{ClipVertex, Geometry};
use crate::dev::kona::back::raster::WVertex;

/// The guard band: x and y are cut where |x| or |y| passes 8 w, so the kept region
/// is eight viewport widths across in all (four either side of the viewport's
/// edge), which keeps window positions inside the `i32` grid.
pub const GUARD: f32 = 8.0;

#[derive(Clone, Copy, Debug, Default)]
pub struct KVertex {
    /// Clip-space and eye-space positions: both linear in the object position,
    /// so clipping interpolates them along an edge exactly.
    pub h: [f32; 4],
    pub e: [f32; 4],
    /// Front and back colour, 0 to 1.
    pub c: [f32; 4],
    pub cb: [f32; 4],
    /// Texture coordinate (s, t, r, q) after the texture matrix.
    pub t: [f32; 4],
    /// The planes the vertex is outside of.
    pub oc: u32,
    /// The edge flag in force when the vertex was submitted: the edge from it to the next vertex of
    /// its polygon is a boundary edge, which a polygon mode of line or point draws.
    pub edge: bool,
}

impl ClipVertex for KVertex {
    #[inline(always)]
    fn h(&self) -> &[f32; 4] {
        &self.h
    }
    #[inline(always)]
    fn e(&self) -> &[f32; 4] {
        &self.e
    }
    #[inline(always)]
    fn cross(a: &Self, b: &Self, da: f32, db: f32) -> Self {
        let t = da / (da - db);
        let l = |x: f32, y: f32| x + (y - x) * t;
        KVertex {
            h: std::array::from_fn(|k| l(a.h[k], b.h[k])),
            e: std::array::from_fn(|k| l(a.e[k], b.e[k])),
            c: std::array::from_fn(|k| l(a.c[k], b.c[k])),
            cb: std::array::from_fn(|k| l(a.cb[k], b.cb[k])),
            t: std::array::from_fn(|k| l(a.t[k], b.t[k])),
            oc: 0,
            // The clipper sets it.
            edge: false,
        }
    }
    const EDGE_FLAGS: bool = true;
    #[inline(always)]
    fn edge(&self) -> bool {
        self.edge
    }
    #[inline(always)]
    fn set_edge(&mut self, edge: bool) {
        self.edge = edge;
    }
}

pub struct KonaGeometry;

impl Geometry for KonaGeometry {
    const VOLUME: u32 = 0x3f;
    #[inline(always)]
    fn volume(h: &[f32; 4], i: usize) -> f32 {
        match i {
            0 => GUARD * h[3] + h[0],
            1 => GUARD * h[3] - h[0],
            2 => GUARD * h[3] + h[1],
            3 => GUARD * h[3] - h[1],
            4 => h[3] + h[2],
            _ => h[3] - h[2],
        }
    }
    #[inline(always)]
    fn divide(h: &[f32; 4]) -> Option<[f32; 3]> {
        let [cx, cy, cz, cw] = *h;
        if !(cw > 0.0) {
            return None;
        }
        let r = 1.0 / cw;
        Some([cx * r, cy * r, cz * r])
    }
    #[inline(always)]
    fn snap(v: f32, subpixel: f32) -> f32 {
        (v * subpixel).round_ties_even() / subpixel
    }
}

/// The window position of a clip-space vertex relative to the window origin:
/// the viewport applied to the divided position, depth through the depth range,
/// colour and texture words pre-divided, all quantised to the vertex bus's
/// precision. None when the vertex has no divide (w at or below zero). With `fogged`, the vertex
/// carries the fog coordinate, the eye-space distance `|z|` (the model's own choice; see
/// `raster::fog`), over w.
pub fn project(viewport: [f32; 4], depth_range: [f32; 2], v: &KVertex, textured: bool, fogged: bool) -> Option<WVertex> {
    let [nx, ny, nz] = KonaGeometry::divide(&v.h)?;
    let r = 1.0 / v.h[3];
    let [vx, vy, vw, vh] = viewport;
    let [dn, df] = depth_range;
    let t = if textured { [v.t[0] * r, v.t[1] * r, v.t[2] * r, v.t[3] * r] } else { [0.0; 4] };
    Some(WVertex {
        x: grid(vx + (nx + 1.0) * 0.5 * vw),
        y: grid(vy + (ny + 1.0) * 0.5 * vh),
        z: depth32(dn + (nz * 0.5 + 0.5) * (df - dn)),
        c: v.c.map(colour13),
        t: [t, [0.0; 4]],
        d: 0.0,
        fog: if fogged { [fog_coordinate(&v.e) * r, r] } else { [0.0; 2] },
    })
}

/// The fog coordinate of an eye-space position: the absolute depth, over the homogeneous w when it is
/// not 1.
#[inline]
pub fn fog_coordinate(e: &[f32; 4]) -> f32 {
    (if e[3] != 0.0 { e[2] / e[3] } else { e[2] }).abs()
}

/// A colour as the word the planes hold: `0xAARRGGBB`, eight bits a component
/// (the model holds a window's colour at eight bits a component, one word a
/// buffer, whatever depth the visual names).
pub fn colour_word(c: [f32; 4]) -> u32 {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    b(c[3]) << 24 | b(c[0]) << 16 | b(c[1]) << 8 | b(c[2])
}

/// Full intensity of a 13-bit colour component, as the vertex bus carries it.
pub const COLOUR_MAX: f32 = 8191.0;

/// A colour component, 0 to 1, as a 13-bit value.
#[inline]
pub fn colour13(c: f32) -> u16 {
    (c.clamp(0.0, 1.0) * COLOUR_MAX + 0.5) as u16
}

/// A window coordinate in pixels on the 1/256 grid, saturating at what an `i32` holds. The geometry
/// engine subtracts 49152.0 (1.5 x 2^15) from window x and y before a vertex leaves it (established
/// as the store-data rule reads it; the operand order is inferred). Within +-16384 of the origin the
/// result lies in the binade [2^15, 2^16), where one single-precision step is exactly 1/256, so that
/// subtract is what puts the position on the 12.8 grid. A single-precision subtract rounds to
/// nearest, ties to even (the one tie test the diagnostic has fits that and truncation alike, so
/// nearest-even is assumed, inferred), and 49152 is a multiple of 1/256, so the position rounds to
/// the nearest 1/256 with ties to even. Beyond +-16384 the engine's grid is coarser or finer and
/// the clip rejects the vertex first; the model rounds the same way.
#[inline]
pub fn grid(v: f32) -> i32 {
    (v * 256.0).round_ties_even() as i32
}

/// A depth fraction, 0 at the near end of the depth range, as 32 bits.
#[inline]
pub fn depth32(z: f32) -> u32 {
    (z.clamp(0.0, 1.0) * 4_294_967_040.0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::gl::vertex::Clip;

    fn kv(x: f32, y: f32, z: f32, w: f32) -> KVertex {
        let mut v = KVertex { h: [x, y, z, w], e: [x, y, z, w], ..Default::default() };
        v.oc = Clip::default().outcode_by::<KonaGeometry, _>(&v);
        v
    }

    /// Near and far are planes here, where VPro's geometry has none.
    #[test]
    fn near_and_far_clip_geometrically_and_x_and_y_only_at_the_guard_band() {
        let c = Clip::default();
        assert_eq!(kv(0.0, 0.0, 0.5, 1.0).oc, 0);
        assert_ne!(kv(0.0, 0.0, 1.5, 1.0).oc & 0x20, 0, "beyond the far plane");
        assert_ne!(kv(0.0, 0.0, -1.5, 1.0).oc & 0x10, 0, "in front of the near plane");
        assert_eq!(kv(5.0, -5.0, 0.0, 1.0).oc, 0, "outside the window, inside the guard band");
        assert_ne!(kv(9.0, 0.0, 0.0, 1.0).oc & 0x2, 0);
        // A triangle straddling the far plane is cut at it.
        let mut v = [kv(0.0, 0.0, 0.0, 1.0), kv(0.0, 0.0, 3.0, 1.0), kv(0.5, 0.0, 0.0, 1.0), KVertex::default()];
        let any = v[..3].iter().fold(0, |o, w| o | w.oc);
        let mut tmp = v;
        let (n, over) = c.clip_polygon_by::<KonaGeometry, _>(&mut v, &mut tmp, 3, any);
        assert!(!over && n == 4, "{n}");
        assert!(v[..n].iter().all(|w| w.h[2] <= w.h[3] + 1e-6));
    }

    #[test]
    fn the_divide_multiplies_by_a_reciprocal_and_refuses_w_at_or_below_zero() {
        let n = KonaGeometry::divide(&[7.0, 0.0, 0.0, 49.0]).unwrap();
        assert_eq!(n[0], 7.0 * (1.0 / 49.0));
        assert!(KonaGeometry::divide(&[1.0, 1.0, 1.0, 0.0]).is_none());
        assert!(KonaGeometry::divide(&[1.0, 1.0, 1.0, -1.0]).is_none());
        assert_eq!(KonaGeometry::snap(-0.001, 256.0), 0.0, "nearest, not toward minus infinity");
        assert_eq!(KonaGeometry::snap(-0.003, 256.0), -1.0 / 256.0);
    }

    #[test]
    fn components_quantise_to_thirteen_bits_and_depth_to_a_fraction() {
        assert_eq!((colour13(0.0), colour13(1.0), colour13(2.0), colour13(-1.0)), (0, 8191, 8191, 0));
        assert_eq!(colour13(0.5), 4096);
        assert_eq!((depth32(0.0), depth32(2.0)), (0, 4_294_967_040));
    }

    #[test]
    fn positions_round_to_the_nearest_step_of_the_grid_ties_to_even() {
        assert_eq!((grid(1.0), grid(1.0 / 256.0), grid(0.999), grid(-0.001), grid(-1.0)), (256, 1, 256, 0, -256));
        // Halfway between steps: the even one, as the engine's single-precision subtract of 49152.0 lands them.
        assert_eq!((grid(0.5 / 256.0), grid(1.5 / 256.0), grid(2.5 / 256.0), grid(-0.5 / 256.0), grid(-1.5 / 256.0)), (0, 2, 2, 0, -2));
        // The subtract itself, within +-16384 of the origin: the same result as rounding the position.
        for v in [100.37f32, 0.001953125 * 3.0, 1279.999, 16383.5, -16383.25, 0.0] {
            let engine = ((v - 49152.0) as f64 + 49152.0) * 256.0;
            assert_eq!(grid(v) as f64, engine, "{v}");
        }
    }
}
