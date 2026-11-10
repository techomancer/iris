//! Texture-coordinate generation: the equations of OpenGL 1.2 section 2.10.4.
//!
//! Each of the four coordinates (s, t, r, q) can be replaced by a value computed from the
//! vertex: a plane's product with the object position (`OBJECT_LINEAR`), with the eye position
//! (`EYE_LINEAR`), or, for s and t, the sphere-map reflection of the view direction about the
//! eye-space normal. The texture matrix runs after, on the result. Single precision, like the
//! rest of the core.

use super::light::norm;

pub const EYE_LINEAR: u32 = 0x2400;
pub const OBJECT_LINEAR: u32 = 0x2401;
pub const SPHERE_MAP: u32 = 0x2402;

/// The generation state of one texture unit. Plain data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexGen {
    /// Bit i is set when coordinate i (s, t, r, q) is generated.
    pub enabled: u32,
    pub mode: [u32; 4],
    /// Object planes, and eye planes as they are kept: already in eye space.
    pub object: [[f32; 4]; 4],
    pub eye: [[f32; 4]; 4],
}

impl Default for TexGen {
    /// OpenGL's start-up state: eye linear, s from x and t from y, nothing enabled. The start-up
    /// mode, 0x2400 (eye linear), is established for the pipe; the planes' start-up values are
    /// OpenGL's and no source gives the pipe's (inferred).
    fn default() -> Self {
        let mut planes = [[0.0; 4]; 4];
        planes[0][0] = 1.0;
        planes[1][1] = 1.0;
        TexGen { enabled: 0, mode: [EYE_LINEAR; 4], object: planes, eye: planes }
    }
}

fn dot4(p: &[f32; 4], v: [f32; 4]) -> f32 {
    p[0] * v[0] + p[1] * v[1] + p[2] * v[2] + p[3] * v[3]
}

impl TexGen {
    /// Whether any coordinate is generated.
    #[inline]
    pub fn active(&self) -> bool {
        self.enabled != 0
    }

    /// Whether a generated coordinate reads the normal.
    #[inline]
    pub fn needs_normal(&self) -> bool {
        (0..2).any(|i| self.enabled >> i & 1 != 0 && self.mode[i] == SPHERE_MAP)
    }

    /// The texture coordinate of a vertex at object position `obj` and eye position `eye`, with
    /// eye-space normal `n` (read only for sphere map), given the current coordinate `st`: each
    /// enabled coordinate is replaced by its generated value.
    pub fn generate(&self, st: [f32; 4], obj: [f32; 4], eye: [f32; 4], n: [f32; 3]) -> [f32; 4] {
        let mut out = st;
        let mut r = None;
        for (i, o) in out.iter_mut().enumerate() {
            if self.enabled >> i & 1 == 0 {
                continue;
            }
            *o = match self.mode[i] {
                OBJECT_LINEAR => dot4(&self.object[i], obj),
                EYE_LINEAR => dot4(&self.eye[i], eye),
                // Sphere map: only s and t can take it, and one reflection serves both.
                _ if i < 2 => {
                    let r = r.get_or_insert_with(|| reflect(eye, n));
                    let m = 2.0 * (r[0] * r[0] + r[1] * r[1] + (r[2] + 1.0) * (r[2] + 1.0)).sqrt();
                    // The reflection straight back at the viewer has no image: the middle.
                    if m > 0.0 { r[i] / m + 0.5 } else { 0.5 }
                }
                _ => *o,
            };
        }
        out
    }
}

/// u - 2 n' (n' . u), u the unit vector from the eye to the vertex, n' the unit normal.
fn reflect(eye: [f32; 4], n: [f32; 3]) -> [f32; 3] {
    let p = if eye[3] != 0.0 { [eye[0] / eye[3], eye[1] / eye[3], eye[2] / eye[3]] } else { [eye[0], eye[1], eye[2]] };
    let u = norm(p);
    let n = norm(n);
    let d = 2.0 * (n[0] * u[0] + n[1] * u[1] + n[2] * u[2]);
    [u[0] - n[0] * d, u[1] - n[1] * d, u[2] - n[2] * d]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_up_state_is_eye_linear_with_the_identity_planes_and_nothing_enabled() {
        let g = TexGen::default();
        assert!(!g.active() && !g.needs_normal());
        assert_eq!(g.mode, [EYE_LINEAR; 4]);
        assert_eq!(g.generate([5.0; 4], [1.0; 4], [2.0; 4], [0.0; 3]), [5.0; 4], "not enabled: the current coordinate");
    }

    #[test]
    fn linear_modes_take_the_plane_product_with_the_object_or_the_eye_position() {
        let mut g = TexGen { enabled: 0b1111, ..Default::default() };
        g.mode = [OBJECT_LINEAR, EYE_LINEAR, OBJECT_LINEAR, EYE_LINEAR];
        g.object[0] = [1.0, 2.0, 3.0, 4.0];
        g.eye[1] = [0.0, 1.0, 0.0, 0.5];
        g.object[2] = [0.0, 0.0, 1.0, 0.0];
        g.eye[3] = [1.0, 1.0, 1.0, 1.0];
        let out = g.generate([9.0; 4], [1.0, 1.0, 1.0, 1.0], [10.0, 20.0, 30.0, 1.0], [0.0; 3]);
        assert_eq!(out, [10.0, 20.5, 1.0, 61.0]);
        // Only the enabled ones are replaced.
        g.enabled = 0b0010;
        assert_eq!(g.generate([9.0; 4], [1.0; 4], [10.0, 20.0, 30.0, 1.0], [0.0; 3]), [9.0, 20.5, 9.0, 9.0]);
    }

    #[test]
    fn sphere_map_reflects_the_view_about_the_normal() {
        let mut g = TexGen { enabled: 0b0011, mode: [SPHERE_MAP; 4], ..Default::default() };
        assert!(g.needs_normal());
        // Looking down -z at a surface facing +z: the reflection goes straight back, the centre.
        let st = g.generate([0.0; 4], [0.0; 4], [0.0, 0.0, -5.0, 1.0], [0.0, 0.0, 1.0]);
        assert_eq!(st[..2], [0.5, 0.5]);
        // A normal turned 45 degrees toward +x reflects the view to +x: s above the centre, t on it.
        let n = [0.5f32.sqrt(), 0.0, 0.5f32.sqrt()];
        let st = g.generate([0.0; 4], [0.0; 4], [0.0, 0.0, -5.0, 1.0], n);
        let s = 0.5 + 1.0 / (2.0 * 2f32.sqrt());
        assert!((st[0] - s).abs() < 1e-6 && (st[1] - 0.5).abs() < 1e-6, "{st:?}");
        // A normal that need not be unit, and an eye position with w: same answer.
        let st2 = g.generate([0.0; 4], [0.0; 4], [0.0, 0.0, -10.0, 2.0], [n[0] * 4.0, 0.0, n[2] * 4.0]);
        assert!((st2[0] - st[0]).abs() < 1e-6 && (st2[1] - st[1]).abs() < 1e-6);
        // R and q cannot be sphere mapped: they keep the current value.
        g.enabled = 0b1111;
        assert_eq!(g.generate([7.0, 7.0, 3.0, 4.0], [0.0; 4], [0.0, 0.0, -5.0, 1.0], [0.0, 0.0, 1.0])[2..], [3.0, 4.0]);
        // The reflection at m = 0: the view meets the surface edge-on from behind.
        let st = g.generate([0.0; 4], [0.0; 4], [0.0, 0.0, 5.0, 1.0], [0.0, 0.0, 1.0]);
        assert_eq!(st[..2], [0.5, 0.5]);
    }
}
