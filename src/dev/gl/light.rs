//! The OpenGL 1.1 lighting equation and fog, per vertex. A board fills
//! `Lighting` from its own protocol (GR2: `src/dev/gr2/gl_light.rs`).
//!
//! Two things vary between boards' libglcore and are parameters here:
//! - specular and spot exponents arrive either as numbers (`shininess`,
//!   `spot_exp`) or as uploaded pow() tables the GE interpolates (GR2:
//!   `spec_table` / `spot_table` set, x mapped to (x - start) * scale);
//! - GR2's libglcore folds the lights' ambient terms into the scene ambient
//!   when it can (`ambient_folded`); otherwise each light adds its own.

pub const MAX_LIGHTS: usize = 8;
pub const SPEC_N: usize = 128;
pub const SPOT_N: usize = 32;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Material {
    pub emission: [f32; 3],
    pub ambient: [f32; 3],
    pub diffuse: [f32; 4],
    pub specular: [f32; 3],
    pub shininess: f32,
    /// Non-zero: the specular factor comes from `spec` (see module docs).
    pub spec_table: u32,
    pub spec_scale: f32,
    pub spec_start: f32,
    pub spec: [f32; SPEC_N],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Light {
    pub ambient: [f32; 3],
    pub diffuse: [f32; 3],
    pub specular: [f32; 3],
    /// Eye space; w = 0: direction to the light.
    pub pos: [f32; 4],
    pub spot_dir: [f32; 3],
    /// linear, quadratic, constant
    pub atten: [f32; 3],
    pub spot_on: u32,
    pub spot_exp: f32,
    /// cos(GL_SPOT_CUTOFF).
    pub spot_cos_cutoff: f32,
    pub spot_table: u32,
    pub spot_scale: f32,
    pub spot_start: f32,
    pub spot: [f32; SPOT_N],
    /// Next enabled light (-1 = end), from `Lighting::head`.
    pub next: i32,
}

/// Lighting and fog state. Plain data, valid when zeroed (see `init`).
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Lighting {
    pub on: u32,
    pub two_sided: u32,
    pub local_viewer: u32,
    /// Scene ambient (with the lights' ambient folded in when
    /// `ambient_folded`).
    pub amb_sum: [f32; 3],
    pub ambient_folded: u32,
    pub mat: [Material; 2],
    pub lights: [Light; MAX_LIGHTS],
    pub head: i32,
    pub count: u32,
    /// Colour material: param 1 emission 2 ambient 3 diffuse 4 specular 5
    /// ambient+diffuse; face 0 front 1 back 2 both.
    pub cmat_param: u32,
    pub cmat_face: u32,
    pub cmat_on: u32,
    pub normal_matrix: [f32; 9],
    pub normalize: u32,
    pub fog_on: u32,
    /// Fog: mode (0 linear, 1 EXP, 2 EXP2), then linear: end, 1 / (end -
    /// start); EXP/EXP2: density; then the colour RGB.
    pub fog_mode: u32,
    pub fog_a: f32,
    pub fog_b: f32,
    pub fog_color: [f32; 3],
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `a` scaled to unit length (unchanged if ~0).
pub fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt();
    if l > 1e-20 { [a[0] / l, a[1] / l, a[2] / l] } else { a }
}

/// Table lookup as the GE7 does it: x mapped to (x - start) * scale,
/// linearly interpolated, 0 below the table.
fn table(t: &[f32], start: f32, scale: f32, x: f32) -> f32 {
    let p = (x - start) * scale;
    if !p.is_finite() || p < 0.0 {
        return 0.0;
    }
    let last = (t.len() - 1) as f32;
    if p >= last {
        return t[t.len() - 1];
    }
    let i = p.floor() as usize;
    let fr = p - i as f32;
    t[i] + (t[i + 1] - t[i]) * fr
}

impl Lighting {
    /// OpenGL's initial state (no lights enabled).
    pub fn init(&mut self) {
        self.head = -1;
        for l in self.lights.iter_mut() {
            l.next = -1;
            l.atten = [0.0, 0.0, 1.0];
            l.pos = [0.0, 0.0, 1.0, 0.0];
            l.spot_dir = [0.0, 0.0, -1.0];
            l.spot_cos_cutoff = -1.0;
        }
        self.lights[0].diffuse = [1.0; 3];
        self.lights[0].specular = [1.0; 3];
        self.amb_sum = [0.2; 3];
        self.normal_matrix = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        for m in self.mat.iter_mut() {
            m.ambient = [0.2; 3];
            m.diffuse = [0.8, 0.8, 0.8, 1.0];
            m.spec_scale = 1.0;
        }
        self.fog_a = 1.0;
        self.fog_b = 1.0;
    }

    /// Enabled lights, in the order of the `head` / `next` chain.
    pub fn enabled(&self) -> impl Iterator<Item = &Light> {
        let mut i = self.head;
        let mut left = self.count.min(MAX_LIGHTS as u32);
        std::iter::from_fn(move || {
            if left == 0 || !(0..MAX_LIGHTS as i32).contains(&i) {
                return None;
            }
            left -= 1;
            let l = &self.lights[i as usize];
            i = l.next;
            Some(l)
        })
    }

    /// Colour material: while tracking is on, a colour command writes the
    /// colour into the tracked properties of the active material, and the
    /// change lasts until the material is loaded again (GL and IRIS GL
    /// lmcolor alike).
    pub fn track_color(&mut self, color: [f32; 4]) {
        if self.cmat_on == 0 {
            return;
        }
        let c3 = [color[0], color[1], color[2]];
        for face in 0..2 {
            if self.cmat_face != 2 && self.cmat_face as usize != face {
                continue;
            }
            let m = &mut self.mat[face];
            match self.cmat_param {
                1 => m.emission = c3,
                2 => m.ambient = c3,
                3 => m.diffuse = color,
                4 => m.specular = c3,
                5 => {
                    m.ambient = c3;
                    m.diffuse = color;
                }
                _ => {}
            }
        }
    }

    /// Lit colours (front, back) of a vertex: `eye` = eye-space position,
    /// `n_obj` = current normal, `color` = current colour.
    pub fn light_vertex(&self, eye: [f32; 4], n_obj: [f32; 3], color: [f32; 4]) -> ([f32; 4], [f32; 4]) {
        let nm = &self.normal_matrix;
        let mut n = [
            nm[0] * n_obj[0] + nm[3] * n_obj[1] + nm[6] * n_obj[2],
            nm[1] * n_obj[0] + nm[4] * n_obj[1] + nm[7] * n_obj[2],
            nm[2] * n_obj[0] + nm[5] * n_obj[1] + nm[8] * n_obj[2],
        ];
        if self.normalize != 0 {
            n = norm(n);
        }
        let p = if eye[3] != 0.0 { [eye[0] / eye[3], eye[1] / eye[3], eye[2] / eye[3]] } else { [eye[0], eye[1], eye[2]] };
        let front = self.shade(0, n, p, color);
        let back = if self.two_sided != 0 { self.shade(1, [-n[0], -n[1], -n[2]], p, color) } else { front };
        (front, back)
    }

    fn shade(&self, face: usize, n: [f32; 3], p: [f32; 3], _color: [f32; 4]) -> [f32; 4] {
        let m = &self.mat[face];
        let per_light_amb = self.ambient_folded == 0;
        let mut c = [0.0f32; 3];
        for k in 0..3 {
            c[k] = m.emission[k] + m.ambient[k] * self.amb_sum[k];
        }
        let v = if self.local_viewer != 0 { norm([-p[0], -p[1], -p[2]]) } else { [0.0, 0.0, 1.0] };
        for l in self.enabled() {
            let (lv, mut att) = if l.pos[3] == 0.0 {
                (norm([l.pos[0], l.pos[1], l.pos[2]]), 1.0)
            } else {
                let d = [l.pos[0] - p[0], l.pos[1] - p[1], l.pos[2] - p[2]];
                let dist = dot(d, d).sqrt();
                let k = l.atten[2] + l.atten[0] * dist + l.atten[1] * dist * dist;
                (norm(d), if k > 0.0 { 1.0 / k } else { 1.0 })
            };
            if l.spot_on != 0 {
                let cos = -dot(lv, l.spot_dir);
                att *= if l.spot_table != 0 {
                    table(&l.spot, l.spot_start, l.spot_scale, cos)
                } else if cos < l.spot_cos_cutoff {
                    0.0
                } else {
                    cos.max(0.0).powf(l.spot_exp)
                };
            }
            if att == 0.0 {
                continue;
            }
            let ndl = dot(n, lv);
            for k in 0..3 {
                let mut t = 0.0;
                if per_light_amb {
                    t += l.ambient[k] * m.ambient[k];
                }
                if ndl > 0.0 {
                    t += ndl * l.diffuse[k] * m.diffuse[k];
                }
                c[k] += att * t;
            }
            if ndl > 0.0 {
                let h = norm([lv[0] + v[0], lv[1] + v[1], lv[2] + v[2]]);
                let ndh = dot(n, h);
                let s = if m.spec_table != 0 {
                    table(&m.spec, m.spec_start, m.spec_scale, ndh)
                } else {
                    ndh.max(0.0).powf(m.shininess)
                };
                for k in 0..3 {
                    c[k] += att * s * l.specular[k] * m.specular[k];
                }
            }
        }
        [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0), m.diffuse[3].clamp(0.0, 1.0)]
    }

    /// Fog factor for eye-space distance `z` (>= 0): 1 = no fog.
    pub fn fog_factor(&self, z: f32) -> f32 {
        let f = match self.fog_mode {
            0 => (self.fog_a - z) * self.fog_b,
            1 => (-self.fog_a * z).exp(),
            _ => (-(self.fog_a * z) * (self.fog_a * z)).exp(),
        };
        f.clamp(0.0, 1.0)
    }

    /// Apply fog to a colour for eye-space position `eye`.
    pub fn fog_color(&self, eye: [f32; 4], c: [f32; 4]) -> [f32; 4] {
        if self.fog_on == 0 {
            return c;
        }
        let z = if eye[3] != 0.0 { (eye[2] / eye[3]).abs() } else { eye[2].abs() };
        let fct = self.fog_factor(z);
        let fc = self.fog_color;
        [
            fct * c[0] + (1.0 - fct) * fc[0],
            fct * c[1] + (1.0 - fct) * fc[1],
            fct * c[2] + (1.0 - fct) * fc[2],
            c[3],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit() -> Lighting {
        let mut lt: Lighting = unsafe { std::mem::zeroed() };
        lt.init();
        lt.on = 1;
        lt.head = 0;
        lt.count = 1;
        lt
    }

    /// The default light 0 (white, from +z) on a face looking at it: the
    /// material's diffuse plus its ambient times the scene and light
    /// ambient (OpenGL 1.1 initial state: 0.2 * 0.2 + 0.8).
    #[test]
    fn default_light_on_a_facing_normal() {
        let lt = lit();
        let (c, _) = lt.light_vertex([0.0, 0.0, -5.0, 1.0], [0.0, 0.0, 1.0], [1.0; 4]);
        for k in 0..3 {
            assert!((c[k] - 0.84).abs() < 1e-5, "{c:?}");
        }
    }

    #[test]
    fn linear_fog_half_way() {
        let mut lt = lit();
        lt.fog_on = 1;
        (lt.fog_mode, lt.fog_a, lt.fog_b, lt.fog_color) = (0, 10.0, 0.1, [1.0, 1.0, 1.0]);
        let c = lt.fog_color([0.0, 0.0, -5.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
        assert!((c[0] - 0.5).abs() < 1e-5);
    }
}
