//! GE7 lighting and fog (HLE), fed by libglcore EXPRESS (gr2_lighting.c,
//! gr2_fog.c). See HQ2.h "OpenGL token addressing" for the token formats.
//!
//! libglcore loads the GE's lighting state through:
//! - key/value parameter pairs on 0x080 (GE parameter address, value; both
//!   floats): 4 = first enabled light, 5 = select light slot, 6 = next
//!   light of the selected slot (-1 = end), 10 = number of enabled lights,
//!   2/3 = front specular table scale/start, 24/25 = back, 7 = table slot for
//!   the next table upload (0 front specular, 1 back, 2 + n spot of light
//!   n), 29/30 = spot table scale/start, 31 = light owning the spot table
//!   (-1 = the selected light has no spot);
//! - light-model pairs on 0x085: 6 = local viewer (3/4/5 select GE lighting
//!   variants; not needed here, see `per_light_ambient`);
//! - tables: specular (128 floats, 0x074) and spot (32 floats, 0x086) of
//!   pow(x, exponent) for x = start + i / scale; entry 0 is 0;
//! - per selected light: ambient 0x0E5, diffuse 0x101, specular 0x102 (RGB),
//!   position 0x07F (eye space; w = 0: normalized direction to the light),
//!   spot direction 0x084, attenuation 0x109 (linear, quadratic, constant);
//! - materials (front / back): emission 0x076/0x077, ambient 0x078/0x079,
//!   diffuse RGBA 0x07A/0x07B, specular 0x07C/0x07D;
//! - 0x075: scene ambient = model ambient, plus the sum of the enabled
//!   lights' ambient when there are no local lights, no local viewer and no
//!   two-sided lighting (__glExpLoadAmbientSum);
//! - 0x081: colour material (param 1 emission 2 ambient 3 diffuse 4 specular
//!   5 ambient+diffuse; face 0 front 1 back 2 both; enable);
//! - 0x10E: 0.0; DATA lighting on; 0x0D4: 1.0; DATA two-sided;
//! - IRIS GL (libgl.so) sends one light colour, 0x07E (RGB), for diffuse and
//!   specular (LCOLOR) instead of 0x101/0x102;
//! - fog: 0x027 enable, 0x028 = mode (0 linear, 1 EXP, 2 EXP2), a, b, RGB.
//!
//! The per-vertex maths is the OpenGL 1.1 lighting equation evaluated with
//! those parameters; specular and spot factors use the uploaded tables the
//! way the GE does (the exponents are only known through them).

use super::f;

pub const T_LIGHT_PARAM: u32 = 0x080;
pub const T_MODEL_PARAM: u32 = 0x085;
pub const T_SPEC_TABLE: u32 = 0x074;
pub const T_SPOT_TABLE: u32 = 0x086;
pub const T_AMBIENT_SUM: u32 = 0x075;
pub const T_MAT_FIRST: u32 = 0x076;
pub const T_MAT_LAST: u32 = 0x07d;
pub const T_LIGHT_AMBIENT: u32 = 0x0e5;
pub const T_LIGHT_DIFFUSE: u32 = 0x101;
pub const T_LIGHT_SPECULAR: u32 = 0x102;
/// IRIS GL light colour (LCOLOR): diffuse and specular of the selected light.
pub const T_IRIS_LCOLOR: u32 = 0x07e;
pub const T_LIGHT_POSITION: u32 = 0x07f;
pub const T_SPOT_DIRECTION: u32 = 0x084;
pub const T_ATTENUATION: u32 = 0x109;
pub const T_COLOR_MATERIAL: u32 = 0x081;
pub const T_MATERIAL_COMMIT: u32 = 0x083;
pub const T_LIGHTING: u32 = 0x10e;
pub const T_TWO_SIDED: u32 = 0x0d4;
pub const T_NORMALIZE: u32 = 0x0e3;
pub const T_NORMALIZE_B: u32 = 0x082;
pub const T_FOG_ON: u32 = 0x027;
pub const T_FOG: u32 = 0x028;
/// Scale libglcore applies to the EXP fog density (__glExpPassFog).
const FOG_EXP_SCALE: f32 = 0.180_464_26;

pub const MAX_LIGHTS: usize = 8;
const SPEC_N: usize = 128;
const SPOT_N: usize = 32;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Material {
    pub emission: [f32; 3],
    pub ambient: [f32; 3],
    pub diffuse: [f32; 4],
    pub specular: [f32; 3],
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
    pub pos: [f32; 4],
    pub spot_dir: [f32; 3],
    /// linear, quadratic, constant
    pub atten: [f32; 3],
    pub spot_on: u32,
    pub spot_scale: f32,
    pub spot_start: f32,
    pub spot: [f32; SPOT_N],
    pub next: i32,
}

/// Lighting and fog state. Plain data, valid when zeroed (see `init`).
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Lighting {
    pub on: u32,
    pub two_sided: u32,
    pub local_viewer: u32,
    pub amb_sum: [f32; 3],
    pub mat: [Material; 2],
    pub lights: [Light; MAX_LIGHTS],
    pub head: i32,
    pub count: u32,
    cur: i32,
    table_slot: i32,
    table_pos: u32,
    /// Light that owns the spot table parameters being loaded.
    spot_light: i32,
    pub cmat_param: u32,
    pub cmat_face: u32,
    pub cmat_on: u32,
    pub normal_matrix: [f32; 9],
    pub normalize: u32,
    pub fog_on: u32,
    pub fog: [f32; 6],
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt();
    if l > 1e-20 { [a[0] / l, a[1] / l, a[2] / l] } else { a }
}

/// Table lookup as the GE does it: x mapped to (x - start) * scale,
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
    pub fn init(&mut self) {
        self.head = -1;
        self.cur = -1;
        self.spot_light = -1;
        for l in self.lights.iter_mut() {
            l.next = -1;
            l.atten = [0.0, 0.0, 1.0];
            l.pos = [0.0, 0.0, 1.0, 0.0];
        }
        self.normal_matrix = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        for m in self.mat.iter_mut() {
            m.ambient = [0.2; 3];
            m.diffuse = [0.8, 0.8, 0.8, 1.0];
            m.spec_scale = 1.0;
        }
    }

    fn light_mut(&mut self, i: i32) -> Option<&mut Light> {
        if (0..MAX_LIGHTS as i32).contains(&i) { Some(&mut self.lights[i as usize]) } else { None }
    }

    /// Words a lighting/fog port takes (None: not a lighting port).
    pub fn port_words(tok: u32) -> Option<u32> {
        Some(match tok {
            T_LIGHT_PARAM | T_MODEL_PARAM => 2,
            T_AMBIENT_SUM | T_LIGHT_AMBIENT | T_LIGHT_DIFFUSE | T_LIGHT_SPECULAR | T_SPOT_DIRECTION
            | T_ATTENUATION | T_COLOR_MATERIAL | T_IRIS_LCOLOR => 3,
            0x07a | 0x07b | T_LIGHT_POSITION => 4,
            T_MAT_FIRST..=T_MAT_LAST => 3,
            T_FOG => 6,
            T_SPEC_TABLE | T_SPOT_TABLE => 1,
            _ => return None,
        })
    }

    /// A complete port write (`b` holds `port_words(tok)` words).
    pub fn port(&mut self, tok: u32, b: &[u32]) {
        let v3 = |b: &[u32]| [f(b[0]), f(b[1]), f(b[2])];
        match tok {
            T_LIGHT_PARAM => self.param(f(b[0]), f(b[1])),
            T_MODEL_PARAM => {
                if f(b[0]) as i32 == 6 {
                    self.local_viewer = (f(b[1]) != 0.0) as u32;
                }
            }
            T_AMBIENT_SUM => self.amb_sum = v3(b),
            T_MAT_FIRST..=T_MAT_LAST => {
                let face = ((tok - T_MAT_FIRST) & 1) as usize;
                let m = &mut self.mat[face];
                match (tok - T_MAT_FIRST) / 2 {
                    0 => m.emission = v3(b),
                    1 => m.ambient = v3(b),
                    2 => m.diffuse = [f(b[0]), f(b[1]), f(b[2]), f(b[3])],
                    _ => m.specular = v3(b),
                }
            }
            T_LIGHT_AMBIENT | T_LIGHT_DIFFUSE | T_LIGHT_SPECULAR | T_LIGHT_POSITION | T_SPOT_DIRECTION
            | T_ATTENUATION | T_IRIS_LCOLOR => {
                let cur = self.cur;
                if let Some(l) = self.light_mut(cur) {
                    match tok {
                        T_LIGHT_AMBIENT => l.ambient = v3(b),
                        T_LIGHT_DIFFUSE => l.diffuse = v3(b),
                        T_LIGHT_SPECULAR => l.specular = v3(b),
                        T_LIGHT_POSITION => l.pos = [f(b[0]), f(b[1]), f(b[2]), f(b[3])],
                        T_SPOT_DIRECTION => l.spot_dir = norm(v3(b)),
                        T_IRIS_LCOLOR => {
                            l.diffuse = v3(b);
                            l.specular = v3(b);
                        }
                        _ => l.atten = v3(b),
                    }
                }
            }
            T_COLOR_MATERIAL => {
                self.cmat_param = b[0];
                self.cmat_face = b[1];
                self.cmat_on = b[2] & 1;
            }
            T_FOG => {
                for (k, w) in b.iter().enumerate().take(6) {
                    self.fog[k] = f(*w);
                }
            }
            T_SPEC_TABLE | T_SPOT_TABLE => {
                let (slot, pos) = (self.table_slot, self.table_pos as usize);
                self.table_pos += 1;
                let v = f(b[0]);
                if tok == T_SPEC_TABLE && (0..2).contains(&slot) && pos < SPEC_N {
                    self.mat[slot as usize].spec[pos] = v;
                } else if tok == T_SPOT_TABLE && pos < SPOT_N {
                    if let Some(l) = self.light_mut(slot - 2) {
                        l.spot[pos] = v;
                    }
                }
            }
            _ => {}
        }
    }

    /// GE parameter write through 0x080.
    fn param(&mut self, addr: f32, val: f32) {
        let (a, vi) = (addr as i32, val as i32);
        match a {
            2 => self.mat[0].spec_scale = val,
            3 => self.mat[0].spec_start = val,
            24 => self.mat[1].spec_scale = val,
            25 => self.mat[1].spec_start = val,
            4 => self.head = vi,
            5 => self.cur = vi,
            6 => {
                let cur = self.cur;
                if let Some(l) = self.light_mut(cur) {
                    l.next = vi;
                }
            }
            7 => {
                self.table_slot = vi;
                self.table_pos = 0;
            }
            10 => self.count = vi.max(0) as u32,
            31 => {
                self.spot_light = vi;
                if vi < 0 {
                    let cur = self.cur;
                    if let Some(l) = self.light_mut(cur) {
                        l.spot_on = 0;
                    }
                } else if let Some(l) = self.light_mut(vi) {
                    l.spot_on = 1;
                }
            }
            29 | 30 => {
                let s = self.spot_light;
                if let Some(l) = self.light_mut(s) {
                    if a == 29 { l.spot_scale = val } else { l.spot_start = val }
                }
            }
            _ => {}
        }
    }

    /// Single-word lighting commands. Returns false if not one.
    pub fn command(&mut self, tok: u32, a: &[u32]) -> bool {
        match tok {
            T_LIGHTING => self.on = a[1] & 1,
            T_TWO_SIDED => self.two_sided = a[1] & 1,
            T_NORMALIZE | T_NORMALIZE_B => self.normalize = a[0] & 1,
            T_FOG_ON => self.fog_on = a[0] & 1,
            T_MATERIAL_COMMIT => {}
            _ => return false,
        }
        true
    }

    /// Enabled lights in the GE's list order.
    fn enabled(&self) -> impl Iterator<Item = &Light> {
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

    /// libglcore folds the lights' ambient into 0x075 unless there are local
    /// lights, a local viewer or two-sided lighting (__glExpLoadAmbientSum).
    fn per_light_ambient(&self) -> bool {
        self.local_viewer != 0 || self.two_sided != 0 || self.enabled().any(|l| l.pos[3] != 0.0)
    }

    /// Material of `face` with colour material applied.
    fn material(&self, face: usize, color: [f32; 4]) -> Material {
        let mut m = self.mat[face];
        if self.cmat_on != 0 && (self.cmat_face == 2 || self.cmat_face as usize == face) {
            let c3 = [color[0], color[1], color[2]];
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
        m
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

    fn shade(&self, face: usize, n: [f32; 3], p: [f32; 3], color: [f32; 4]) -> [f32; 4] {
        let m = self.material(face, color);
        let per_light_amb = self.per_light_ambient();
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
                att *= table(&l.spot, l.spot_start, l.spot_scale, cos);
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
                let s = table(&m.spec, m.spec_start, m.spec_scale, dot(n, h));
                for k in 0..3 {
                    c[k] += att * s * l.specular[k] * m.specular[k];
                }
            }
        }
        [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0), m.diffuse[3].clamp(0.0, 1.0)]
    }

    /// Fog factor for eye-space distance `z` (>= 0): 1 = no fog.
    fn fog_factor(&self, z: f32) -> f32 {
        let fg = &self.fog;
        let mode = fg[0].to_bits();
        let f = match mode {
            0 => (fg[1] - z) * fg[2],
            1 => (-(fg[1] / FOG_EXP_SCALE) * z).exp(),
            // EXP2: density scale constant not yet confirmed (unverified).
            _ => {
                let d = fg[1] / FOG_EXP_SCALE;
                (-(d * z) * (d * z)).exp()
            }
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
        let fc = [self.fog[3], self.fog[4], self.fog[5]];
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

    fn fw(v: f32) -> u32 {
        v.to_bits()
    }

    /// IRIS GL atlantis (libgl.so): light 0 is set up with 0x080 (5, 0),
    /// 0x07E colour (1, 1, 1) and 0x07F position (0, 1, 0, 0); without 0x07E
    /// the light contributed nothing and the fish were drawn with the scene
    /// ambient only.
    #[test]
    fn iris_lcolor_sets_diffuse_and_specular() {
        let mut lt: Lighting = unsafe { std::mem::zeroed() };
        lt.init();
        lt.port(T_LIGHT_PARAM, &[fw(4.0), fw(0.0)]);
        lt.port(T_LIGHT_PARAM, &[fw(10.0), fw(1.0)]);
        lt.port(T_LIGHT_PARAM, &[fw(5.0), fw(0.0)]);
        lt.port(T_IRIS_LCOLOR, &[fw(1.0), fw(1.0), fw(1.0)]);
        lt.port(T_LIGHT_POSITION, &[fw(0.0), fw(1.0), fw(0.0), fw(0.0)]);
        lt.port(0x07a, &[fw(0.46), fw(0.66), fw(0.795), fw(1.0)]);
        lt.port(0x078, &[fw(0.0), fw(0.1), fw(0.2)]);
        lt.port(T_AMBIENT_SUM, &[fw(0.4), fw(0.4), fw(0.4)]);
        assert_eq!(lt.lights[0].specular, [1.0; 3]);
        let (c, _) = lt.light_vertex([0.0, 0.0, -5.0, 1.0], [0.0, 1.0, 0.0], [1.0; 4]);
        let want = [0.46, 0.66 + 0.04, 0.795 + 0.08];
        for k in 0..3 {
            assert!((c[k] - want[k]).abs() < 1e-4, "channel {k}: {c:?} vs {want:?}");
        }
    }
}
