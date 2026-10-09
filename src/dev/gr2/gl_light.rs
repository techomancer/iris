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
//! This file is the loader: it decodes those ports into the shared GL
//! core's `Lighting` (`crate::dev::gl::light`), which evaluates the OpenGL
//! 1.1 lighting equation per vertex. On GR2 the specular and spot factors
//! always come from the uploaded tables (the exponents are only known
//! through them), and the scene ambient is folded the way
//! __glExpLoadAmbientSum decides (`refold`).

use super::f;
use crate::dev::gl::light::{norm, Light, Lighting, MAX_LIGHTS, SPEC_N, SPOT_N};

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


/// Where the GE's lighting loads are going: the selected light (0x080 5),
/// the table being uploaded and how far (0x080 7), and the light owning the
/// spot table parameters (0x080 31). Plain data, valid zeroed after `init`.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Loader {
    cur: i32,
    table_slot: i32,
    table_pos: u32,
    spot_light: i32,
}

fn light_mut(lt: &mut Lighting, i: i32) -> Option<&mut Light> {
    if (0..MAX_LIGHTS as i32).contains(&i) { Some(&mut lt.lights[i as usize]) } else { None }
}

/// libglcore folds the lights' ambient into 0x075 unless there are local
/// lights, a local viewer or two-sided lighting (__glExpLoadAmbientSum).
/// Kept current after every load, since any of those can change.
fn refold(lt: &mut Lighting) {
    let per_light = lt.local_viewer != 0 || lt.two_sided != 0 || lt.enabled().any(|l| l.pos[3] != 0.0);
    lt.ambient_folded = (!per_light) as u32;
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

/// Single-word lighting commands. Returns false if not one.
pub fn command(lt: &mut Lighting, tok: u32, a: &[u32]) -> bool {
    match tok {
        T_LIGHTING => lt.on = a[1] & 1,
        T_TWO_SIDED => lt.two_sided = a[1] & 1,
        T_NORMALIZE | T_NORMALIZE_B => lt.normalize = a[0] & 1,
        T_FOG_ON => lt.fog_on = a[0] & 1,
        T_MATERIAL_COMMIT => {}
        _ => return false,
    }
    refold(lt);
    true
}

impl Loader {
    /// `lt` freshly `Lighting::init`ed: the GE's starting state. Unlike
    /// OpenGL's initial state the scene ambient and light 0's colours start
    /// at 0 (libglcore loads them; kept as the HLE has always had them).
    pub fn init(&mut self, lt: &mut Lighting) {
        self.cur = -1;
        self.table_slot = 0;
        self.table_pos = 0;
        self.spot_light = -1;
        lt.amb_sum = [0.0; 3];
        lt.lights[0].diffuse = [0.0; 3];
        lt.lights[0].specular = [0.0; 3];
        for m in lt.mat.iter_mut() {
            m.spec_table = 1;
        }
        for l in lt.lights.iter_mut() {
            l.spot_table = 1;
        }
        refold(lt);
    }

    /// A complete port write (`b` holds `port_words(tok)` words).
    pub fn port(&mut self, lt: &mut Lighting, tok: u32, b: &[u32]) {
        let v3 = |b: &[u32]| [f(b[0]), f(b[1]), f(b[2])];
        match tok {
            T_LIGHT_PARAM => self.param(lt, f(b[0]), f(b[1])),
            T_MODEL_PARAM => {
                if f(b[0]) as i32 == 6 {
                    lt.local_viewer = (f(b[1]) != 0.0) as u32;
                }
            }
            T_AMBIENT_SUM => lt.amb_sum = v3(b),
            T_MAT_FIRST..=T_MAT_LAST => {
                let face = ((tok - T_MAT_FIRST) & 1) as usize;
                let m = &mut lt.mat[face];
                match (tok - T_MAT_FIRST) / 2 {
                    0 => m.emission = v3(b),
                    1 => m.ambient = v3(b),
                    2 => m.diffuse = [f(b[0]), f(b[1]), f(b[2]), f(b[3])],
                    _ => m.specular = v3(b),
                }
            }
            T_LIGHT_AMBIENT | T_LIGHT_DIFFUSE | T_LIGHT_SPECULAR | T_LIGHT_POSITION | T_SPOT_DIRECTION
            | T_ATTENUATION | T_IRIS_LCOLOR => {
                if let Some(l) = light_mut(lt, self.cur) {
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
                lt.cmat_param = b[0];
                lt.cmat_face = b[1];
                lt.cmat_on = b[2] & 1;
            }
            // Mode (an integer), then linear: end, 1 / (end - start);
            // EXP/EXP2: the density as libglcore scales it.
            T_FOG => {
                lt.fog_mode = b[0];
                let a = f(b[1]);
                // EXP2: density scale constant not yet confirmed (unverified).
                lt.fog_a = if b[0] == 0 { a } else { a / FOG_EXP_SCALE };
                lt.fog_b = f(b[2]);
                lt.fog_color = v3(&b[3..]);
            }
            T_SPEC_TABLE | T_SPOT_TABLE => {
                let (slot, pos) = (self.table_slot, self.table_pos as usize);
                self.table_pos += 1;
                let v = f(b[0]);
                if tok == T_SPEC_TABLE && (0..2).contains(&slot) && pos < SPEC_N {
                    lt.mat[slot as usize].spec[pos] = v;
                } else if tok == T_SPOT_TABLE && pos < SPOT_N {
                    if let Some(l) = light_mut(lt, slot - 2) {
                        l.spot[pos] = v;
                    }
                }
            }
            _ => {}
        }
        refold(lt);
    }

    /// GE parameter write through 0x080.
    fn param(&mut self, lt: &mut Lighting, addr: f32, val: f32) {
        let (a, vi) = (addr as i32, val as i32);
        match a {
            2 => lt.mat[0].spec_scale = val,
            3 => lt.mat[0].spec_start = val,
            24 => lt.mat[1].spec_scale = val,
            25 => lt.mat[1].spec_start = val,
            4 => lt.head = vi,
            5 => self.cur = vi,
            6 => {
                if let Some(l) = light_mut(lt, self.cur) {
                    l.next = vi;
                }
            }
            7 => {
                self.table_slot = vi;
                self.table_pos = 0;
            }
            10 => lt.count = vi.max(0) as u32,
            31 => {
                self.spot_light = vi;
                if vi < 0 {
                    if let Some(l) = light_mut(lt, self.cur) {
                        l.spot_on = 0;
                    }
                } else if let Some(l) = light_mut(lt, vi) {
                    l.spot_on = 1;
                }
            }
            29 | 30 => {
                if let Some(l) = light_mut(lt, self.spot_light) {
                    if a == 29 { l.spot_scale = val } else { l.spot_start = val }
                }
            }
            _ => {}
        }
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
        let mut ld: Loader = unsafe { std::mem::zeroed() };
        lt.init();
        ld.init(&mut lt);
        ld.port(&mut lt, T_LIGHT_PARAM, &[fw(4.0), fw(0.0)]);
        ld.port(&mut lt, T_LIGHT_PARAM, &[fw(10.0), fw(1.0)]);
        ld.port(&mut lt, T_LIGHT_PARAM, &[fw(5.0), fw(0.0)]);
        ld.port(&mut lt, T_IRIS_LCOLOR, &[fw(1.0), fw(1.0), fw(1.0)]);
        ld.port(&mut lt, T_LIGHT_POSITION, &[fw(0.0), fw(1.0), fw(0.0), fw(0.0)]);
        ld.port(&mut lt, 0x07a, &[fw(0.46), fw(0.66), fw(0.795), fw(1.0)]);
        ld.port(&mut lt, 0x078, &[fw(0.0), fw(0.1), fw(0.2)]);
        ld.port(&mut lt, T_AMBIENT_SUM, &[fw(0.4), fw(0.4), fw(0.4)]);
        assert_eq!(lt.lights[0].specular, [1.0; 3]);
        let (c, _) = lt.light_vertex([0.0, 0.0, -5.0, 1.0], [0.0, 1.0, 0.0], [1.0; 4]);
        let want = [0.46, 0.66 + 0.04, 0.795 + 0.08];
        for k in 0..3 {
            assert!((c[k] - want[k]).abs() < 1e-4, "channel {k}: {c:?} vs {want:?}");
        }
    }
}
