//! The GL client's commands: state into the context, triangles out.
//!
//! This is the GL half of what the geometry engines do with the stream. State
//! commands (`state`) update the context's `Gl`; a begin opens a primitive,
//! vertex commands transform, light and assemble it, and an end closes it.
//! Each primitive is clipped in clip space against the view volume by the GL
//! core (`crate::dev::gl`, with `vertex::KonaGeometry`), divided, put through
//! the viewport and quantised to the vertex bus's precision (`WVertex`), and
//! appended to a `Prims` as triangles with the draw state they run under. There
//! is no clip in window space: x and y are cut only at a guard band, and set-up
//! cuts the rest with the job's rectangle.
//!
//! Assembly. Triangles, strips, fans, quads, quad strips and polygons are broken
//! into independent triangles, a polygon of more than three vertices into one
//! fan from its first. That the engines split polygons this way is inferred;
//! what is established is that the bus carries strips and fans. Points and
//! lines are clipped and then drawn as triangles (a point as a square or a
//! disc, a line as a rectangle about its segment, with the distance along it for
//! the line stipple). A polygon whose face is in the line or point mode is drawn
//! as lines between its boundary-edge vertices or as points at them.
//!
//! Flat shading takes the colour of the vertex GL names (the last of a
//! triangle, the first of a polygon). Two-sided lighting picks the back colour
//! by the polygon's projected orientation. Culling is by that orientation
//! too. The half-pixel offset is not applied here (see `raster::setup`).
//!
//! The double forms of the colour, normal, texture-coordinate and vertex tokens
//! are narrowed to single precision once, as the command arrives, and run the
//! float forms' path (see `narrow`).
//!
//! User clip planes are tested in eye space with the core's clip planes 6 to 11,
//! along with the view volume, and a plane cuts a primitive wherever the view
//! volume would (see `GlPath::begin`).
//!
//! Texture coordinates are the current ones, or, for the coordinates enabled for
//! generation, computed from the vertex (`crate::dev::gl::texgen`), and then go
//! through the texture matrix (inferred: no source states the
//! arithmetic). The set of generated coordinates is read once, at
//! a primitive's begin.
//!
//! Not taken, counted by selector: the colour-index double form, the
//! generation parameters of the point and line distance planes, colour material by token, the
//! IRIS GL compatibility tokens, vertex arrays, display-list tokens and the
//! texture commands other than the coordinate and the environment's mode and
//! colour. Fog is taken: the state is kept, each vertex carries its fog
//! coordinate and the per-pixel stage applies it. Texture loads and binds are
//! not this module's: the back end hands them to `texture`, and a bound 1D or 2D
//! texture is sampled in `raster`.

pub mod pixels;
pub mod query;
pub mod state;
pub mod texture;
pub mod vertex;

use crate::dev::gl::math;
use crate::dev::kona::defs::{gl, *};
use crate::dev::gl::vertex::{Clip, MAX_POLY};
use super::raster::{ClearReq, Prims, Rect, SmoothPoint, SmoothPoly, WVertex};
use state::{f, Gl, Taken};
use std::collections::BTreeMap;
use vertex::{colour13, colour_word, depth32, grid, project, KVertex, KonaGeometry};

pub const BEGIN_FIRST: u32 = BEGIN_PRIMITIVE_POINTS;
pub const BEGIN_LAST: u32 = BEGIN_PRIMITIVE_POLYGON;
pub const END: u32 = END_OPEN_PRIMITIVE;
/// Selectors of every texture-coordinate form the front end frames.
pub const TEXCOORD: [u32; 5] = [SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SINGLE_FLOAT, SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT, SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SIGNED_SHORT, SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT, SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT];

/// Whether `selector` is one of the commands `GlPath::command` is given.
pub fn is_geometry(selector: u32) -> bool {
    matches!(selector, CLEAR_SELECTED_BUFFERS | SET_COLOUR_CLEAR_VALUE | SET_DEPTH_CLEAR_VALUE | SELECT_DRAW_BUFFER | SELECT_CURRENT_MATRIX_STACK..=TRANSLATE_FLOATS | SET_VIEWPORT | SET_SCISSOR_BOX | ENABLE_CAPABILITY | DISABLE_CAPABILITY | SET_FLAT_OR_SMOOTH_SHADING | SET_FACE_CULLING | SET_FRONT_FACE_WINDING | SET_DEPTH_TEST | SET_ALPHA_TEST | SET_STENCIL_TEST | SET_STENCIL_OPERATIONS | SET_STENCIL_WRITE_MASK | SET_BLEND_FACTORS | SET_CONSTANT_BLEND_COLOUR | SET_LOGIC_OPERATION | SET_COLOUR_WRITE_MASK | SET_DEPTH_WRITE_MASK | SET_POLYGON_OFFSET | SET_STENCIL_CLEAR_VALUE | SET_LINE_WIDTH | SET_POINT_SIZE | SET_LINE_STIPPLE | SET_MULTISAMPLE_PATTERN | SET_MULTISAMPLE_COVERAGE_MASK | SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM | SET_TEXTURE_ENVIRONMENT_PARAMETER_INTEGER_FORM | SET_LIGHT_PARAMETER_FLOAT_FORM | SET_LIGHT_PARAMETER_INTEGER_FORM | SET_LIGHTING_MODEL_PARAMETER_FLOAT_FORM | SET_LIGHTING_MODEL_PARAMETER_INTEGER_FORM | SET_MATERIAL_PARAMETER_FLOAT_FORM | BEGIN_FIRST..=BEGIN_LAST | END | SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_BYTE..=SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_32_BIT_INTEGER | SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_BYTE..=SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_SHORT | SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SIGNED_SHORT | SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_DOUBLE | SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_DOUBLE | SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_DOUBLE | SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_DOUBLE | SUBMIT_VERTEX_2_COMPONENTS_DOUBLE..=SUBMIT_VERTEX_4_COMPONENTS_SIGNED_SHORT | SET_USER_CLIP_PLANE | TEXTURE_COORDINATE_GENERATION_FLOAT_FORM | SET_CURRENT_EDGE_FLAG)
}

/// The float form a double-precision colour, normal, texture-coordinate or vertex token runs as,
/// with its words narrowed to single precision. A double is two words, the high one first (the
/// order of the depth clear's double, established; for these tokens inferred); the word counts
/// are established for the texture coordinates and inferred for the colour, normal and
/// vertex, and a count that differs is not narrowed. The
/// colour double is narrowed and not scaled or clamped at entry (established; the clamp to 0 to 1 comes later, in the
/// colour-clamp pass).
fn narrow(selector: u32, a: &[u32]) -> Option<(u32, [u32; 4])> {
    let float_form = match selector {
        SET_CURRENT_COLOUR_3_COMPONENTS_DOUBLE => SET_CURRENT_COLOUR_3_COMPONENTS_SINGLE_FLOAT,
        SET_CURRENT_COLOUR_4_COMPONENTS_DOUBLE => SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT,
        SET_CURRENT_NORMAL_3_COMPONENTS_DOUBLE => SET_CURRENT_NORMAL_3_COMPONENTS_SINGLE_FLOAT,
        SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_DOUBLE => SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SINGLE_FLOAT,
        SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_DOUBLE => SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT,
        SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_DOUBLE => SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT,
        SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_DOUBLE => SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT,
        SUBMIT_VERTEX_2_COMPONENTS_DOUBLE => SUBMIT_VERTEX_2_COMPONENTS_SINGLE_FLOAT,
        SUBMIT_VERTEX_3_COMPONENTS_DOUBLE => SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT,
        SUBMIT_VERTEX_4_COMPONENTS_DOUBLE => SUBMIT_VERTEX_4_COMPONENTS_SINGLE_FLOAT,
        _ => return None,
    };
    if a.len() % 2 != 0 || a.len() > 8 {
        return None;
    }
    let mut out = [0u32; 4];
    for (o, w) in out.iter_mut().zip(a.chunks_exact(2)) {
        *o = (f64::from_bits((w[0] as u64) << 32 | w[1] as u64) as f32).to_bits();
    }
    Some((float_form, out))
}

/// A vertex as feedback reports it: window position, depth as a fraction, 1 / w, colour and texture.
fn feed_vertex(gl: &Gl, v: &KVertex) -> query::Fv {
    let [vx, vy, vw, vh] = gl.raster.viewport;
    let [dn, df] = gl.raster.depth_range;
    let r = if v.h[3] != 0.0 { 1.0 / v.h[3] } else { 0.0 };
    query::Fv { x: vx + (v.h[0] * r + 1.0) * 0.5 * vw, y: vy + (v.h[1] * r + 1.0) * 0.5 * vh, z: dn + (v.h[2] * r * 0.5 + 0.5) * (df - dn), w: r, c: v.c, t: v.t }
}

/// `v` with every edge a boundary edge. A triangle or polygon takes its edges from the vertices' flags.
/// A quad strip's pieces take the constant flag word 0xc1400000 whatever the flags (the constant is
/// established, its meaning, both outer edges of each triangle on and the diagonal off, is inferred),
/// which draws a quad's four sides. A triangle strip and fan combine their vertices' words with no
/// per-edge mask, and which of their edges draw in line mode is unknown: the model draws every edge of
/// a piece, inner ones too (its own choice).
fn all_edges<const N: usize>(mut v: [KVertex; N]) -> [KVertex; N] {
    v.iter_mut().for_each(|w| w.edge = true);
    v
}

/// What the path did, for the monitor and the census.
#[derive(Clone, Debug, Default)]
pub struct GlCounts {
    pub commands: u64,
    /// Selectors seen that are geometry commands but whose form is not taken.
    pub unmodelled: BTreeMap<u32, u64>,
    /// glEnable and glDisable capabilities the path does not read.
    pub caps: BTreeMap<u32, u64>,
    pub errors: u64,
    pub begins: [u64; 10],
    pub texcoord: [u64; 5],
    pub vertices: u64,
    pub primitives: u64,
    pub clears: u64,
    /// Primitives the volume clip cut, that vanished in it, that had a vertex
    /// with no window position, and that culling dropped.
    pub clipped: u64,
    pub clipped_away: u64,
    pub no_window_position: u64,
    pub culled: u64,
    /// Points and lines that survived clipping and were handed to set-up as
    /// triangles, and lines of no length, which draw nothing.
    pub points: u64,
    pub lines: u64,
    pub empty_lines: u64,
    /// Window placements whose ID word is in none of the ranges the board encodes.
    pub odd_window_ids: u64,
    /// Vertices outside a vertex command's begin, which draw nothing.
    pub stray: u64,
    /// With `census` on: each vertex's place against the window, once.
    pub behind: u64,
    pub outside_volume: u64,
    pub outside_window: u64,
    pub inside: u64,
}

impl GlCounts {
    pub fn note(&mut self, selector: u32) {
        *self.unmodelled.entry(selector).or_default() += 1;
    }
}

/// The GL path: the scratch it assembles in.
pub struct GlPath {
    pub counts: GlCounts,
    /// Measure every vertex against the window (an extra divide each).
    #[cfg(test)]
    pub census: bool,
    /// The managed area, which a draw's clip rectangle is cut to.
    screen: Rect,
    clip: Clip,
    /// The primitive's vertices so far, and the strip's or fan's held ones.
    prim: Vec<KVertex>,
    /// Vertices of the open primitive that have been taken.
    seen: u32,
    /// For the open primitive: whether texture coordinates are generated, and whether that reads the
    /// normal. Decided at begin, so a vertex tests a flag and no more.
    texgen: bool,
    texgen_normal: bool,
}

impl GlPath {
    /// The managed area, which a draw's clip rectangle is cut to.
    pub fn screen(&self) -> Rect {
        self.screen
    }

    /// A path for a managed area of `screen` pixels, width and height.
    pub fn new(screen: (u32, u32)) -> GlPath {
        GlPath { counts: GlCounts::default(), #[cfg(test)] census: false, screen: Rect { x0: 0, y0: 0, x1: screen.0 as i32, y1: screen.1 as i32 }, clip: Clip::default(), prim: Vec::with_capacity(64), seen: 0, texgen: false, texgen_normal: false }
    }

    /// One GL command. `packed` is the opening word's top half, which the
    /// byte, short and packed forms carry data in.
    pub fn command(&mut self, gl: &mut Gl, selector: u32, a: &[u32], packed: u32, prims: &mut Prims) {
        self.counts.commands += 1;
        let narrowed = narrow(selector, a);
        let (selector, a) = match &narrowed {
            Some((float_form, w)) => (*float_form, &w[..a.len() / 2]),
            None => (selector, a),
        };
        let taken = match selector {
            BEGIN_FIRST..=BEGIN_LAST => {
                self.begin(gl, selector);
                Taken::Yes
            }
            END => {
                self.end(gl, prims);
                Taken::Yes
            }
            CLEAR_SELECTED_BUFFERS if a.len() == 1 => {
                self.clear(gl, a[0], prims);
                Taken::Yes
            }
            SUBMIT_VERTEX_2_COMPONENTS_DOUBLE..=SUBMIT_VERTEX_4_COMPONENTS_SIGNED_SHORT => self.vertex(gl, selector, a, packed, prims),
            SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_BYTE..=SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_32_BIT_INTEGER | SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_BYTE..=SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_SHORT | SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SIGNED_SHORT | SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_EDGE_FLAG => self.attribute(gl, selector, a, packed),
            SELECT_CURRENT_MATRIX_STACK..=TRANSLATE_FLOATS if selector != SET_DEPTH_RANGE => gl.matrix(selector, a),
            _ => gl.state(selector, a),
        };
        match taken {
            Taken::Yes => {}
            Taken::Unmodelled if matches!(selector, ENABLE_CAPABILITY | DISABLE_CAPABILITY) => *self.counts.caps.entry(a[0]).or_default() += 1,
            Taken::Unmodelled => self.counts.note(selector),
            Taken::Error(e) => {
                self.counts.errors += 1;
                if gl.error == 0 {
                    gl.error = e;
                }
            }
        }
    }

    /// glClear: the colour and depth planes of the window, to the clear values.
    /// The mask's bits are OpenGL's (colour 0x4000, depth 0x100, stencil 0x400);
    /// accumulation is not cleared yet. Each plane clears under its write mask.
    fn clear(&mut self, gl: &mut Gl, mask: u32, prims: &mut Prims) {
        self.counts.clears += 1;
        let ds = gl.draw_state(self.screen);
        prims.clears.push(ClearReq {
            rect: ds.clip,
            colour: (mask & gl::COLOR_BUFFER_BIT != 0).then(|| colour_word(gl.clear_colour)),
            depth: (mask & gl::DEPTH_BUFFER_BIT != 0).then(|| depth32(gl.clear_depth)),
            stencil: (mask & gl::STENCIL_BUFFER_BIT != 0).then_some(gl.clear_stencil as u8),
            colour_mask: ds.colour_mask,
            depth_mask: ds.depth_mask,
            stencil_mask: ds.stencil.write_mask,
        });
        prims.state = Some(ds);
    }

    fn begin(&mut self, gl: &mut Gl, mode: u32) {
        // `mode` is the begin token itself.
        if gl.in_begin {
            // GL keeps the first error until it is read.
            if gl.error == 0 {
                gl.error = gl::INVALID_OPERATION;
            }
            self.counts.errors += 1;
            return;
        }
        gl.in_begin = true;
        gl.prim = mode;
        // The clip planes in force for this primitive, decided once here, not per vertex.
        self.clip.user = gl.user_plane;
        self.clip.user_on = gl.user_on;
        self.texgen = (gl.raster.texture_2d || gl.raster.texture_3d) && gl.texgen.active();
        self.texgen_normal = self.texgen && gl.texgen.needs_normal();
        self.counts.begins[(mode - BEGIN_FIRST) as usize] += 1;
        self.prim.clear();
        self.seen = 0;
    }

    fn end(&mut self, gl: &mut Gl, prims: &mut Prims) {
        if !gl.in_begin {
            // GL keeps the first error until it is read.
            if gl.error == 0 {
                gl.error = gl::INVALID_OPERATION;
            }
            self.counts.errors += 1;
            return;
        }
        match gl.prim {
            BEGIN_PRIMITIVE_LINE_LOOP if self.prim.len() >= 2 => {
                let (a, b) = (*self.prim.last().expect("two"), self.prim[0]);
                self.line(gl, a, b, prims);
            }
            BEGIN_PRIMITIVE_POLYGON if self.prim.len() >= 3 => {
                let p = std::mem::take(&mut self.prim);
                self.polygon(gl, &p, 0, prims);
                self.prim = p;
            }
            _ => {}
        }
        gl.in_begin = false;
        self.prim.clear();
    }

    /// A colour, normal, texture coordinate or edge flag: the current values.
    fn attribute(&mut self, gl: &mut Gl, selector: u32, a: &[u32], packed: u32) -> Taken {
        let cur = |gl: &mut Gl, c: [f32; 4]| {
            gl.color = c;
            gl.light.track_color(c);
        };
        match (selector, a.len()) {
            (SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, 4) => cur(gl, [f(a[0]), f(a[1]), f(a[2]), f(a[3])]),
            (SET_CURRENT_COLOUR_3_COMPONENTS_SINGLE_FLOAT, 3) => cur(gl, [f(a[0]), f(a[1]), f(a[2]), 1.0]),
            (SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_BYTE, 1) | (SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_BYTE, 1) => {
                let b = |v: u32| (v & 0xff) as f32 / 255.0;
                let alpha = if selector == SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_BYTE { b(a[0] >> 16) } else { 1.0 };
                cur(gl, [b(packed >> 24), b(packed >> 16), b(a[0] >> 24), alpha]);
            }
            (SET_CURRENT_NORMAL_3_COMPONENTS_SINGLE_FLOAT, 3) => gl.normal = [f(a[0]), f(a[1]), f(a[2])],
            (SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_32_BIT_INTEGER, 3) => gl.normal = [a[0] as i32 as f32 / 2147483647.0, a[1] as i32 as f32 / 2147483647.0, a[2] as i32 as f32 / 2147483647.0].map(|v| v.max(-1.0)),
            (SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SINGLE_FLOAT, 1) | (SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT, 2) | (SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT, 3) | (SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT, 4) | (SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SIGNED_SHORT, 1) => {
                let i = TEXCOORD.iter().position(|&s| s == selector).expect("listed");
                self.counts.texcoord[i] += 1;
                gl.texcoord = match selector {
                    SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SINGLE_FLOAT => [f(a[0]), 0.0, 0.0, 1.0],
                    SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT => [f(a[0]), f(a[1]), 0.0, 1.0],
                    SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT => [f(a[0]), f(a[1]), f(a[2]), 1.0],
                    SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT => [f(a[0]), f(a[1]), f(a[2]), f(a[3])],
                    // s << 16 | selector, then t << 16; shorts convert as integers.
                    _ => [(packed >> 16) as i16 as f32, (a[0] >> 16) as i16 as f32, 0.0, 1.0],
                };
            }
            (SET_CURRENT_EDGE_FLAG, 1) => gl.edge_flag = a[0] != 0,
            _ => return Taken::Unmodelled,
        }
        Taken::Yes
    }

    /// A vertex: transform, light, take into the open primitive.
    fn vertex(&mut self, gl: &mut Gl, selector: u32, a: &[u32], packed: u32, prims: &mut Prims) -> Taken {
        let p = match (selector, a.len()) {
            (SUBMIT_VERTEX_2_COMPONENTS_SINGLE_FLOAT, 2) => [f(a[0]), f(a[1]), 0.0, 1.0],
            (SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, 3) => [f(a[0]), f(a[1]), f(a[2]), 1.0],
            (SUBMIT_VERTEX_4_COMPONENTS_SINGLE_FLOAT, 4) => [f(a[0]), f(a[1]), f(a[2]), f(a[3])],
            (SUBMIT_VERTEX_2_COMPONENTS_SIGNED_32_BIT_INTEGER, 2) => [a[0] as i32 as f32, a[1] as i32 as f32, 0.0, 1.0],
            (SUBMIT_VERTEX_3_COMPONENTS_SIGNED_32_BIT_INTEGER, 3) => [a[0] as i32 as f32, a[1] as i32 as f32, a[2] as i32 as f32, 1.0],
            (SUBMIT_VERTEX_2_COMPONENTS_SIGNED_SHORT, 1) => [(packed >> 16) as i16 as f32, (a[0] >> 16) as i16 as f32, 0.0, 1.0],
            (SUBMIT_VERTEX_3_COMPONENTS_SIGNED_SHORT, 1) => [(packed >> 16) as i16 as f32, (a[0] >> 16) as i16 as f32, a[0] as i16 as f32, 1.0],
            _ => return Taken::Unmodelled,
        };
        if !gl.in_begin {
            self.counts.stray += 1;
            return Taken::Yes;
        }
        self.counts.vertices += 1;
        let v = self.transform(gl, p);
        self.take(gl, v, prims);
        Taken::Yes
    }

    fn transform(&mut self, gl: &mut Gl, p: [f32; 4]) -> KVertex {
        let e = math::xform(gl.mv.get(), p);
        let h = math::xform(&*gl.proj.get(), e);
        let (mut c, mut cb) = (gl.color, gl.color);
        if gl.raster.lighting {
            gl.refresh_normal_matrix();
            (c, cb) = gl.light.light_vertex(e, gl.normal, gl.color);
        }
        let t = if gl.raster.texture_2d || gl.raster.texture_3d {
            let st = if self.texgen {
                let n = if self.texgen_normal { gl.eye_normal() } else { [0.0; 3] };
                gl.texgen.generate(gl.texcoord, p, e, n)
            } else {
                gl.texcoord
            };
            math::xform(gl.tex.get(), st)
        } else {
            [0.0; 4]
        };
        let mut v = KVertex { h, e, c, cb, t, oc: 0, edge: gl.edge_flag };
        v.oc = self.clip.outcode_by::<KonaGeometry, _>(&v);
        #[cfg(test)]
        if self.census {
            self.measure(gl, &v);
        }
        v
    }

    /// Where a vertex falls against the window, for the census (tests only).
    #[cfg(test)]
    fn measure(&mut self, gl: &Gl, v: &KVertex) {
        let Some(p) = project(gl.raster.viewport, gl.raster.depth_range, v, false, false) else {
            self.counts.behind += 1;
            return;
        };
        if v.oc != 0 {
            self.counts.outside_volume += 1;
            return;
        }
        let r = Self::window_rect(gl);
        let (x, y) = (p.x as f32 / 256.0, p.y as f32 / 256.0);
        if x < r[0] || x > r[2] || y < r[1] || y > r[3] {
            self.counts.outside_window += 1;
        } else {
            self.counts.inside += 1;
        }
    }

    /// The window the census measures against: the scissor when it is on, else
    /// the window the kernel placed.
    #[cfg(test)]
    fn window_rect(gl: &Gl) -> [f32; 4] {
        let r = &gl.raster;
        if r.scissor_test {
            let s = r.scissor;
            [s[0] as f32, s[1] as f32, (s[0] + s[2]) as f32, (s[1] + s[3]) as f32]
        } else {
            let c = gl.clip;
            [(c.x0 - gl.origin[0]) as f32, (c.y0 - gl.origin[1]) as f32, (c.x1 - gl.origin[0]) as f32, (c.y1 - gl.origin[1]) as f32]
        }
    }

    /// Add a vertex to the open primitive and draw what it completes.
    fn take(&mut self, gl: &mut Gl, v: KVertex, prims: &mut Prims) {
        self.seen += 1;
        let n = self.prim.len();
        match gl.prim {
            BEGIN_PRIMITIVE_POINTS => self.point(gl, v, prims),
            BEGIN_PRIMITIVE_INDEPENDENT_LINES => {
                if n == 1 {
                    let a = self.prim[0];
                    self.prim.clear();
                    self.line(gl, a, v, prims);
                    return;
                }
                self.prim.push(v);
            }
            BEGIN_PRIMITIVE_LINE_LOOP | BEGIN_PRIMITIVE_LINE_STRIP => {
                // A strip keeps the newest vertex; a loop keeps the first and the
                // newest, the first for the segment that closes it.
                if n >= 1 {
                    let a = *self.prim.last().expect("one");
                    self.line(gl, a, v, prims);
                }
                self.prim.truncate(if gl.prim == BEGIN_PRIMITIVE_LINE_LOOP { 1 } else { 0 });
                self.prim.push(v);
            }
            BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES => {
                self.prim.push(v);
                if self.prim.len() == 3 {
                    let t = [self.prim[0], self.prim[1], self.prim[2]];
                    self.prim.clear();
                    self.polygon(gl, &t, 2, prims);
                }
            }
            BEGIN_PRIMITIVE_TRIANGLE_STRIP => {
                self.prim.push(v);
                let n = self.prim.len();
                if n >= 3 {
                    let t = if n % 2 == 1 { [self.prim[n - 3], self.prim[n - 2], self.prim[n - 1]] } else { [self.prim[n - 2], self.prim[n - 3], self.prim[n - 1]] };
                    self.polygon(gl, &all_edges(t), 2, prims);
                    if n > 3 {
                        self.prim.remove(0);
                    }
                }
            }
            BEGIN_PRIMITIVE_TRIANGLE_FAN => {
                self.prim.push(v);
                if self.prim.len() >= 3 {
                    let t = [self.prim[0], self.prim[1], self.prim[2]];
                    self.polygon(gl, &all_edges(t), 2, prims);
                    self.prim.remove(1);
                }
            }
            BEGIN_PRIMITIVE_INDEPENDENT_QUADS => {
                self.prim.push(v);
                if self.prim.len() == 4 {
                    let q = [self.prim[0], self.prim[1], self.prim[2], self.prim[3]];
                    self.prim.clear();
                    self.polygon(gl, &q, 3, prims);
                }
            }
            BEGIN_PRIMITIVE_QUAD_STRIP => {
                self.prim.push(v);
                if self.prim.len() >= 4 && self.prim.len() % 2 == 0 {
                    let n = self.prim.len();
                    let q = [self.prim[n - 4], self.prim[n - 3], self.prim[n - 1], self.prim[n - 2]];
                    self.polygon(gl, &all_edges(q), 3, prims);
                    self.prim.drain(..2);
                }
            }
            _ => {
                if self.prim.len() < MAX_POLY - 12 {
                    self.prim.push(v);
                }
            }
        }
    }

    fn point(&mut self, gl: &mut Gl, v: KVertex, prims: &mut Prims) {
        self.counts.primitives += 1;
        if v.oc != 0 {
            self.counts.clipped_away += 1;
            return;
        }
        if gl.feed.mode != query::RENDER {
            let fv = feed_vertex(gl, &v);
            gl.feed.primitive(query::POINT, &[fv]);
            return;
        }
        self.emit_point(gl, &v, false, prims);
    }

    /// A point inside the volume, handed to set-up as a square or a disc. `of_polygon` is a vertex of a
    /// polygon drawn in the point mode, which follows the polygon-smooth switch and not the point one.
    fn emit_point(&mut self, gl: &mut Gl, v: &KVertex, of_polygon: bool, prims: &mut Prims) {
        let Some(p) = project(gl.raster.viewport, gl.raster.depth_range, v, false, gl.raster.fog.on) else {
            self.counts.no_window_position += 1;
            return;
        };
        self.counts.points += 1;
        let size = gl.raster.point_size.max(0.0);
        let (cx, cy) = (p.x as f32 / 256.0, p.y as f32 / 256.0);
        let at = |dx: f32, dy: f32| WVertex { x: grid(cx + dx), y: grid(cy + dy), ..p };
        let mut ds = gl.draw_state(self.screen);
        // Polygon offset reaches a polygon drawn as points under its own enable, and never a point
        // primitive (established). The model has no enable for the polygon-mode case, so neither is offset.
        ds.offset = None;
        let smooth = if of_polygon { ds.smooth.polygons } else { ds.smooth.points };
        if smooth && size > 0.0 {
            // Smoothed: the square about the disc, whose pixels `raster::smooth` weighs by the share of
            // each the disc covers. Its centre is the vertex's position as the sample plane has it.
            let r = size / 2.0 + 0.5;
            let quad = [at(-r, -r), at(r, -r), at(r, r), at(-r, r)];
            prims.discs.push(SmoothPoint { quad, x: cx, y: cy, r: size / 2.0 });
        } else if size <= POINT_PIXEL_MAX {
            // One pixel: a unit square about the vertex, which holds exactly one sample.
            let (a, b, c, d) = (at(-0.5, -0.5), at(0.5, -0.5), at(0.5, 0.5), at(-0.5, 0.5));
            prims.tris.extend([[a, b, c], [a, c, d]]);
        } else {
            // A disc: sixteen triangles about the centre (Iris's own form).
            let r = size / 2.0;
            let rim = |k: usize| {
                let t = k as f32 * std::f32::consts::TAU / 16.0;
                at(r * t.cos(), r * t.sin())
            };
            prims.tris.extend((0..16).map(|k| [at(0.0, 0.0), rim(k), rim(k + 1)]));
        }
        prims.state = Some(ds);
    }

    fn line(&mut self, gl: &mut Gl, mut a: KVertex, b: KVertex, prims: &mut Prims) {
        self.counts.primitives += 1;
        if gl.raster.flat {
            a.c = b.c;
        }
        let any = a.oc | b.oc;
        let (a, b) = if any != 0 {
            self.counts.clipped += 1;
            match self.clip.clip_line_by::<KonaGeometry, _>(a, b, any) {
                Some(p) => p,
                None => {
                    self.counts.clipped_away += 1;
                    return;
                }
            }
        } else {
            (a, b)
        };
        if gl.feed.mode != query::RENDER {
            let (fa, fb) = (feed_vertex(gl, &a), feed_vertex(gl, &b));
            gl.feed.primitive(query::LINE, &[fa, fb]);
            return;
        }
        self.emit_line(gl, &a, &b, false, prims);
    }

    /// A segment inside the volume, handed to set-up as the rectangle about it.
    /// A line inside the volume. `of_polygon` is an edge of a polygon drawn in the line mode, which follows
    /// the polygon-smooth switch and not the line one (inferred).
    fn emit_line(&mut self, gl: &mut Gl, a: &KVertex, b: &KVertex, of_polygon: bool, prims: &mut Prims) {
        let (vp, dr, fog) = (gl.raster.viewport, gl.raster.depth_range, gl.raster.fog.on);
        let (Some(pa), Some(pb)) = (project(vp, dr, a, false, fog), project(vp, dr, b, false, fog)) else {
            self.counts.no_window_position += 1;
            return;
        };
        let (ax, ay, bx, by) = (pa.x as f32 / 256.0, pa.y as f32 / 256.0, pb.x as f32 / 256.0, pb.y as f32 / 256.0);
        let (dx, dy) = (bx - ax, by - ay);
        let len = (dx * dx + dy * dy).sqrt();
        if !(len > 0.0) {
            self.counts.empty_lines += 1;
            return;
        }
        self.counts.lines += 1;
        // The rectangle of half-width w / 2 about the segment: two triangles; `d` is the
        // distance along the line, restarting at each segment (a strip's stipple does not
        // continue across segments here).
        let h = gl.raster.line_width.max(0.0) / 2.0;
        let (nx, ny) = (-dy / len * h, dx / len * h);
        let corner = |base: &WVertex, x: f32, y: f32, d: f32| WVertex { x: grid(x), y: grid(y), d, ..*base };
        let (a0, a1) = (corner(&pa, ax + nx, ay + ny, 0.0), corner(&pa, ax - nx, ay - ny, 0.0));
        let (b0, b1) = (corner(&pb, bx + nx, by + ny, len), corner(&pb, bx - nx, by - ny, len));
        let mut ds = gl.draw_state(self.screen);
        // Polygon offset reaches a polygon drawn as lines under its own enable, and never a line
        // primitive (established). The model has no enable for the polygon-mode case, so neither is offset.
        ds.offset = None;
        if if of_polygon { ds.smooth.polygons } else { ds.smooth.lines } {
            // Smoothed: the rectangle whole, for set-up to weigh by coverage.
            let start = prims.aa_verts.len() as u32;
            prims.aa_verts.extend([a0, a1, b1, b0]);
            prims.aa.push(SmoothPoly { start, n: 4, line: true });
        } else {
            prims.lines.extend([[a0, a1, b1], [a0, b1, b0]]);
        }
        prims.state = Some(ds);
    }

    /// A convex polygon of three or four or more vertices, whose flat colour is
    /// vertex `prov`'s.
    fn polygon(&mut self, gl: &mut Gl, poly: &[KVertex], prov: usize, prims: &mut Prims) {
        // The common primitive, a triangle or a quad wholly inside the volume,
        // needs workspace for four vertices, not for the most a clip can make.
        if poly.len() <= 4 && poly.iter().all(|w| w.oc == 0) {
            self.polygon_in::<4>(gl, poly, prov, prims);
        } else {
            self.polygon_in::<MAX_POLY>(gl, poly, prov, prims);
        }
    }

    /// `polygon` with room for `K` vertices through the volume clip.
    fn polygon_in<const K: usize>(&mut self, gl: &mut Gl, poly: &[KVertex], prov: usize, prims: &mut Prims) {
        self.counts.primitives += 1;
        let n = poly.len();
        let mut v = [KVertex::default(); K];
        v[..n].copy_from_slice(poly);
        if gl.raster.flat {
            let (c, cb) = (v[prov].c, v[prov].cb);
            v[..n].iter_mut().for_each(|w| (w.c, w.cb) = (c, cb));
        }
        let any = v[..n].iter().fold(0, |o, w| o | w.oc);
        let mut n = n;
        if any != 0 {
            self.counts.clipped += 1;
            let mut tmp = [KVertex::default(); K];
            n = self.clip.clip_polygon_by::<KonaGeometry, _>(&mut v, &mut tmp, n, any).0;
            if n < 3 {
                self.counts.clipped_away += 1;
                return;
            }
        }
        let tex = gl.raster.texture_2d || gl.raster.texture_3d;
        let (vp, dr, fog) = (gl.raster.viewport, gl.raster.depth_range, gl.raster.fog.on);
        let mut p = [WVertex::default(); K];
        for i in 0..n {
            let Some(q) = project(vp, dr, &v[i], tex, fog) else {
                self.counts.no_window_position += 1;
                return;
            };
            p[i] = q;
        }
        // Orientation: positive area is counter-clockwise in a window whose y
        // runs up, front-facing unless the front face is clockwise (OpenGL's
        // own convention). The recorded demos set the cull face to back (0x405,
        // 23,484 times) and the front face twice in the whole recording (once
        // each way), so what is culled is the back faces of a counter-clockwise
        // front; which face the hardware calls back is not tested. The area is
        // that of the quantised positions, the ones the triangles are drawn from.
        let area: i64 = (0..n).map(|i| p[i].x as i64 * p[(i + 1) % n].y as i64 - p[(i + 1) % n].x as i64 * p[i].y as i64).sum();
        let back = (area < 0) != gl.raster.front_cw;
        let r = &gl.raster;
        // A polygon of no area faces neither way: it is not culled, and set-up counts it as having no area.
        if area != 0 && r.cull && (r.cull_mode == gl::FRONT_AND_BACK || (r.cull_mode == gl::FRONT) != back) {
            self.counts.culled += 1;
            return;
        }
        // A feedback or select pass takes what survived clipping and culling, and draws nothing.
        if gl.feed.mode != query::RENDER {
            let fvs: Vec<query::Fv> = v[..n].iter().map(|w| feed_vertex(gl, w)).collect();
            gl.feed.primitive(query::POLYGON, &fvs);
            return;
        }
        // The pipe subtracts the line mode from the face's mode: zero is line, below is point and above is
        // fill, whatever the value (established), so an unchecked mode word still draws as one of the three.
        let drawn = (gl.raster.polygon_mode[back as usize] as i32).wrapping_sub(gl::LINE as i32);
        if drawn <= 0 {
            self.outline(gl, &v[..n], back, if drawn == 0 { gl::LINE } else { gl::POINT }, prims);
            return;
        }
        if back && gl.light.two_sided != 0 {
            for i in 0..n {
                p[i].c = v[i].cb.map(colour13);
            }
        }
        // A fan from the first vertex. A smoothed polygon goes whole, for set-up to cut into the fan and
        // weigh by coverage (`raster::smooth`).
        if gl.draw_state(self.screen).smooth.polygons {
            let start = prims.aa_verts.len() as u32;
            prims.aa_verts.extend_from_slice(&p[..n]);
            prims.aa.push(SmoothPoly { start, n: n as u32, line: false });
        } else {
            for i in 1..n - 1 {
                prims.tris.push([p[0], p[i], p[i + 1]]);
            }
        }
        prims.state = Some(gl.draw_state(self.screen));
    }
}

impl GlPath {
    /// A polygon drawn as its boundary edges (`mode` is `gl::LINE`) or its vertices (`gl::POINT`),
    /// from the polygon after clipping and culling. An edge or a vertex is drawn when its vertex's
    /// edge flag is set: the flag of the vertex an edge starts from, and one clipping added is set
    /// (the shared clipper's rule). Lines and points take the line and point state, not the
    /// polygon's: width, size and stipple, and the polygon stipple does not apply. The colours are
    /// the face's, as for a fill. Smoothing follows the polygon-smooth switch, not the line and point
    /// ones (inferred). The pipe applies polygon offset to these modes under their own enables
    /// (established); the model keeps no enables for them and applies none (a limit).
    fn outline(&mut self, gl: &mut Gl, v: &[KVertex], back: bool, mode: u32, prims: &mut Prims) {
        let n = v.len();
        let back_colour = back && gl.light.two_sided != 0;
        let face = |w: &KVertex| {
            let mut w = *w;
            if back_colour {
                w.c = w.cb;
            }
            w
        };
        for i in 0..n {
            if !v[i].edge {
                continue;
            }
            if mode == gl::POINT {
                let a = face(&v[i]);
                self.emit_point(gl, &a, true, prims);
            } else {
                let (a, b) = (face(&v[i]), face(&v[(i + 1) % n]));
                self.emit_line(gl, &a, &b, true, prims);
            }
        }
    }
}

#[cfg(test)]
mod tests;
