//! Queries, the error query, select and feedback.
//!
//! A query is a short token and an answer in the context's return page: a completion word
//! (word 0, which the pipe writes last), the count of data words, then the data from byte 8
//! (the definitions table's section on queries, which establishes all three). The client waits
//! on the completion word, so every query token the model knows of is answered; one it cannot answer fully gets an empty or zero reply and is counted,
//! which is the table's own inference of what keeps the client from hanging. Where the table
//! grades a layout inferred or unknown, the choice is made where the code makes it and
//! counted. The choices:
//!
//! - Current values (0x03b): the table's established layout, in the order the pipe pushes the
//!   current attributes: colour at byte 0, colour index at 16, normal at 20, texture coordinates
//!   at 32 and the edge flag at 48. The colour index is zero (the model keeps none). The edge
//!   flag word is a single-precision float, which the client truncates to an integer for integer
//!   and boolean reads (established), so a set flag is 1.0 and a clear one 0.0; the engine
//!   handler's own bit pattern is not decoded (inferred).
//! - Raster position (0x03d): colour at 0, index at 16, texture coordinates at 20, position at 36
//!   (window x, y, depth as a fraction, 1), the valid flag at 52 (an integer word) and the distance
//!   at 56. The index, texture coordinates and distance are zero (the model keeps none).
//! - Generic state (0x04c): the 171 names the table lists, each with its reply's word count; the
//!   values the context keeps, as floats (the depth range and depth clear value as doubles, as the
//!   firmware sends them), zeros of the right length for a name it keeps nothing for, counted by
//!   name; a name with no row is an empty reply and an invalid-enumerant error.
//! - Light and material replies carry what `Lighting` keeps: three components of an ambient,
//!   diffuse (material) or specular colour, so the fourth reads as 1; a spot cutoff is the
//!   arc cosine of the stored cosine.
//! - The line-mode limit table (0x0ee) is a value the recordings show another model of the board
//!   returning (eight rows of 0x2000, 0x2000, 0, 0). The hardware's is not known.
//! - Select and feedback follow GL's own layouts for the words the table does not decode:
//!   feedback is the record code, then (polygons) the vertex count, then each vertex as the
//!   feedback type says; a select record is a marker word of 1, the name count, the nearest and
//!   farthest depth and the names. The table establishes the record codes and sizes. Inferred:
//!   the order of the values in a vertex, the vertex scaling (x and y by 0.0625, z by 2^-20, which
//!   take the engine's sixteenths of a pixel and its 20-bit depth back to a pixel and a fraction;
//!   see `defs::reply`) and the select depths, unsigned 32-bit integers clamped at 4294967040.0.
//!   A hit is a primitive that survives clipping and culling. Names are pushed up to 64 deep (a
//!   model cap). Output past the buffer the client gave sets the overflow flag in the completion's
//!   status.

use super::state::Gl;
use crate::dev::kona::defs::*;
use crate::dev::kona::stream::Stream;
use std::collections::BTreeMap;

pub const RENDER: u32 = 0x1c00;
pub const FEEDBACK: u32 = 0x1c01;
pub const SELECT: u32 = 0x1c02;
/// Feedback types.
pub const FEEDBACK_2D: u32 = 0x600;
pub const FEEDBACK_3D: u32 = 0x601;
pub const FEEDBACK_3D_COLOUR: u32 = 0x602;
pub const FEEDBACK_3D_COLOUR_TEXTURE: u32 = 0x603;
pub const FEEDBACK_4D_COLOUR_TEXTURE: u32 = 0x604;
/// Feedback record codes.
pub const PASS_THROUGH: u32 = 0x700;
pub const POINT: u32 = 0x701;
pub const LINE: u32 = 0x702;
pub const POLYGON: u32 = 0x703;
/// The completion's status bit for a full buffer.
pub const OVERFLOW_BIT: u32 = 0x800;
/// The most names the stack holds (the model's cap).
pub const NAME_STACK_DEPTH: usize = 64;

/// What a feedback or select pass keeps while it runs, and until the host collects it.
#[derive(Clone, Debug, PartialEq)]
pub struct Feed {
    pub mode: u32,
    pub kind: u32,
    pub names: Vec<u32>,
    /// A select hit not yet written: the depth range so far, and whether the names it
    /// would write are the stack's now.
    pub open: Option<(u32, u32)>,
    pub words: Vec<u32>,
    /// The buffer the client gave: words that fit. None: no start seen.
    pub limit: Option<usize>,
    pub overflow: bool,
    /// A bracket was opened (a start) while a feedback or select mode was current: its finish
    /// takes the result. The client switches the mode before it finishes the one it leaves, so
    /// the first finish of a pass belongs to the mode before.
    pub bracket: bool,
}

impl Default for Feed {
    fn default() -> Self {
        Feed { mode: RENDER, kind: FEEDBACK_2D, names: Vec::new(), open: None, words: Vec::new(), limit: None, overflow: false, bracket: false }
    }
}

impl Feed {
    pub fn visit<S: Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { mode: _, kind: _, names: _, open: _, words: _, limit: _, overflow: _, bracket: _ } = self;
        s.u32(&mut self.mode);
        s.u32(&mut self.kind);
        s.vec_u32(&mut self.names);
        s.opt(&mut self.open, |s, v| {
            s.u32(&mut v.0);
            s.u32(&mut v.1);
        });
        s.vec_u32(&mut self.words);
        let mut limit = self.limit.map_or(u32::MAX, |l| l as u32);
        s.u32(&mut limit);
        self.limit = (limit != u32::MAX).then_some(limit as usize);
        s.bool(&mut self.overflow);
        s.bool(&mut self.bracket);
    }

    /// A start: the buffer's size in bytes, and an empty result.
    pub fn start(&mut self, bytes: u32) {
        self.words.clear();
        self.open = None;
        self.overflow = false;
        self.bracket = self.mode != RENDER;
        self.limit = Some(bytes as usize / 4);
    }

    fn push(&mut self, words: &[u32]) {
        let room = self.limit.unwrap_or(usize::MAX).saturating_sub(self.words.len());
        if words.len() > room {
            self.overflow = true;
            self.words.extend_from_slice(&words[..room]);
        } else {
            self.words.extend_from_slice(words);
        }
    }

    /// A select hit written out: the marker, the name count, the depths, the names.
    fn close(&mut self) {
        if let Some((near, far)) = self.open.take() {
            let mut r = vec![1, self.names.len() as u32, near, far];
            r.extend_from_slice(&self.names);
            self.push(&r);
        }
    }

    /// A primitive that survived clipping and culling, with its window vertices, in the active mode.
    /// Returns whether the pass took it (and so nothing is drawn).
    pub fn primitive(&mut self, code: u32, v: &[Fv]) -> bool {
        match self.mode {
            FEEDBACK => {
                let mut r = vec![code];
                if code == POLYGON {
                    r.push(v.len() as u32);
                }
                for p in v {
                    r.extend(p.words(self.kind));
                }
                self.push(&r);
                true
            }
            SELECT => {
                let (near, far) = v.iter().fold((u32::MAX, 0), |(n, f), p| {
                    let z = (p.z.clamp(0.0, 1.0) * reply::SELECT_DEPTH_MAX) as u32;
                    (n.min(z), f.max(z))
                });
                self.open = Some(self.open.map_or((near, far), |(n, f)| (n.min(near), f.max(far))));
                true
            }
            _ => false,
        }
    }

    /// A name-stack token. Select mode only: in the other modes the words are dropped.
    fn names(&mut self, selector: u32, a: &[u32]) -> Option<u32> {
        if self.mode != SELECT {
            return None;
        }
        self.close();
        match selector {
            CLEAR_SELECTION_NAME_STACK => self.names.clear(),
            REPLACE_TOP_SELECTION_NAME => match self.names.last_mut() {
                Some(t) => *t = a.first().copied().unwrap_or(0),
                None => return Some(0x502),
            },
            PUSH_SELECTION_NAME if self.names.len() >= NAME_STACK_DEPTH => return Some(0x503),
            PUSH_SELECTION_NAME => self.names.push(a.first().copied().unwrap_or(0)),
            _ => {
                if self.names.pop().is_none() {
                    return Some(0x504);
                }
            }
        }
        None
    }

    /// Take the result for the host: what was collected and whether the buffer overflowed. The
    /// mode is left as it is.
    pub fn take(&mut self) -> (Vec<u32>, bool) {
        self.close();
        self.bracket = false;
        self.limit = None;
        (std::mem::take(&mut self.words), std::mem::take(&mut self.overflow))
    }
}

/// A vertex as feedback reports it: window position, depth as a fraction, 1 / w, colour and texture.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fv {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
    pub c: [f32; 4],
    pub t: [f32; 4],
}

impl Fv {
    /// The vertex as the pipe's formatter writes it (inferred, see `defs::reply`): the engine holds
    /// window x and y in sixteenths of a pixel and depth over the 20-bit window range, and the
    /// formatter scales them back.
    fn words(&self, kind: u32) -> Vec<u32> {
        let b = |v: f32| v.to_bits();
        let xy = |v: f32| v * 16.0 * reply::FEEDBACK_XY_SCALE;
        let z = self.z * reply::WINDOW_DEPTH_RANGE * reply::FEEDBACK_Z_SCALE;
        let mut r = vec![b(xy(self.x)), b(xy(self.y))];
        match kind {
            FEEDBACK_2D => {}
            FEEDBACK_3D => r.push(b(z)),
            FEEDBACK_3D_COLOUR => {
                r.push(b(z));
                r.extend(self.c.map(b));
            }
            FEEDBACK_3D_COLOUR_TEXTURE => {
                r.push(b(z));
                r.extend(self.c.map(b));
                r.extend(self.t.map(b));
            }
            _ => {
                r.extend([b(z), b(self.w)]);
                r.extend(self.c.map(b));
                r.extend(self.t.map(b));
            }
        }
        r
    }
}

/// What the queries did.
#[derive(Clone, Debug, Default)]
pub struct Counts {
    pub answered: u64,
    /// Recognised query tokens answered with an empty reply, by selector.
    pub empty: BTreeMap<u32, u64>,
    /// Generic-state names answered with one zero word.
    pub unknown_state: BTreeMap<u32, u64>,
    /// Tokens nothing sends (the integer and double query forms), refused by the pipe.
    pub unsent: u64,
    pub hits: u64,
    pub feedback_values: u64,
    pub select_errors: u64,
    pub passes: u64,
    pub overflows: u64,
    /// Replies with no context current or no return page registered, which are not posted.
    pub unposted: u64,
    /// Push-attributes tokens, each answered with a bare completion.
    pub pushes: u64,
    /// Pushes onto a full stack, which the pipe also closes.
    pub pushes_full: u64,
    /// Pushes whose mask had a bit the pipe reads back, which the model answers with no data.
    pub pushes_unread: u64,
    pub pops: u64,
}

impl Counts {
    pub fn render(&self) -> String {
        format!(
            "{} answered, {} empty by selector {:?}, {} unknown state names {:?}, {} unsent forms, {} not posted; attribute pushes {} ({} onto a full stack, {} with a read-back left empty), pops {}; select and feedback {} passes, {} words, {} hits, {} overflowed",
            self.answered,
            self.empty.values().sum::<u64>(),
            self.empty.iter().map(|(k, v)| format!("{k:#x} x {v}")).collect::<Vec<_>>(),
            self.unknown_state.values().sum::<u64>(),
            self.unknown_state.iter().map(|(k, v)| format!("{k:#x} x {v}")).collect::<Vec<_>>(),
            self.unsent,
            self.unposted,
            self.pushes,
            self.pushes_full,
            self.pushes_unread,
            self.pops,
            self.passes,
            self.feedback_values,
            self.hits,
            self.overflows
        )
    }
}

/// The tokens `answer` handles: every query token the table lists, including those that
/// nothing sends.
pub fn is_query(selector: u32) -> bool {
    matches!(
        selector,
        QUERY_CURRENT_VERTEX_ATTRIBUTES
            | QUERY_CLIP_PLANE
            | QUERY_RASTER_POSITION_STATE
            | QUERY_PENDING_ERROR
            | QUERY_STATE_FLOAT
            | QUERY_STATE_INTEGER
            | QUERY_LIGHT_FLOAT
            | QUERY_LIGHT_INTEGER
            | QUERY_MATERIAL_FLOAT
            | QUERY_MATERIAL_INTEGER
            | QUERY_PIXEL_MAP_FLOAT
            | QUERY_POLYGON_STIPPLE
            | QUERY_STATE_GENERIC
            | QUERY_TEXTURE_ENVIRONMENT_FLOAT
            | QUERY_TEXTURE_ENVIRONMENT_INTEGER
            | QUERY_TEXTURE_GENERATION_FLOAT
            | QUERY_COLOUR_TABLE
            | QUERY_COLOUR_TABLE_PARAMETERS
            | QUERY_LINE_MODE_LIMIT_TABLE
            | QUERY_CONVOLUTION_FILTER
            | QUERY_SEPARABLE_FILTER
            | QUERY_PIXEL_TRANSFORM_PARAMETERS_INTEGER
            | QUERY_HISTOGRAM
            | QUERY_HISTOGRAM_PARAMETERS
            | QUERY_MIN_MAX_RESULTS
            | QUERY_MIN_MAX_PARAMETERS
            | QUERY_ENGINE_SIDE_TEXTURE_ATTRIBUTES
            | QUERY_DEFORMATION_STATE
    )
}

/// The tokens of the select and feedback modes that act on the context.
pub fn is_mode(selector: u32) -> bool {
    matches!(selector, SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES | SELECT_FEEDBACK_RECORD_TYPE | CLEAR_SELECTION_NAME_STACK | REPLACE_TOP_SELECTION_NAME | PUSH_SELECTION_NAME | POP_SELECTION_NAME | INSERT_FEEDBACK_PASS_THROUGH_MARKER)
}

/// A mode, type, name-stack or pass-through token. Returns whether it was one.
pub fn mode_command(gl: &mut Gl, c: &mut Counts, selector: u32, a: &[u32]) -> bool {
    match (selector, a) {
        (SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES, &[mode]) => {
            if matches!(mode, RENDER | FEEDBACK | SELECT) {
                gl.feed.close();
                gl.feed.mode = mode;
                c.passes += (mode != RENDER) as u64;
            } else if gl.error == 0 {
                gl.error = 0x500;
            }
            true
        }
        (SELECT_FEEDBACK_RECORD_TYPE, &[kind]) => {
            if (FEEDBACK_2D..=FEEDBACK_4D_COLOUR_TEXTURE).contains(&kind) {
                gl.feed.kind = kind;
            } else if gl.error == 0 {
                gl.error = 0x500;
            }
            true
        }
        (CLEAR_SELECTION_NAME_STACK | REPLACE_TOP_SELECTION_NAME | PUSH_SELECTION_NAME | POP_SELECTION_NAME, _) => {
            if let Some(e) = gl.feed.names(selector, a) {
                c.select_errors += 1;
                if gl.error == 0 {
                    gl.error = e;
                }
            }
            true
        }
        // A pass-through marker acts only in feedback mode.
        (INSERT_FEEDBACK_PASS_THROUGH_MARKER, &[value]) => {
            if gl.feed.mode == FEEDBACK {
                gl.feed.push(&[PASS_THROUGH, value]);
            }
            true
        }
        _ => false,
    }
}

/// The attribute stack's depth: a push onto a full one overflows.
pub const ATTRIBUTE_STACK_DEPTH: u32 = 16;
/// The error a push onto a full stack leaves (the GL stack-overflow error).
const STACK_OVERFLOW: u32 = 0x503;
/// The mask bits the pipe's handler does not test: the evaluator bit and the list bit. Every other
/// bit is tested and some send data to the host.
const MASK_UNTESTED: u32 = 0x10000 | 0x20000;

/// A push-attributes token (0xc9). The pipe opens a read-back before it looks at the mask and closes it
/// on every path, a push onto a full stack included, and the close writes the completion word, so every
/// push is answered and the client's wait on it ends (inferred: the table grades the bracket's
/// always completing as inferred, and so the literal completion value, as for every query). The reply carries no data: the model reads nothing back for the
/// mask bits that do send some, and counts those pushes. A full stack sets the stack-overflow error
/// (inferred from the handler's call to its overflow routine) and does not grow.
pub fn push_attributes(gl: &mut Gl, c: &mut Counts, mask: u32) -> Vec<u32> {
    c.pushes += 1;
    c.pushes_unread += (mask & !MASK_UNTESTED != 0) as u64;
    if gl.attrib_depth >= ATTRIBUTE_STACK_DEPTH {
        c.pushes_full += 1;
        if gl.error == 0 {
            gl.error = STACK_OVERFLOW;
        }
    } else {
        gl.attrib_depth += 1;
    }
    Vec::new()
}

/// A pop-attributes token (0xc8). Nothing is known to wait on it, so nothing is posted; the depth
/// only keeps a later push honest about a full stack.
pub fn pop_attributes(gl: &mut Gl, c: &mut Counts) {
    c.pops += 1;
    gl.attrib_depth = gl.attrib_depth.saturating_sub(1);
}

fn floats(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The reply words to a query (after the count), and whether the model could answer it.
pub fn answer(gl: &mut Gl, c: &mut Counts, selector: u32, a: &[u32]) -> Vec<u32> {
    let out = match (selector, a) {
        (QUERY_CURRENT_VERTEX_ATTRIBUTES, _) => {
            let mut r = [0u32; reply::CURRENT_WORDS];
            r[reply::CURRENT_COLOUR..][..4].copy_from_slice(&floats(&gl.color));
            r[reply::CURRENT_NORMAL..][..3].copy_from_slice(&floats(&gl.normal));
            r[reply::CURRENT_TEXTURE..][..4].copy_from_slice(&floats(&gl.texcoord));
            r[reply::CURRENT_EDGE_FLAG] = (gl.edge_flag as u32 as f32).to_bits();
            r.to_vec()
        }
        (QUERY_RASTER_POSITION_STATE, _) => {
            let p = gl.pix.pos;
            let mut r = [0u32; reply::RASTER_WORDS];
            r[reply::RASTER_COLOUR..][..4].copy_from_slice(&floats(&p.colour));
            r[reply::RASTER_POSITION..][..4].copy_from_slice(&floats(&[p.x, p.y, p.z as f32 / 4_294_967_040.0, 1.0]));
            r[reply::RASTER_VALID] = p.valid as u32;
            r.to_vec()
        }
        (QUERY_PENDING_ERROR, _) => vec![std::mem::take(&mut gl.error)],
        // The plane, four floats (established; that the pipe keeps it in eye space is inferred from
        // OpenGL's query). A name outside 0x3000 to 0x3005 is the client's error to
        // raise (it checks the name before it asks); one that arrives anyway is answered with nothing.
        (QUERY_CLIP_PLANE, &[name]) if (name.wrapping_sub(gl::CLIP_PLANE0) as usize) < gl::CLIP_PLANES => floats(&gl.user_plane[(name - gl::CLIP_PLANE0) as usize]),
        (QUERY_CLIP_PLANE, _) => {
            *c.empty.entry(selector).or_default() += 1;
            if gl.error == 0 {
                gl.error = gl::INVALID_ENUM;
            }
            Vec::new()
        }
        (QUERY_LIGHT_FLOAT, &[n, name]) => light(gl, n, name),
        (QUERY_MATERIAL_FLOAT, &[face, name]) => material(gl, face, name),
        (QUERY_PIXEL_MAP_FLOAT, &[name, start, count]) => pixel_map(gl, name, start, count),
        (QUERY_POLYGON_STIPPLE, _) => gl.raster.polygon_stipple.to_vec(),
        (QUERY_STATE_GENERIC, &[name]) => state(gl, c, name),
        (QUERY_TEXTURE_ENVIRONMENT_FLOAT, &[name]) => match name {
            0x2200 => vec![gl.raster.env_mode],
            0x2201 => floats(&gl.raster.env_colour),
            _ => vec![0; 4],
        },
        (QUERY_LINE_MODE_LIMIT_TABLE, _) => (0..8).flat_map(|_| [0x2000, 0x2000, 0, 0]).collect(),
        // Anything else the table lists: nothing the model keeps answers it.
        (QUERY_STATE_FLOAT | QUERY_STATE_INTEGER | QUERY_LIGHT_INTEGER | QUERY_MATERIAL_INTEGER | QUERY_TEXTURE_ENVIRONMENT_INTEGER, _) => {
            c.unsent += 1;
            Vec::new()
        }
        _ => {
            *c.empty.entry(selector).or_default() += 1;
            Vec::new()
        }
    };
    c.answered += 1;
    out
}

fn light(gl: &Gl, n: u32, name: u32) -> Vec<u32> {
    let Some(l) = gl.light.lights.get(n as usize) else { return vec![0; 4] };
    let rgba = |c: [f32; 3]| floats(&[c[0], c[1], c[2], 1.0]);
    match name {
        0x1200 => rgba(l.ambient),
        0x1201 => rgba(l.diffuse),
        0x1202 => rgba(l.specular),
        0x1203 => floats(&l.pos),
        0x1204 => floats(&l.spot_dir),
        0x1205 => floats(&[l.spot_exp]),
        0x1206 => floats(&[l.spot_cos_cutoff.clamp(-1.0, 1.0).acos().to_degrees()]),
        0x1207 => floats(&[l.atten[2]]),
        0x1208 => floats(&[l.atten[0]]),
        0x1209 => floats(&[l.atten[1]]),
        _ => vec![0; 4],
    }
}

fn material(gl: &Gl, face: u32, name: u32) -> Vec<u32> {
    let m = &gl.light.mat[(face == gl::BACK) as usize];
    let rgba = |c: [f32; 3]| floats(&[c[0], c[1], c[2], 1.0]);
    match name {
        0x1200 => rgba(m.ambient),
        0x1201 => floats(&m.diffuse),
        0x1202 => rgba(m.specular),
        0x1600 => rgba(m.emission),
        0x1601 => floats(&[m.shininess]),
        _ => vec![0; 3],
    }
}

/// One 16-byte entry a map entry, the value in the component's slot (0 for index and red).
fn pixel_map(gl: &Gl, name: u32, start: u32, count: u32) -> Vec<u32> {
    use crate::dev::kona::defs::pixel::*;
    let slot = match name {
        n if (MAP_INDEX_TO_RED..MAP_INDEX_TO_RED + 4).contains(&n) => (n - MAP_INDEX_TO_RED) as usize,
        n if (MAP_RED_TO_RED..MAP_RED_TO_RED + 4).contains(&n) => (n - MAP_RED_TO_RED) as usize,
        _ => 0,
    };
    let Some(m) = gl.pix.maps.get(&name) else { return vec![0; 4 * count.min(512) as usize] };
    (start..start.saturating_add(count.min(512)))
        .flat_map(|i| {
            let v = m.values.get(i as usize).copied().unwrap_or(0.0);
            let v = if m.max > 0.0 { v / m.max } else { v };
            let mut e = [0u32; 4];
            e[slot] = v.to_bits();
            e
        })
        .collect()
}

/// The words of a double as two reply words, high first (the client's element type for the depth range
/// and the depth clear value, which the firmware answers in two words each).
fn double(v: f32) -> [u32; 2] {
    let b = (v as f64).to_bits();
    [(b >> 32) as u32, b as u32]
}

/// The reply to a generic state name: the table's rows give the word count of each of the 171 names the
/// pipe answers (`defs::state_query`); a name with no row is answered with an empty reply and an
/// invalid-enumerant error, as the firmware does. The values are what the context keeps, as floats; a row
/// the model keeps nothing for replies zeros of the right length and is counted by name, as is a name with
/// no row.
fn state(gl: &mut Gl, c: &mut Counts, name: u32) -> Vec<u32> {
    let Some(words) = state_query::words(name) else {
        *c.unknown_state.entry(name).or_default() += 1;
        if gl.error == 0 {
            gl.error = 0x500;
        }
        return Vec::new();
    };
    let r = &gl.raster;
    let mut out = match name {
        0xba6 => floats(gl.mv.get()),
        0xba7 => floats(gl.proj.get()),
        0xba8 => floats(gl.tex.get()),
        0xba0 => floats(&[gl.mode as f32]),
        0xba2 => floats(&r.viewport),
        0xb70 => r.depth_range.iter().flat_map(|&v| double(v)).collect(),
        0xc10 => floats(&r.scissor.map(|v| v as f32)),
        0xc22 => floats(&gl.clear_colour),
        0xb73 => double(gl.clear_depth).to_vec(),
        0xb91 => floats(&[gl.clear_stencil as f32]),
        0xb80 => floats(&gl.pix.accum_clear),
        0xbc2 => floats(&[r.alpha_ref]),
        0x8005 => floats(&r.blend_colour),
        0xb21 => floats(&[r.line_width]),
        0xb11 => floats(&[r.point_size]),
        // A capability's enable bit, as an integer word (the table says the client reads booleans by the low byte).
        0xbd0 => vec![r.dither as u32],
        // The fog rows: the enable, then mode, density, start, end and colour as floats (the colour's
        // fourth word is not kept).
        0xb60 => vec![r.fog.on as u32],
        // The polygon modes, front then back, as numbers.
        0xb40 => floats(&r.polygon_mode.map(|m| m as f32)),
        0xb65 => floats(&[r.fog.mode as f32]),
        0xb62 => floats(&[r.fog.density]),
        0xb63 => floats(&[r.fog.start]),
        0xb64 => floats(&[r.fog.end]),
        0xb66 => floats(&r.fog.colour),
        _ => {
            *c.unknown_state.entry(name).or_default() += 1;
            Vec::new()
        }
    };
    out.resize(words, 0);
    out
}

/// The bytes of a reply: the count, then the words.
pub fn reply_bytes(words: &[u32]) -> Vec<u8> {
    std::iter::once(words.len() as u32).chain(words.iter().copied()).flat_map(u32::to_be_bytes).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_writes_a_record_per_name_change_with_the_depths_it_saw() {
        let mut f = Feed::default();
        f.start(1024);
        let mut c = Counts::default();
        let mut gl = Gl::default();
        gl.feed = f;
        mode_command(&mut gl, &mut c, SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES, &[SELECT]);
        mode_command(&mut gl, &mut c, CLEAR_SELECTION_NAME_STACK, &[0]);
        mode_command(&mut gl, &mut c, PUSH_SELECTION_NAME, &[7]);
        let v = |z: f32| Fv { z, ..Fv::default() };
        assert!(gl.feed.primitive(POLYGON, &[v(0.25), v(0.5)]));
        assert!(gl.feed.primitive(POLYGON, &[v(0.75)]));
        // A new name closes the open hit.
        mode_command(&mut gl, &mut c, REPLACE_TOP_SELECTION_NAME, &[9]);
        assert!(gl.feed.primitive(POINT, &[v(1.0)]));
        mode_command(&mut gl, &mut c, SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES, &[RENDER]);
        let (words, overflow) = gl.feed.take();
        assert!(!overflow);
        let z = |x: f32| (x * 4_294_967_040.0) as u32;
        assert_eq!(words, vec![1, 1, z(0.25), z(0.75), 7, 1, 1, z(1.0), z(1.0), 9]);
    }

    #[test]
    fn feedback_records_follow_the_type_and_overflow_sets_the_flag() {
        let mut gl = Gl::default();
        let mut c = Counts::default();
        gl.feed.start(4 * 8);
        mode_command(&mut gl, &mut c, SELECT_FEEDBACK_RECORD_TYPE, &[FEEDBACK_3D]);
        mode_command(&mut gl, &mut c, SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES, &[FEEDBACK]);
        let v = Fv { x: 1.0, y: 2.0, z: 0.5, ..Fv::default() };
        mode_command(&mut gl, &mut c, INSERT_FEEDBACK_PASS_THROUGH_MARKER, &[3.0f32.to_bits()]);
        gl.feed.primitive(LINE, &[v, v]);
        // Two words of the pass-through, then the line's code and the first vertex (3 words); the rest does not fit.
        let (w, overflow) = gl.feed.take();
        assert!(overflow);
        assert_eq!(w.len(), 8);
        assert_eq!(&w[..2], &[PASS_THROUGH, 3.0f32.to_bits()]);
        let z = 0.5 * reply::WINDOW_DEPTH_RANGE * reply::FEEDBACK_Z_SCALE;
        assert_eq!(&w[2..6], &[LINE, 1.0f32.to_bits(), 2.0f32.to_bits(), z.to_bits()]);
    }

    #[test]
    fn name_stack_errors_and_modes_other_than_select_drop_the_tokens() {
        let mut gl = Gl::default();
        let mut c = Counts::default();
        mode_command(&mut gl, &mut c, PUSH_SELECTION_NAME, &[1]);
        assert!(gl.feed.names.is_empty() && gl.error == 0, "in render mode the words are dropped");
        mode_command(&mut gl, &mut c, SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES, &[SELECT]);
        mode_command(&mut gl, &mut c, POP_SELECTION_NAME, &[0]);
        assert_eq!(gl.error, 0x504);
        for i in 0..=NAME_STACK_DEPTH {
            mode_command(&mut gl, &mut c, PUSH_SELECTION_NAME, &[i as u32]);
        }
        assert_eq!(gl.feed.names.len(), NAME_STACK_DEPTH);
        assert_eq!(c.select_errors, 2);
    }

    #[test]
    fn current_values_and_the_error_query() {
        let mut gl = Gl::default();
        let mut c = Counts::default();
        gl.color = [0.1, 0.2, 0.3, 0.4];
        gl.texcoord = [5.0, 6.0, 7.0, 8.0];
        gl.normal = [0.0, 1.0, 0.0];
        gl.edge_flag = false;
        let r = answer(&mut gl, &mut c, QUERY_CURRENT_VERTEX_ATTRIBUTES, &[0]);
        assert_eq!(r.len(), 13);
        assert_eq!(f32::from_bits(r[0]), 0.1);
        assert_eq!([f32::from_bits(r[5]), f32::from_bits(r[6]), f32::from_bits(r[7])], [0.0, 1.0, 0.0], "the normal at byte 20");
        assert_eq!(r.iter().skip(8).take(4).map(|w| f32::from_bits(*w)).collect::<Vec<_>>(), [5.0, 6.0, 7.0, 8.0], "all four texture coordinates at byte 32");
        assert_eq!(r[12], 0, "the edge flag at byte 48, a clear flag the float 0.0");
        gl.edge_flag = true;
        assert_eq!(answer(&mut gl, &mut c, QUERY_CURRENT_VERTEX_ATTRIBUTES, &[0])[12], 1.0f32.to_bits(), "a set one the float 1.0");
        gl.error = 0x501;
        assert_eq!(answer(&mut gl, &mut c, QUERY_PENDING_ERROR, &[0xdbdb_dbdb]), vec![0x501]);
        assert_eq!(answer(&mut gl, &mut c, QUERY_PENDING_ERROR, &[0xdbdb_dbdb]), vec![0], "a read clears it");
        assert_eq!(answer(&mut gl, &mut c, QUERY_STATE_GENERIC, &[0x1234]), Vec::<u32>::new(), "no row: an empty reply");
        assert_eq!((c.unknown_state[&0x1234], gl.error), (1, 0x500), "and an invalid-enumerant error");
        gl.error = 0;
        assert_eq!(answer(&mut gl, &mut c, QUERY_STATE_GENERIC, &[0xb44]), vec![0], "a row the model keeps nothing for: a zero word");
        assert_eq!(answer(&mut gl, &mut c, QUERY_STATE_GENERIC, &[0xb53]), vec![0; 4], "of the row's length");
        assert_eq!(c.unknown_state[&0xb53], 1);
        gl.raster.fog.end = 7.0;
        gl.raster.fog.on = true;
        assert_eq!((answer(&mut gl, &mut c, QUERY_STATE_GENERIC, &[0xb64]), answer(&mut gl, &mut c, QUERY_STATE_GENERIC, &[0xb60])), (vec![7.0f32.to_bits()], vec![1]), "the fog rows answer from the context");
        gl.raster.depth_range = [0.25, 0.75];
        gl.clear_depth = 0.5;
        let doubles = |r: Vec<u32>| r.chunks(2).map(|w| f64::from_bits((w[0] as u64) << 32 | w[1] as u64)).collect::<Vec<_>>();
        assert_eq!(doubles(answer(&mut gl, &mut c, QUERY_STATE_GENERIC, &[0xb70])), [0.25, 0.75], "two doubles, four words");
        assert_eq!(doubles(answer(&mut gl, &mut c, QUERY_STATE_GENERIC, &[0xb73])), [0.5]);
        assert_eq!(answer(&mut gl, &mut c, QUERY_LINE_MODE_LIMIT_TABLE, &[0]).len(), 32);
        assert_eq!(reply_bytes(&[7]), vec![0, 0, 0, 1, 0, 0, 0, 7]);
    }

    #[test]
    fn the_raster_position_reply_tiles_its_fifteen_words() {
        let mut gl = Gl::default();
        let mut c = Counts::default();
        gl.pix.pos = crate::dev::kona::back::gl::pixels::RasterPos { valid: true, x: 3.0, y: 4.0, z: 0, colour: [0.25, 0.5, 0.75, 1.0] };
        let r = answer(&mut gl, &mut c, QUERY_RASTER_POSITION_STATE, &[0]);
        assert_eq!(r.len(), 15);
        let f = |i: usize| f32::from_bits(r[i]);
        assert_eq!([f(0), f(1), f(2), f(3)], [0.25, 0.5, 0.75, 1.0], "colour at byte 0");
        assert_eq!([f(9), f(10), f(11), f(12)], [3.0, 4.0, 0.0, 1.0], "position at byte 36");
        assert_eq!(r[13], 1, "valid at byte 52");
        assert_eq!(r[14], 0, "distance at byte 56");
        gl.pix.pos.valid = false;
        assert_eq!(answer(&mut gl, &mut c, QUERY_RASTER_POSITION_STATE, &[0])[13], 0);
    }

    #[test]
    fn feedback_vertices_are_scaled_back_from_the_engines_units() {
        let v = Fv { x: 100.5, y: 7.25, z: 1.0, ..Fv::default() };
        let w = v.words(FEEDBACK_3D);
        assert_eq!(f32::from_bits(w[0]), 100.5, "x: sixteenths of a pixel times 0.0625");
        assert_eq!(f32::from_bits(w[1]), 7.25);
        assert_eq!(f32::from_bits(w[2]), 1_048_575.0 / 1_048_576.0, "z: the 20-bit window range times 2^-20");
    }

    #[test]
    fn select_depths_are_unsigned_and_clamp_at_the_float_below_2_to_the_32() {
        let mut f = Feed::default();
        f.start(64);
        f.mode = SELECT;
        let v = |z: f32| Fv { z, ..Fv::default() };
        f.primitive(POINT, &[v(3.0)]);
        f.primitive(POINT, &[v(-1.0)]);
        let (w, _) = f.take();
        assert_eq!(&w[2..4], &[0, 4_294_967_040], "below 0 and above 1 clamp");
    }

    #[test]
    fn the_select_and_feedback_state_round_trips_through_a_stream() {
        use crate::dev::kona::stream::{Reader, Writer};
        let mut f = Feed::default();
        f.start(64);
        f.mode = SELECT;
        f.names = vec![1, 2];
        f.open = Some((3, 4));
        f.words = vec![9, 8];
        f.bracket = true;
        let mut w = Writer::default();
        f.visit(&mut w);
        let mut g = Feed::default();
        let mut r = Reader::new(&w.out);
        g.visit(&mut r);
        assert!(r.finish().is_ok());
        assert_eq!(g, f);
    }
}
