//! Queries, the error query, select and feedback.
//!
//! A query is a short token and an answer in the context's return page: a completion word
//! (word 0, which the pipe writes last), the count of data words, then the data from byte 8
//! (the definitions table's section on queries, which establishes all three). The client waits
//! on the completion word, so every query token the model frames is answered; one it cannot answer
//! fully gets an empty or zero reply and is counted, which is the table's own inference of what
//! keeps the client from hanging. That holds only for a token the front end frames (`tables::GL`
//! gives each query its word count): the words of an unframed token never reach `answer`, the
//! front end drops them as a desynchronised stream, and the client waits for a reply that is
//! never posted. Where the table grades a layout inferred or unknown, the choice is made where
//! the code makes it and counted. The choices:
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
//! - Generic state (0x04c): the 171 names the table lists, each with its reply's word count. A
//!   row the model holds state for answers from the context in the type the client reads it as
//!   (the return-data descriptor inventory: signed 32-bit integers for the enables, enumerants,
//!   masks, depths and the viewport and scissor box, binary32 for sizes, widths, fog and the
//!   matrices, scaled binary32 for colours, and binary64 for the depth range and depth clear value,
//!   which the firmware sends in two words each). A row the model keeps nothing for (hints, the
//!   colour-index write mask, the colour-material choice, colour tables and convolution) is zeros of
//!   the right length, counted by name; a name with no row is an empty reply and an invalid-enumerant
//!   error.
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
        // The mode is a signed integer word whichever getter asked: the guest reads it as one (a float's
        // bits came back as 1174668288 from glGetTexEnviv), as it does the generation mode below.
        (QUERY_TEXTURE_ENVIRONMENT_FLOAT, &[name]) => match name {
            0x2200 => vec![gl.raster.env_mode],
            0x2201 => floats(&gl.raster.env_colour),
            _ => vec![0; 4],
        },
        (QUERY_TEXTURE_GENERATION_FLOAT, &[coord, name]) => texgen(gl, coord, name),
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

/// A coordinate's generation mode as an integer word, or its object or eye plane (four floats, the eye plane
/// as the pipe kept it, in eye space); anything else is four zeros, as for a light.
fn texgen(gl: &Gl, coord: u32, name: u32) -> Vec<u32> {
    let Some(i) = coord.checked_sub(gl::TEXTURE_COORD_S).filter(|&i| i < 4).map(|i| i as usize) else { return vec![0; 4] };
    match name {
        gl::TEXTURE_GEN_MODE => vec![gl.texgen.mode[i]],
        gl::OBJECT_PLANE => floats(&gl.texgen.object[i]),
        gl::EYE_PLANE => floats(&gl.texgen.eye[i]),
        _ => vec![0; 4],
    }
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

/// A signed 32-bit word of a reply.
fn int(v: impl Into<i64>) -> u32 {
    v.into() as i32 as u32
}

/// A boolean row's word: an integer 0 or 1 (the client reads a boolean by the low byte of the word).
fn flag(v: bool) -> Vec<u32> {
    vec![v as u32]
}

/// The words of a generic state row the model holds the state for, or None for a row it keeps nothing
/// for. Each row answers in the native type the client reads it as (the return-data descriptor
/// inventory, which the tests restate): a signed 32-bit integer for the enables, enumerants, masks,
/// stack depths, the viewport and the scissor box; a binary32 for the sizes, widths, the fog
/// parameters and the matrices; a binary32 the client scales for a colour, an accumulation clear
/// value and the alpha reference; two binary64 for the depth range and one for the depth clear value.
/// The client's integer getter does not change the type (it converts the native words itself), so an
/// integer row is never sent as a float's bits.
fn held(gl: &Gl, name: u32) -> Option<Vec<u32>> {
    let r = &gl.raster;
    let bit = |bits: u32, i: u32| bits >> i & 1 != 0;
    let front_or_back = |mask: Option<u32>| match mask {
        Some(0) => 0,
        Some(1) => gl::FRONT,
        Some(2) => gl::BACK,
        Some(_) => gl::FRONT_AND_BACK,
        None if gl.double => gl::BACK,
        None => gl::FRONT,
    };
    Some(match name {
        // Points, lines and polygons.
        0xb10 => flag(r.point_smooth),
        0xb11 => floats(&[r.point_size]),
        0xb20 => flag(r.line_smooth),
        0xb21 => floats(&[r.line_width]),
        0xb24 => flag(r.line_stipple_on),
        0xb25 => vec![r.line_stipple.1 as u32],
        0xb26 => vec![r.line_stipple.0],
        0xb40 => r.polygon_mode.to_vec(),
        0xb41 => flag(r.polygon_smooth),
        0xb42 => flag(r.polygon_stipple_on),
        0xb44 => flag(r.cull),
        0xb45 => vec![r.cull_mode],
        0xb46 => vec![if r.front_cw { gl::CW } else { gl::CCW }],
        // Lighting. The model-ambient colour's alpha is the one the model keeps no value for.
        0xb50 => flag(r.lighting),
        0xb51 => flag(gl.light.local_viewer != 0),
        0xb52 => flag(gl.light.two_sided != 0),
        0xb53 => floats(&[gl.light.amb_sum[0], gl.light.amb_sum[1], gl.light.amb_sum[2], 1.0]),
        0xb54 => vec![if r.flat { gl::FLAT } else { gl::SMOOTH }],
        0xb57 => flag(gl.light.cmat_on != 0),
        // Fog: the enable, then density, start, end as floats, the mode as an enumerant and the colour
        // (its fourth word is not kept).
        0xb60 => flag(r.fog.on),
        0xb62 => floats(&[r.fog.density]),
        0xb63 => floats(&[r.fog.start]),
        0xb64 => floats(&[r.fog.end]),
        0xb65 => vec![r.fog.mode],
        0xb66 => floats(&[r.fog.colour[0], r.fog.colour[1], r.fog.colour[2], 0.0]),
        // Depth, accumulation and stencil.
        0xb70 => r.depth_range.iter().flat_map(|&v| double(v)).collect(),
        0xb71 => flag(r.depth_test),
        0xb72 => flag(r.depth_mask),
        0xb73 => double(gl.clear_depth).to_vec(),
        0xb74 => vec![r.depth_func],
        0xb80 => floats(&gl.pix.accum_clear),
        0xb90 => flag(r.stencil_test),
        0xb91 => vec![gl.clear_stencil],
        0xb92 => vec![r.stencil_func],
        0xb93 => vec![r.stencil_read_mask],
        0xb94 => vec![r.stencil_ops[0]],
        0xb95 => vec![r.stencil_ops[1]],
        0xb96 => vec![r.stencil_ops[2]],
        0xb97 => vec![r.stencil_ref],
        0xb98 => vec![r.stencil_write_mask],
        // Transformation.
        0xba0 => vec![gl.mode],
        0xba1 => flag(gl.light.normalize != 0),
        0xba2 => r.viewport.map(|v| int(v as i32)).to_vec(),
        0xba3 => vec![gl.mv.top + 1],
        0xba4 => vec![gl.proj.top + 1],
        0xba5 => vec![gl.tex.top + 1],
        0xba6 => floats(gl.mv.get()),
        0xba7 => floats(gl.proj.get()),
        0xba8 => floats(gl.tex.get()),
        0xbb0 => vec![gl.attrib_depth],
        // Fragment operations.
        0xbc0 => flag(r.alpha_test),
        0xbc1 => vec![r.alpha_func],
        0xbc2 => floats(&[r.alpha_ref]),
        0xbd0 => flag(r.dither),
        0xbe0 => vec![r.blend_factors[1]],
        0xbe1 => vec![r.blend_factors[0]],
        0xbe2 => flag(r.blend),
        0xbf0 => vec![r.logic_op],
        0xbf2 => flag(r.logic_op_on),
        // Buffers, scissor and clear values.
        0xc01 => vec![front_or_back(gl.draw)],
        0xc02 => vec![front_or_back(gl.pix.read)],
        0xc10 => r.scissor.map(int).to_vec(),
        0xc11 => flag(r.scissor_test),
        0xc20 => floats(&[gl.clear_index]),
        0xc22 => floats(&gl.clear_colour),
        0xc23 => r.colour_mask.map(|m| m as u32).to_vec(),
        // Texture generation's four enables.
        0xc60..=0xc63 => flag(bit(gl.texgen.enabled, name - 0xc60)),
        // A pixel map's size, one while none was loaded (OpenGL's start-up size).
        0xcb0..=0xcb9 => vec![gl.pix.maps.get(&(name - 0x40)).map_or(1, |m| m.values.len().max(1) as u32)],
        // The select name stack's depth: kept in select mode only.
        0xd70 => vec![gl.feed.names.len() as u32],
        0xde1 => flag(r.texture_2d),
        0x3000..=0x3005 => flag(bit(gl.user_on, name - 0x3000)),
        0x4000..=0x4007 => flag(bit(gl.light_bits, name - 0x4000)),
        0x8005 => floats(&r.blend_colour),
        // The post-convolution then post-colour-matrix scales and biases, which the pixel transfer keeps.
        0x801c..=0x8023 => floats(&[gl.pix.post[(name - 0x801c) as usize]]),
        0x80b4..=0x80bb => floats(&[gl.pix.post[8 + (name - 0x80b4) as usize]]),
        0x806f => flag(r.texture_3d),
        0x8037 => flag(r.offset_fill),
        0x8038 => floats(&[r.offset[0]]),
        // Multisample: the enables, the mask's value and invert flag and the pattern.
        0x809d => flag(r.ms.enable),
        0x809e => flag(r.ms.alpha_to_mask),
        0x809f => flag(r.ms.alpha_to_one),
        0x80a0 => flag(r.ms.mask_on),
        0x80aa => floats(&[r.ms.mask_value]),
        0x80ab => flag(r.ms.mask_invert),
        0x80ac => vec![r.ms.pattern],
        _ => return None,
    })
}

/// The reply to a generic state name: the table's rows give the word count of each of the 171 names the
/// pipe answers (`defs::state_query`); a name with no row is answered with an empty reply and an
/// invalid-enumerant error, as the firmware does. A row the model holds state for answers from the
/// context in its native type (`held`); a row the model keeps nothing for replies zeros of the right
/// length and is counted by name, as is a name with no row.
fn state(gl: &mut Gl, c: &mut Counts, name: u32) -> Vec<u32> {
    let Some(words) = state_query::words(name) else {
        *c.unknown_state.entry(name).or_default() += 1;
        if gl.error == 0 {
            gl.error = 0x500;
        }
        return Vec::new();
    };
    let mut out = held(gl, name).unwrap_or_else(|| {
        *c.unknown_state.entry(name).or_default() += 1;
        Vec::new()
    });
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
        assert_eq!(answer(&mut gl, &mut c, QUERY_STATE_GENERIC, &[0xc50]), vec![0], "a row the model keeps nothing for: a zero word");
        assert_eq!(answer(&mut gl, &mut c, QUERY_STATE_GENERIC, &[0x817e]), vec![0; 4], "of the row's length");
        assert_eq!(c.unknown_state[&0x817e], 1);
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

    fn set(gl: &mut Gl, selector: u32, a: &[u32]) {
        assert_eq!(gl.state(selector, a), super::super::state::Taken::Yes);
    }

    #[test]
    fn a_light_query_reads_the_light_by_number_and_name() {
        let mut gl = Gl::default();
        let mut c = Counts::default();
        let w = |v: f32| v.to_bits();
        set(&mut gl, SET_LIGHT_PARAMETER_FLOAT_FORM, &[gl::LIGHT0 + 3, gl::DIFFUSE, w(0.25), w(0.5), w(0.75), w(1.0)]);
        set(&mut gl, SET_LIGHT_PARAMETER_FLOAT_FORM, &[gl::LIGHT0 + 3, gl::CONSTANT_ATTENUATION, w(2.0), 0, 0, 0]);
        let r = answer(&mut gl, &mut c, QUERY_LIGHT_FLOAT, &[3, 0x1201]);
        assert_eq!(r, vec![w(0.25), w(0.5), w(0.75), w(1.0)], "diffuse, with the fourth component 1");
        assert_eq!(answer(&mut gl, &mut c, QUERY_LIGHT_FLOAT, &[3, 0x1207]), vec![w(2.0)], "constant attenuation: one word");
        assert_eq!(answer(&mut gl, &mut c, QUERY_LIGHT_FLOAT, &[9, 0x1201]), vec![0; 4], "no such light: zeros");
        assert_eq!(c.answered, 3);
        assert!(c.empty.is_empty() && c.unsent == 0, "answered by the model, not counted as empty");
    }

    #[test]
    fn a_material_query_reads_the_face_and_name() {
        let mut gl = Gl::default();
        let mut c = Counts::default();
        let w = |v: f32| v.to_bits();
        set(&mut gl, SET_MATERIAL_PARAMETER_FLOAT_FORM, &[gl::BACK, gl::SHININESS, w(33.0), 0, 0, 0]);
        set(&mut gl, SET_MATERIAL_PARAMETER_FLOAT_FORM, &[gl::FRONT, gl::EMISSION, w(0.5), w(0.25), w(0.125), 0]);
        assert_eq!(answer(&mut gl, &mut c, QUERY_MATERIAL_FLOAT, &[gl::BACK, 0x1601]), vec![w(33.0)]);
        assert_eq!(answer(&mut gl, &mut c, QUERY_MATERIAL_FLOAT, &[gl::FRONT, 0x1600]), vec![w(0.5), w(0.25), w(0.125), w(1.0)]);
        assert_ne!(answer(&mut gl, &mut c, QUERY_MATERIAL_FLOAT, &[gl::FRONT, 0x1601]), vec![w(33.0)], "the faces are kept apart");
    }

    #[test]
    fn a_texture_generation_query_reads_the_coordinate_mode_and_planes() {
        let mut gl = Gl::default();
        let mut c = Counts::default();
        let w = |v: f32| v.to_bits();
        gl.texgen.mode[2] = crate::dev::gl::texgen::OBJECT_LINEAR;
        gl.texgen.object[2] = [1.0, 2.0, 3.0, 4.0];
        gl.texgen.eye[1] = [5.0, 6.0, 7.0, 8.0];
        assert_eq!(answer(&mut gl, &mut c, QUERY_TEXTURE_GENERATION_FLOAT, &[0x2002, 0x2500]), vec![0x2401], "the mode, an integer word");
        assert_eq!(answer(&mut gl, &mut c, QUERY_TEXTURE_GENERATION_FLOAT, &[0x2002, 0x2501]), vec![w(1.0), w(2.0), w(3.0), w(4.0)]);
        assert_eq!(answer(&mut gl, &mut c, QUERY_TEXTURE_GENERATION_FLOAT, &[0x2001, 0x2502]), vec![w(5.0), w(6.0), w(7.0), w(8.0)]);
        assert_eq!(answer(&mut gl, &mut c, QUERY_TEXTURE_GENERATION_FLOAT, &[0x2009, 0x2500]), vec![0; 4], "no such coordinate");
        assert!(c.empty.is_empty(), "answered by the model");
    }

    #[test]
    fn a_texture_environment_query_reads_the_mode_as_an_integer_and_the_colour() {
        let mut gl = Gl::default();
        let mut c = Counts::default();
        let w = |v: f32| v.to_bits();
        set(&mut gl, SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM, &[0x2200, w(7681.0), 0, 0, 0]);
        set(&mut gl, SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM, &[0x2201, w(0.1), w(0.2), w(0.3), w(0.4)]);
        assert_eq!(answer(&mut gl, &mut c, QUERY_TEXTURE_ENVIRONMENT_FLOAT, &[0x2200]), vec![7681], "an integer word");
        assert_eq!(answer(&mut gl, &mut c, QUERY_TEXTURE_ENVIRONMENT_FLOAT, &[0x2201]), vec![w(0.1), w(0.2), w(0.3), w(0.4)]);
    }

    #[test]
    fn the_polygon_stipple_and_pixel_map_queries_answer_from_the_context() {
        let mut gl = Gl::default();
        let mut c = Counts::default();
        let mut rows = [0u32; 32];
        rows[0] = 0xdead_beef;
        rows[31] = 0x0f0f_0f0f;
        gl.set_polygon_stipple(rows);
        let r = answer(&mut gl, &mut c, QUERY_POLYGON_STIPPLE, &[0]);
        assert_eq!((r.len(), r[0], r[31]), (32, 0xdead_beef, 0x0f0f_0f0f), "the 32 rows");
        let m = answer(&mut gl, &mut c, QUERY_PIXEL_MAP_FLOAT, &[crate::dev::kona::defs::pixel::MAP_INDEX_TO_RED, 0, 2]);
        assert_eq!(m.len(), 8, "four words an entry");
        assert!(c.empty.is_empty());
    }

    #[test]
    fn the_integer_and_generic_float_forms_nothing_sends_are_answered_and_counted() {
        let mut gl = Gl::default();
        let mut c = Counts::default();
        for s in [QUERY_STATE_FLOAT, QUERY_STATE_INTEGER, QUERY_LIGHT_INTEGER, QUERY_MATERIAL_INTEGER, QUERY_TEXTURE_ENVIRONMENT_INTEGER] {
            assert!(is_query(s));
            assert_eq!(answer(&mut gl, &mut c, s, &[0, 0]), Vec::<u32>::new());
        }
        assert_eq!((c.answered, c.unsent), (5, 5));
    }

    fn row(gl: &mut Gl, c: &mut Counts, name: u32) -> Vec<u32> {
        answer(gl, c, QUERY_STATE_GENERIC, &[name])
    }

    /// Every capability the pipe holds reads back through its generic state row, as an integer boolean (the
    /// client reads a boolean by the word's low byte; a float's bits, 0x3f800000, would read as false).
    #[test]
    fn an_enable_reads_back_through_its_state_row_as_an_integer() {
        let mut gl = Gl::default();
        let mut c = Counts::default();
        let caps = [
            0xb10, 0xb20, 0xb24, 0xb41, 0xb42, 0xb44, 0xb50, 0xb57, 0xb60, 0xb71, 0xb90, 0xba1, 0xbc0, 0xbd0, 0xbe2, 0xbf2, 0xc11, 0xc60, 0xc61, 0xc62, 0xc63, 0xde1, 0x806f, 0x8037, 0x809d, 0x809e, 0x809f, 0x80a0, 0x3000, 0x3001, 0x3002, 0x3003, 0x3004, 0x3005, 0x4000, 0x4001, 0x4002, 0x4003, 0x4004, 0x4005, 0x4006,
            0x4007,
        ];
        for cap in caps {
            assert_eq!(gl.enable(cap, true), super::super::state::Taken::Yes, "{cap:#x}");
            assert_eq!(row(&mut gl, &mut c, cap), vec![1], "{cap:#x} on");
            gl.enable(cap, false);
            assert_eq!(row(&mut gl, &mut c, cap), vec![0], "{cap:#x} off");
        }
        // One enable does not touch its neighbours.
        gl.enable(0x4002, true);
        assert_eq!((row(&mut gl, &mut c, 0x4001), row(&mut gl, &mut c, 0x4002), row(&mut gl, &mut c, 0x3002)), (vec![0], vec![1], vec![0]));
        assert!(c.unknown_state.is_empty(), "every one of them is a row the model keeps");
    }

    /// The integer rows answer in integers and the float rows in floats: the client reads each in the type its
    /// descriptor names, so a viewport sent as float bits reads as 1077936128.
    #[test]
    fn an_integer_row_is_an_integer_and_a_float_row_a_float() {
        let mut gl = Gl::default();
        let mut c = Counts::default();
        let w = |v: f32| v.to_bits();
        set(&mut gl, SET_VIEWPORT, &[w(3.0), w(5.0), w(77.0), w(66.0)]);
        assert_eq!(row(&mut gl, &mut c, 0xba2), vec![3, 5, 77, 66], "viewport");
        gl.matrix(SELECT_CURRENT_MATRIX_STACK, &[gl::TEXTURE]);
        assert_eq!(row(&mut gl, &mut c, 0xba0), vec![0x1702], "matrix mode");
        set(&mut gl, SET_SCISSOR_BOX, &[1, 2, 30, 40]);
        assert_eq!(row(&mut gl, &mut c, 0xc10), vec![1, 2, 30, 40], "scissor box");
        set(&mut gl, SET_STENCIL_CLEAR_VALUE, &[5]);
        set(&mut gl, SET_STENCIL_TEST, &[0x202, 3, 0xf]);
        set(&mut gl, SET_STENCIL_OPERATIONS, &[0x1e02, 0x1e03, 0x150a]);
        set(&mut gl, SET_STENCIL_WRITE_MASK, &[0xaa]);
        let stencil: Vec<u32> = [0xb91, 0xb92, 0xb97, 0xb93, 0xb94, 0xb95, 0xb96, 0xb98].iter().flat_map(|&n| row(&mut gl, &mut c, n)).collect();
        assert_eq!(stencil, vec![5, 0x202, 3, 0xf, 0x1e02, 0x1e03, 0x150a, 0xaa], "stencil");
        set(&mut gl, SET_FRONT_FACE_WINDING, &[gl::CW]);
        set(&mut gl, SET_FACE_CULLING, &[gl::FRONT]);
        set(&mut gl, SET_FLAT_OR_SMOOTH_SHADING, &[gl::FLAT]);
        assert_eq!((row(&mut gl, &mut c, 0xb46), row(&mut gl, &mut c, 0xb45), row(&mut gl, &mut c, 0xb54)), (vec![0x900], vec![gl::FRONT], vec![0x1d00]));
        set(&mut gl, SET_POLYGON_FILL_MODE, &[gl::FRONT, gl::LINE]);
        set(&mut gl, SET_POLYGON_FILL_MODE, &[gl::BACK, gl::POINT]);
        assert_eq!(row(&mut gl, &mut c, 0xb40), vec![0x1b01, 0x1b00], "polygon mode, front then back");
        set(&mut gl, SET_BLEND_FACTORS, &[0x302, 0x303]);
        assert_eq!((row(&mut gl, &mut c, 0xbe1), row(&mut gl, &mut c, 0xbe0)), (vec![0x302], vec![0x303]), "source then destination");
        set(&mut gl, SET_COLOUR_WRITE_MASK, &[1, 0, 1, 0]);
        assert_eq!(row(&mut gl, &mut c, 0xc23), vec![1, 0, 1, 0]);
        set(&mut gl, SET_LOGIC_OPERATION, &[0x1506]);
        assert_eq!(row(&mut gl, &mut c, 0xbf0), vec![0x1506]);
        gl.mv.top = 2;
        assert_eq!((row(&mut gl, &mut c, 0xba3), row(&mut gl, &mut c, 0xba4), row(&mut gl, &mut c, 0xbb0)), (vec![3], vec![1], vec![0]), "stack depths count the entries");
        // Floats stay floats.
        set(&mut gl, SET_LINE_WIDTH, &[w(2.5), 0]);
        set(&mut gl, SET_POINT_SIZE, &[w(4.0), 0]);
        assert_eq!((row(&mut gl, &mut c, 0xb21), row(&mut gl, &mut c, 0xb11)), (vec![w(2.5)], vec![w(4.0)]));
        assert_eq!(row(&mut gl, &mut c, 0xba6), floats(gl.mv.get()), "a matrix");
        // A colour is four floats the client scales, and the alpha reference one.
        set(&mut gl, SET_COLOUR_CLEAR_VALUE, &[w(0.25), w(0.5), w(0.75), w(1.0)]);
        set(&mut gl, SET_ALPHA_TEST, &[0x204, w(0.5)]);
        assert_eq!((row(&mut gl, &mut c, 0xc22), row(&mut gl, &mut c, 0xbc2), row(&mut gl, &mut c, 0xbc1)), (vec![w(0.25), w(0.5), w(0.75), w(1.0)], vec![w(0.5)], vec![0x204]));
        set(&mut gl, SET_CONSTANT_BLEND_COLOUR, &[w(0.125), 0, 0, w(1.0)]);
        assert_eq!(row(&mut gl, &mut c, 0x8005), vec![w(0.125), 0, 0, w(1.0)]);
        // The depth range, as two doubles.
        set(&mut gl, SET_DEPTH_RANGE, &dbl(&[0.25, 0.75]));
        let doubles = |r: Vec<u32>| r.chunks(2).map(|p| f64::from_bits((p[0] as u64) << 32 | p[1] as u64)).collect::<Vec<_>>();
        assert_eq!(doubles(row(&mut gl, &mut c, 0xb70)), [0.25, 0.75]);
        assert!(c.unknown_state.is_empty());
    }

    fn dbl(v: &[f64]) -> Vec<u32> {
        v.iter().flat_map(|d| [(d.to_bits() >> 32) as u32, d.to_bits() as u32]).collect()
    }

    #[test]
    fn the_buffer_rows_follow_the_draw_and_read_buffers_and_a_pixel_maps_size_its_values() {
        let mut gl = Gl::default();
        let mut c = Counts::default();
        assert_eq!((row(&mut gl, &mut c, 0xc01), row(&mut gl, &mut c, 0xc02)), (vec![gl::FRONT], vec![gl::FRONT]), "a single-buffered window starts on the front");
        gl.double = true;
        assert_eq!(row(&mut gl, &mut c, 0xc01), vec![gl::BACK], "and a double-buffered one on the back");
        set(&mut gl, SELECT_DRAW_BUFFER, &[gl::FRONT_AND_BACK]);
        assert_eq!(row(&mut gl, &mut c, 0xc01), vec![gl::FRONT_AND_BACK]);
        set(&mut gl, SELECT_DRAW_BUFFER, &[0]);
        assert_eq!(row(&mut gl, &mut c, 0xc01), vec![0], "none");
        assert_eq!(row(&mut gl, &mut c, 0xcb2), vec![1], "no map loaded: OpenGL's start-up size");
        gl.pix.maps.insert(0xc72, crate::dev::kona::back::gl::pixels::Map { max: 1.0, values: vec![0.0; 16] });
        assert_eq!((row(&mut gl, &mut c, 0xcb2), row(&mut gl, &mut c, 0xcb3)), (vec![16], vec![1]), "the size of the map that was loaded");
        gl.feed.names = vec![4, 5];
        assert_eq!(row(&mut gl, &mut c, 0xd70), vec![2], "the name stack's depth");
    }

    /// A reply is as long as the row says, and a row nothing holds state for is the zeros of that length and
    /// counted by name (the model keeps no hint, index mask, colour-material choice, colour table or
    /// convolution state, and the polygon-offset units' native type is awaiting a policy).
    #[test]
    fn a_row_with_no_state_behind_it_is_zeros_of_its_length_and_counted() {
        let gl = Gl::default();
        let mut c = Counts::default();
        let mut gl2 = gl.clone();
        let mut not_kept = Vec::new();
        for &(name, words) in state_query::ROWS.iter() {
            match held(&gl, name) {
                Some(w) => assert_eq!(w.len(), words as usize, "{name:#x}: the row's word count"),
                None => not_kept.push(name),
            }
            let r = row(&mut gl2, &mut c, name);
            assert_eq!(r.len(), words as usize, "{name:#x}");
        }
        assert_eq!(not_kept.len(), 48);
        for hint in [0xc50, 0xc51, 0xc52, 0xc53, 0xc54, 0x2a00, 0x8009, 0x8010, 0x8011, 0x8012, 0x80b1] {
            assert!(not_kept.contains(&hint), "{hint:#x}");
            assert_eq!(c.unknown_state.get(&hint), Some(&1), "{hint:#x}");
        }
        assert!(c.unknown_state.keys().all(|n| not_kept.contains(n)));
        assert_eq!(gl2.error, 0, "a row the model has no state for is not an error");
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
