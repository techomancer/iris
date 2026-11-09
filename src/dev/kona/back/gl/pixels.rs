//! GL pixel operations: the pixel-store, transfer and zoom state, draw, read and copy of
//! pixel rectangles, bitmaps and the accumulation buffer.
//!
//! The tokens and their layouts are the definitions table's section on GL pixel
//! operations. What the table grades established is taken as stated. Where it grades
//! a thing inferred or unknown, this model makes the simplest choice and says so at the
//! choice; each is counted in `Counts`, which `kona status` shows. The choices:
//!
//! - The pixel path is the standard OpenGL one: unpack, scale and bias, index shift and
//!   offset, the pixel maps, then a fragment at each destination pixel (zoom by replication)
//!   through the fragment operations after ownership and the stipples, with the raster
//!   position's colour and depth. The table records the pipe's state but not which stages
//!   it runs, so the stages are GL's (approximated). Post-convolution and post-colour-matrix
//!   scale and bias are stored and not applied (counted).
//! - A packed type's component layout is the public one (the table does not record the
//!   pipe's unpacker). Format 8 (the ABGR extension) is read and written with its components
//!   in the order alpha, blue, green, red, in the plain types only. Formats 9 and 10 (CMYK and
//!   CMYKA) are sized by the pipe's component table (four and five components) so the words
//!   that follow a header are consumed, and refused. Type codes 14 to 20 are refused (their
//!   element sizes are known, their bit layouts are not).
//! - A colour-index image has no index window to land in: it goes through the
//!   index-to-colour pixel maps when they are loaded, else as an eight-bit grey (the
//!   model's own).
//! - Accumulation keeps one float per component in the layout's accumulation words (three
//!   words, or four with alpha). The storage format in frame-buffer memory is unknown, so the
//!   floats are the model's own. An operation is one pass, except that a value of exactly 1.0
//!   (which does not fit the 13-bit value field, inferred) runs accumulate and add twice at 0.5,
//!   load as a load at 0.5 then an accumulate at 0.5, and multiply by 1.0 as nothing; a return
//!   is always one pass (established). A return writes the draw buffers under the colour
//!   mask and ownership, and no blend or test.
//! - A read's reply is the packed image at offset 8 of the return page when it fits
//!   `READ_REPLY_LIMIT` bytes, led by the destination's byte offset in its first word. A
//!   read that would go by DMA, a video-source copy and the convolution queries' wider
//!   replies are not modelled (counted).
//! - The overlap query (token 0x110) answers whether the copy's source and destination
//!   rectangles meet. The convolution overlap query's reply (0x115) is that bit and nothing
//!   else; the single-component query (0x116) answers a zero word (the table says it is a
//!   result record's mode word, whose contents are unknown).
//! - The table's word-count column gives tokens 0x005, 0x006, 0x03a and 0x111 seven, seven,
//!   seven and one words. The front end frames them with eight, nine, eight and two, which is
//!   what the recording and the table's text show (the recording's bitmaps are eight words
//!   and an image); the model follows the framer.

use super::state::{f, Gl};
use super::vertex::{colour13, depth32};
use crate::dev::gl::math;
use crate::dev::kona::back::raster::ops::ownership_plane;
use crate::dev::kona::back::raster::pixel::Planes;
use crate::dev::kona::back::raster::DrawState;
use crate::dev::kona::back::store::{PixelStore, WordArray, DID_PLANE};
use crate::dev::kona::defs::pixel::*;
use crate::dev::kona::defs::*;
use crate::dev::kona::stream::Stream;
use std::collections::BTreeMap;

/// Most values a pixel map keeps. The pipe's and the client's table sizes are unknown (model cap).
pub const MAP_LIMIT: u32 = 1 << 16;

/// A pixel-store record: the unpack side's or the pack side's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    pub swap_bytes: bool,
    pub lsb_first: bool,
    pub row_length: u32,
    pub skip_rows: u32,
    pub skip_pixels: u32,
    pub alignment: u32,
    /// Skip images, image height, skip volumes and image depth, kept and unused (no volumes).
    pub volume: [u32; 4],
}

impl Default for Record {
    fn default() -> Self {
        Record { swap_bytes: false, lsb_first: false, row_length: 0, skip_rows: 0, skip_pixels: 0, alignment: 4, volume: [0; 4] }
    }
}

impl Record {
    /// The record display-list images are read with: unpadded.
    fn list() -> Record {
        Record { alignment: 1, ..Record::default() }
    }

    /// Set the field a name in a group (0 to 5 from the group's base) or a volume name
    /// stands for. An alignment other than 1, 2, 4 and 8 is stored and gives the pipe a zero
    /// mask (inferred from its table), which `mask` returns.
    fn set(&mut self, index: u32, value: u32) {
        match index {
            0 => self.swap_bytes = value != 0,
            1 => self.lsb_first = value != 0,
            2 => self.row_length = value,
            3 => self.skip_rows = value,
            4 => self.skip_pixels = value,
            5 => self.alignment = value,
            _ => {}
        }
    }

    fn mask(&self) -> usize {
        match self.alignment {
            2 => 1,
            4 => 3,
            8 => 7,
            _ => 0,
        }
    }

    fn visit<S: Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { swap_bytes: _, lsb_first: _, row_length: _, skip_rows: _, skip_pixels: _, alignment: _, volume: _ } = self;
        s.bool(&mut self.swap_bytes);
        s.bool(&mut self.lsb_first);
        s.u32(&mut self.row_length);
        s.u32(&mut self.skip_rows);
        s.u32(&mut self.skip_pixels);
        s.u32(&mut self.alignment);
        s.u32s(&mut self.volume);
    }
}

/// A pixel map's values and the form they arrived in (float, or an unsigned integer of 32 or 16 bits,
/// which a colour lookup divides down to 0 to 1).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Map {
    pub max: f32,
    pub values: Vec<f32>,
}

impl Map {
    fn at(&self, v: f32) -> f32 {
        let n = self.values.len();
        let i = (v.clamp(0.0, 1.0) * (n - 1) as f32 + 0.5) as usize;
        let raw = self.values[i.min(n - 1)];
        if self.max > 0.0 { raw / self.max } else { raw }
    }

    /// An integer lookup (index or stencil), clamped to the table.
    fn index(&self, i: i64) -> i64 {
        let n = self.values.len() as i64;
        self.values[i.clamp(0, n - 1) as usize] as i64
    }
}

/// The current raster position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RasterPos {
    pub valid: bool,
    /// Window x and y, and depth as 32 bits.
    pub x: f32,
    pub y: f32,
    pub z: u32,
    pub colour: [f32; 4],
}

impl Default for RasterPos {
    fn default() -> Self {
        RasterPos { valid: true, x: 0.0, y: 0.0, z: 0, colour: [1.0; 4] }
    }
}

/// What a context's pixel operations keep.
#[derive(Clone, Debug, PartialEq)]
pub struct Pixels {
    pub unpack: Record,
    pub pack: Record,
    pub scale: [f32; 4],
    pub bias: [f32; 4],
    pub depth_scale: f32,
    pub depth_bias: f32,
    pub index_shift: i32,
    pub index_offset: i32,
    pub map_colour: bool,
    pub map_stencil: bool,
    /// Post-convolution scales then biases, post-colour-matrix scales then biases.
    pub post: [f32; 16],
    pub zoom: [f32; 2],
    pub maps: BTreeMap<u32, Map>,
    /// Between a suspend's save and its restore the fixed-function stages are off.
    pub suspended: bool,
    /// Which record a suspend switched the unpack side to: 0 the context's own.
    pub records: u32,
    /// The read buffer: 1 front, 2 back; None is the draw default.
    pub read: Option<u32>,
    pub pos: RasterPos,
    pub accum_clear: [f32; 4],
    /// A draw announced by DMA waits here for its packet: the header words.
    pub pending: Option<[u32; 8]>,
}

impl Default for Pixels {
    fn default() -> Self {
        let mut post = [0.0; 16];
        post[..4].fill(1.0);
        post[8..12].fill(1.0);
        Pixels {
            unpack: Record::default(),
            pack: Record::default(),
            scale: [1.0; 4],
            bias: [0.0; 4],
            depth_scale: 1.0,
            depth_bias: 0.0,
            index_shift: 0,
            index_offset: 0,
            map_colour: false,
            map_stencil: false,
            post,
            zoom: [1.0; 2],
            maps: BTreeMap::new(),
            suspended: false,
            records: 0,
            read: None,
            pos: RasterPos::default(),
            accum_clear: [0.0; 4],
            pending: None,
        }
    }
}

impl Pixels {
    pub fn visit<S: Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { unpack: _, pack: _, scale: _, bias: _, depth_scale: _, depth_bias: _, index_shift: _, index_offset: _, map_colour: _, map_stencil: _, post: _, zoom: _, maps: _, suspended: _, records: _, read: _, pos: _, accum_clear: _, pending: _ } = self;
        self.unpack.visit(s);
        self.pack.visit(s);
        s.f32s(&mut self.scale);
        s.f32s(&mut self.bias);
        s.f32(&mut self.depth_scale);
        s.f32(&mut self.depth_bias);
        s.i32(&mut self.index_shift);
        s.i32(&mut self.index_offset);
        s.bool(&mut self.map_colour);
        s.bool(&mut self.map_stencil);
        s.f32s(&mut self.post);
        s.f32s(&mut self.zoom);
        let mut n = self.maps.len() as u32;
        s.u32(&mut n);
        if s.loading() {
            self.maps.clear();
            for _ in 0..n.min(64) {
                if !s.ok() {
                    break;
                }
                let (mut name, mut max, mut bits) = (0u32, 0f32, Vec::new());
                s.u32(&mut name);
                s.f32(&mut max);
                s.vec_u32(&mut bits);
                self.maps.insert(name, Map { max, values: bits.into_iter().map(f32::from_bits).collect() });
            }
        } else {
            for (name, m) in &self.maps {
                let (mut name, mut max, mut bits) = (*name, m.max, m.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                s.u32(&mut name);
                s.f32(&mut max);
                s.vec_u32(&mut bits);
            }
        }
        s.bool(&mut self.suspended);
        s.u32(&mut self.records);
        s.opt(&mut self.read, |s, v| s.u32(v));
        s.bool(&mut self.pos.valid);
        s.f32(&mut self.pos.x);
        s.f32(&mut self.pos.y);
        s.u32(&mut self.pos.z);
        s.f32s(&mut self.pos.colour);
        s.f32s(&mut self.accum_clear);
        let mut has = self.pending.is_some();
        s.bool(&mut has);
        if has {
            let mut h = self.pending.unwrap_or_default();
            s.u32s(&mut h);
            self.pending = Some(h);
        } else {
            self.pending = None;
        }
    }

    /// The unpack record in force: a suspend can switch to a fixed one.
    pub fn unpack_in_force(&self) -> Record {
        match self.records {
            SUSPEND_USE_DEFAULT_RECORDS => Record::default(),
            SUSPEND_USE_LIST_RECORD => Record::list(),
            _ => self.unpack,
        }
    }
}

/// What the pixel operations did, for `kona status`.
#[derive(Clone, Debug, Default)]
pub struct Counts {
    pub draws: u64,
    pub reads: u64,
    pub copies: u64,
    pub bitmaps: u64,
    pub accumulations: u64,
    pub fragments: u64,
    /// Operations refused for a format or type the model does not know, or an image too short.
    pub refused: u64,
    /// Draws, copies and bitmaps with an invalid raster position, which draw nothing.
    pub invalid_position: u64,
    /// Reads that would go by DMA, which are not modelled.
    pub read_by_dma: u64,
    /// Queries answered with a model choice (overlap, convolution overlap, single component).
    pub queries: u64,
    /// Transfers that had post-convolution or post-colour-matrix terms set, which are not applied.
    pub post_stages_ignored: u64,
    /// Colour-index images drawn without an index-to-colour map.
    pub index_without_map: u64,
    /// Pixel maps longer than `MAP_LIMIT`, or named in none of the groups.
    pub maps_refused: u64,
    /// Parameter names the transfer or store tokens did not recognise.
    pub unknown_names: u64,
    /// Tokens nothing sends (the float and integer sibling forms), seen anyway.
    pub unsent: u64,
    /// Current-raster-position token 0x10e, whose words the table does not describe.
    pub list_positions: u64,
    /// Draw announced by DMA whose packet arrived.
    pub dma_draws: u64,
}

impl Counts {
    pub fn render(&self) -> String {
        format!(
            "{} draws ({} by DMA), {} reads, {} copies, {} bitmaps, {} accumulations; {} fragments; {} refused, {} with no valid raster position, {} reads by DMA not modelled, {} queries answered by the model, {} transfers with post stages not applied, {} index images without a map, {} maps refused, {} unknown names, {} unsent tokens, {} list positions",
            self.draws, self.dma_draws, self.reads, self.copies, self.bitmaps, self.accumulations, self.fragments, self.refused, self.invalid_position, self.read_by_dma, self.queries, self.post_stages_ignored, self.index_without_map, self.maps_refused, self.unknown_names, self.unsent, self.list_positions
        )
    }
}

/// The tokens `command` takes.
pub fn is_state(selector: u32) -> bool {
    matches!(
        selector,
        SET_PIXEL_STORE_PARAMETER_INTEGER_FORM
            | SET_PIXEL_STORE_PARAMETER_FLOAT_FORM
            | SET_PIXEL_TRANSFER_PARAMETER_FLOAT_FORM
            | SET_PIXEL_TRANSFER_PARAMETER_INTEGER_FORM
            | SET_PIXEL_ZOOM_FACTORS
            | SUSPEND_PIXEL_MODE_STATE_AROUND_TRANSFER
            | LOAD_PIXEL_MAP_FLOAT_VALUES
            | LOAD_PIXEL_MAP_UNSIGNED_32_BIT_VALUES
            | LOAD_PIXEL_MAP_UNSIGNED_16_BIT_VALUES
            | SELECT_READ_BUFFER
            | SET_ACCUMULATION_BUFFER_CLEAR_VALUE
            | SET_RASTER_POSITION_2_COMPONENTS_DOUBLE..=SET_RASTER_POSITION_4_COMPONENTS_SHORT
            | SET_CURRENT_RASTER_POSITION
    )
}

/// The tokens whose data run `Job::Data` brings back.
pub fn is_data(selector: u32) -> bool {
    matches!(selector, LOAD_PIXEL_MAP_FLOAT_VALUES | LOAD_PIXEL_MAP_UNSIGNED_32_BIT_VALUES | LOAD_PIXEL_MAP_UNSIGNED_16_BIT_VALUES | DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA | DRAW_BITMAP_FROM_DISPLAY_LIST | DRAW_BITMAP)
}

/// The tokens that read or write the planes: they wait for the drawing queued before them.
pub fn is_plane(selector: u32) -> bool {
    matches!(selector, DRAW_BITMAP_FROM_DISPLAY_LIST | DRAW_BITMAP | DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA | COPY_PIXEL_RECTANGLE_WITHIN_FRAME_BUFFER | READ_PIXEL_RECTANGLE_BACK_TO_HOST | ACCUMULATION_BUFFER_OPERATION | ASK_WHETHER_COPY_SOURCE_OVERLAPS_ITS_DESTINATION | ASK_WHETHER_CONVOLUTION_COPY_OVERLAPS | RESOLVE_WHETHER_PIXEL_FORMAT_IS_SINGLE_COMPONENT)
}

/// Pixel-state tokens: stored in the context, no plane touched. False for a token not
/// in `is_state`. (The stored values are GL's names and values as the client sends them.)
pub fn command(gl: &mut Gl, c: &mut Counts, selector: u32, a: &[u32]) -> bool {
    let p = &mut gl.pix;
    match (selector, a) {
        (SET_PIXEL_STORE_PARAMETER_INTEGER_FORM, &[name, value]) => {
            let g = |base: u32| (name >= base && name < base + STORE_NAMES).then(|| name - base);
            match name {
                UNPACK_SKIP_IMAGES | UNPACK_IMAGE_HEIGHT | UNPACK_SKIP_VOLUMES | UNPACK_IMAGE_DEPTH => p.unpack.volume[[UNPACK_SKIP_IMAGES, UNPACK_IMAGE_HEIGHT, UNPACK_SKIP_VOLUMES, UNPACK_IMAGE_DEPTH].iter().position(|&n| n == name).unwrap_or(0)] = value,
                PACK_SKIP_IMAGES | PACK_IMAGE_HEIGHT | PACK_SKIP_VOLUMES | PACK_IMAGE_DEPTH => p.pack.volume[[PACK_SKIP_IMAGES, PACK_IMAGE_HEIGHT, PACK_SKIP_VOLUMES, PACK_IMAGE_DEPTH].iter().position(|&n| n == name).unwrap_or(0)] = value,
                _ => match (g(UNPACK_BASE), g(PACK_BASE)) {
                    (Some(i), _) => p.unpack.set(i, value),
                    (_, Some(i)) => p.pack.set(i, value),
                    _ => c.unknown_names += 1,
                },
            }
            true
        }
        (SET_PIXEL_STORE_PARAMETER_FLOAT_FORM | SET_PIXEL_TRANSFER_PARAMETER_INTEGER_FORM, _) => {
            c.unsent += 1;
            true
        }
        (SET_PIXEL_TRANSFER_PARAMETER_FLOAT_FORM, &[name, value]) => {
            let v = f(value);
            match name {
                RED_SCALE => p.scale[0] = v,
                GREEN_SCALE => p.scale[1] = v,
                BLUE_SCALE => p.scale[2] = v,
                ALPHA_SCALE => p.scale[3] = v,
                RED_BIAS => p.bias[0] = v,
                GREEN_BIAS => p.bias[1] = v,
                BLUE_BIAS => p.bias[2] = v,
                ALPHA_BIAS => p.bias[3] = v,
                DEPTH_SCALE => p.depth_scale = v,
                DEPTH_BIAS => p.depth_bias = v,
                // The client rounds half away from zero and sends an integer.
                INDEX_SHIFT => p.index_shift = value as i32,
                INDEX_OFFSET => p.index_offset = value as i32,
                MAP_COLOUR => p.map_colour = value != 0,
                MAP_STENCIL => p.map_stencil = value != 0,
                n if (POST_CONVOLUTION_SCALE..POST_CONVOLUTION_SCALE + 8).contains(&n) => p.post[(n - POST_CONVOLUTION_SCALE) as usize] = v,
                n if (POST_COLOUR_MATRIX_SCALE..POST_COLOUR_MATRIX_SCALE + 8).contains(&n) => p.post[8 + (n - POST_COLOUR_MATRIX_SCALE) as usize] = v,
                _ => c.unknown_names += 1,
            }
            true
        }
        (SET_PIXEL_ZOOM_FACTORS, &[x, y]) => {
            p.zoom = [f(x), f(y)];
            true
        }
        // Word 0 says which record the unpack side uses, and bit 29 of the mode word switches to it
        // (the client's own records otherwise). Mode zero restores; bit 30 saves, and turns the
        // fixed-function stages off.
        (SUSPEND_PIXEL_MODE_STATE_AROUND_TRANSFER, &[selector, mode]) => {
            p.suspended = mode & SUSPEND_SAVE != 0;
            p.records = if mode & SUSPEND_SWITCH_RECORDS != 0 && matches!(selector, SUSPEND_USE_DEFAULT_RECORDS | SUSPEND_USE_LIST_RECORD) { selector } else { 0 };
            true
        }
        (LOAD_PIXEL_MAP_FLOAT_VALUES | LOAD_PIXEL_MAP_UNSIGNED_32_BIT_VALUES | LOAD_PIXEL_MAP_UNSIGNED_16_BIT_VALUES, _) => true,
        (SELECT_READ_BUFFER, &[mask]) => {
            p.read = Some(match mask {
                gl::FRONT | gl::FRONT_LEFT | gl::RIGHT => 1,
                gl::BACK | gl::BACK_LEFT => 2,
                _ => 1,
            });
            true
        }
        (SET_ACCUMULATION_BUFFER_CLEAR_VALUE, &[r, g, b, al]) => {
            p.accum_clear = [f(r), f(g), f(b), f(al)];
            true
        }
        (SET_CURRENT_RASTER_POSITION, _) => {
            c.list_positions += 1;
            true
        }
        (SET_RASTER_POSITION_2_COMPONENTS_FLOAT, &[x, y]) => raster_position(gl, [f(x), f(y), 0.0, 1.0]),
        (SET_RASTER_POSITION_3_COMPONENTS_FLOAT, &[x, y, z]) => raster_position(gl, [f(x), f(y), f(z), 1.0]),
        (SET_RASTER_POSITION_4_COMPONENTS_FLOAT, &[x, y, z, w]) => raster_position(gl, [f(x), f(y), f(z), f(w)]),
        (SET_RASTER_POSITION_2_COMPONENTS_32_BIT_INTEGER, &[x, y]) => raster_position(gl, [x as i32 as f32, y as i32 as f32, 0.0, 1.0]),
        (SET_RASTER_POSITION_3_COMPONENTS_32_BIT_INTEGER, &[x, y, z]) => raster_position(gl, [x as i32 as f32, y as i32 as f32, z as i32 as f32, 1.0]),
        (SET_RASTER_POSITION_4_COMPONENTS_32_BIT_INTEGER, &[x, y, z, w]) => raster_position(gl, [x as i32 as f32, y as i32 as f32, z as i32 as f32, w as i32 as f32]),
        // A short in each word's low half (inferred: the table counts one word a component).
        (SET_RASTER_POSITION_2_COMPONENTS_SHORT, &[x, y]) => raster_position(gl, [x as i16 as f32, y as i16 as f32, 0.0, 1.0]),
        (SET_RASTER_POSITION_3_COMPONENTS_SHORT, &[x, y, z]) => raster_position(gl, [x as i16 as f32, y as i16 as f32, z as i16 as f32, 1.0]),
        (SET_RASTER_POSITION_4_COMPONENTS_SHORT, &[x, y, z, w]) => raster_position(gl, [x as i16 as f32, y as i16 as f32, z as i16 as f32, w as i16 as f32]),
        // The double forms: two words a component, high word first (as the matrix doubles are).
        (SET_RASTER_POSITION_2_COMPONENTS_DOUBLE, &[..]) if a.len() == 4 => raster_position(gl, [double(a, 0), double(a, 1), 0.0, 1.0]),
        (SET_RASTER_POSITION_3_COMPONENTS_DOUBLE, &[..]) if a.len() == 6 => raster_position(gl, [double(a, 0), double(a, 1), double(a, 2), 1.0]),
        (SET_RASTER_POSITION_4_COMPONENTS_DOUBLE, &[..]) if a.len() == 8 => raster_position(gl, [double(a, 0), double(a, 1), double(a, 2), double(a, 3)]),
        _ => false,
    }
}

fn double(a: &[u32], i: usize) -> f32 {
    f64::from_bits((a[2 * i] as u64) << 32 | a[2 * i + 1] as u64) as f32
}

/// Take a pixel map: four header words and the values, as the framer hands them (`a` is the
/// header, `words` the values). Only the map names the table lists are kept.
pub fn load_map(gl: &mut Gl, c: &mut Counts, selector: u32, a: &[u32], words: &[u32]) {
    let (max, value): (f32, fn(u32) -> f32) = match selector {
        LOAD_PIXEL_MAP_FLOAT_VALUES => (0.0, f),
        LOAD_PIXEL_MAP_UNSIGNED_32_BIT_VALUES => (4_294_967_295.0, |w| w as f32),
        _ => (65_535.0, |w| (w & 0xffff) as f32),
    };
    let name = a.first().copied().unwrap_or(0);
    let known = (MAP_INDEX_TO_INDEX..MAP_INDEX_TO_INDEX + 2 + 4 + 4).contains(&name);
    if !known || words.is_empty() || words.len() as u32 > MAP_LIMIT {
        c.maps_refused += 1;
        return;
    }
    gl.pix.maps.insert(name, Map { max, values: words.iter().map(|&w| value(w)).collect() });
}

/// The current raster position from an object-space position: through the model-view and
/// projection matrices, valid when inside the clip volume; colour the current colour (unlit:
/// the model's own, as lighting at the raster position is not modelled).
fn raster_position(gl: &mut Gl, p: [f32; 4]) -> bool {
    let e = math::xform(gl.mv.get(), p);
    let h = math::xform(gl.proj.get(), e);
    let w = h[3];
    let valid = w > 0.0 && h[..3].iter().all(|v| v.abs() <= w);
    let [vx, vy, vw, vh] = gl.raster.viewport;
    let [dn, df] = gl.raster.depth_range;
    let (nx, ny, nz) = if w > 0.0 { (h[0] / w, h[1] / w, h[2] / w) } else { (0.0, 0.0, 0.0) };
    gl.pix.pos = RasterPos { valid, x: vx + (nx + 1.0) * 0.5 * vw, y: vy + (ny + 1.0) * 0.5 * vh, z: depth32(dn + (nz * 0.5 + 0.5) * (df - dn)), colour: gl.color };
    true
}

// ---- Images ----

/// A pixel rectangle's components, by the kind of data.
#[derive(Clone, Debug, PartialEq)]
pub enum Plane {
    Colour(Vec<[f32; 4]>),
    Index(Vec<i64>),
    Stencil(Vec<i64>),
    Depth(Vec<f32>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub plane: Plane,
}

/// Components in a pixel of `format`, or None for a code the model does not know.
fn components(format: u32) -> Option<usize> {
    match format {
        FORMAT_RED | FORMAT_GREEN | FORMAT_BLUE | FORMAT_ALPHA | FORMAT_LUMINANCE | FORMAT_COLOUR_INDEX | FORMAT_STENCIL | FORMAT_DEPTH => Some(1),
        FORMAT_LUMINANCE_ALPHA => Some(2),
        FORMAT_RGB => Some(3),
        FORMAT_RGBA | FORMAT_ABGR | FORMAT_CMYK => Some(4),
        FORMAT_CMYKA => Some(5),
        _ => None,
    }
}

/// Bytes in an element of `ty`, or the whole pixel for a packed type (flag second).
fn element_bytes(ty: u32) -> Option<(usize, bool)> {
    match ty {
        TYPE_BYTE | TYPE_UNSIGNED_BYTE => Some((1, false)),
        TYPE_SHORT | TYPE_UNSIGNED_SHORT => Some((2, false)),
        TYPE_INT | TYPE_UNSIGNED_INT | TYPE_FLOAT => Some((4, false)),
        TYPE_3_3_2 => Some((1, true)),
        TYPE_4_4_4_4 | TYPE_5_5_5_1 => Some((2, true)),
        TYPE_8_8_8_8 | TYPE_10_10_10_2 => Some((4, true)),
        _ => None,
    }
}

/// Bytes a pixel takes (a bitmap's is one bit: reported as 0 here and handled by the callers).
fn pixel_bytes(format: u32, ty: u32) -> Option<usize> {
    let n = components(format)?;
    if ty == TYPE_BITMAP {
        return (n == 1).then_some(0);
    }
    let (b, packed) = element_bytes(ty)?;
    if packed {
        // The ABGR extension's packed layouts are not recorded.
        if format == FORMAT_ABGR {
            return None;
        }
        // A packed type is a whole pixel: RGB for the 3-3-2 form, RGBA for the others.
        let want = if ty == TYPE_3_3_2 { 3 } else { 4 };
        return (n == want).then_some(b);
    }
    Some(b * n)
}

/// An element's bytes as an unsigned integer in the client's order, swapped if asked.
fn word_at(bytes: &[u8], at: usize, n: usize, swap: bool) -> Option<u32> {
    let s = bytes.get(at..at + n)?;
    let mut v = 0u32;
    if swap {
        for &b in s.iter().rev() {
            v = v << 8 | b as u32;
        }
    } else {
        for &b in s {
            v = v << 8 | b as u32;
        }
    }
    Some(v)
}

/// An element as a number in 0 to 1 (signed types: -1 to 1 by GL's rule), or for an index or
/// stencil image the integer itself.
fn element(bytes: &[u8], at: usize, ty: u32, swap: bool, integer: bool) -> Option<f32> {
    let (n, _) = element_bytes(ty)?;
    let w = word_at(bytes, at, n, swap)?;
    Some(match ty {
        TYPE_FLOAT => f32::from_bits(w),
        _ if integer => match ty {
            TYPE_BYTE => w as u8 as i8 as f32,
            TYPE_SHORT => w as u16 as i16 as f32,
            TYPE_INT => w as i32 as f32,
            _ => w as f32,
        },
        TYPE_UNSIGNED_BYTE => w as f32 / 255.0,
        TYPE_UNSIGNED_SHORT => w as f32 / 65535.0,
        TYPE_UNSIGNED_INT => (w as f64 / 4_294_967_295.0) as f32,
        TYPE_BYTE => (2.0 * (w as u8 as i8) as f32 + 1.0) / 255.0,
        TYPE_SHORT => (2.0 * (w as u16 as i16) as f32 + 1.0) / 65535.0,
        TYPE_INT => ((2.0 * (w as i32) as f64 + 1.0) / 4_294_967_295.0) as f32,
        _ => return None,
    })
}

/// A packed pixel's components, public layouts: red, green, blue, alpha from the high bits.
fn unpack_packed(ty: u32, w: u32) -> [f32; 4] {
    let g = |shift: u32, bits: u32| ((w >> shift) & ((1 << bits) - 1)) as f32 / ((1u32 << bits) - 1) as f32;
    match ty {
        TYPE_3_3_2 => [g(5, 3), g(2, 3), g(0, 2), 1.0],
        TYPE_4_4_4_4 => [g(12, 4), g(8, 4), g(4, 4), g(0, 4)],
        TYPE_5_5_5_1 => [g(11, 5), g(6, 5), g(1, 5), g(0, 1)],
        TYPE_8_8_8_8 => [g(24, 8), g(16, 8), g(8, 8), g(0, 8)],
        _ => [g(22, 10), g(12, 10), g(2, 10), g(0, 2)],
    }
}

/// Bytes a draw's header has the stream carry. The pipe computes this from the unpack state.
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub w: u32,
    pub h: u32,
    pub format: u32,
    pub ty: u32,
    pub offset: u32,
    pub dma: bool,
    pub row_mode: u32,
    pub column: u32,
}

impl Header {
    /// The eight words of a pixel draw's header (section 7.15).
    pub fn parse(a: &[u32]) -> Option<Header> {
        let a: &[u32; 8] = a.get(..8)?.try_into().ok()?;
        Some(Header { w: a[0], h: a[1], format: a[2], ty: a[3], offset: a[4] & 3, dma: a[5] != 0, row_mode: a[6], column: a[7] })
    }

    fn geometry(&self, rec: &Record) -> Option<Geometry> {
        let p = pixel_bytes(self.format, self.ty)?;
        let bits = self.ty == TYPE_BITMAP;
        // A row length below the width puts a row's end before its start: no size, and the draw is refused (counted).
        if rec.row_length != 0 && rec.row_length < self.w {
            return None;
        }
        let rl = if rec.row_length > 0 { rec.row_length } else { self.w } as usize;
        let r = if bits { rl.div_ceil(8) } else { rl * p };
        let mask = rec.mask();
        let s = (r + mask) & !mask;
        // The unpack state is the guest's: a size past what fits is no size, and the draw is refused (counted).
        let skip = (rec.skip_rows as usize).checked_mul(s)?.checked_add(if bits { rec.skip_pixels as usize / 8 } else { rec.skip_pixels as usize * p })?;
        Some(Geometry { p, bits, row: r, stride: s, start: (self.offset as usize).checked_add(skip)?, bit_skip: if bits { rec.skip_pixels as usize % 8 } else { 0 } })
    }

    /// The words the image takes in the stream, or None for a format or type the model does not know.
    pub fn image_words(&self, rec: &Record) -> Option<u32> {
        let g = self.geometry(rec)?;
        let h = self.h as usize;
        if h == 0 || self.w == 0 {
            return Some(0);
        }
        if self.row_mode == 0 {
            let total = g.stride.checked_mul(h)?.checked_sub(g.stride.checked_sub(g.row)?)?;
            u32::try_from(g.start.checked_add(total)?.div_ceil(4) - g.start / 4).ok()
        } else {
            // A row's first byte sits at `(start + j * stride) % 4`, which repeats every four rows.
            let row = |j: usize| (g.start.wrapping_add(j.wrapping_mul(g.stride)) % 4 + g.row).div_ceil(4) as u64;
            let (cycles, rest) = (h / 4, h % 4);
            let sum = (cycles as u64).checked_mul((0..4).map(row).sum::<u64>())?.checked_add((0..rest).map(row).sum::<u64>())?;
            u32::try_from(sum).ok()
        }
    }
}

struct Geometry {
    p: usize,
    bits: bool,
    row: usize,
    stride: usize,
    start: usize,
    bit_skip: usize,
}

/// Decode a draw's image from its stream words.
pub fn unpack(h: &Header, rec: &Record, bytes: &[u8]) -> Option<Image> {
    if matches!(h.format, FORMAT_CMYK | FORMAT_CMYKA) {
        return None;
    }
    let g = h.geometry(rec)?;
    let (w, ht) = (h.w as usize, h.h as usize);
    let packed = element_bytes(h.ty).is_some_and(|e| e.1);
    let esize = element_bytes(h.ty).map_or(0, |e| e.0);
    let n = components(h.format)?;
    // Where row `j` starts in `bytes`.
    let mut row_at = Vec::with_capacity(ht);
    let mut block = 0usize;
    for j in 0..ht {
        if h.row_mode == 0 {
            row_at.push(g.start % 4 + j * g.stride);
        } else {
            let o = (g.start + j * g.stride) % 4;
            row_at.push(block + o);
            block += (o + g.row).div_ceil(4) * 4;
        }
    }
    let swap = rec.swap_bytes;
    let int_plane = matches!(h.format, FORMAT_COLOUR_INDEX | FORMAT_STENCIL);
    let mut colour = Vec::new();
    let mut ints = Vec::new();
    let mut depth = Vec::new();
    for j in 0..ht {
        for i in 0..w {
            if g.bits {
                let bit = g.bit_skip + i;
                let byte = *bytes.get(row_at[j] + bit / 8)?;
                let v = if rec.lsb_first { byte >> (bit % 8) & 1 } else { byte >> (7 - bit % 8) & 1 };
                if int_plane {
                    ints.push(v as i64);
                } else if h.format == FORMAT_DEPTH {
                    depth.push(v as f32);
                } else {
                    colour.push(to_colour(h.format, &[v as f32; 4]));
                }
                continue;
            }
            let at = row_at[j] + i * g.p;
            if packed {
                let c = unpack_packed(h.ty, word_at(bytes, at, esize, swap)?);
                colour.push(if h.ty == TYPE_3_3_2 || h.format == FORMAT_RGB { [c[0], c[1], c[2], 1.0] } else { c });
                continue;
            }
            if int_plane {
                ints.push(element(bytes, at, h.ty, swap, true)?.round() as i64);
                continue;
            }
            let mut c = [0.0f32; 4];
            for (k, slot) in c.iter_mut().enumerate().take(n) {
                *slot = element(bytes, at + k * esize, h.ty, swap, false)?;
            }
            if h.format == FORMAT_DEPTH {
                depth.push(c[0]);
            } else {
                colour.push(to_colour(h.format, &c));
            }
        }
    }
    Some(Image {
        w,
        h: ht,
        plane: match h.format {
            FORMAT_COLOUR_INDEX => Plane::Index(ints),
            FORMAT_STENCIL => Plane::Stencil(ints),
            FORMAT_DEPTH => Plane::Depth(depth),
            _ => Plane::Colour(colour),
        },
    })
}

/// The components read from a pixel of `format`, as red, green, blue and alpha.
fn to_colour(format: u32, c: &[f32; 4]) -> [f32; 4] {
    match format {
        FORMAT_RED => [c[0], 0.0, 0.0, 1.0],
        FORMAT_GREEN => [0.0, c[0], 0.0, 1.0],
        FORMAT_BLUE => [0.0, 0.0, c[0], 1.0],
        FORMAT_ALPHA => [0.0, 0.0, 0.0, c[0]],
        FORMAT_RGB => [c[0], c[1], c[2], 1.0],
        FORMAT_LUMINANCE => [c[0], c[0], c[0], 1.0],
        FORMAT_LUMINANCE_ALPHA => [c[0], c[0], c[0], c[1]],
        FORMAT_ABGR => [c[3], c[2], c[1], c[0]],
        _ => *c,
    }
}

/// The bytes of a packed row of `w` pixels and the stride between rows, under the pack record:
/// None for a format or type the packer does not know.
pub fn row_bytes(w: usize, format: u32, ty: u32, rec: &Record) -> Option<(usize, usize)> {
    let p = pixel_bytes(format, ty)?;
    let rl = if rec.row_length > 0 { rec.row_length as usize } else { w };
    let r = if ty == TYPE_BITMAP { rl.div_ceil(8) } else { rl * p };
    let mask = rec.mask();
    Some((r, (r + mask) & !mask))
}

/// The most bytes `pack` will build. A read's size is the guest's (the pack record's skips and a row length
/// are its pixel-store state), so it is bounded before anything is allocated; the largest bracket buffer
/// the host side takes is this size too.
pub const MAX_PACK_BYTES: u64 = 1 << 26;

/// The bytes `pack` builds for a `w` by `h` image, before it pads them to whole words: the leading skips,
/// the rows and the last row. None for a format or type the packer does not know, or for a size past what
/// 64 bits hold.
pub fn packed_len(w: usize, h: usize, format: u32, ty: u32, low: u32, rec: &Record) -> Option<u64> {
    if matches!(format, FORMAT_CMYK | FORMAT_CMYKA) {
        return None;
    }
    let p = pixel_bytes(format, ty)? as u64;
    components(format)?;
    let (r, s) = row_bytes(w, format, ty, rec)?;
    let (r, s) = (r as u64, s as u64);
    let skip = if ty == TYPE_BITMAP { rec.skip_pixels as u64 / 8 } else { rec.skip_pixels as u64 * p };
    let first = (low as u64).checked_add((rec.skip_rows as u64).checked_mul(s)?)?.checked_add(skip)?;
    first.checked_add((h.max(1) as u64 - 1).checked_mul(s)?)?.checked_add(r.max(1))
}

/// Encode an image back to the host's layout under a pack record: the bytes from the
/// destination's low two address bits on, padded to whole words. None for an image
/// that would take more than `MAX_PACK_BYTES`.
pub fn pack(img: &Image, format: u32, ty: u32, low: u32, rec: &Record) -> Option<Vec<u8>> {
    let len = packed_len(img.w, img.h, format, ty, low, rec).filter(|&n| n <= MAX_PACK_BYTES)?;
    let p = pixel_bytes(format, ty)?;
    let n = components(format)?;
    let bits = ty == TYPE_BITMAP;
    let (esize, packed) = element_bytes(ty).unwrap_or((0, false));
    let (r, s) = row_bytes(img.w, format, ty, rec)?;
    let first = low as usize + rec.skip_rows as usize * s + if bits { rec.skip_pixels as usize / 8 } else { rec.skip_pixels as usize * p };
    let mut out = vec![0u8; len as usize];
    let put = |out: &mut Vec<u8>, at: usize, n: usize, v: u32| {
        for k in 0..n {
            let b = if rec.swap_bytes { v >> (8 * k) } else { v >> (8 * (n - 1 - k)) } as u8;
            out[at + k] = b;
        }
    };
    for j in 0..img.h {
        for i in 0..img.w {
            let k = j * img.w + i;
            let at = first + j * s + if bits { 0 } else { i * p };
            if bits {
                let v = match &img.plane {
                    Plane::Index(v) | Plane::Stencil(v) => v[k] & 1 != 0,
                    Plane::Depth(d) => d[k] >= 0.5,
                    Plane::Colour(c) => c[k][0] >= 0.5,
                };
                let bit = (rec.skip_pixels as usize % 8) + i;
                let m = if rec.lsb_first { 1 << (bit % 8) } else { 0x80 >> (bit % 8) };
                if v {
                    out[first + j * s + bit / 8] |= m;
                }
                continue;
            }
            // The components in the format's order.
            let comps: Vec<f32> = match &img.plane {
                Plane::Colour(c) => {
                    let c = c[k];
                    match format {
                        FORMAT_RED => vec![c[0]],
                        FORMAT_GREEN => vec![c[1]],
                        FORMAT_BLUE => vec![c[2]],
                        FORMAT_ALPHA => vec![c[3]],
                        FORMAT_RGB => vec![c[0], c[1], c[2]],
                        FORMAT_RGBA => c.to_vec(),
                        FORMAT_ABGR => vec![c[3], c[2], c[1], c[0]],
                        // Luminance is the sum of red, green and blue (GL's rule).
                        FORMAT_LUMINANCE => vec![(c[0] + c[1] + c[2]).clamp(0.0, 1.0)],
                        _ => vec![(c[0] + c[1] + c[2]).clamp(0.0, 1.0), c[3]],
                    }
                }
                Plane::Depth(d) => vec![d[k]],
                Plane::Index(v) | Plane::Stencil(v) => vec![v[k] as f32],
            };
            if packed {
                let g = |c: f32, bits: u32| ((c.clamp(0.0, 1.0) * ((1u32 << bits) - 1) as f32 + 0.5) as u32) & ((1 << bits) - 1);
                let c = &comps;
                let a = c.get(3).copied().unwrap_or(1.0);
                let w = match ty {
                    TYPE_3_3_2 => g(c[0], 3) << 5 | g(c[1], 3) << 2 | g(c[2], 2),
                    TYPE_4_4_4_4 => g(c[0], 4) << 12 | g(c[1], 4) << 8 | g(c[2], 4) << 4 | g(a, 4),
                    TYPE_5_5_5_1 => g(c[0], 5) << 11 | g(c[1], 5) << 6 | g(c[2], 5) << 1 | g(a, 1),
                    TYPE_8_8_8_8 => g(c[0], 8) << 24 | g(c[1], 8) << 16 | g(c[2], 8) << 8 | g(a, 8),
                    _ => g(c[0], 10) << 22 | g(c[1], 10) << 12 | g(c[2], 10) << 2 | g(a, 2),
                };
                put(&mut out, at, esize, w);
                continue;
            }
            let int_plane = matches!(img.plane, Plane::Index(_) | Plane::Stencil(_));
            for (m, &v) in comps.iter().enumerate().take(n) {
                let w: u32 = match ty {
                    TYPE_FLOAT => v.to_bits(),
                    _ if int_plane => v as i64 as u32,
                    TYPE_UNSIGNED_BYTE => (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32,
                    TYPE_UNSIGNED_SHORT => (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u32,
                    TYPE_UNSIGNED_INT => (v.clamp(0.0, 1.0) as f64 * 4_294_967_295.0 + 0.5) as u32,
                    TYPE_BYTE => (((v.clamp(0.0, 1.0) * 255.0 - 1.0) / 2.0).round() as i32) as u32,
                    TYPE_SHORT => (((v.clamp(0.0, 1.0) * 65535.0 - 1.0) / 2.0).round() as i32) as u32,
                    _ => (((v.clamp(0.0, 1.0) as f64 * 4_294_967_295.0 - 1.0) / 2.0).round() as i64) as u32,
                };
                put(&mut out, at + m * esize, esize, w);
            }
        }
    }
    out.resize(out.len().div_ceil(4) * 4, 0);
    Some(out)
}

// ---- The planes ----

/// Where fragments land: the draw state's planes, with ownership resolved.
struct Canvas<'a> {
    store: &'a PixelStore,
    ds: DrawState,
    planes: Planes<'a>,
    did: Option<WordArray<'a>>,
    pitch: usize,
}

impl<'a> Canvas<'a> {
    fn new(store: &'a PixelStore, mut ds: DrawState) -> Canvas<'a> {
        // A pixel rectangle is not textured, and the model does not stipple it (see the module doc).
        ds.texture = None;
        ds.polygon_stipple = None;
        let did = ds.ownership.is_some().then(|| store.array(DID_PLANE)).flatten();
        Canvas { store, planes: Planes::new(store, &ds), ds, did, pitch: store.width() as usize }
    }

    /// The plane index of window pixel (`x`, `y`), when it is inside the clip rectangle and owned.
    fn at(&self, x: i64, y: i64) -> Option<usize> {
        let (px, py) = (self.ds.origin[0] as i64 + x, self.ds.origin[1] as i64 + y);
        let c = &self.ds.clip;
        if px < c.x0 as i64 || px >= c.x1 as i64 || py < c.y0 as i64 || py >= c.y1 as i64 {
            return None;
        }
        let index = py as usize * self.pitch + px as usize;
        match &self.ds.ownership {
            None => Some(index),
            // SAFETY: the pixel is inside the clip rectangle, which is inside the screen.
            Some(o) => o.owns(ownership_plane(self.did.as_ref().map_or(0, |d| unsafe { d.load(index) }))).then_some(index),
        }
    }

    fn frag(&self, c: &mut Counts, x: i64, y: i64, colour: [f32; 4], z: u32) {
        if let Some(index) = self.at(x, y) {
            c.fragments += 1;
            let src = colour.map(|v| colour13(v) as u32);
            match &self.ds.samples {
                None => self.planes.shade(&self.ds, index, src, z),
                // A pixel rectangle covers every sample of its pixel, at one depth.
                Some(ms) => self.planes.shade_samples(&self.ds, index, src, |_| z, (1u32 << ms.count) - 1),
            }
        }
    }
}

/// A plane word `0xAARRGGBB` as red, green, blue and alpha in 0 to 1, at the word's own eight bits.
fn components8(w: u32) -> [f32; 4] {
    [w >> 16 & 0xff, w >> 8 & 0xff, w & 0xff, w >> 24 & 0xff].map(|v| v as f32 / 255.0)
}

/// The colour word a read or accumulation takes its pixels from.
fn read_word(gl: &Gl, ds: &DrawState) -> Option<usize> {
    let l = gl.layout.as_ref()?;
    let mask = gl.pix.read.unwrap_or(if gl.double { 2 } else { 1 });
    let shown = if gl.double { gl.shown as u32 } else { 0 };
    let word = if mask & 1 != 0 { l.colour.start + shown } else { l.colour.start + 1 - shown };
    let _ = ds;
    (word < l.colour.end).then_some(word as usize)
}

impl Pixels {
    /// The transfer stages for a colour.
    fn colour(&self, mut c: [f32; 4]) -> [f32; 4] {
        if self.suspended {
            return c;
        }
        for k in 0..4 {
            c[k] = (c[k] * self.scale[k] + self.bias[k]).clamp(0.0, 1.0);
        }
        if self.map_colour {
            for k in 0..4 {
                if let Some(m) = self.maps.get(&(MAP_RED_TO_RED + k as u32)) {
                    c[k] = m.at(c[k]);
                }
            }
        }
        c
    }

    fn depth(&self, d: f32) -> f32 {
        if self.suspended { d } else { (d * self.depth_scale + self.depth_bias).clamp(0.0, 1.0) }
    }

    /// An index or stencil value through the shift, offset and (for the maps' flags) the table.
    fn integer(&self, v: i64, map: u32, mapped: bool) -> i64 {
        if self.suspended {
            return v;
        }
        let s = self.index_shift;
        let v = (if s >= 0 { v << s.min(40) } else { v >> (-s).min(40) }) + self.index_offset as i64;
        match self.maps.get(&map) {
            Some(m) if mapped => m.index(v),
            _ => v,
        }
    }

    fn index_colour(&self, i: i64, c: &mut Counts) -> [f32; 4] {
        let m: Vec<_> = (0..4).map(|k| self.maps.get(&(MAP_INDEX_TO_RED + k))).collect();
        if m.iter().all(|m| m.is_some()) && self.map_colour {
            let v = |k: usize| m[k].map_or(0.0, |m| m.index(i) as f32 / if m.max > 0.0 { m.max } else { 1.0 });
            return [v(0), v(1), v(2), v(3)];
        }
        c.index_without_map += 1;
        let g = (i & 0xff) as f32 / 255.0;
        [g, g, g, 1.0]
    }
}

/// Fragments for `img` with its lower-left pixel at the raster position, zoomed.
fn emit(gl: &Gl, c: &mut Counts, canvas: &Canvas<'_>, img: &Image) {
    let p = &gl.pix;
    let pos = p.pos;
    let (rx, ry) = (pos.x.floor() as i64, pos.y.floor() as i64);
    let [zx, zy] = p.zoom;
    let span = |origin: i64, i: usize, z: f32| {
        let (a, b) = ((origin as f32 + i as f32 * z).floor() as i64, (origin as f32 + (i + 1) as f32 * z).floor() as i64);
        (a.min(b), a.max(b))
    };
    let (rc, rz) = (pos.colour, pos.z);
    for j in 0..img.h {
        let (y0, y1) = span(ry, j, zy);
        for i in 0..img.w {
            let (x0, x1) = span(rx, i, zx);
            let k = j * img.w + i;
            for y in y0..y1.max(y0) {
                for x in x0..x1.max(x0) {
                    match &img.plane {
                        Plane::Colour(col) => canvas.frag(c, x, y, p.colour(col[k]), rz),
                        Plane::Index(v) => {
                            let col = p.index_colour(p.integer(v[k], MAP_INDEX_TO_INDEX, p.map_colour), c);
                            canvas.frag(c, x, y, col, rz)
                        }
                        Plane::Depth(d) => canvas.frag(c, x, y, rc, depth32(p.depth(d[k]))),
                        Plane::Stencil(v) => {
                            let value = p.integer(v[k], MAP_STENCIL_TO_STENCIL, p.map_stencil) as u32 as u8;
                            if let (Some(index), true, true) = (canvas.at(x, y), canvas.ds.stencil_bits != 0, canvas.ds.samples.is_some()) {
                                canvas.planes.set_stencil(&canvas.ds, index, value, canvas.ds.stencil.write_mask);
                            } else if let (Some(index), true) = (canvas.at(x, y), canvas.ds.stencil_bits != 0) {
                                if let Some(a) = canvas.store.array(canvas.ds.stencil_word as usize) {
                                    let wm = canvas.ds.stencil.write_mask;
                                    // SAFETY: the index is inside the clip rectangle, inside the screen.
                                    unsafe { a.store(index, ((a.load(index) as u8 & !wm) | (value & wm)) as u32) };
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if p.post[..4] != [1.0; 4] || p.post[4..8] != [0.0; 4] || p.post[8..12] != [1.0; 4] || p.post[12..] != [0.0; 4] {
        c.post_stages_ignored += 1;
    }
}

/// The rectangle of the read buffer, as an image of `kind`. Pixels outside the window read zero.
fn capture(gl: &Gl, canvas: &Canvas<'_>, x: i32, y: i32, w: usize, h: usize, kind: u32) -> Option<Image> {
    let ds = &canvas.ds;
    let colour = (kind == COPY_COLOUR).then(|| read_word(gl, ds).and_then(|word| canvas.store.array(word))).flatten();
    let plain = ds.samples.is_none();
    let depth = (kind == COPY_DEPTH && ds.depth_bits != 0 && plain).then(|| canvas.store.array(ds.depth_word as usize)).flatten();
    let stencil = (kind == COPY_STENCIL && ds.stencil_bits != 0 && plain).then(|| canvas.store.array(ds.stencil_word as usize)).flatten();
    let mut col = Vec::new();
    let mut ints = Vec::new();
    let mut zs = Vec::new();
    for j in 0..h {
        for i in 0..w {
            let at = canvas.at(x as i64 + i as i64, y as i64 + j as i64);
            // Each load below is of a pixel `at` found inside the clip rectangle, inside the screen.
            match kind {
                COPY_COLOUR => col.push(match (at, &colour) {
                    // SAFETY: `a` is inside the clip rectangle, inside the screen.
                    (Some(a), Some(arr)) => components8(unsafe { arr.load(a) }),
                    _ => [0.0; 4],
                }),
                COPY_DEPTH => zs.push(match (at, &depth) {
                    // SAFETY: as above.
                    (Some(a), Some(arr)) => (unsafe { arr.load(a) }) as f32 / 4_294_967_040.0,
                    // A multisampled window's depth is in its samples: a read takes the first.
                    (Some(a), None) => canvas.planes.sample_depth(ds, a).map_or(0.0, |z| z as f32 / 4_294_967_040.0),
                    _ => 0.0,
                }),
                _ => ints.push(match (at, &stencil) {
                    // SAFETY: as above.
                    (Some(a), Some(arr)) => (unsafe { arr.load(a) }) as i64 & 0xff,
                    (Some(a), None) => canvas.planes.sample_stencil(a).map_or(0, |v| v as i64),
                    _ => 0,
                }),
            }
        }
    }
    Some(Image { w, h, plane: match kind { COPY_COLOUR => Plane::Colour(col), COPY_DEPTH => Plane::Depth(zs), _ => Plane::Stencil(ints) } })
}

/// What a plane-touching token needs of the board around it.
pub struct Env<'a> {
    pub store: &'a PixelStore,
    pub ds: DrawState,
}

/// What a token answered.
#[derive(Debug, PartialEq)]
pub enum Out {
    Nothing,
    /// Bytes for the return page, at offset 8.
    Reply(Vec<u8>),
}

/// The words a draw, bitmap or map token asks the stream for after its framed words; None for
/// a token that carries none. Maps load the values the header names; a draw's image is sized from
/// the unpack record in force, and a draw announced by DMA has none in the stream.
pub fn run_words(gl: &Gl, selector: u32, a: &[u32]) -> Option<u32> {
    match selector {
        LOAD_PIXEL_MAP_FLOAT_VALUES | LOAD_PIXEL_MAP_UNSIGNED_32_BIT_VALUES | LOAD_PIXEL_MAP_UNSIGNED_16_BIT_VALUES => a.get(1).copied().map(|n| n.min(MAP_LIMIT)),
        DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA => {
            let h = Header::parse(a)?;
            Some(if h.dma { 0 } else { h.image_words(&gl.pix.unpack_in_force()).unwrap_or(0) })
        }
        // The small bitmap's word count includes the byte count, which is in the framed words.
        DRAW_BITMAP_FROM_DISPLAY_LIST if a.len() == 8 => Some(a[6].saturating_sub(1)),
        DRAW_BITMAP => {
            let h = large_bitmap(a)?;
            Some(if h.dma { 0 } else { h.image_words(&gl.pix.unpack_in_force()).unwrap_or(0) })
        }
        _ => None,
    }
}

/// The large bitmap's header: width, height, byte offset, DMA flag, row mode, then four floats.
pub fn large_bitmap(a: &[u32]) -> Option<Header> {
    let a: &[u32; 9] = a.get(..9)?.try_into().ok()?;
    Some(Header { w: a[0], h: a[1], format: FORMAT_COLOUR_INDEX, ty: TYPE_BITMAP, offset: a[2] & 3, dma: a[3] != 0, row_mode: a[4], column: 0 })
}

/// A plane-touching token with no data run: copy, read, accumulate and the queries.
pub fn plane_command(gl: &mut Gl, c: &mut Counts, env: &Env<'_>, selector: u32, a: &[u32]) -> Out {
    let canvas = Canvas::new(env.store, env.ds);
    match (selector, a) {
        (COPY_PIXEL_RECTANGLE_WITHIN_FRAME_BUFFER, &[x, y, w, h, _offset, kind, _overlap]) => {
            c.copies += 1;
            if !gl.pix.pos.valid {
                c.invalid_position += 1;
                return Out::Nothing;
            }
            let Some(img) = capture(gl, &canvas, x as i32, y as i32, w as usize, h as usize, kind) else {
                c.refused += 1;
                return Out::Nothing;
            };
            emit(gl, c, &canvas, &img);
            Out::Nothing
        }
        (READ_PIXEL_RECTANGLE_BACK_TO_HOST, &[x, y, w, h, format, ty, low]) => {
            c.reads += 1;
            let kind = match format {
                FORMAT_DEPTH => COPY_DEPTH,
                FORMAT_STENCIL => COPY_STENCIL,
                _ => COPY_COLOUR,
            };
            let Some(mut img) = capture(gl, &canvas, x as i32, y as i32, w as usize, h as usize, kind) else {
                c.refused += 1;
                return Out::Nothing;
            };
            let p = &gl.pix;
            match &mut img.plane {
                Plane::Colour(col) => col.iter_mut().for_each(|v| *v = p.colour(*v)),
                Plane::Depth(d) => d.iter_mut().for_each(|v| *v = p.depth(*v)),
                Plane::Stencil(v) => v.iter_mut().for_each(|v| *v = p.integer(*v, MAP_STENCIL_TO_STENCIL, p.map_stencil)),
                Plane::Index(_) => {}
            }
            // A colour-index read has no index plane to read: it is refused.
            let Some(bytes) = (format != FORMAT_COLOUR_INDEX).then(|| pack(&img, format, ty, low & 3, &p.pack)).flatten() else {
                c.refused += 1;
                return Out::Nothing;
            };
            if bytes.len() > READ_REPLY_LIMIT {
                c.read_by_dma += 1;
                return Out::Nothing;
            }
            Out::Reply(bytes)
        }
        (ACCUMULATION_BUFFER_OPERATION, &[op, value]) => {
            c.accumulations += 1;
            accumulate(gl, c, &canvas, op, f(value));
            Out::Nothing
        }
        (ASK_WHETHER_COPY_SOURCE_OVERLAPS_ITS_DESTINATION | ASK_WHETHER_CONVOLUTION_COPY_OVERLAPS, &[x, y, w, h]) => {
            c.queries += 1;
            let p = &gl.pix;
            let (dx, dy) = (p.pos.x.floor() as i64, p.pos.y.floor() as i64);
            let (sx, sy) = (x as i32 as i64, y as i32 as i64);
            let (zw, zh) = ((w as f32 * p.zoom[0].abs()) as i64, (h as f32 * p.zoom[1].abs()) as i64);
            let meets = sx < dx + zw && dx < sx + w as i64 && sy < dy + zh && dy < sy + h as i64;
            let mut reply = vec![0u8; if selector == ASK_WHETHER_CONVOLUTION_COPY_OVERLAPS { 40 } else { 4 }];
            reply[..4].copy_from_slice(&(if meets { OVERLAP_BIT } else { 0 }).to_be_bytes());
            Out::Reply(reply)
        }
        (RESOLVE_WHETHER_PIXEL_FORMAT_IS_SINGLE_COMPONENT, &[..]) => {
            c.queries += 1;
            Out::Reply(vec![0; 4])
        }
        _ => {
            c.refused += 1;
            Out::Nothing
        }
    }
}

/// A draw's header and its image words (or, for a DMA draw, the bytes the packet carried).
pub fn draw(gl: &mut Gl, c: &mut Counts, env: &Env<'_>, hdr: &[u32; 8], bytes: &[u8]) {
    c.draws += 1;
    let Some(h) = Header::parse(hdr) else { return };
    let Some(img) = unpack(&h, &gl.pix.unpack_in_force(), bytes) else {
        c.refused += 1;
        return;
    };
    if !gl.pix.pos.valid {
        c.invalid_position += 1;
        return;
    }
    let canvas = Canvas::new(env.store, env.ds);
    // A strip of a wide image starts at its column (inferred from the table's word 7).
    let saved = gl.pix.pos.x;
    gl.pix.pos.x += h.column as f32 * gl.pix.zoom[0];
    emit(gl, c, &canvas, &img);
    gl.pix.pos.x = saved;
}

/// A bitmap: a set bit is a fragment of the raster position's colour at the position less the
/// origin; the position then moves. `rec` reads the image: the small form's rows are packed.
pub fn bitmap(gl: &mut Gl, c: &mut Counts, env: &Env<'_>, h: &Header, rec: &Record, orig: [f32; 2], mv: [f32; 2], bytes: &[u8]) {
    c.bitmaps += 1;
    if !gl.pix.pos.valid {
        c.invalid_position += 1;
        return;
    }
    let Some(Image { plane: Plane::Index(bits), w, .. }) = unpack(h, rec, bytes) else {
        c.refused += 1;
        return;
    };
    let canvas = Canvas::new(env.store, env.ds);
    let pos = gl.pix.pos;
    let (x0, y0) = ((pos.x - orig[0]).floor() as i64, (pos.y - orig[1]).floor() as i64);
    for (k, _) in bits.iter().enumerate().filter(|(_, &b)| b != 0) {
        canvas.frag(c, x0 + (k % w) as i64, y0 + (k / w) as i64, pos.colour, pos.z);
    }
    gl.pix.pos.x += mv[0];
    gl.pix.pos.y += mv[1];
}

/// The passes an accumulation operation runs for `v`, each an operation and its value. Only a value
/// of exactly 1.0 splits: accumulate and add run twice at 0.5, load is a load at 0.5 then an
/// accumulate at 0.5, and multiply by 1.0 does nothing. Every other value, and a return, is one pass.
fn accumulation_passes(op: u32, v: f32) -> Vec<(u32, f32)> {
    match (op, v == 1.0) {
        (ACCUM | ACCUM_ADD, true) => vec![(op, 0.5), (op, 0.5)],
        (ACCUM_LOAD, true) => vec![(ACCUM_LOAD, 0.5), (ACCUM, 0.5)],
        (ACCUM_MULT, true) => Vec::new(),
        _ => vec![(op, v)],
    }
}

/// One accumulation operation over the window (the draw state's clip rectangle).
fn accumulate(gl: &Gl, c: &mut Counts, canvas: &Canvas<'_>, op: u32, v: f32) {
    let Some(l) = &gl.layout else { return };
    if l.accumulation.is_empty() || !matches!(op, ACCUM | ACCUM_LOAD | ACCUM_RETURN | ACCUM_MULT | ACCUM_ADD) {
        return;
    }
    let passes = accumulation_passes(op, v);
    let words: Vec<WordArray<'_>> = l.accumulation.clone().filter_map(|w| canvas.store.array(w as usize)).collect();
    let colour = read_word(gl, &canvas.ds).and_then(|w| canvas.store.array(w));
    let ds = &canvas.ds;
    let r = ds.clip;
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            let Some(index) = canvas.at((x - ds.origin[0]) as i64, (y - ds.origin[1]) as i64) else { continue };
            // SAFETY: the pixel is inside the clip rectangle, inside the screen.
            let get = |k: usize| words.get(k).map_or(1.0, |a| f32::from_bits(unsafe { a.load(index) }));
            let put = |k: usize, v: f32| {
                if let Some(a) = words.get(k) {
                    // SAFETY: as above.
                    unsafe { a.store(index, v.to_bits()) };
                }
            };
            // SAFETY: as above.
            let src = colour.map_or([0.0; 4], |a| components8(unsafe { a.load(index) }));
            for &(op, v) in &passes {
                match op {
                    ACCUM | ACCUM_LOAD => (0..words.len().min(4)).for_each(|k| put(k, if op == ACCUM { get(k) } else { 0.0 } + v * src[k])),
                    ACCUM_MULT => (0..words.len()).for_each(|k| put(k, get(k) * v)),
                    ACCUM_ADD => (0..words.len()).for_each(|k| put(k, get(k) + v)),
                    _ => {
                        // Return: the accumulated colour times the value, to the draw buffers under the colour mask.
                        let out = [0, 1, 2, 3].map(|k| (get(k) * v).clamp(0.0, 1.0));
                        for &w in &ds.colour[..ds.colours as usize] {
                            if let Some(a) = canvas.store.array(w as usize) {
                                // SAFETY: as above.
                                let old = unsafe { a.load(index) };
                                let keep = (!ds.colour_mask[0] as u32 * 0xff) << 16 | (!ds.colour_mask[1] as u32 * 0xff) << 8 | (!ds.colour_mask[2] as u32 * 0xff) | (!ds.colour_mask[3] as u32 * 0xff) << 24;
                                let word = out.map(|v| (v * 255.0 + 0.5) as u32);
                                let new = word[3] << 24 | word[0] << 16 | word[1] << 8 | word[2];
                                // SAFETY: as above.
                                unsafe { a.store(index, (new & !keep) | (old & keep)) };
                            }
                        }
                    }
                }
            }
            c.fragments += 1;
        }
    }
}

/// glClear's accumulation bit: the accumulation words of the window go to the clear value, one
/// component a word, under ownership and the clip rectangle.
pub fn clear_accumulation(gl: &Gl, c: &mut Counts, env: &Env<'_>) {
    let Some(l) = &gl.layout else { return };
    let canvas = Canvas::new(env.store, env.ds);
    let r = env.ds.clip;
    for w in l.accumulation.clone() {
        let Some(a) = env.store.array(w as usize) else { continue };
        let v = gl.pix.accum_clear[(w - l.accumulation.start) as usize].to_bits();
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                if let Some(index) = canvas.at((x - env.ds.origin[0]) as i64, (y - env.ds.origin[1]) as i64) {
                    // SAFETY: the pixel is inside the clip rectangle, inside the screen.
                    unsafe { a.store(index, v) };
                    c.fragments += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::kona::stream::{Reader, Writer};

    fn header(w: u32, h: u32, format: u32, ty: u32, row_mode: u32) -> Header {
        Header { w, h, format, ty, offset: 0, dma: false, row_mode, column: 0 }
    }

    /// The size rule of the table: a 3 x 2 RGB image of bytes is rows of nine bytes padded to
    /// twelve under alignment 4, 21 bytes in all; under alignment 1 it is 18.
    #[test]
    fn the_image_length_follows_the_unpack_state() {
        let h = header(3, 2, FORMAT_RGB, TYPE_UNSIGNED_BYTE, 0);
        assert_eq!(h.image_words(&Record::default()), Some(6));
        assert_eq!(h.image_words(&Record { alignment: 1, ..Record::default() }), Some(5));
        // Skips move the start: two rows of 12, and the 4-byte word boundary rounds the end up.
        assert_eq!(h.image_words(&Record { skip_rows: 1, ..Record::default() }), Some(6));
        assert_eq!(h.image_words(&Record { skip_pixels: 1, alignment: 1, ..Record::default() }), Some(6));
        // Row mode 1 counts each row's words alone; a row of nine bytes starting at a word edge is three.
        assert_eq!(header(3, 2, FORMAT_RGB, TYPE_UNSIGNED_BYTE, 1).image_words(&Record::default()), Some(6));
        // An unknown format (14, which the pipe counts as no components) or type is no length, and an empty image carries nothing.
        assert_eq!(header(1, 1, 14, TYPE_BYTE, 0).image_words(&Record::default()), None);
        assert_eq!(header(1, 1, FORMAT_RGBA, 14, 0).image_words(&Record::default()), None);
        assert_eq!(header(0, 5, FORMAT_RGBA, TYPE_BYTE, 0).image_words(&Record::default()), Some(0));
        // A bitmap row of ten pixels is two bytes.
        assert_eq!(header(10, 3, FORMAT_COLOUR_INDEX, TYPE_BITMAP, 0).image_words(&Record { alignment: 1, ..Record::default() }), Some(2));
    }

    /// A guest's unpack state whose sizes overflow is no image size: the draw is refused, not a panic or a wrapped length.
    #[test]
    fn an_unpack_state_whose_sizes_overflow_has_no_image_size() {
        let big = Record { row_length: u32::MAX, skip_rows: u32::MAX, skip_pixels: u32::MAX, alignment: 8, ..Record::default() };
        assert_eq!(header(4, u32::MAX, FORMAT_RGBA, TYPE_FLOAT, 0).image_words(&big), None);
        assert_eq!(header(4, u32::MAX, FORMAT_RGBA, TYPE_FLOAT, 1).image_words(&big), None);
        assert_eq!(header(4, u32::MAX, FORMAT_RGBA, TYPE_FLOAT, 1).image_words(&Record::default()), None);
    }

    /// An unpack row length below the width is no size, not an underflow.
    #[test]
    fn a_row_length_below_the_width_has_no_image_size() {
        let rec = Record { row_length: 10, ..Record::default() };
        assert_eq!(header(100, 4, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, 0).image_words(&rec), None);
        assert_eq!(header(100, 4, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, 1).image_words(&rec), None);
        assert!(unpack(&header(100, 4, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, 0), &rec, &[0; 2000]).is_none());
        assert_eq!(header(100, 4, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, 0).image_words(&Record { row_length: 100, ..Record::default() }), Some(400));
    }

    fn colours() -> Image {
        Image { w: 3, h: 2, plane: Plane::Colour(vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.25, 0.5, 0.75, 1.0], [0.0; 4]]) }
    }

    /// What a pack writes, an unpack of the same record reads back (to the type's precision),
    /// in every layout the table lists: each format and type, alignment, row length, skips and swap.
    #[test]
    fn pack_and_unpack_round_trip() {
        let img = colours();
        let records = [Record::default(), Record { alignment: 1, ..Record::default() }, Record { row_length: 5, skip_rows: 1, skip_pixels: 2, alignment: 8, ..Record::default() }, Record { swap_bytes: true, alignment: 2, ..Record::default() }];
        for rec in records {
            for (format, ty, eps) in [(FORMAT_RGBA, TYPE_UNSIGNED_BYTE, 1.0 / 255.0), (FORMAT_RGBA, TYPE_UNSIGNED_SHORT, 1e-4), (FORMAT_RGBA, TYPE_UNSIGNED_INT, 1e-6), (FORMAT_RGBA, TYPE_FLOAT, 0.0), (FORMAT_RGBA, TYPE_BYTE, 2.0 / 255.0), (FORMAT_RGBA, TYPE_SHORT, 1e-4), (FORMAT_RGBA, TYPE_INT, 1e-6), (FORMAT_RGBA, TYPE_8_8_8_8, 1.0 / 255.0), (FORMAT_RGBA, TYPE_4_4_4_4, 0.07), (FORMAT_RGBA, TYPE_5_5_5_1, 0.5), (FORMAT_RGBA, TYPE_10_10_10_2, 0.34), (FORMAT_RGB, TYPE_3_3_2, 0.2), (FORMAT_RGB, TYPE_UNSIGNED_BYTE, 1.0 / 255.0)] {
                let bytes = pack(&img, format, ty, 0, &rec).unwrap_or_else(|| panic!("{format} {ty}"));
                let h = header(3, 2, format, ty, 0);
                let words = h.image_words(&rec).unwrap() as usize;
                assert!(bytes.len() >= words * 4, "{format} {ty}: {} bytes, {words} words", bytes.len());
                // The stream starts at the word holding the first pixel.
                let from = h.geometry(&rec).unwrap().start / 4 * 4;
                let back = unpack(&h, &rec, &bytes[from..]).unwrap_or_else(|| panic!("{format} {ty}"));
                let (Plane::Colour(a), Plane::Colour(b)) = (&img.plane, &back.plane) else { panic!() };
                for (x, y) in a.iter().zip(b) {
                    let want = if format == FORMAT_RGB { [x[0], x[1], x[2], 1.0] } else { *x };
                    // The 3-3-2 and 5-5-5-1 forms hold too little of green or alpha to compare more tightly.
                    assert!(want.iter().zip(y).all(|(p, q)| (p - q).abs() <= eps + 1e-6), "{format} {ty} {rec:?}: {want:?} read back as {y:?}");
                }
            }
        }
    }

    /// Luminance reads as red plus green plus blue, clamped; index, stencil and depth round trip too.
    #[test]
    fn other_formats_pack_by_their_own_rules() {
        let img = Image { w: 2, h: 1, plane: Plane::Colour(vec![[0.25, 0.25, 0.25, 1.0], [0.9, 0.9, 0.9, 0.5]]) };
        let b = pack(&img, FORMAT_LUMINANCE, TYPE_UNSIGNED_BYTE, 0, &Record { alignment: 1, ..Record::default() }).unwrap();
        assert_eq!(&b[..2], &[191, 255]);
        let idx = Image { w: 3, h: 1, plane: Plane::Index(vec![0, 7, 200]) };
        let rec = Record { alignment: 1, ..Record::default() };
        let bytes = pack(&idx, FORMAT_COLOUR_INDEX, TYPE_UNSIGNED_BYTE, 0, &rec).unwrap();
        assert_eq!(unpack(&header(3, 1, FORMAT_COLOUR_INDEX, TYPE_UNSIGNED_BYTE, 0), &rec, &bytes).unwrap().plane, Plane::Index(vec![0, 7, 200]));
        let z = Image { w: 2, h: 1, plane: Plane::Depth(vec![0.0, 1.0]) };
        let bytes = pack(&z, FORMAT_DEPTH, TYPE_UNSIGNED_SHORT, 0, &rec).unwrap();
        assert_eq!(&bytes[..4], &[0, 0, 0xff, 0xff]);
        // A bitmap packs most significant bit first, or least when asked.
        let bits = Image { w: 10, h: 1, plane: Plane::Index(vec![1, 0, 0, 0, 0, 0, 0, 1, 1, 0]) };
        assert_eq!(&pack(&bits, FORMAT_COLOUR_INDEX, TYPE_BITMAP, 0, &rec).unwrap()[..2], &[0x81, 0x80]);
        assert_eq!(&pack(&bits, FORMAT_COLOUR_INDEX, TYPE_BITMAP, 0, &Record { lsb_first: true, ..rec }).unwrap()[..2], &[0x81, 0x01]);
    }

    /// The skips in a pack record are the guest's, so the packer sizes its buffer with checked arithmetic and
    /// refuses one past its cap: a skip of 2^32 rows must not ask the allocator for terabytes.
    #[test]
    fn a_huge_pack_skip_is_no_buffer() {
        let img = Image { w: 8000, h: 1, plane: Plane::Colour(vec![[0.0; 4]; 8000]) };
        let rec = Record { alignment: 1, ..Record::default() };
        assert_eq!(packed_len(8000, 1, FORMAT_RED, TYPE_UNSIGNED_SHORT, 0, &rec), Some(16000));
        for skipped in [Record { skip_rows: u32::MAX, ..rec }, Record { skip_pixels: u32::MAX, ..rec }, Record { skip_rows: 1 << 20, row_length: 1 << 20, ..rec }] {
            assert!(pack(&img, FORMAT_RED, TYPE_UNSIGNED_SHORT, 0, &skipped).is_none(), "{skipped:?}");
        }
        // A skip that fits is built, with the image after it.
        let b = pack(&img, FORMAT_RED, TYPE_UNSIGNED_SHORT, 0, &Record { skip_rows: 2, ..rec }).unwrap();
        assert_eq!(b.len(), 3 * 16000);
        assert_eq!(packed_len(1, 1, FORMAT_RED, TYPE_UNSIGNED_BYTE, 0, &Record { skip_rows: u32::MAX, row_length: u32::MAX, ..rec }), Some(u32::MAX as u64 * u32::MAX as u64 + u32::MAX as u64));
    }

    /// The extension formats: ABGR keeps its components in the order alpha, blue, green, red; CMYK and CMYKA
    /// are sized by the pipe's component table (four and five) so a header's image words are consumed, and refused.
    #[test]
    fn the_extension_formats_have_their_sizes_and_abgr_its_order() {
        let rec = Record { alignment: 1, ..Record::default() };
        let rgba = Image { w: 1, h: 1, plane: Plane::Colour(vec![[1.0, 0.5, 0.25, 0.0]]) };
        let b = pack(&rgba, FORMAT_ABGR, TYPE_UNSIGNED_BYTE, 0, &rec).unwrap();
        assert_eq!(&b[..4], &[0, 64, 128, 255], "alpha, blue, green, red");
        assert_eq!(unpack(&header(1, 1, FORMAT_ABGR, TYPE_UNSIGNED_BYTE, 0), &rec, &b).unwrap().plane, Plane::Colour(vec![[1.0, 128.0 / 255.0, 64.0 / 255.0, 0.0]]));
        assert_eq!(pixel_bytes(FORMAT_ABGR, TYPE_UNSIGNED_SHORT), Some(8));
        assert_eq!(pixel_bytes(FORMAT_ABGR, TYPE_8_8_8_8), None, "its packed layouts are not recorded");
        assert_eq!((pixel_bytes(FORMAT_CMYK, TYPE_UNSIGNED_BYTE), pixel_bytes(FORMAT_CMYKA, TYPE_UNSIGNED_BYTE)), (Some(4), Some(5)));
        assert!(unpack(&header(1, 1, FORMAT_CMYK, TYPE_UNSIGNED_BYTE, 0), &rec, &[0; 4]).is_none());
        assert!(pack(&rgba, FORMAT_CMYKA, TYPE_UNSIGNED_BYTE, 0, &rec).is_none());
        assert_eq!(header(3, 2, FORMAT_CMYKA, TYPE_UNSIGNED_BYTE, 0).image_words(&rec), Some(8), "3 x 2 pixels of 5 bytes, rows packed by 1");
    }

    /// The destination's low address bits lead the buffer, and a row of 3 bytes pads to the alignment.
    #[test]
    fn a_read_starts_at_the_destinations_byte_offset() {
        let img = Image { w: 3, h: 2, plane: Plane::Colour(vec![[1.0; 4]; 6]) };
        let b = pack(&img, FORMAT_RED, TYPE_UNSIGNED_BYTE, 2, &Record::default()).unwrap();
        assert_eq!(b, vec![0, 0, 255, 255, 255, 0, 255, 255, 255, 0, 0, 0]);
    }

    /// The transfer stages: scale and bias with a clamp, the index shift and offset, and the
    /// maps only when their flag is on.
    #[test]
    fn transfer_scale_bias_and_maps() {
        let mut p = Pixels::default();
        p.scale = [0.5, 1.0, 1.0, 1.0];
        p.bias = [0.0, 0.5, 0.0, 0.0];
        assert_eq!(p.colour([1.0, 0.75, 0.0, 1.0]), [0.5, 1.0, 0.0, 1.0]);
        p.map_colour = true;
        p.maps.insert(MAP_RED_TO_RED, Map { max: 0.0, values: vec![1.0, 0.0] });
        assert_eq!(p.colour([1.0, 0.0, 0.0, 1.0])[0], 0.0, "a two-entry map flips red");
        p.suspended = true;
        assert_eq!(p.colour([1.0, 0.75, 0.0, 1.0]), [1.0, 0.75, 0.0, 1.0], "suspended, nothing applies");
        p.suspended = false;
        p.index_shift = -1;
        p.index_offset = 3;
        assert_eq!(p.integer(10, MAP_INDEX_TO_INDEX, false), 8);
        p.maps.insert(MAP_STENCIL_TO_STENCIL, Map { max: 0.0, values: vec![9.0, 8.0, 7.0] });
        assert_eq!(p.integer(0, MAP_STENCIL_TO_STENCIL, true), 7, "a lookup past the table takes the last entry");
    }

    /// Only a value of exactly 1.0 makes an accumulation operation two passes.
    #[test]
    fn only_a_value_of_one_splits_an_accumulation_into_passes() {
        assert_eq!(accumulation_passes(ACCUM, 1.0), vec![(ACCUM, 0.5), (ACCUM, 0.5)]);
        assert_eq!(accumulation_passes(ACCUM_ADD, 1.0), vec![(ACCUM_ADD, 0.5), (ACCUM_ADD, 0.5)]);
        assert_eq!(accumulation_passes(ACCUM_LOAD, 1.0), vec![(ACCUM_LOAD, 0.5), (ACCUM, 0.5)]);
        assert_eq!(accumulation_passes(ACCUM_MULT, 1.0), vec![]);
        assert_eq!(accumulation_passes(ACCUM_RETURN, 1.0), vec![(ACCUM_RETURN, 1.0)]);
        for op in [ACCUM, ACCUM_LOAD, ACCUM_MULT, ACCUM_ADD, ACCUM_RETURN] {
            for v in [0.0, 0.5, 0.999_999_9, 1.000_000_1, 2.0, -1.0] {
                assert_eq!(accumulation_passes(op, v), vec![(op, v)], "{op:#x} by {v}");
            }
        }
    }

    /// What the context keeps of the pixel operations is saved and restored.
    #[test]
    fn the_pixel_state_round_trips_through_a_stream() {
        let mut p = Pixels::default();
        p.unpack.alignment = 1;
        p.pack.swap_bytes = true;
        p.scale[2] = 0.25;
        p.zoom = [2.0, -1.0];
        p.maps.insert(MAP_INDEX_TO_RED, Map { max: 65535.0, values: vec![1.0, 2.0, 3.0] });
        p.read = Some(2);
        p.pos = RasterPos { valid: false, x: 3.5, y: 4.5, z: 9, colour: [0.1, 0.2, 0.3, 0.4] };
        p.pending = Some([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut w = Writer::default();
        p.visit(&mut w);
        let mut q = Pixels::default();
        let mut r = Reader::new(&w.out);
        q.visit(&mut r);
        assert!(r.finish().is_ok());
        assert_eq!(q, p);
    }
}
