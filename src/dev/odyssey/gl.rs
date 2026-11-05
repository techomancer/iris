//! The colour, depth, stencil, fog and texture hooks the board gives the shared
//! rasterizer, and the state its GL draws are made under.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.


use super::pipe::backend::DestMem;
use crate::devlog::LogModule;
use super::pipe::state::FragmentWords;
use super::pipe::PipeDraw;
use super::trace::TRACE_DRAW;
use super::board::{
    Board, ODSY_CLIP_RECTS,
};
use super::consts::{
    colour, gl_cmd, pattern, prog, tex, write, ODSY_AUX_OVERLAY_BUFFER, ODSY_DEPTH24_MAX, ODSY_DRAW_WORD_STATE,
    ODSY_ID_COMPARE_ENABLE, ODSY_ID_COMPARE_MASK_SHIFT, ODSY_INDEX_MASK,
};
use super::geometry::{gl_transpose, odyssey_f32};
use super::assembly::{vtx, Assembly, Prim, PrimSink, XfState, CLIP_PLANES, POLYGON, TRIANGLES};
use super::draw::{RasterOp, CLEAR_COLOUR, CLEAR_DEPTH, CLEAR_STENCIL, MAX_TARGETS, RASTER_CLEAR};
use super::clip::ClipRect;
use super::pipe::setup::{LiveDraw, Transform};
use super::pipe::slot::DrawState;
use super::pipe::texture::{Target, Unit};
use super::raster_unit::{
    primitive_cost, rect_cost, LINE_PHASE_ON_RASTER, RasterCtx,
};

impl Board {
    /// Whether a GL fragment's colour is an index (prog::COLOUR_MODE).
    pub fn gl_index_mode(&self) -> bool {
        self.colour_mode[0] & colour::MODE_INDEX != 0
    }
}

/// A colour index carried through the shared rasterizer, which knows only
/// RGBA: bits 7:0 of the index ride in blue and bits 11:8 in green, so at
/// the rules' eight colour bits the stored colour's low twelve bits are the
/// index again (gl_store_colour()). Smooth shading then
/// interpolates the index as GL does, within one run of 256.
pub(super) fn gl_index_rgba(index: u32) -> [f32; 4] {
    [0.0, ((index >> 8) & 0xf) as f32 / 255.0f32, (index & 0xff) as f32 / 255.0f32, 1.0]
}

/// A float index (glIndexf, glClearIndex): its integer part, wrapped to the
/// twelve index bits the way a write keeps only the buffer's bits.
pub(super) fn float_index(bits: u32) -> u32 {
    let mut f = f32::from_bits(bits);

    if !f.is_finite() {
        return 0;
    }
    // The integer part, wrapped to the buffer's twelve bits without an
    // out-of-range conversion: fmodf is exact.
    f = f.floor() % 4096.0f32;
    if f < 0.0 {
        f += 4096.0f32;
    }
    (f as u32) & ODSY_INDEX_MASK
}

impl Board {
    /// Whether GL draws go to the id-plane destination: it is selected and the
    /// kernel has bound it as the draw drawable (0x608, bit 31 aside), as it
    /// does for flight's overlay context (0x80142000; OBSERVED). The X server
    /// selects the same destination for its own id writes, which a GL
    /// context's draws do not follow.
    ///
    /// APPROXIMATION: this rule reconciles those two observations; it is not a
    /// known board mechanism. An equally consistent alternative is that GL
    /// draws simply follow the drawable 0x608 binds, whatever 0x1017 last
    /// selected.
    pub fn gl_to_aux(&mut self) -> bool {
        self.dest_is_id_planes() && (self.draw_word & !ODSY_DRAW_WORD_STATE) == self.dest_base
    }

    /// A GL draw aimed at the id-plane destination that has no modelled home.
    pub fn gl_aux_refused(&mut self, what: &str) {
        if !self.diag.gl_aux_reported {
            self.diag.gl_aux_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: {} to the id-plane destination is not modeled; dropped", what);
        }
    }

    /// What GlPlanes declines, reported once, from the draw rather than from
    /// a pixel: the conditions are fixed for the whole draw, and GlPlanes
    /// then only stores. colour says whether the draw writes colour at
    /// all.
    pub fn gl_report_refusals(&mut self, colour: bool) {
        let index_mask = self.gl_write_mask & write::INDEX_MASK;
        let mask = (self.gl_write_mask >> write::COLOUR_SHIFT) & write::COLOUR_MASK;
        let buffers = self.draw_buffers();

        if !colour {
            return;
        }
        if !self.gl_index_mode() {
            if buffers != 0 && self.gl_to_aux() {
                self.gl_aux_refused("an RGBA draw");
            }
            return;
        }
        if index_mask != write::INDEX_MASK && !self.diag.gl_index_mask_logged {
            self.diag.gl_index_mask_logged = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: index write mask {:#05x} is decoded but not applied", index_mask);
        }
        if mask != 0 && (buffers & !(1u32 << ODSY_AUX_OVERLAY_BUFFER)) != 0 && self.gl_to_aux() {
            self.gl_aux_refused("a GL id-byte write");
        }
    }
}

/// The board state a GL draw tests and stores its pixels under, fixed for
/// the whole draw: the destination's fields in board memory, as the pixel
/// mode places them (view::DestViews), and the window's tests.
#[derive(Clone, Copy)]
pub(super) struct GlPixelState {
    pub mem: DestMem,
    pub id_compare: u32,
    pub clip_rect: [ClipRect; ODSY_CLIP_RECTS as usize],
    pub clip_kept: u32,
    pub write_mask: u32,
    /// Board::gl_to_aux() for the draw.
    pub to_aux: bool,
}

/// A GL draw's destination and pixel state as a raster job takes them, and
/// the texel memory.
#[derive(Clone, Copy)]
pub(super) struct GlPixelJob {
    pub state: GlPixelState,
    pub obj: super::vram::VramPtr,
}

// SAFETY: the texel span is the board's memory, which outlives the job;
// the raster unit owns it while the job runs (Board::raster_wait).
unsafe impl Send for GlPixelJob {}

/// A primitive's fields other than its borrows, for a raster job to own.
#[derive(Clone, Copy)]
struct PrimFields {
    context: u32,
    mode: u32,
    count: u32,
    back_colours: bool,
    whole_polygon: bool,
    line_phase: Option<u32>,
    line_phase_reset: bool,
}

impl PrimFields {
    fn of(p: &Prim) -> Self {
        PrimFields {
            context: p.context,
            mode: p.mode,
            count: p.count,
            back_colours: p.back_colours,
            whole_polygon: p.whole_polygon,
            line_phase: p.line_phase,
            line_phase_reset: p.line_phase_reset,
        }
    }

    fn prim<'a>(&self, vertices: &'a [f32], xf: &'a XfState) -> Prim<'a> {
        let f = *self;
        Prim {
            context: f.context,
            mode: f.mode,
            count: f.count,
            vertices,
            back_colours: f.back_colours,
            xf,
            whole_polygon: f.whole_polygon,
            line_phase: f.line_phase,
            line_phase_reset: f.line_phase_reset,
        }
    }
}

/// One GL primitive set up for the raster unit: everything Board::gl_draw()
/// read of the board, owned.
///
/// The raster unit hands a finished job back (RasterQueue::give_back) and
/// the front end fills it again for a later draw, keeping its vertex
/// buffer: allocating it per draw on the front end and
/// freeing them on the raster thread had the two threads contending for
/// the allocator.
pub(super) struct GlDrawJob {
    /// The transform side's depth row (Board::gl_depth_row()).
    depth: DepthRow,
    /// Colour-index drawing (prog::COLOUR_MODE).
    index: bool,
    /// An opaque polygon stipple's background colour.
    background: Option<[f64; 4]>,
    /// The draw state slot the draw was issued under (Slots::issue()),
    /// held until the job is given back.
    pub slot: *const DrawState,
    state: XfState,
    vertices: Vec<f32>,
    fields: PrimFields,
    pixels: GlPixelJob,
    /// The fragment pipeline's state words (pipe/state.rs).
    words: FragmentWords,
}

impl GlDrawJob {
    /// Let go of the draw state slot: the draw is done with it.
    pub fn release(&mut self) {
        if !self.slot.is_null() {
            // SAFETY: the slot was issued for this job and not yet released.
            unsafe { DrawState::release(self.slot) };
            self.slot = std::ptr::null();
        }
    }
}

impl Drop for GlDrawJob {
    fn drop(&mut self) {
        self.release();
    }
}

// SAFETY: the slot pointer is a board slot the job holds (Slots).
unsafe impl Send for GlDrawJob {}

/// What a draw's context borrows besides its job, kept on the heap so that
/// the shader thread can read it after the raster thread is done with the
/// draw.
#[repr(C, align(64))]
pub(super) struct DrawFrame {
    words_stipple: Option<[u32; 32]>,
    pipe: Option<Box<PipeDraw>>,
    pub live: LiveDraw,
}

impl GlDrawJob {
    /// Draw it; the line-stipple phase assembly keeps.
    pub fn run(&self, ctx: &mut RasterCtx) -> Option<u32> {
        let (phase, frame) = self.run_frame(ctx, false);

        if frame.pipe.as_ref().is_some_and(|p| p.used.get()) {
            // The ring's slots name the frame: store them before it goes.
            ctx.drain_back();
        }
        phase
    }

    /// Draw it on the raster thread, its quads shaded and stored on the
    /// shader and back-end threads where there are some: the job back when
    /// the draw is done with it, or None when the ring holds it until the
    /// back end is past the draw's quads (RasterCtx::reclaim_shade() gives
    /// it back then).
    pub fn run_owned(self: Box<Self>, ctx: &mut RasterCtx) -> Option<Box<Self>> {
        let (_, frame) = self.run_frame(ctx, true);

        if frame.pipe.as_ref().is_some_and(|p| p.used.get()) {
            let ring = ctx.ring.clone().expect("the ring the draw's quads went into");

            ring.hold(self, frame);
            return None;
        }
        drop(frame);
        Some(self)
    }

    fn run_frame(&self, ctx: &mut RasterCtx, allow_shader: bool) -> (Option<u32>, Box<DrawFrame>) {
        use super::pipe::state::PolygonMode;

        // SAFETY: the frame borrows the job's heap fields, and the job
        // outlives it: the frame is dropped first, or the ring keeps both
        // and drops the frame first (QuadRing::reclaim()).
        let job: &'static GlDrawJob = unsafe { &*(self as *const GlDrawJob) };
        let mut frame = Box::new(DrawFrame {
            words_stipple: job.words.raster.polygon_stipple(),
            pipe: None,
            live: LiveDraw::default(),
        });
        let fr = &mut *frame;
        let mut d = job.fields.prim(&job.vertices, &job.state);
        if d.line_phase == Some(LINE_PHASE_ON_RASTER) {
            d.line_phase = Some(ctx.line_phase);
        }
        let xf = Transform { st: &job.state, depth_row: job.depth.row, divisor: job.depth.divisor, back_colours: d.back_colours };
        // SAFETY: the job holds its slot until it is given back.
        let st: &'static DrawState = unsafe { &*job.slot };
        let tex = Unit { regs: &st.tex, latch: &st.latch, mem: job.pixels.obj };
        let mut p = PipeDraw::new(&job.words, tex, job.pixels.state.mem, job.index, job.pixels.state.to_aux, xf.depth_range(), job.depth.slack, ctx.ring.as_ref());
        if p.ring.is_none() {
            ctx.drain_back();
        }
        // Filled triangles alone go to the tiles, whose owners walk, shade
        // and store them; a draw with anything else shades here and puts
        // its quads in the tiles already shaded.
        let fills = d.mode >= TRIANGLES
            && d.mode <= POLYGON
            && !p.prims.polygon_smooth
            && p.prims.modes == [PolygonMode::Fill, PolygonMode::Fill];
        p.shade = allow_shader && fills && p.ring.as_ref().is_some_and(|r| r.shader());
        if p.ring.is_some() && !p.shade {
            // The tiles store behind this thread, so the depth it would
            // pre-test against may be stale; the tiles' own walks read
            // their stores.
            p.raster.pretest = None;
        }
        let p = &**fr.pipe.insert(p);
        super::pipe::setup::run(p, &mut d, xf, fr.words_stipple.as_ref(), job.background, &fr.live);
        if let Some(phase) = d.line_phase {
            ctx.line_phase = phase;
        }
        let phase = d.line_phase;
        (phase, frame)
    }
}

impl Board {
    /// The board state a GL draw's pixels pass under.
    pub fn gl_pixel_state(&mut self) -> GlPixelState {
        GlPixelState {
            mem: DestMem::of(self.vram.ptr(), &self.dest_views()),
            id_compare: self.win.id_compare,
            clip_rect: self.win.clip_rect,
            clip_kept: self.win.clip_kept,
            write_mask: self.gl_write_mask,
            to_aux: self.gl_to_aux(),
        }
    }

}

impl Board {
    /// How far past [0, 1] a fragment's depth may round and still be drawn,
    /// clamped: 16 units in the last place of a single near 1 (gl_cmd::DEPTH_SLACK
    /// in consts.rs). APPROXIMATION: the depth row and block 16 arrive as
    /// singles, and a library that maps the far plane to 0.99999999 of the range
    /// leaves less margin than their rounding takes. A board does its
    /// arithmetic in its own precision and the clip it applies at the far plane
    /// is UNKNOWN.
    ///
    /// The depth row over the block-16 row, in units of the client's [0, 1]
    /// depth (Board::gl_load_combined()). The shared context's model-view is
    /// the identity, so its eye coordinates are the object coordinates the
    /// combined matrix takes. Block 16 is the divisor only when all four of its
    /// words are finite and the fourth is not zero; otherwise the sign-only
    /// fallback divides by a constant.
    pub fn gl_depth_row(&mut self) -> DepthRow {
        let scale = 2.0 * gl_cmd::DEPTH_S * self.gl_depth_scale();
        let mut block16 = [0.0f64; 4];
        let mut row = [0.0f64; 4];
        let mut divisor = true;

        for k in 0..4usize {
            row[k] = f64::from(odyssey_f32(self.gl_combined[8 + k])) / scale;
            block16[k] = f64::from(odyssey_f32(self.gl_block16[k]));
            divisor &= block16[k].is_finite();
        }
        DepthRow {
            row,
            divisor: (divisor && block16[3] != 0.0).then(|| block16.map(|b| b / block16[3])),
            slack: gl_cmd::DEPTH_SLACK,
        }
    }
}

/// The transform side's depth for a draw: the depth row and divisor, each
/// applied to the eye coordinates over w, and the rounding slack.
#[derive(Clone, Copy, Default)]
pub(super) struct DepthRow {
    pub row: [f64; 4],
    pub divisor: Option<[f64; 4]>,
    pub slack: f64,
}

impl Board {
    /// An opaque polygon stipple (0x043f bit 4 with shape 0, the GL stipple):
    /// clear bits take the 0x0c21 constant colour, as the 2D path's opaque
    /// patterns do (Board::pattern_background()). That the board paints the
    /// clear bits here as it does for the 2D quads is INFERRED: it is the only
    /// reading under which the X server's FillOpaqueStippled sequence draws an
    /// opaque stipple.
    pub fn gl_stipple_background(&mut self) -> Option<[f64; 4]> {
        if (self.stream.pattern_shape & pattern::SHAPE_OPAQUE) == 0 {
            return None;
        }
        let c = if self.gl_index_mode() {
            gl_index_rgba(self.pattern_background(true))
        } else {
            let pixel = self.pattern_background(false);

            [0, 1, 2, 3].map(|k| if k == 3 { 1.0 } else { ((pixel >> (24 - 8 * k)) & 0xff) as f32 / 255.0f32 })
        };
        Some(c.map(f64::from))
    }
}

impl Board {
    /// The vertex's texture coordinates from the board's words, with the
    /// linear texgen planes where they apply, and the texture matrix.
    pub fn texture_vertex(&mut self) {
        let mut strq: [u32; 4] = [self.tex.coord[0], self.tex.coord[1], self.tex.coord_r, self.tex.coord_q];
        let one_d = (self.tex_regs().path & (tex::MODE_1D | tex::MODE_2D)) == tex::MODE_1D;
        // Captured linear planes already include the eye-space transform.
        let position_word = self.cmd.tag == gl_cmd::VERTEX_2I
            || self.cmd.tag == gl_cmd::VERTEX_2F
            || self.cmd.tag == gl_cmd::VERTEX_3F
            || self.cmd.tag == gl_cmd::VERTEX_3I
            || self.cmd.tag == gl_cmd::VERTEX_4F;
        if position_word && (self.geom.linear_texgen != 0 || self.geom.sphere_texgen != 0) {
            let mut vertex = [
                f64::from(odyssey_f32(self.cmd.operand[0])),
                f64::from(odyssey_f32(self.cmd.operand[1])),
                if self.cmd.tag != gl_cmd::VERTEX_2I && self.cmd.tag != gl_cmd::VERTEX_2F { f64::from(odyssey_f32(self.cmd.operand[2])) } else { 0.0 },
                if self.cmd.tag == gl_cmd::VERTEX_4F { f64::from(odyssey_f32(self.cmd.operand[3])) } else { 1.0 },
            ];

            if self.cmd.tag == gl_cmd::VERTEX_2I || self.cmd.tag == gl_cmd::VERTEX_3I {
                vertex[0] = (self.cmd.operand[0] as i32) as f64;
                vertex[1] = (self.cmd.operand[1] as i32) as f64;
            }
            if self.cmd.tag == gl_cmd::VERTEX_3I {
                vertex[2] = (self.cmd.operand[2] as i32) as f64;
            }
            if self.geom.sphere_texgen != 0 {
                let [ss, tt] = self.sphere_map(&vertex);
                if self.geom.sphere_texgen & 1 != 0 {
                    strq[0] = ss.to_bits();
                }
                if self.geom.sphere_texgen & 2 != 0 {
                    strq[1] = tt.to_bits();
                }
            }
            for axis in 0..4usize {
                let plane = if axis == 3 {
                    self.tex.linear_plane_q
                } else if axis == 2 {
                    self.tex.linear_plane_r
                } else {
                    self.tex.linear_plane[axis]
                };
                let mut value = 0.0f32;

                if (self.geom.linear_texgen & self.tex.linear_valid & (1u32 << axis)) == 0 {
                    continue;
                }
                for k in 0..4usize {
                    value = (value as f64 + f64::from(odyssey_f32(plane[k])) * vertex[k]) as f32;
                }
                strq[axis] = value.to_bits();
            }
        }
        let rows = self.gl_texture_matrix;
        let mut matrix = gl_transpose(&rows);
        // Input T may transform S, but transformed T cannot affect 1D sampling.
        if one_d {
            for i in 0..4usize {
                matrix[i * 4 + 1] = 0.0;
            }
        }
        self.assembly.texture = matrix;
        self.assembly.texcoord = strq.map(f32::from_bits);
    }
}

impl Board {
    /// GL_SPHERE_MAP's s and t for a vertex at object position obj: the eye
    /// direction u to the vertex reflected about the eye-space normal n,
    /// r = u - 2 (n . u) n, and m = 2 sqrt(rx^2 + ry^2 + (rz + 1)^2), then
    /// s = rx / m + 1/2, t = ry / m + 1/2 (OpenGL 1.2, 2.10.4). The normal is
    /// the current normal through the normal matrix, renormalised under
    /// GL_NORMALIZE, as the lighting takes it.
    pub fn sphere_map(&self, obj: &[f64; 4]) -> [f32; 2] {
        let f = |w: u32| f32::from_bits(w) as f64;
        let mv = &self.geom.modelview;
        let nm = &self.geom.normal_matrix;
        let normal = self.geom.normal.map(f);
        let eye: [f64; 3] = std::array::from_fn(|r| (0..4).map(|k| f(mv[r * 4 + k]) * obj[k]).sum());
        let mut n: [f64; 3] = std::array::from_fn(|r| (0..3).map(|c| f(nm[r * 4 + c]) * normal[c]).sum());
        let unit = |v: [f64; 3]| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if l > 0.0 { v.map(|c| c / l) } else { v }
        };
        if self.geom.normalize {
            n = unit(n);
        }
        let u = unit(eye);
        let nu = n[0] * u[0] + n[1] * u[1] + n[2] * u[2];
        let r: [f64; 3] = std::array::from_fn(|k| u[k] - 2.0 * nu * n[k]);
        let m = 2.0 * (r[0] * r[0] + r[1] * r[1] + (r[2] + 1.0) * (r[2] + 1.0)).sqrt();
        if m == 0.0 {
            return [0.5, 0.5];
        }
        [(r[0] / m + 0.5) as f32, (r[1] / m + 0.5) as f32]
    }
}

/// A draw's estimated raster work in pixels, from its vertices' window
/// positions (`primitive_cost`).
fn gl_draw_cost(d: &Prim, sizes: [u32; 2]) -> u32 {
    let points = d.vertices.chunks_exact(vtx::FLOATS).take(d.count as usize).map(|v| {
        let clip = [v[vtx::POSITION], v[vtx::POSITION + 1], v[vtx::POSITION + 2], v[vtx::POSITION + 3]];
        if !(clip[3] > 0.0) {
            return None;
        }
        let win = d.xf.clip_to_window(&clip, super::pipe::setup::SUB as f32)?;
        Some((win[0] as f64, win[1] as f64))
    });
    let size = |w: u32| (w & super::consts::line::POINT_SIZE_MASK) as f32 / super::consts::line::POINT_SIZE_SCALE;

    primitive_cost(d.mode, points, size(sizes[1]), size(sizes[0]))
}

impl Board {
    /// Draw one assembled primitive: set up as a draw job from the board's
    /// words and the transform state, and run on the raster unit (now, when
    /// it has no thread of its own).
    pub fn draw_prim(&mut self, d: &mut Prim) {
        // A job the raster unit has finished with, if one is back.
        let spare = self.raster.take_draw();
        let words = self.fragment_words();
        let textured = self.tex_regs().target() != Target::None;
        let slot = self.issue_state();

        self.gl_report_refusals(true);
        let depth = self.gl_depth_row();
        let index = self.gl_index_mode();
        let background = self.gl_stipple_background();
        let pixels = self.gl_pixel_job();
        let state = *d.xf;
        let job = match spare {
            Some(mut j) => {
                j.depth = depth;
                j.index = index;
                j.background = background;
                j.slot = slot;
                j.state = state;
                j.vertices.clear();
                j.vertices.extend_from_slice(d.vertices);
                j.fields = PrimFields::of(d);
                j.pixels = pixels;
                j.words = words;
                j
            }
            None => Box::new(GlDrawJob {
                depth,
                index,
                background,
                slot,
                state,
                vertices: d.vertices.to_vec(),
                fields: PrimFields::of(d),
                pixels,
                words,
            }),
        };
        let cost = gl_draw_cost(d, self.line_point_words);
        let tracing = self.trace.on(TRACE_DRAW);
        if !self.raster.threaded() {
            d.line_phase = self.raster.run_now(|ctx| job.run(ctx));
            self.raster.give_back(job);
            if tracing {
                self.trace_draw(d, textured, "drawn");
            }
            return;
        }
        if tracing {
            self.trace_draw(d, textured, &format!("queued cost {cost}"));
        }
        // The raster unit draws it later, and keeps the line-stipple phase
        // the draw advances.
        self.raster.submit_draw(job, cost);
        if d.line_phase.is_some() {
            d.line_phase = Some(LINE_PHASE_ON_RASTER);
        }
    }

    /// The board's planes and pixel state as a raster job takes them.
    pub fn gl_pixel_job(&mut self) -> GlPixelJob {
        GlPixelJob { state: self.gl_pixel_state(), obj: self.vram.ptr() }
    }

    /// glClear's raster op, and the pattern rectangle's runs: the raster
    /// unit runs ops in order with the draw state of now.
    pub fn gl_raster_ops(&mut self, ops: Vec<RasterOp>) {
        if self.trace.on(TRACE_DRAW) {
            for op in &ops {
                self.trace_raster_op(op);
            }
        }
        let index = self.gl_index_mode();
        let pixels = self.gl_pixel_job();

        if let Some(op) = ops.first() {
            self.gl_report_refusals(op.kind != RASTER_CLEAR || (op.clear_buffers & CLEAR_COLOUR) != 0);
        }
        let cost = ops.iter().fold(0u32, |c, op| c.saturating_add(rect_cost(op.x0, op.y0, op.x1, op.y1, op.kind == RASTER_CLEAR)));
        // Clears go through the quad ring, in order behind the draws before
        // them, with no drain; without a back-end thread they store here.
        self.raster.submit_ring(cost, move |ctx| {
            let ring = ctx.ring.clone();
            for op in &ops {
                if op.kind != RASTER_CLEAR {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: raster op of kind {} is not executed", op.kind);
                    continue;
                }
                let Some(clear) = clear_draw(index, &pixels, op) else {
                    continue;
                };
                let st = &pixels.state;
                let Some(ring) = &ring else {
                    for y in op.y0..=op.y1 {
                        for r in &st.clip_rect[..st.clip_kept as usize] {
                            let (x0, x1) = (op.x0.max(r.x0), op.x1.min(r.x1));
                            if y >= r.y0 && y <= r.y1 && x0 <= x1 {
                                clear.span(y, x0, x1);
                            }
                        }
                    }
                    continue;
                };
                // One rectangle a clip rectangle, which the tiles it covers
                // clear their part of.
                let p: *const super::pipe::backend::ClearDraw = &*clear;
                for r in &st.clip_rect[..st.clip_kept as usize] {
                    let (x0, y0, x1, y1) = (op.x0.max(r.x0), op.y0.max(r.y0), op.x1.min(r.x1), op.y1.min(r.y1));
                    if x0 <= x1 && y0 <= y1 {
                        ring.push_rect(x0, y0, x1, y1, p);
                    }
                }
                ring.hold_clear(clear);
            }
        });
    }

    /// glClear's raster op.
    pub fn gl_raster(&mut self, op: &RasterOp) {
        self.gl_raster_ops(vec![*op]);
    }

}

impl PrimSink for Board {
    fn draw(&mut self, p: &mut Prim) {
        self.draw_prim(p);
    }
}

impl Board {
    /// Run f on the board's primitive assembly, with the board as the sink
    /// for the pieces it sends: the assembly is out of the board for the
    /// call, so that a draw can borrow the rest of it.
    pub fn assemble(&mut self, f: impl FnOnce(&mut Assembly, &mut Board)) {
        let mut a = std::mem::take(&mut self.assembly);

        f(&mut a, self);
        self.assembly = a;
    }

    /// One vertex at object position pos, `components` of it given, into
    /// the open primitive.
    pub fn assemble_vertex(&mut self, pos: &[f32], components: u32) {
        self.assemble(|a, b| a.vertex(b, pos, components));
    }

    /// User clip plane i, from its eye-space words, in the object
    /// coordinates the transform state's identity model-view makes eye
    /// coordinates: p_obj[k] = sum_r p_eye[r] M[r][k], M the geometry
    /// engine's row-major model-view.
    pub fn load_clip_plane(&mut self, i: u32) {
        let mut eye = [0.0f64; 4];
        let mut obj = [0.0f64; 4];

        for r in 0..4usize {
            eye[r] = f64::from(odyssey_f32(self.geom.clip_eye[i as usize * 4 + r]));
        }
        for r in 0..4usize {
            for k in 0..4usize {
                obj[k] += eye[r] * f64::from(odyssey_f32(self.geom.modelview[r * 4 + k]));
            }
        }
        self.assembly.xf.clip_plane[i as usize] = obj.map(|v| v as f32);
    }

    /// Which user clip planes are tested, from the path word's thermometer
    /// count n (ShaderWords::clip_planes()): planes 0 to n - 1.
    pub fn load_clip_planes(&mut self) {
        let n = self.tex_regs().clip_planes();

        self.assembly.xf.clip_planes = (1u32 << n.min(CLIP_PLANES)) - 1;
    }
}

/// The lighting words matched by value/// The lighting words matched by value (gl_cmd::NORMAL_3F and the rest).
pub(super) static GL_LIGHT_WORDS: [u32; 16] = [
    gl_cmd::NORMAL_3F,
    gl_cmd::MATERIAL_AMBIENT,
    gl_cmd::MATERIAL_DIFFUSE,
    gl_cmd::MATERIAL_SPECULAR,
    gl_cmd::MATERIAL_EMISSION,
    gl_cmd::BACK_AMBIENT,
    gl_cmd::BACK_DIFFUSE,
    gl_cmd::BACK_SPECULAR,
    gl_cmd::BACK_EMISSION,
    gl_cmd::SHININESS_SQRT,
    gl_cmd::BACK_SHININESS_SQRT,
    gl_cmd::SHININESS_DERIVED,
    gl_cmd::LIGHT_COLOUR_LOAD,
    gl_cmd::LIGHT_VEC_LOAD,
    gl_cmd::CLIP_PLANE_LOAD,
    prog::RUN_ONCE,
];

/// The texture-coordinate and generation fragments.
pub(super) static UCODE_TEXGEN: [u32; 14] = [
    0x033, 0x04d, 0x052, 0x057, 0x05c, 0x061, 0x066, 0x06b, 0x06c, 0x077, 0x07b, 0x07f, 0x091, 0x09f,
];

pub(super) fn prog_value(word: u32) -> u32 {
    (word & prog::LOW_MASK) | (((word >> prog::HIGH_SHIFT) & 1) << 10)
}

pub(super) fn is_program_word(word: u32) -> bool {
    let cls = word >> prog::CLASS_SHIFT;

    if (word & prog::SHAPE) != prog::TAG {
        return false;
    }
    if (prog::PROGRAM_HALFWORD..=prog::PROGRAM_HALFWORD_LAST).contains(&cls) {
        return true;
    }
    if (prog::REG_FIRST..=prog::REG_LAST).contains(&cls)
        && (word & gl_cmd::MATRIX_SELECT_SHAPE) != gl_cmd::MATRIX_SELECT_TAG
    {
        return true;
    }
    matches!(
        cls,
        prog::COLOUR_MODE
            | prog::COLOUR_MODE_B
            | prog::SET_UCODE
            | prog::PROGRAM_ADDR
            | prog::CONST_SELECT
            | prog::ENTRY_29
            | prog::ENTRY_2A
            | prog::ENTRY_2C
            | prog::LIGHT_VEC_SELECT
            | prog::LIGHT_COLOUR_SELECT
            | prog::LIST_INDEX
            | prog::COLOUR_MATERIAL
            | prog::CONST_STORE
    )
}

pub(super) fn ucode_is_texgen(a: u32) -> bool {
    UCODE_TEXGEN.contains(&a)
}

/// A clear's record for the back end (`pipe::backend::ClearDraw`): the
/// colour as an 8-bit store takes it (floor(256 c), at most 255), or in
/// colour-index mode the index under the index mask; the colour mask the
/// planes keep, the depth word and the stencil value under its write mask;
/// None where it stores nothing. An overlay context's clear
/// (Board::gl_to_aux) stores only the overlay byte, which the destination's
/// pixel mode places.
fn clear_draw(index: bool, pixels: &GlPixelJob, op: &RasterOp) -> Option<Box<super::pipe::backend::ClearDraw>> {
    use super::pipe::backend::quantise8;

    if op.clip.n == 0 && op.clip.empty {
        return None;
    }
    let st = &pixels.state;
    // Colour-index windows store no alpha.
    let rgba_mask = op.clear_colour_mask & if index { 0x7 } else { 0xf };
    let n = op.ntargets.min(MAX_TARGETS) as usize;
    let mut targets = [false; 2];
    for &t in &op.targets[..n] {
        if (t as usize) < 2 {
            targets[t as usize] = true;
        }
    }
    if st.to_aux {
        targets = super::pipe::aux_targets(targets, index);
    }
    let c = op.clear_rgba.map(|c| quantise8(c as f64));
    // The word's bits the clear changes.
    let colour_mask = if index {
        st.write_mask & write::INDEX_MASK
    } else {
        (0..4).filter(|k| rgba_mask & (1 << k) != 0).fold(0u32, |m, k| m | 0xff << (24 - 8 * k))
    };
    let colour = (op.clear_buffers & CLEAR_COLOUR != 0 && targets != [false; 2] && colour_mask != 0).then(|| {
        if index {
            // The index from its two carrier components (gl_index_rgba()).
            return (c[1] << 8 | c[2]) & ODSY_INDEX_MASK;
        }
        super::pipe::backend::pack_rgba8(c)
    });
    let depth = (op.clear_buffers & CLEAR_DEPTH != 0).then_some(op.clear_depth24 & ODSY_DEPTH24_MAX);
    let stencil = (op.clear_buffers & CLEAR_STENCIL != 0).then_some(((op.clear_stencil & 0xff) as u8, (op.stencil_mask & 0xff) as u8));
    if colour.is_none() && depth.is_none() && stencil.is_none() {
        return None;
    }
    let id_mask = if st.id_compare & ODSY_ID_COMPARE_ENABLE != 0 { (st.id_compare >> ODSY_ID_COMPARE_MASK_SHIFT) as u8 } else { 0 };

    Some(Box::new(super::pipe::backend::ClearDraw {
        planes: st.mem,
        targets,
        colour,
        colour_mask: if colour_mask == (if index { write::INDEX_MASK } else { u32::MAX }) { u32::MAX } else { colour_mask },
        depth,
        stencil,
        id_value: st.id_compare as u8,
        id_mask,
    }))
}
