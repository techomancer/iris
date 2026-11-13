//! Primitive assembly: the vertices the geometry engine's words deliver,
//! each with the current colour, texture coordinates and edge flag, into
//! the pieces the drawing pipeline takes (`Prim`), each sent as soon as its
//! last vertex arrives, so a context switch between two vertices loses
//! nothing; and the transform state the pipeline's setup maps them by
//! (`XfState`), in the shared T&L code's terms (`dev::gl`).
//!
//! The geometry engine lights each vertex before it gets here
//! (`Board::gl_vertex_colour`), and the combined matrix carries the
//! model-view, so the model-view is the identity and the projection the
//! combined matrix's window map (`Board::gl_load_combined`).

use crate::devlog::LogModule;
use crate::dev::gl::math::{xform, IDENT};
use crate::dev::gl::vertex::{NearW, Viewport, Wv};

/// The board's primitive types (the 0x0400 begin word), in assembly order.
pub(super) const POINTS: u32 = 0;
pub(super) const LINES: u32 = 1;
pub(super) const LINE_LOOP: u32 = 2;
pub(super) const LINE_STRIP: u32 = 3;
pub(super) const TRIANGLES: u32 = 4;
pub(super) const TRIANGLE_STRIP: u32 = 5;
pub(super) const TRIANGLE_FAN: u32 = 6;
pub(super) const QUADS: u32 = 7;
pub(super) const QUAD_STRIP: u32 = 8;
pub(super) const POLYGON: u32 = 9;

/// The user clip planes.
pub(super) const CLIP_PLANES: u32 = 6;

/// The most vertices assembly holds between pieces of a primitive: a quad
/// is sent once its fourth arrives, so three wait.
pub(super) const HELD_VERTICES: u32 = 3;

/// A vertex as assembly hands it on: floats at these offsets.
pub(super) mod vtx {
    pub const FLOATS: usize = 28;
    pub const POSITION: usize = 0;
    pub const COLOR: usize = 4;
    /// The vertex's material index (0: the geometry engine lights it first).
    pub const MATERIAL: usize = 11;
    pub const TEXCOORD: usize = 12;
    /// The edge flag's bits.
    pub const EDGE: usize = 17;
    pub const BACK_COLOR: usize = 18;
    /// The separate specular colour (GL_SEPARATE_SPECULAR_COLOR), RGB, and
    /// the back face's; zero otherwise.
    pub const SPECULAR: usize = 22;
    pub const BACK_SPECULAR: usize = 25;
}

/// Reports, once per process (a machine has one VPro board), that the w
/// plane clipped a primitive: preclip is off by default, the library stores
/// positions untested, and the board's behaviour at w <= 0 is unknown. iris
/// clips there so nothing reaches the integer edge functions. Call it from
/// a clip's slow path only.
pub(super) fn report_w_plane_clip() {
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::dlog!(LogModule::Odyssey, "sgi-odyssey: a primitive reached w <= 0 and was clipped; the board's behaviour there is unknown");
    }
}

/// What setup maps a primitive's vertices by: object to eye by the
/// model-view, eye to clip by the projection, clip to window by the
/// viewport, window depth through the depth range, and the user clip
/// planes, in eye coordinates, those whose bit is set enabled.
#[derive(Clone, Copy)]
pub(super) struct XfState {
    /// Column-major.
    pub modelview: [f32; 16],
    pub projection: [f32; 16],
    pub viewport: [i32; 4],
    pub depth_range: [f32; 2],
    pub clip_planes: u32,
    pub clip_plane: [[f32; 4]; CLIP_PLANES as usize],
}

impl Default for XfState {
    fn default() -> Self {
        XfState { modelview: IDENT, projection: IDENT, viewport: [0; 4], depth_range: [0.0, 1.0], clip_planes: 0, clip_plane: [[0.0; 4]; CLIP_PLANES as usize] }
    }
}

impl XfState {
    /// An eye position's clip coordinates by the projection, in single
    /// precision through the shared transform.
    pub fn eye_to_clip(&self, eye: [f32; 4]) -> [f32; 4] {
        xform(&self.projection, eye)
    }

    /// The viewport as the shared viewport mapping takes it, depth [0, 1]
    /// left normalized: the depth range maps it per fragment.
    fn mapping(&self) -> Viewport {
        let [x, y, w, h] = self.viewport.map(|v| v as f32);
        let mut vp = Viewport::default();

        vp.set(x, y, w, h, 0.0, 1.0, 1.0);
        vp
    }

    /// Clip coordinates to window x and y on the grid of 1 / subpixel of a
    /// pixel, and normalized depth, (z / w + 1) / 2, in single precision
    /// through the shared VPro geometry (dev::gl::vertex::NearW): the divide
    /// is a nearest-rounded reciprocal and a multiply, and the position
    /// truncates onto the grid. None for a w of zero or less.
    pub fn clip_to_window(&self, clip: &[f32; 4], subpixel: f32) -> Option<[f32; 3]> {
        let v = self.mapping().project_by::<NearW, 0, 0>(Wv::<0, 0> { h: *clip, ..Default::default() }, [0.0, 0.0], subpixel);

        (v.ok != 0).then_some([v.x, v.y, v.z])
    }
}

/// One piece of a primitive, or a whole polygon, as the pipeline takes it.
#[derive(Clone, Copy)]
pub(super) struct Prim<'a> {
    /// The board's context identity.
    pub context: u32,
    /// POINTS to POLYGON.
    pub mode: u32,
    pub count: u32,
    /// count * vtx::FLOATS floats, the texture coordinates through the
    /// texture matrix, ready for clipping.
    pub vertices: &'a [f32],
    /// The vertices carry back-face colours for a back-facing polygon
    /// (two-sided lighting).
    pub back_colours: bool,
    pub xf: &'a XfState,
    /// A whole polygon assembled under per-face polygon modes, which the
    /// raster words give.
    pub whole_polygon: bool,
    /// The standalone line stipple's phase, which the draw advances in
    /// place; None counts a phase local to the draw.
    pub line_phase: Option<u32>,
    /// With a borrowed phase, reset a local candidate before counting.
    pub line_phase_reset: bool,
}

impl<'a> Prim<'a> {
    pub fn new(mode: u32, count: u32, vertices: &'a [f32], xf: &'a XfState) -> Self {
        Prim { context: 0, mode, count, vertices, back_colours: false, xf, whole_polygon: false, line_phase: None, line_phase_reset: false }
    }
}

/// Where assembly's pieces go: drawn, or queued to draw, before the call
/// returns, so what a piece points at need only live that long.
pub(super) trait PrimSink {
    /// Draw one piece; a line stipple's phase comes back in p.line_phase.
    fn draw(&mut self, p: &mut Prim);
}

/// A polygon's vertices, kept until the end of the primitive.
#[derive(Clone, Default)]
pub(super) struct PolygonBuffer {
    pub vertices: Vec<f32>,
    pub count: u32,
    /// The buffer could not grow: the polygon is dropped.
    pub failed: bool,
}

impl PolygonBuffer {
    fn release(&mut self) {
        *self = PolygonBuffer::default();
    }

    /// Room for count vertices; false where it cannot be had.
    fn reserve(&mut self, count: usize) -> bool {
        let floats = count.checked_mul(vtx::FLOATS);

        match floats {
            Some(f) if f <= self.vertices.len() => true,
            Some(f) => {
                let want = f.max(8 * vtx::FLOATS).max(2 * self.vertices.len());

                self.vertices.try_reserve_exact(want - self.vertices.len()).is_ok() && {
                    self.vertices.resize(want, 0.0);
                    true
                }
            }
            None => false,
        }
    }
}

/// The board's primitive assembly and its transform state.
#[derive(Clone)]
pub(super) struct Assembly {
    /// The board's context identity.
    pub hw_id: u32,
    pub xf: XfState,
    /// The texture matrix, column-major, applied to each vertex's texture
    /// coordinates as it arrives.
    pub texture: [f32; 16],
    pub colour: [f32; 4],
    /// The back face's colour under two-sided lighting, else the colour.
    pub back_colour: [f32; 4],
    /// The vertices carry back colours (Prim::back_colours).
    pub two_sided: bool,
    /// The separate specular colours, front and back (vtx::SPECULAR).
    pub specular: [f32; 3],
    pub back_specular: [f32; 3],
    pub texcoord: [f32; 4],
    pub edge: u32,
    pub polygon: PolygonBuffer,
    /// The primitive open since its begin, POINTS to POLYGON.
    pub begin_mode: u32,
    pub in_begin: bool,
    /// Vertices waiting for the rest of their piece, nheld of them.
    pub held: [[f32; vtx::FLOATS]; HELD_VERTICES as usize],
    pub nheld: u32,
    /// Set while the next triangle of an open strip is an odd one, whose
    /// first two vertices are sent swapped.
    pub strip_odd: bool,
    /// The standalone line stipple's phase, carried across GL_LINES pieces.
    pub line_phase: u32,
}

impl Default for Assembly {
    fn default() -> Self {
        Assembly {
            hw_id: 0,
            xf: XfState::default(),
            texture: IDENT,
            colour: [1.0; 4],
            back_colour: [1.0; 4],
            two_sided: false,
            specular: [0.0; 3],
            back_specular: [0.0; 3],
            texcoord: [0.0, 0.0, 0.0, 1.0],
            edge: 1,
            polygon: PolygonBuffer::default(),
            begin_mode: 0,
            in_begin: false,
            held: [[0.0; vtx::FLOATS]; HELD_VERTICES as usize],
            nheld: 0,
            strip_odd: false,
            line_phase: 0,
        }
    }
}

impl Assembly {
    /// Start over for context hw_id: identity matrices, white, no primitive.
    pub fn reset(&mut self, hw_id: u32) {
        *self = Assembly { hw_id, ..Assembly::default() };
    }

    /// The current texture coordinates through the texture matrix.
    fn texture_coord(&self) -> [f32; 4] {
        xform(&self.texture, self.texcoord)
    }

    /// One vertex at object position `pos` (`components` of it given, the
    /// rest 0 and w 1) with the current attributes, not drawn. The board
    /// keeps these to restart a primitive after a context switch.
    pub fn capture_vertex(&self, mode: u32, pos: &[f32], components: u32) -> [f32; vtx::FLOATS] {
        let mut out = [0.0; vtx::FLOATS];

        out[vtx::POSITION + 3] = 1.0;
        for i in 0..(components as usize).min(pos.len()) {
            out[vtx::POSITION + i] = pos[i];
        }
        out[vtx::COLOR..vtx::COLOR + 4].copy_from_slice(&self.colour);
        out[vtx::TEXCOORD..vtx::TEXCOORD + 4].copy_from_slice(&self.texture_coord());
        out[vtx::EDGE] = f32::from_bits(self.edge);
        out[vtx::BACK_COLOR..vtx::BACK_COLOR + 4].copy_from_slice(&self.back_colour);
        out[vtx::SPECULAR..vtx::SPECULAR + 3].copy_from_slice(&self.specular);
        out[vtx::BACK_SPECULAR..vtx::BACK_SPECULAR + 3].copy_from_slice(&self.back_specular);
        // Points and lines take the colour sum here: only filled triangles
        // carry the specular apart to the shader.
        if mode <= LINE_STRIP {
            for k in 0..3 {
                out[vtx::COLOR + k] = (out[vtx::COLOR + k] + out[vtx::SPECULAR + k]).min(1.0);
                out[vtx::BACK_COLOR + k] = (out[vtx::BACK_COLOR + k] + out[vtx::BACK_SPECULAR + k]).min(1.0);
                out[vtx::SPECULAR + k] = 0.0;
                out[vtx::BACK_SPECULAR + k] = 0.0;
            }
        }
        // A strip's and a fan's inner edges are outline edges under the
        // polygon modes.
        if mode == TRIANGLE_STRIP || mode == TRIANGLE_FAN || mode == QUAD_STRIP {
            out[vtx::EDGE] = f32::from_bits(1);
        }
        out
    }

    /// Send one piece of the open primitive: count of v.
    fn submit(&mut self, sink: &mut dyn PrimSink, mode: u32, v: &[&[f32; vtx::FLOATS]]) {
        let mut verts = [0.0f32; 4 * vtx::FLOATS];

        for (i, src) in v.iter().enumerate() {
            verts[i * vtx::FLOATS..][..vtx::FLOATS].copy_from_slice(*src);
        }
        let mut p = Prim::new(mode, v.len() as u32, &verts[..v.len() * vtx::FLOATS], &self.xf);

        p.context = self.hw_id;
        p.back_colours = self.two_sided;
        p.line_phase = (mode == LINES).then_some(self.line_phase);
        p.line_phase_reset = self.begin_mode == LINES;
        sink.draw(&mut p);
        if let Some(phase) = p.line_phase {
            self.line_phase = phase;
        }
    }

    fn hold(&mut self, n: u32, v: &[f32; vtx::FLOATS]) {
        self.held[n as usize] = *v;
    }

    /// Open a primitive of `mode`.
    pub fn begin(&mut self, mode: u32) {
        if self.in_begin {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: a begin inside a primitive is ignored");
            return;
        }
        if mode > POLYGON {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: a begin of primitive type {} is ignored", mode);
            return;
        }
        self.line_phase = 0;
        self.in_begin = true;
        self.begin_mode = mode;
        self.nheld = 0;
        self.strip_odd = false;
    }

    /// Close the open primitive: a polygon draws whole, and a line loop
    /// closes from its last vertex to its first.
    pub fn end(&mut self, sink: &mut dyn PrimSink) {
        if !self.in_begin {
            return;
        }
        if self.begin_mode == POLYGON {
            if !self.polygon.failed && self.polygon.count >= 3 {
                let polygon = std::mem::take(&mut self.polygon);
                let mut p = Prim::new(POLYGON, polygon.count, &polygon.vertices, &self.xf);

                p.context = self.hw_id;
                p.back_colours = self.two_sided;
                p.whole_polygon = true;
                sink.draw(&mut p);
            }
            self.polygon.release();
        }
        if self.begin_mode == LINE_LOOP && self.nheld == 2 {
            let (h1, h0) = (self.held[1], self.held[0]);
            self.submit(sink, LINES, &[&h1, &h0]);
        }
        self.discard_primitive();
        self.line_phase = 0;
    }

    /// One vertex of the open primitive with the current attributes. Every
    /// piece is sent as soon as its last vertex arrives; vertices that
    /// complete no piece before the end draw nothing.
    pub fn vertex(&mut self, sink: &mut dyn PrimSink, pos: &[f32], components: u32) {
        let (h0, h1, h2) = (self.held[0], self.held[1], self.held[2]);
        let n = self.nheld;

        if !self.in_begin {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: a vertex outside a primitive is ignored");
            return;
        }
        let cur = self.capture_vertex(self.begin_mode, pos, components);
        match self.begin_mode {
            POLYGON => {
                let p = &mut self.polygon;

                if p.failed {
                    return;
                }
                if p.count == u32::MAX || !p.reserve(p.count as usize + 1) {
                    p.release();
                    p.failed = true;
                    return;
                }
                let at = p.count as usize * vtx::FLOATS;
                p.vertices[at..at + vtx::FLOATS].copy_from_slice(&cur);
                p.count += 1;
                return;
            }
            POINTS => {
                self.submit(sink, POINTS, &[&cur]);
                return;
            }
            LINES if n >= 1 => {
                self.submit(sink, LINES, &[&h0, &cur]);
                self.nheld = 0;
                return;
            }
            LINE_STRIP if n >= 1 => {
                self.submit(sink, LINES, &[&h0, &cur]);
                self.hold(0, &cur);
                return;
            }
            LINE_LOOP if n >= 1 => {
                // Slot 0 keeps the first vertex for the closing segment.
                let prev = self.held[n as usize - 1];
                self.submit(sink, LINES, &[&prev, &cur]);
                self.hold(1, &cur);
                self.nheld = 2;
                return;
            }
            TRIANGLES if n >= 2 => {
                self.submit(sink, TRIANGLES, &[&h0, &h1, &cur]);
                self.nheld = 0;
                return;
            }
            TRIANGLE_STRIP if n >= 2 => {
                if self.strip_odd {
                    self.submit(sink, TRIANGLES, &[&h1, &h0, &cur]);
                } else {
                    self.submit(sink, TRIANGLES, &[&h0, &h1, &cur]);
                }
                self.strip_odd = !self.strip_odd;
                self.hold(0, &h1);
                self.hold(1, &cur);
                return;
            }
            TRIANGLE_FAN if n >= 2 => {
                self.submit(sink, TRIANGLES, &[&h0, &h1, &cur]);
                self.hold(1, &cur);
                return;
            }
            QUADS if n >= 3 => {
                self.submit(sink, QUADS, &[&h0, &h1, &h2, &cur]);
                self.nheld = 0;
                return;
            }
            QUAD_STRIP if n >= 3 => {
                self.submit(sink, QUAD_STRIP, &[&h0, &h1, &h2, &cur]);
                self.hold(0, &h2);
                self.hold(1, &cur);
                self.nheld = 2;
                return;
            }
            _ => {}
        }
        self.hold(n, &cur);
        self.nheld = n + 1;
    }

    /// Resume a primitive a context switch interrupted: begin `mode` and
    /// take k of the vertices it held, oldest first, without drawing. count
    /// is how many vertices had been sent since its begin. Refused, with a
    /// log, inside a primitive and for a (k, count, mode) that cannot occur.
    pub fn resume_primitive(&mut self, mode: u32, v: &[[f32; vtx::FLOATS]], k: u32, count: u32) -> bool {
        if self.in_begin {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: a resume inside a primitive is ignored");
            return false;
        }
        if mode >= POLYGON || k > HELD_VERTICES || v.len() < k as usize || !resume_held_ok(mode, k, count) {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: resume of primitive type {} with {} held of {} sent refused", mode, k, count);
            return false;
        }
        // The line phase is left alone, unlike at a begin.
        self.in_begin = true;
        self.begin_mode = mode;
        self.nheld = k;
        for i in 0..k {
            self.hold(i, &v[i as usize]);
        }
        // The strip's next triangle faces as its count's parity says.
        self.strip_odd = mode == TRIANGLE_STRIP && k == 2 && count & 1 != 0;
        true
    }

    /// Close an interrupted primitive without drawing anything.
    pub fn discard_primitive(&mut self) {
        if self.begin_mode == POLYGON {
            self.polygon.release();
        }
        self.in_begin = false;
        self.begin_mode = 0;
        self.nheld = 0;
        self.strip_odd = false;
    }
}

/// Whether a primitive of this mode can hold k vertices after count sent
/// since its begin. Only count mod 2, 3 and 4 and k itself decide it, so a
/// count right only modulo 12 is enough.
pub(super) fn resume_held_ok(mode: u32, k: u32, count: u32) -> bool {
    match mode {
        POINTS => k == 0,
        LINES => k == count % 2,
        TRIANGLES => k == count % 3,
        QUADS => k == count % 4,
        QUAD_STRIP => {
            if k >= 2 {
                k == 2 + count % 2
            } else {
                k == count
            }
        }
        LINE_STRIP => k <= 1 && (k == 1 || count == 0),
        // Strips of triangles, fans, loops: first and last.
        _ => k <= 2 && (k == 2 || k == count),
    }
}
