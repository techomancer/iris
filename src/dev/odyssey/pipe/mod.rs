//! The fragment pipeline from the board's own state: the rasteriser, the
//! shader and the back end, each reading the words the board holds for it
//! (`clanker/plans/odyssey-quad-pipeline.md`).
//!
//! `state` holds the three state records and their decoders. The stages
//! follow: `backend` (planes and packers), `raster` (primitives to quads)
//! and `shader` (quads to colours).

pub(super) mod backend;
pub(super) mod fixed;
pub(super) mod interp;
#[cfg(test)]
pub(super) mod encode;
pub(super) mod prims;
pub(super) mod raster;
pub(super) mod setup;
pub(super) mod shader;
pub(super) mod state;
pub(super) mod texture;
pub(super) mod ring;
pub(super) mod slot;

use super::depth_word::{word_for_fraction, word_for_fraction32};
use backend::{DestMem, BackendState};
use raster::RasterState;
use shader::ShaderState;
use state::FragmentWords;
use std::sync::Arc;
use ring::{BackQuad, BackendDraw, QuadRing};

/// One draw's state for the three stages and the planes the back end
/// writes, built by the draw job from its words.
#[repr(C, align(64))]
pub(crate) struct PipeDraw {
    pub raster: raster::RasterState,
    pub shader: shader::ShaderState,
    pub backend: backend::BackendState,
    pub planes: backend::DestMem,
    /// The polygon offset of filled primitives, in depth-word steps.
    pub fill_steps: i64,
    /// What lines, points and polygon modes take from the raster words.
    pub prims: prims::PrimState,
    /// The draw's texture unit, and whether its target is on.
    pub tex: texture::Unit<'static>,
    pub textured: bool,
    /// The quad ring, when the back end runs on its own thread; otherwise
    /// the stores run inline.
    pub ring: Option<Arc<QuadRing>>,
    /// The draw's back-end state as the ring's slots name it (the draw's
    /// frame keeps it until the back end is past them, QuadRing::hold).
    pub back_draw: BackendDraw,
    /// The shader thread shades the draw's filled triangles.
    pub shade: bool,
    /// Some quad of the draw went into the ring.
    pub used: std::cell::Cell<bool>,
}

impl PipeDraw {
    /// Store a shaded quad (q.draw is set here): on the back-end thread, or
    /// now.
    #[inline]
    pub fn emit(&self, q: &mut BackQuad) {
        if q.cov == 0 {
            return;
        }
        match &self.ring {
            Some(r) => {
                q.draw = &self.back_draw;
                r.push_shaded(q);
                self.used.set(true);
            }
            None => (self.backend.store)(&self.backend, &self.planes, q),
        }
    }
}

impl PipeDraw {
    /// A draw's pipeline state from its words, over board memory: an RGBA
    /// or a colour-index draw, and whether it is an overlay context's
    /// (Board::gl_to_aux), whose destination is the window-id buffer. The
    /// transform side supplies the depth range and the depth's rounding
    /// slack.
    #[allow(clippy::too_many_arguments, reason = "the stages' inputs")]
    pub fn new(w: &FragmentWords, tex: texture::Unit<'static>, planes: DestMem, index: bool, to_aux: bool, depth_range: [f64; 2], slack: f64, ring: Option<&Arc<QuadRing>>) -> Box<PipeDraw> {
        let mut backend = BackendState::of(&w.backend, true, !index);
        if to_aux {
            backend.targets = aux_targets(backend.targets, index);
        }
        backend.store = backend.select(&planes);
        let depth = (planes.depth.v.kind != super::view::Kind::None).then_some(planes.depth);
        let mut raster = RasterState::of(
            &w.raster,
            w.backend.dest.colour[0].width,
            planes.did,
            depth_range,
            slack,
            word_for_fraction,
            depth,
            backend.depth_test && !backend.stencil,
        );
        raster.depth_code32 = Some(word_for_fraction32);


        Box::new(PipeDraw {
            raster,
            shader: ShaderState::of(&w.shader, tex.regs),
            backend,
            planes,
            fill_steps: if w.raster.offset_fill() { w.raster.offset_steps() as i64 } else { 0 },
            prims: prims::PrimState::of(&w.raster),
            tex,
            textured: tex.regs.target() != texture::Target::None,
            ring: ring.cloned(),
            back_draw: BackendDraw { state: backend, planes },
            shade: false,
            used: std::cell::Cell::new(false),
        })
    }
}

/// The colour targets of an overlay context's draw, whose destination is
/// the window-id buffer: an index goes to buffer B, the overlay byte; the
/// id byte (buffer A) is the X server's alone, and RGBA has no home there.
pub(crate) fn aux_targets(targets: [bool; 2], index: bool) -> [bool; 2] {
    [false, index && targets[1]]
}

