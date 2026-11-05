//! The window clip: up to sixteen rectangles a drawing keeps.

/// Up to 16 inclusive rectangles, in plane coordinates. With n 0 the list
/// is no clip, unless empty is set: then it keeps no pixel.
pub(super) const CLIP_MAX: usize = 16;

#[derive(Clone, Copy, Default, Debug)]
pub(super) struct ClipRect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

#[derive(Clone, Copy, Default, Debug)]
pub(super) struct ClipList {
    pub n: u32,
    /// With n 0: true keeps nothing, false keeps everything.
    pub empty: bool,
    pub r: [ClipRect; CLIP_MAX],
}

