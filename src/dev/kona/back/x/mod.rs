//! The X server's integer path, drawing inline.
//!
//! `state` and `gc`: the graphics context and the drawing state loaded around
//! it; `fill`: fills and the stipple they use (`stipple` is its cache);
//! `line`: segments and polylines; `pixel`: pixel writes, reads and copies.

pub mod fill;
pub mod gc;
pub mod line;
pub mod pixel;
pub mod state;
pub mod stipple;
