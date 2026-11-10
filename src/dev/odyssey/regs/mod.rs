//! Register and command-word layouts, as named fields.
//!
//! `buzz` has the command opcodes and the image, copy, pixel-mode, texture
//! and sampler words of the command stream. `dma` has the DMA kick words and
//! the descriptor's entry, page and tag words. Each field carries a grade
//! (established, inferred or unknown) in its doc comment: a decode here is
//! only as certain as that grade says.
//!
//! Both files are generated and are vendored here byte for byte, banner
//! line included. Do not edit them. When their source changes, replace the
//! files with the regenerated ones and fix up the callers, so a layout
//! change shows up as a compile error and not as a silent difference.

pub mod buzz;
pub mod dma;
