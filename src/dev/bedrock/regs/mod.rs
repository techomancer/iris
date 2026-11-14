//! Register offsets, one module per Bedrock block, from the block's `BASE`
//! in the node's register window. `bedrock.rs` is generated: do not edit
//! it. Refresh it as
//! clanker/devices/refreshing-the-generated-register-files.md says.

#[path = "bedrock.rs"]
mod defs;
#[cfg(test)]
mod tests;

pub use self::defs::{extra, ii as io, lb, md, ni, pi, pi::PI1_BASE, windows, xb};
pub use crate::dev::hub::regs::IIO_PROTECT_OVRRD_KEY;
pub use crate::dev::hub::WINDOW_SIZE;
