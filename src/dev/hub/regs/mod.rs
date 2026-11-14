//! Register offsets in the Hub's window, for every unit. `hub.rs` is
//! generated: do not edit it. Refresh it as
//! clanker/devices/refreshing-the-generated-register-files.md says.

mod hub;
#[cfg(test)]
mod tests;

pub use self::hub::*;
