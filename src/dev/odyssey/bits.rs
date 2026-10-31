//! Bit-field helpers for register and command words.

/// Bits `start .. start + len` of `v`, sign-extended.
pub fn sextract32(v: u32, start: u32, len: u32) -> i32 {
    debug_assert!(len > 0 && start + len <= 32);
    ((v << (32 - len - start)) as i32) >> (32 - len)
}

/// `v` with bits `start .. start + len` replaced by `field`.
pub fn deposit32(v: u32, start: u32, len: u32, field: u32) -> u32 {
    debug_assert!(len > 0 && start + len <= 32);
    let mask = (u32::MAX >> (32 - len)) << start;
    (v & !mask) | ((field << start) & mask)
}
