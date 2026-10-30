//! Bit-field helpers for register and command words, and byte-order reads.

/// Bits `start .. start + len` of `v`, sign-extended.
pub(super) fn sextract32(v: u32, start: u32, len: u32) -> i32 {
    debug_assert!(len > 0 && start + len <= 32);
    ((v << (32 - len - start)) as i32) >> (32 - len)
}

/// `v` with bits `start .. start + len` replaced by `field`.
pub(super) fn deposit32(v: u32, start: u32, len: u32, field: u32) -> u32 {
    debug_assert!(len > 0 && start + len <= 32);
    let mask = (u32::MAX >> (32 - len)) << start;
    (v & !mask) | ((field << start) & mask)
}

/// Big- and little-endian words from the front of `b`, which must hold them.
#[inline]
pub(super) fn be16(b: &[u8]) -> u16 {
    u16::from_be_bytes([b[0], b[1]])
}
#[inline]
pub(super) fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}
#[inline]
pub(super) fn be64(b: &[u8]) -> u64 {
    u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}
#[inline]
pub(super) fn le16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}
#[inline]
pub(super) fn le32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
