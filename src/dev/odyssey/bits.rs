//! Bit-field helpers for register and command words, and byte-order reads.

/// Bits `start .. start + len` of `v`, sign-extended.
pub(super) fn sextract32(v: u32, start: u32, len: u32) -> i32 {
    debug_assert!(len > 0 && start + len <= 32);
    ((v << (32 - len - start)) as i32) >> (32 - len)
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

/// C's max and min: the second operand wins a tie and a NaN on either side,
/// where f64::max and f64::min give the number. Arithmetic ported from C is
/// written against this order.
#[inline(always)]
pub(super) fn cmax<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

#[inline(always)]
pub(super) fn cmin<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}
