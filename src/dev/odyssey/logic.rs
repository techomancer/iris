//! The sixteen logical operations a pixel write can combine through.

// The sixteen logic ops, in OpenGL's order (GL_CLEAR + op).
pub(super) const LOGIC_CLEAR: u32 = 0;
pub(super) const LOGIC_AND: u32 = 1;
pub(super) const LOGIC_AND_REVERSE: u32 = 2;
pub(super) const LOGIC_COPY: u32 = 3;
pub(super) const LOGIC_AND_INVERTED: u32 = 4;
pub(super) const LOGIC_NOOP: u32 = 5;
pub(super) const LOGIC_XOR: u32 = 6;
pub(super) const LOGIC_OR: u32 = 7;
pub(super) const LOGIC_NOR: u32 = 8;
pub(super) const LOGIC_EQUIV: u32 = 9;
pub(super) const LOGIC_INVERT: u32 = 10;
pub(super) const LOGIC_OR_REVERSE: u32 = 11;
pub(super) const LOGIC_COPY_INVERTED: u32 = 12;
pub(super) const LOGIC_OR_INVERTED: u32 = 13;
pub(super) const LOGIC_NAND: u32 = 14;

pub(super) fn logic_op(op: u8, src: u32, dst: u32) -> u32 {
    let s = src;
    let d = dst;
    match (op & 0xf) as u32 {
        LOGIC_CLEAR => 0,
        LOGIC_AND => s & d,
        LOGIC_AND_REVERSE => s & !d,
        LOGIC_COPY => s,
        LOGIC_AND_INVERTED => !s & d,
        LOGIC_NOOP => d,
        LOGIC_XOR => s ^ d,
        LOGIC_OR => s | d,
        LOGIC_NOR => !(s | d),
        LOGIC_EQUIV => !(s ^ d),
        LOGIC_INVERT => !d,
        LOGIC_OR_REVERSE => s | !d,
        LOGIC_COPY_INVERTED => !s,
        LOGIC_OR_INVERTED => !s | d,
        LOGIC_NAND => !(s & d),
        _ => !0u32, // LOGIC_SET
    }
}
