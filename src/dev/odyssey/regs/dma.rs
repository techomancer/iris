// generated from odsy 83fbfa1b67bf DMA_DECODE.h - do not edit, regenerate in odsy
//!
//! DMA-engine descriptor words of the Odyssey board: the kick words, the
//! transfer entry and its tag.
//!
//! Grades on each item say what is known about it: `established` is what the
//! shipped driver software does, read in full; `inferred` is a reasoned
//! reading; `unknown` is not derived. None of them is a measurement of the
//! silicon.

#![allow(dead_code)]

/// established.
pub const DONE_FLAG_BIT: u32 = 11;
/// established.
pub const KICK_ADDR_BITS: u32 = 48;
/// established.
pub const ENTRY_BYTES: u32 = 128;
/// established.
pub const ENTRY_HDR_BYTES: u32 = 32;
/// established.
pub const PAGE_ENTRY_BYTES: u32 = 8;
/// established.
pub const ENTRY_PAGES_MAX: u32 = 12;
/// established: the largest chunk a page entry is built from; whether it is a page size or only a software chunk size: inferred.
pub const PAGE_CHUNK_BYTES: u32 = 0x4000;
/// 147456; established as the split rule.
pub const ENTRY_MAX_BYTES: u32 = 0x24000;
/// established.
pub const TAG_CMD_LOW16: u32 = 0xa5a;
/// established.
pub const TAG_CMD_COUNT: u32 = 2;
/// decoded from the word above.
pub const TAG_CMD_ID: u32 = 0x25a;

/// Values of `id` in `KickAHi` that the header names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KickAHiId {
    /// established.
    Plain,
    /// established.
    Variant,
    /// A value the notes do not name.
    Other(u32),
}

impl KickAHiId {
    pub const fn from_bits(v: u32) -> Self {
        match v {
            0xe01 => Self::Plain,
            0xe03 => Self::Variant,
            _ => Self::Other(v as u32),
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::Plain => 0xe01,
            Self::Variant => 0xe03,
            Self::Other(v) => v as u32,
        }
    }
}

/// Values of `read_dir` in `EntryW7` that the header names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryW7ReadDir {
    /// established.
    Write,
    /// established.
    Read,
    /// A value the notes do not name.
    Other(u8),
}

impl EntryW7ReadDir {
    pub const fn from_bits(v: u32) -> Self {
        match v {
            0x0 => Self::Write,
            0x1 => Self::Read,
            _ => Self::Other(v as u8),
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::Write => 0x0,
            Self::Read => 0x1,
            Self::Other(v) => v as u32,
        }
    }
}

/// Values of `transform` in `EntryW7` that the header names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryW7Transform {
    /// established.
    Plain,
    /// established.
    Transform,
    /// A value the notes do not name.
    Other(u8),
}

impl EntryW7Transform {
    pub const fn from_bits(v: u32) -> Self {
        match v {
            0x0 => Self::Plain,
            0x1 => Self::Transform,
            _ => Self::Other(v as u8),
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::Plain => 0x0,
            Self::Transform => 0x1,
            Self::Other(v) => v as u32,
        }
    }
}

/// Values of `marker` in `TagW1` that the header names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TagW1Marker {
    /// established.
    Tag,
    /// A value the notes do not name.
    Other(u8),
}

impl TagW1Marker {
    pub const fn from_bits(v: u32) -> Self {
        match v {
            0x5a => Self::Tag,
            _ => Self::Other(v as u8),
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::Tag => 0x5a,
            Self::Other(v) => v as u32,
        }
    }
}

/// Values of `flag` in `TagW0` that the header names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TagW0Flag {
    /// established.
    Set,
    /// A value the notes do not name.
    Other(u8),
}

impl TagW0Flag {
    pub const fn from_bits(v: u32) -> Self {
        match v {
            0x1 => Self::Set,
            _ => Self::Other(v as u8),
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::Set => 0x1,
            Self::Other(v) => v as u32,
        }
    }
}

/// The `KICK_A_HI` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KickAHi(u32);

impl KickAHi {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 11:0, established.
    pub const fn id(self) -> KickAHiId {
        KickAHiId::from_bits(self.0 & 0xfff)
    }
}

/// The `KICK_A_LO` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KickALo(u32);

impl KickALo {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 16:0, established.
    pub const fn addr_lo(self) -> u32 {
        self.0 & 0x1ffff
    }

    /// Bits 29:23, established.
    pub const fn addr_hi(self) -> u32 {
        (self.0 >> 23) & 0x7f
    }
}

/// The `KICK_B_HI` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KickBHi(u32);

impl KickBHi {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 16:0, established.
    pub const fn addr_lo(self) -> u32 {
        self.0 & 0x1ffff
    }

    /// Bits 29:23, established.
    pub const fn addr_hi(self) -> u32 {
        (self.0 >> 23) & 0x7f
    }
}

/// The `KICK_B_LO` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KickBLo(u32);

impl KickBLo {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 31:0, established.
    pub const fn zero(self) -> u32 {
        self.0
    }
}

/// The `ENTRY_W7` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryW7(u32);

impl EntryW7 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bit 24, established.
    pub const fn read_dir(self) -> EntryW7ReadDir {
        EntryW7ReadDir::from_bits((self.0 >> 24) & 0x1)
    }

    /// Bit 25, established.
    pub const fn fixed25(self) -> bool {
        (self.0 >> 25) & 1 != 0
    }

    /// Bit 27, established.
    pub const fn transform(self) -> EntryW7Transform {
        EntryW7Transform::from_bits((self.0 >> 27) & 0x1)
    }

    /// Bit 28, established.
    pub const fn fixed28(self) -> bool {
        (self.0 >> 28) & 1 != 0
    }

    /// Bit 29, established.
    pub const fn fixed29(self) -> bool {
        (self.0 >> 29) & 1 != 0
    }

    /// Bit 30, established.
    pub const fn differs(self) -> bool {
        (self.0 >> 30) & 1 != 0
    }

    /// Bit 31, inferred.
    pub const fn irq_after(self) -> bool {
        (self.0 >> 31) & 1 != 0
    }
}

/// The `PAGE_W0` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageW0(u32);

impl PageW0 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 31:0, established.
    pub const fn addr(self) -> u32 {
        self.0
    }
}

/// The `PAGE_W1` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageW1(u32);

impl PageW1 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 31:16, established.
    pub const fn addr_lo(self) -> u32 {
        self.0 >> 16
    }

    /// Bits 15:0, inferred.
    pub const fn count(self) -> u32 {
        self.0 & 0xffff
    }
}

/// The `TAG_W0` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TagW0(u32);

impl TagW0 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 14:0, inferred.
    pub const fn slot(self) -> u32 {
        self.0 & 0x7fff
    }

    /// Bit 15, established.
    pub const fn flag(self) -> TagW0Flag {
        TagW0Flag::from_bits((self.0 >> 15) & 0x1)
    }
}

/// The `TAG_W1` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TagW1(u32);

impl TagW1 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 7:0, established.
    pub const fn tag(self) -> u32 {
        self.0 & 0xff
    }

    /// Bits 15:8, established.
    pub const fn marker(self) -> TagW1Marker {
        TagW1Marker::from_bits((self.0 >> 8) & 0xff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn testvec_kick_a_hi_e01() {
        let w = KickAHi::new(0xe01);
        assert_eq!(w.id().bits(), 3585, "id");
    }

    #[test]
    fn testvec_kick_a_hi_e03() {
        let w = KickAHi::new(0xe03);
        assert_eq!(w.id().bits(), 3587, "id");
    }

    #[test]
    fn testvec_tag_w1_5a00() {
        let w = TagW1::new(0x5a00);
        assert_eq!(w.marker().bits(), 90, "marker");
        assert_eq!(w.tag(), 0, "tag");
    }
}
