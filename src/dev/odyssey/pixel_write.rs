//! The library's image-write codes: word 7's format (bits 4:0) and type
//! (bits 8:5), and what bytes a pixel of each takes and means.
//!
//! Datasheet: "Pixel transfer headers from the OpenGL library", "Image-write
//! table". ESTABLISHED there as what the library sends. A packed type is
//! always sent under the format whose component order it has in memory, so
//! the format lists the components most significant first; the `_REV`
//! variants are other formats, not a bit.
//!
//! What the board does with the bytes is INFERRED from the names: single
//! channels widen to eight bits by the usual GL scaling, a luminance or
//! intensity source spreads over red, green and blue, and a component the
//! source lacks is 0, or full scale for alpha.

/// A source component.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Comp {
    Red,
    Green,
    Blue,
    Alpha,
    /// Luminance: red, green and blue together.
    Lum,
    /// Intensity: red, green, blue and alpha together.
    Intensity,
    Index,
}

/// One field of a packed word: component, width in bits, shift.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub comp: Comp,
    pub width: u8,
    pub shift: u8,
}

#[derive(Clone, Debug)]
pub enum WriteKind {
    /// Whole components in order, `comp_bytes` bytes each, big-endian.
    Comps { comps: [Comp; 4], n: u8, comp_bytes: u8 },
    /// One big-endian word of fields.
    Packed { fields: [Field; 4], n: u8 },
}

/// What a write code names: the bytes a pixel takes and how to read them.
#[derive(Clone, Debug)]
pub struct WriteForm {
    pub bytes: u32,
    pub kind: WriteKind,
}

const fn f(comp: Comp, width: u8, shift: u8) -> Field {
    Field { comp, width, shift }
}

const NONE: Field = f(Comp::Red, 0, 0);

/// The form of write code `code` (word 7 & 0x1ff), or None for a code this
/// model does not take: depth and stencil (format 0, which goes to its own
/// planes), the 4:2:2 formats 0xa and 0xe, and anything not in the table.
pub fn write_form(code: u32) -> Option<WriteForm> {
    use Comp::*;
    let format = code & 0x1f;
    let ty = (code >> 5) & 0xf;
    let comps: &[Comp] = match format {
        1 => &[Index],
        2 => &[Lum],
        3 => &[Intensity],
        4 => &[Red],
        5 => &[Green],
        6 => &[Blue],
        7 => &[Alpha],
        8 => &[Lum, Alpha],
        9 => &[Red, Green, Blue],
        0xb => &[Red, Green, Blue, Alpha],
        0xc => &[Alpha, Blue, Green, Red],
        0xd => &[Blue, Green, Red],
        0x11 => &[Blue, Green, Red, Alpha],
        _ => &[],
    };
    // Whole components: type 2 bytes, 3 shorts, 4 words; 0xb and 0xc are
    // bytes written into a 12-bit or an 8-bit RGBA drawable.
    let comp_bytes = match ty {
        2 | 0xb | 0xc => 1,
        3 => 2,
        4 => 4,
        _ => 0,
    };
    if comp_bytes != 0 && !comps.is_empty() {
        let mut c = [Red; 4];
        c[..comps.len()].copy_from_slice(comps);
        let n = comps.len() as u8;
        return Some(WriteForm { bytes: n as u32 * comp_bytes as u32, kind: WriteKind::Comps { comps: c, n, comp_bytes } });
    }
    // Packed types. The format orders the fields from the top bit.
    let order: &[Comp] = match format {
        9 | 0xd => match format {
            9 => &[Red, Green, Blue],
            _ => &[Blue, Green, Red],
        },
        0xb => &[Red, Green, Blue, Alpha],
        0xc => &[Alpha, Blue, Green, Red],
        0x11 => &[Blue, Green, Red, Alpha],
        0x10 => &[Alpha, Red, Green, Blue],
        _ => return None,
    };
    let widths: &[u8] = match (ty, order.len()) {
        (5, 3) => {
            if order[0] == Blue { &[2, 3, 3] } else { &[3, 3, 2] }
        }
        (0xa, 3) => &[5, 6, 5],
        (6, 4) => &[4, 4, 4, 4],
        (7, 4) => {
            if order[0] == Alpha { &[1, 5, 5, 5] } else { &[5, 5, 5, 1] }
        }
        (8, 4) => {
            if order[0] == Alpha { &[2, 10, 10, 10] } else { &[10, 10, 10, 2] }
        }
        _ => return None,
    };
    let total: u8 = widths.iter().sum();
    let mut fields = [NONE; 4];
    let mut shift = total;

    for (i, (&comp, &width)) in order.iter().zip(widths).enumerate() {
        shift -= width;
        fields[i] = f(comp, width, shift);
    }
    Some(WriteForm { bytes: total.div_ceil(8) as u32, kind: WriteKind::Packed { fields, n: order.len() as u8 } })
}

/// A component value `v` of `bits` bits as an 8-bit channel, by the GL
/// scaling and rounding.
fn scale8(v: u64, bits: u32) -> u32 {
    let max = (1u64 << bits) - 1;

    ((v * 255 + max / 2) / max) as u32
}

impl WriteForm {
    /// Whether a pixel of this form carries an alpha (or intensity) value the
    /// board stores, rather than the model's 255.
    pub fn has_alpha(&self) -> bool {
        let any = |c: &[Comp]| c.iter().any(|&c| c == Comp::Alpha || c == Comp::Intensity);

        match &self.kind {
            WriteKind::Comps { comps, n, .. } => any(&comps[..*n as usize]),
            WriteKind::Packed { fields, n } => fields[..*n as usize].iter().any(|f| f.comp == Comp::Alpha),
        }
    }

    pub fn is_index(&self) -> bool {
        matches!(&self.kind, WriteKind::Comps { comps, .. } if comps[0] == Comp::Index)
    }

    /// The pixel in `b` (at least `bytes` long) as the plane word 0xRRGGBBAA,
    /// or for an index form the raw index.
    pub fn decode(&self, b: &[u8]) -> u32 {
        let (mut r, mut g, mut bl, mut a) = (0u32, 0u32, 0u32, 0xffu32);
        let mut put = |comp: Comp, v: u32| match comp {
            Comp::Red => r = v,
            Comp::Green => g = v,
            Comp::Blue => bl = v,
            Comp::Alpha => a = v,
            Comp::Lum => {
                r = v;
                g = v;
                bl = v;
            }
            Comp::Intensity => {
                r = v;
                g = v;
                bl = v;
                a = v;
            }
            Comp::Index => r = v,
        };

        match &self.kind {
            WriteKind::Comps { comps, n, comp_bytes } => {
                for i in 0..*n as usize {
                    let at = i * *comp_bytes as usize;
                    let v = match comp_bytes {
                        1 => b[at] as u32,
                        2 => scale8(u16::from_be_bytes([b[at], b[at + 1]]) as u64, 16),
                        _ => scale8(u32::from_be_bytes(b[at..at + 4].try_into().unwrap()) as u64, 32),
                    };

                    if comps[i] == Comp::Index {
                        // The raw value: 12 index bits.
                        let raw = match comp_bytes {
                            1 => b[at] as u32,
                            2 => u16::from_be_bytes([b[at], b[at + 1]]) as u32,
                            _ => u32::from_be_bytes(b[at..at + 4].try_into().unwrap()),
                        };
                        return raw & super::ODSY_INDEX_MASK;
                    }
                    put(comps[i], v);
                }
            }
            WriteKind::Packed { fields, n } => {
                let mut word = 0u64;

                for &byte in &b[..self.bytes as usize] {
                    word = word << 8 | byte as u64;
                }
                for fl in &fields[..*n as usize] {
                    let v = (word >> fl.shift) & ((1u64 << fl.width) - 1);

                    put(fl.comp, scale8(v, fl.width as u32));
                }
            }
        }
        (r << 24) | (g << 16) | (bl << 8) | a
    }
}
