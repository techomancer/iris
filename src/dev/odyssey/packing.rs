//! The texture packing check (feature `vprodiag`): how each upload into
//! board memory packed its texels, checked against every texture level
//! descriptor later named over that memory. A texture packed one way and
//! sampled another round-trips through glGetTexImage unnoticed, so the
//! guest's own tests cannot see it; `vpro status` reports what this finds.

use super::board::Board;
use super::view::Kind;
use super::vram::{Format, ADDR_MASK, PITCH_MASK};
use crate::devlog::LogModule;

/// How many uploads the packing check remembers.
const UPLOADS_KEPT: usize = 64;
/// How many disagreements it keeps as text.
const MISMATCHES_KEPT: usize = 16;

/// How an upload into board memory laid its texels out: the surface's
/// base, row of tiles and pixel size, the field's packing, and what the
/// guest sent (word 4 and the write code, or the path).
#[derive(Clone, Copy, Debug)]
pub(super) struct UploadPacking {
    pub base: u32,
    pub pitch: u32,
    pub log2: u8,
    pub codec: super::field::Codec,
    pub word4: u32,
    pub code: u32,
    pub how: &'static str,
    pub checked: bool,
}

/// Whether two codecs store a word the same way: a field of eight bytes or
/// more holds four 12-bit lanes whatever its FIELD_FMT (field::texel_pack).
fn same_packing(a: super::field::Codec, b: super::field::Codec) -> bool {
    use super::field::Codec;
    let lanes = |c| matches!(c, Codec::Lanes | Codec::Texel { bytes: 8.., .. });
    a == b || (lanes(a) && lanes(b))
}

/// The packing a view's kind stores, None for a depth view.
pub(super) fn kind_codec(k: Kind) -> Option<super::field::Codec> {
    use super::field::{Codec, FIELD_FOUR};
    match k {
        Kind::Rgba8 => Some(Codec::Texel { bytes: 4, field: FIELD_FOUR }),
        Kind::Byte => Some(Codec::Byte),
        Kind::Coded(c) => Some(c),
        _ => None,
    }
}

impl Board {
    /// Remember how an upload into board memory stored its texels.
    pub fn note_upload(&mut self, p: UploadPacking) {
        let u = &mut self.diag.packing.uploads;
        u.retain(|q| q.base != p.base);
        if u.len() >= UPLOADS_KEPT {
            u.remove(0);
        }
        u.push(p);
    }

    /// Note an upload through the pixel mode's selected surface (a FIFO
    /// image into a scratch surface, a board-built mip level).
    pub fn note_selected_upload(&mut self, how: &'static str, code: u32) {
        let f = Format::decode(self.pixel_mode[2]);
        if let Some(codec) = kind_codec(self.dest_views().colour[0].kind) {
            self.note_upload(UploadPacking {
                base: self.pixel_mode[0] & ADDR_MASK,
                pitch: self.pixel_mode[1] & PITCH_MASK,
                log2: f.fb_log2,
                codec,
                word4: self.pixel_mode[2],
                code,
                how,
                checked: false,
            });
        }
    }

    /// A level descriptor (words 0 and 1) named over memory an upload
    /// wrote: the level's storage (address, row and format index) should
    /// be the one the upload stored in. A disagreement means the texels
    /// were packed one way and are sampled another; it is counted, kept
    /// for `vpro status` and logged. Each upload is checked once.
    pub fn check_level_packing(&mut self, word0: u32, word1: u32) {
        let base = word0 & ADDR_MASK;
        if base == 0 {
            return;
        }
        let Some(i) = self.diag.packing.uploads.iter().position(|q| q.base == base && !q.checked) else {
            return;
        };
        let p = self.diag.packing.uploads[i];
        self.diag.packing.uploads[i].checked = true;
        let Some((s, f)) = super::field::level_surface(word0, word1) else {
            return;
        };
        self.diag.packing.checked += 1;
        if s.bpp_log2 == p.log2 && same_packing(f.codec, p.codec) && s.pitch == p.pitch {
            return;
        }
        self.diag.packing.mismatch_count += 1;
        // One line for each kind of disagreement: the level's format word
        // and the upload's path, word 4 and code.
        let kind = format!(
            "level format {:#010x} ({} bytes, {:?}) over a {} upload (word 4 {:#010x}, code {:#x}) stored as {} bytes, {:?}",
            word1 & !0xfff, 1u32 << s.bpp_log2, f.codec, p.how, p.word4, p.code, 1u32 << p.log2, p.codec
        );
        let line = format!("{kind}; level {:#x} pitch {}, upload pitch {}", word0, s.pitch, p.pitch);
        crate::dlog!(LogModule::Odyssey, "sgi-odyssey: texture packing disagrees: {}", line);
        if self.diag.packing.mismatches.len() < MISMATCHES_KEPT && !self.diag.packing.mismatches.iter().any(|m| m.starts_with(&kind)) {
            self.diag.packing.mismatches.push(line);
        }
    }
}

/// The check's state, in the board's reports.
#[derive(Default)]
pub(super) struct PackingCheck {
    /// The packing of the latest uploads into board memory, by base.
    pub uploads: Vec<UploadPacking>,
    /// Level descriptors checked against an upload, and those whose
    /// storage disagrees with it (the first few kept as text).
    pub checked: u32,
    pub mismatch_count: u32,
    pub mismatches: Vec<String>,
}
