//! The back end: a fragment's stencil and depth tests and their
//! operations, the destination read, the blend or logical operation, the
//! colour mask, and the stores, from `BackendWords` into the planes through
//! packers that build each plane's pixel once and store it with one write.
//!
//! The colour arithmetic is integer at 12 bits a component, as the
//! hardware blends (porting guide): a fragment arrives as 12-bit values
//! (floor(4096 c) limited to 4095), an 8-bit destination widens to 12 bits
//! by repeating its top bits, the blend takes products of 12-bit values
//! rounded to 12 bits, and an 8-bit plane stores the top eight. An
//! unblended fragment so stores floor(256 c), as the frames establish; how
//! the board rounds a blend is not shown. The logical operation works on
//! the stored width.
//!
//! Everything branches on the words at run time. Specialisation comes
//! later.

use super::state::{BackendWords, BlendEquation, BlendFactor, Compare, StencilOp};
use super::super::consts::{write::INDEX_MASK, ODSY_DEPTH24_MAX, ODSY_DRAW_MASK_BUFFER0, ODSY_DRAW_MASK_BUFFER1};
use super::super::logic::logic_op;
use super::super::view::{DestViews, Rows};
use super::super::vram::VramPtr;

/// A clear's work at the back end: what it stores, under which masks, as
/// its span slots name it (`ring::KIND_SPAN`). The colour is the stored
/// word (an RGBA colour's bytes, or an index); `colour_mask` names the
/// word's bits that change.
#[repr(C, align(64))]
pub(crate) struct ClearDraw {
    pub planes: DestMem,
    pub targets: [bool; 2],
    pub colour: Option<[u8; 4]>,
    pub colour_mask: u32,
    /// The depth word stored.
    pub depth: Option<u32>,
    /// The stencil value stored under the stencil write mask.
    pub stencil: Option<(u8, u8)>,
    /// The id test: the id bytes' value and mask (0: no test).
    pub id_value: u8,
    pub id_mask: u8,
}

// SAFETY: as DestMem.
unsafe impl Send for ClearDraw {}

impl ClearDraw {
    /// Clear pixels x0 to x1 of row y (inside one of the window's clip
    /// rectangles).
    pub fn span(&self, y: i32, x0: i32, x1: i32) {
        let p = &self.planes;
        let full = self.colour_mask == u32::MAX;

        if self.id_mask == 0 && self.stencil.is_none() && (self.colour.is_none() || full) {
            // Nothing per pixel: whole rows.
            if let Some(z) = self.depth {
                p.depth.fill4(y, x0, x1, pack_z24(z));
            }
            if let Some(c) = self.colour {
                for (b, rows) in p.colour.iter().enumerate() {
                    if self.targets[b] {
                        rows.fill4(y, x0, x1, c);
                    }
                }
            }
            return;
        }
        for x in x0..=x1 {
            if self.id_mask != 0 && p.did.load1(x, y).is_none_or(|id| (id ^ self.id_value) & self.id_mask != 0) {
                continue;
            }
            if let Some(z) = self.depth {
                p.depth.store4(x, y, pack_z24(z));
            }
            if let Some((v, m)) = self.stencil {
                if let Some(old) = p.stencil.load1(x, y) {
                    p.stencil.store1(x, y, (old & !m) | (v & m));
                }
            }
            if let Some(c) = self.colour {
                for (b, rows) in p.colour.iter().enumerate() {
                    if !self.targets[b] {
                        continue;
                    }
                    let out = if full {
                        c
                    } else {
                        let Some(d) = rows.load_word(x, y) else {
                            continue;
                        };
                        ((u32::from_be_bytes(c) & self.colour_mask) | (d & !self.colour_mask)).to_be_bytes()
                    };
                    rows.store4(x, y, out);
                }
            }
        }
    }
}

/// The memory a draw writes, field by field, as the board's registers
/// place it (view::DestViews).
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(crate) struct DestMem {
    pub colour: [Rows; 2],
    pub depth: Rows,
    pub stencil: Rows,
    pub did: Rows,
    pub overlay: Rows,
}

impl DestMem {
    pub fn of(mem: VramPtr, d: &DestViews) -> DestMem {
        DestMem {
            colour: [Rows::new(mem, d.colour[0]), Rows::new(mem, d.colour[1])],
            depth: Rows::new(mem, d.depth),
            stencil: Rows::new(mem, d.stencil),
            did: Rows::new(mem, d.did),
            overlay: Rows::new(mem, d.overlay),
        }
    }
}

// Packers: a plane's pixel from its components, in the bytes the board's
// planes hold (big-endian words, red in the high byte, as Board::allocate
// lays them out), and back.

/// An RGBA8 pixel's bytes from its 8-bit components.
#[inline(always)]
pub(crate) fn pack_rgba8(c: [u32; 4]) -> [u8; 4] {
    [c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8]
}

#[inline(always)]
pub(crate) fn unpack_rgba8(b: [u8; 4]) -> [u32; 4] {
    [b[0] as u32, b[1] as u32, b[2] as u32, b[3] as u32]
}

/// A 24-bit depth word's bytes in its 32-bit plane word; bits 31:24 are
/// not depth and are stored as zero.
#[inline(always)]
pub(crate) fn pack_z24(z: u32) -> [u8; 4] {
    let z = z & ODSY_DEPTH24_MAX;

    [0, (z >> 16) as u8, (z >> 8) as u8, z as u8]
}

#[inline(always)]
pub(crate) fn unpack_z24(b: [u8; 4]) -> u32 {
    ((b[1] as u32) << 16) | ((b[2] as u32) << 8) | b[3] as u32
}

/// The colour index a fragment carries: bits 7:0 ride in blue and 11:8
/// in green (gl::gl_index_rgba), here as the 12-bit components the
/// shader hands on.
#[inline(always)]
pub(crate) fn index_of(src: [u32; 4]) -> u32 {
    (((src[1] >> 4) & 0xf) << 8) | (src[2] >> 4)
}

/// floor(256 c) limited to 255, 0 for anything not above 0 (established on
/// the frames for 8-bit stores).
#[inline(always)]
pub(crate) fn quantise8(c: f64) -> u32 {
    if !(c > 0.0) {
        0
    } else if c >= 1.0 {
        255
    } else {
        ((c * 256.0) as u32).min(255)
    }
}

/// floor(4096 c) limited to 4095, 0 for anything not above 0: a
/// component's 12-bit value, whose top eight bits are quantise8(c).
#[inline(always)]
pub(crate) fn quantise12(c: f64) -> u32 {
    if !(c > 0.0) {
        0
    } else if c >= 1.0 {
        4095
    } else {
        ((c * 4096.0) as u32).min(4095)
    }
}

/// The 12-bit top of 12.12 colour (`pipe::fixed`), clamped.
#[inline(always)]
pub(crate) fn colour12(c: i32) -> u32 {
    (super::fixed::clamp(c) >> 12) as u32
}

/// An 8-bit stored value widened to 12 bits by repeating its top bits.
#[inline(always)]
fn widen8(v: u32) -> u32 {
    (v << 4) | (v >> 4)
}

/// A product of 12-bit values (4095 is one), rounded to 12 bits.
#[inline(always)]
fn mul12(a: u32, b: u32) -> u32 {
    let p = a * b;

    (p + (p >> 12) + 2048) >> 12
}

/// A blend factor's 12-bit value for component k, from the source and
/// destination components and the constant colour.
#[inline(always)]
fn factor(f: BlendFactor, k: usize, other_is_dst: bool, s: &[u32; 4], d: &[u32; 4], c: &[u32; 4]) -> u32 {
    let other = if other_is_dst { d } else { s };

    match f {
        BlendFactor::Zero => 0,
        BlendFactor::One => 4095,
        BlendFactor::OtherColour => other[k],
        BlendFactor::SrcAlpha => s[3],
        BlendFactor::DstAlpha => d[3],
        BlendFactor::ConstColour => c[k],
        BlendFactor::ConstAlpha => c[3],
        BlendFactor::OneMinusOtherColour => 4095 - other[k],
        BlendFactor::OneMinusSrcAlpha => 4095 - s[3],
        BlendFactor::OneMinusDstAlpha => 4095 - d[3],
        BlendFactor::OneMinusConstColour => 4095 - c[k],
        BlendFactor::OneMinusConstAlpha => 4095 - c[3],
        BlendFactor::SrcAlphaSaturate => if k == 3 { 4095 } else { s[3].min(4095 - d[3]) },
        BlendFactor::Unknown(_) => 0,
    }
}

/// What the back end decides once per draw from its words.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub(crate) struct BackendState {
    pub depth_test: bool,
    pub depth_func: Compare,
    pub depth_write: bool,
    /// The stencil test, where the drawable has stencil planes.
    pub stencil: bool,
    pub stencil_func: Compare,
    pub stencil_ref: u32,
    pub stencil_value_mask: u32,
    pub stencil_ops: [StencilOp; 3],
    pub stencil_write_mask: u32,
    pub blend: bool,
    pub blend_src: BlendFactor,
    pub blend_dst: BlendFactor,
    pub blend_equation: BlendEquation,
    /// The constant colour's 12-bit components.
    pub const_colour: [u32; 4],
    pub logic: Option<u8>,
    /// The colour mask's four bits, red to alpha; in index mode 0xf when
    /// the index mask lets any bit change.
    pub colour_mask: u32,
    /// The index bits a colour-index store changes.
    pub index_mask: u32,
    /// Alpha is stored (RGBA mode); index mode stores an index instead.
    pub rgba: bool,
    /// Which colour buffers a store reaches.
    pub targets: [bool; 2],
    /// Whether a store reads the destination first.
    pub reads: bool,
}

impl BackendState {
    /// The state of a draw under words w, for a drawable with stencil
    /// planes or not.
    pub fn of(w: &BackendWords, has_stencil: bool, rgba: bool) -> BackendState {
        let stencil = has_stencil && w.stencil_test_on();
        let smax = 0xff;
        let stencil_ref = w.stencil_ref().min(smax);
        let const_colour = w.const_colour().map(|c| (c as u32).min(4095));
        // The blend equation's logic-op code is the logical operation too
        // (the library sends GL_LOGIC_OP as the equation).
        let logic = (rgba && w.colour_logic_op_on()
            || !rgba && w.index_logic_op_on()
            || w.blend_on() && w.blend_equation() == BlendEquation::LogicOp)
            .then(|| w.logic_op() as u8);
        let blend = rgba && w.blend_on() && logic.is_none();
        let colour_mask = if rgba { w.colour_mask_bits() } else { if w.index_mask() != 0 { 0xf } else { 0 } };
        let targets = [w.draw_mask & ODSY_DRAW_MASK_BUFFER0 != 0, w.draw_mask & ODSY_DRAW_MASK_BUFFER1 != 0];

        BackendState {
            depth_test: w.depth_test_on(),
            depth_func: w.depth_func(),
            depth_write: w.depth_test_on() && w.depth_write(),
            stencil,
            stencil_func: w.stencil_func(),
            stencil_ref,
            stencil_value_mask: w.stencil_value_mask(),
            stencil_ops: w.stencil_ops(),
            stencil_write_mask: w.stencil_write_mask(),
            blend,
            blend_src: w.blend_src(),
            blend_dst: w.blend_dst(),
            blend_equation: w.blend_equation(),
            const_colour,
            logic,
            colour_mask,
            index_mask: w.index_mask(),
            rgba,
            targets,
            reads: if rgba { blend || logic.is_some() || colour_mask != 0xf } else { logic.is_some() || w.index_mask() != INDEX_MASK },
        }
    }

    /// The stencil operation op on the value old, under the write mask.
    #[inline(always)]
    fn stencil_value(&self, op: StencilOp, old: u32) -> Option<u32> {
        let max = 0xffu32;
        let value = match op {
            StencilOp::Keep => return None,
            StencilOp::Zero => 0,
            StencilOp::Replace => self.stencil_ref,
            StencilOp::Incr => if old < max { old + 1 } else { max },
            StencilOp::Decr => old.saturating_sub(1),
            StencilOp::Invert => !old & max,
            StencilOp::Unknown(_) => return None,
        };
        let mask = self.stencil_write_mask & max;

        Some((old & !mask) | (value & mask))
    }

    /// The blend of the fragment's 12-bit components over the
    /// destination's, in 12 bits.
    #[inline(always)]
    fn blend(&self, s: [u32; 4], d: [u32; 4]) -> [u32; 4] {
        let mut out = [0u32; 4];

        for k in 0..4 {
            let sf = factor(self.blend_src, k, true, &s, &d, &self.const_colour);
            let df = factor(self.blend_dst, k, false, &s, &d, &self.const_colour);
            let (a, b) = (mul12(s[k], sf) as i32, mul12(d[k], df) as i32);
            let v = match self.blend_equation {
                BlendEquation::Min => s[k].min(d[k]) as i32,
                BlendEquation::Max => s[k].max(d[k]) as i32,
                BlendEquation::Subtract => a - b,
                BlendEquation::ReverseSubtract => b - a,
                _ => a + b,
            };

            out[k] = v.clamp(0, 4095) as u32;
        }
        out
    }

    /// The 8-bit colour a fragment of 12-bit components stores over the
    /// 8-bit dest, after the blend or logical operation and the colour
    /// mask.
    #[inline(always)]
    pub fn colour_result(&self, src: [u32; 4], dst: [u32; 4]) -> [u32; 4] {
        let mut out = if self.blend { self.blend(src, dst.map(widen8)) } else { src }.map(|c| c >> 4);

        if let Some(op) = self.logic {
            for k in 0..4 {
                out[k] = logic_op(op, out[k], dst[k]) & 0xff;
            }
        }
        for k in 0..4 {
            if self.colour_mask & (1 << k) == 0 {
                out[k] = dst[k];
            }
        }
        out
    }

#[cfg(test)]
    /// One fragment at plane pixel (x, y) with depth word zq (the polygon
    /// offset applied by the rasteriser) and colour rgba, through the tests
    /// and into memory. The rasteriser has already applied the window's clip list,
    /// the scissor and the id test.
    #[inline]
    pub fn fragment(&self, planes: &DestMem, x: i32, y: i32, zq: u32, rgba: [f64; 4]) {
        self.fragment_q(planes, x, y, zq, rgba.map(quantise12));
    }

    /// `fragment` with the colour already quantised to 12 bits a component.
    #[inline]
    pub fn fragment_q(&self, planes: &DestMem, x: i32, y: i32, zq: u32, src: [u32; 4]) {
        // Which of the stencil operations the fragment takes: 0 where the
        // stencil test fails, 1 the depth test, 2 where it passes.
        let mut outcome = 2usize;
        let mut old_stencil = 0u32;

        if self.stencil {
            old_stencil = planes.stencil.load1(x, y).unwrap_or(0) as u32;
            if !self.stencil_func.test(self.stencil_ref & self.stencil_value_mask, old_stencil & self.stencil_value_mask) {
                outcome = 0;
            }
        }
        if outcome == 2 && self.depth_test {
            let stored = planes.depth.load4(x, y).map_or(0, unpack_z24);

            if !self.depth_func.test(zq, stored) {
                outcome = 1;
            }
        }
        if outcome != 2 {
            if self.stencil {
                if let Some(v) = self.stencil_value(self.stencil_ops[outcome], old_stencil) {
                    planes.stencil.store1(x, y, v as u8);
                }
            }
            return;
        }
        if self.depth_write {
            planes.depth.store4(x, y, pack_z24(zq));
        }
        if self.stencil {
            if let Some(v) = self.stencil_value(self.stencil_ops[2], old_stencil) {
                planes.stencil.store1(x, y, v as u8);
            }
        }
        if self.colour_mask == 0 {
            return;
        }
        if !self.rgba {
            self.store_index(planes, x, y, index_of(src));
            return;
        }

        for (b, rows) in planes.colour.iter().enumerate() {
            if !self.targets[b] {
                continue;
            }
            if !self.reads {
                rows.store4(x, y, pack_rgba8(src.map(|c| c >> 4)));
                continue;
            }
            let Some(dst) = rows.load4(x, y).map(unpack_rgba8) else {
                continue;
            };

            rows.store4(x, y, pack_rgba8(self.colour_result(src, dst)));
        }
    }

    /// A colour index into the targets: through the index logical
    /// operation and under the index mask, in the field's own bits.
    #[inline]
    fn store_index(&self, planes: &DestMem, x: i32, y: i32, index: u32) {
        for (b, rows) in planes.colour.iter().enumerate() {
            if !self.targets[b] {
                continue;
            }
            if !self.reads {
                rows.store_word(x, y, index);
                continue;
            }
            let Some(old) = rows.load_word(x, y) else {
                continue;
            };
            let v = self.logic.map_or(index, |op| logic_op(op, index, old));

            rows.store_word(x, y, (v & self.index_mask) | (old & !self.index_mask));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::state::BackendWords;
    use super::super::super::consts::{mode, write};
    use super::super::super::view::DestViews;
    use super::super::super::vram::Vram;

    /// A small board memory the tests own: a 16 x 8 destination laid out
    /// as Board::test_screen()'s window (double-buffered 8:8:8:8 with depth
    /// in 16-byte pixels), depth far, and a window-id buffer after it.
    struct TestPlanes {
        mem: Vram,
        d: DestViews,
    }

    const W: u32 = 16;
    const H: u32 = 8;

    impl TestPlanes {
        fn new() -> TestPlanes {
            let mut t = TestPlanes { mem: Vram::new(1 << 16), d: DestViews::of(&[0x100, 1, 0x0040_6d50, 0], 0x800, 1) };
            let p = t.planes();
            for y in 0..H as i32 {
                for x in 0..W as i32 {
                    p.depth.store4(x, y, pack_z24(ODSY_DEPTH24_MAX));
                }
            }
            t
        }

        fn planes(&mut self) -> DestMem {
            DestMem::of(self.mem.ptr(), &self.d)
        }

        fn colour(&mut self, b: usize, x: u32, y: u32) -> [u8; 4] {
            self.planes().colour[b].load4(x as i32, y as i32).unwrap()
        }

        fn depth(&mut self, x: u32, y: u32) -> u32 {
            unpack_z24(self.planes().depth.load4(x as i32, y as i32).unwrap())
        }
    }

    fn words() -> BackendWords {
        BackendWords { write_mask: write::DEPTH | (0xf << write::COLOUR_SHIFT) | (0xff << write::STENCIL_SHIFT), draw_mask: 1, ..Default::default() }
    }

    #[test]
    fn packers_round_trip_and_lay_red_in_the_high_byte() {
        assert_eq!(pack_rgba8([0x12, 0x34, 0x56, 0x78]), [0x12, 0x34, 0x56, 0x78]);
        assert_eq!(unpack_rgba8(pack_rgba8([1, 2, 3, 4])), [1, 2, 3, 4]);
        assert_eq!(pack_z24(0xff_abcdef), [0, 0xab, 0xcd, 0xef]);
        assert_eq!(unpack_z24(pack_z24(0x123456)), 0x123456);
        assert_eq!(index_of([0, 0xa0, 0xbc0, 0]), 0xabc);
        assert_eq!(quantise8(0.5), 128);
        assert_eq!(quantise8(1.0), 255);
        assert_eq!(quantise8(0.9999), 255);
        assert_eq!(quantise8(-0.1), 0);
        assert_eq!(quantise8(f64::NAN), 0);
    }

    #[test]
    fn a_plain_store_writes_colour_and_depth_once() {
        let mut t = TestPlanes::new();
        let w = BackendWords { mode: mode::DEPTH_TEST, depth_func: 1, ..words() };
        let s = BackendState::of(&w, false, true);
        let p = t.planes();

        assert!(!s.reads && s.depth_test && s.depth_write);
        s.fragment(&p, 3, 2, 0x100000, [1.0, 0.5, 0.25, 1.0]);
        assert_eq!(t.colour(0, 3, 2), [255, 128, 64, 255]);
        assert_eq!(t.depth(3, 2), 0x100000);
        assert_eq!(t.colour(1, 3, 2), [0; 4], "buffer 1 is not a target");
        // A farther fragment fails LESS; a nearer one passes.
        let p = t.planes();
        s.fragment(&p, 3, 2, 0x100001, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(t.colour(0, 3, 2), [255, 128, 64, 255]);
        let p = t.planes();
        s.fragment(&p, 3, 2, 0x0fffff, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(t.colour(0, 3, 2), [0, 0, 0, 255]);
        assert_eq!(t.depth(3, 2), 0x0fffff);
    }

    #[test]
    fn blending_over_the_destination_matches_the_arithmetic() {
        let mut t = TestPlanes::new();
        // SRC_ALPHA, ONE_MINUS_SRC_ALPHA, ADD.
        let w = BackendWords { mode: mode::BLEND, blend: [(0x18 << 6) | 0x39, 0, 0], ..words() };
        let s = BackendState::of(&w, false, true);
        let p = t.planes();

        assert!(s.blend && s.reads);
        s.fragment(&p, 1, 1, 0, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(t.colour(0, 1, 1), [255, 255, 255, 255]);
        let p = t.planes();
        s.fragment(&p, 1, 1, 0, [0.0, 0.0, 0.0, 0.5]);
        // In 12 bits: half (2048) of black over 2047/4095 of white (4095):
        // 2047, stored 127; alpha 2048 * 2048 + 4095 * 2047 rounded to
        // 12 bits is 1024 + 2047 = 3071, whose top eight bits are 191.
        assert_eq!(t.colour(0, 1, 1), [127, 127, 127, 191]);
        // MIN keeps the smaller of the two.
        let w = BackendWords { mode: mode::BLEND, blend: [(1 << 13) | (0x08 << 6) | 0x08, 0, 0], ..words() };
        let s = BackendState::of(&w, false, true);
        let p = t.planes();
        s.fragment(&p, 1, 1, 0, [1.0, 0.0, 0.5, 1.0]);
        assert_eq!(t.colour(0, 1, 1), [127, 0, 127, 191]);
    }

    #[test]
    fn the_logical_operation_and_the_colour_mask_read_the_destination() {
        let mut t = TestPlanes::new();
        let w = BackendWords { mode: mode::COLOR_LOGIC_OP, logic_op: 6, ..words() }; // XOR
        let s = BackendState::of(&w, false, true);
        let p = t.planes();

        s.fragment(&p, 2, 2, 0, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(t.colour(0, 2, 2), [255, 255, 255, 255]);
        let p = t.planes();
        s.fragment(&p, 2, 2, 0, [0.5, 0.0, 1.0, 0.0]);
        assert_eq!(t.colour(0, 2, 2), [127, 255, 0, 255]);
        // Only green may change.
        let w = BackendWords { write_mask: write::DEPTH | (0x2 << write::COLOUR_SHIFT), ..words() };
        let s = BackendState::of(&w, false, true);
        let p = t.planes();
        s.fragment(&p, 2, 2, 0, [0.0, 0.5, 0.0, 0.0]);
        assert_eq!(t.colour(0, 2, 2), [127, 128, 0, 255]);
        // Nothing may change: no read, no write.
        let w = BackendWords { write_mask: write::DEPTH, ..words() };
        let s = BackendState::of(&w, false, true);
        let p = t.planes();
        s.fragment(&p, 2, 2, 0, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(t.colour(0, 2, 2), [127, 128, 0, 255]);
    }

    #[test]
    fn stencil_operations_follow_the_outcome_under_the_write_mask() {
        let mut t = TestPlanes::new();
        // GEQUAL ref 2 mask 0xff; ops: fail ZERO, zfail INCR, zpass REPLACE; depth LESS.
        let stencil = (6 << 25) | (1 << 22) | (3 << 19) | (2 << 16) | (0xff << 8) | 2;
        let w = BackendWords { mode: mode::DEPTH_TEST | mode::STENCIL_TEST, depth_func: 1, stencil, ..words() };
        let s = BackendState::of(&w, true, true);
        let p = t.planes();
        p.stencil.store1(0, 0, 3); // (0, 0): 2 >= 3 fails -> ZERO
        p.stencil.store1(1, 0, 1); // (1, 0): passes; depth fails -> INCR to 2
        p.stencil.store1(2, 0, 1); // (2, 0): passes; depth passes -> REPLACE with 2
        for i in 0..3 {
            p.depth.store4(i, 0, pack_z24(0x10));
        }
        s.fragment(&p, 0, 0, 0x20, [1.0; 4]);
        s.fragment(&p, 1, 0, 0x20, [1.0; 4]);
        s.fragment(&p, 2, 0, 0x08, [1.0; 4]);
        assert_eq!([0, 1, 2].map(|x| p.stencil.load1(x, 0).unwrap()), [0, 2, 2]);
        assert_eq!(t.colour(0, 0, 0), [0; 4]);
        assert_eq!(t.colour(0, 1, 0), [0; 4]);
        assert_eq!(t.colour(0, 2, 0), [255; 4]);
        assert_eq!(t.depth(2, 0), 0x08);
        // The write mask keeps bit 1 of the old value.
        let w = BackendWords { write_mask: write::DEPTH | (0xf << write::COLOUR_SHIFT) | (0xfd << write::STENCIL_SHIFT), ..w };
        let s = BackendState::of(&w, true, true);
        let p = t.planes();
        p.stencil.store1(3, 0, 0xff); // passes 2 >= 0xff? no: fails -> ZERO under mask 0xfd keeps bit 1
        s.fragment(&p, 3, 0, 0x20, [1.0; 4]);
        assert_eq!(p.stencil.load1(3, 0), Some(0x02));
        // No stencil planes: the test is off whatever the word says.
        let s = BackendState::of(&w, false, true);
        assert!(!s.stencil);
    }

    #[test]
    fn index_mode_stores_no_alpha_and_uses_the_index_logic_op() {
        let w = BackendWords { mode: mode::INDEX_LOGIC_OP | mode::BLEND, logic_op: 3, write_mask: write::DEPTH | 0xfff, ..words() };
        let s = BackendState::of(&w, false, false);

        assert!(!s.rgba && !s.blend);
        assert_eq!(s.logic, Some(3));
        assert_eq!(s.colour_mask, 0xf);
        let w = BackendWords { write_mask: write::DEPTH, ..w };
        assert_eq!(BackendState::of(&w, false, false).colour_mask, 0);
    }
}
