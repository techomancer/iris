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
//! A draw picks its quad store once (`BackendState::store`, `select`): a
//! function monomorphised over the depth comparison and write, the colour
//! operation (replace, blend through a blend function monomorphised over
//! its factors and equation, or none) and the colour field's format,
//! addressing memory directly. Stencil, colour-index stores, partial colour
//! masks, logical operations other than copy and the other field formats
//! take `fragment_q`, which branches on the words at run time.

use super::ring::BackQuad;
use super::state::{BackendWords, BlendEquation, BlendFactor, Compare, StencilOp};
use super::super::consts::{write::INDEX_MASK, ODSY_DEPTH24_MAX, ODSY_DRAW_MASK_BUFFER0, ODSY_DRAW_MASK_BUFFER1};
use super::super::logic::logic_op;
use super::super::field::{Codec, FIELD_PACKED};
use super::super::view::{DestViews, Kind, Rows};
use super::super::vram::VramPtr;

/// A clear's work at the back end: what it stores, under which masks, as
/// its rectangles name it (`ring::KIND_RECT`). The colour is the stored
/// word (an RGBA colour, or an index); `colour_mask` names the word's bits
/// that change.
#[repr(C, align(64))]
pub(crate) struct ClearDraw {
    pub planes: DestMem,
    pub targets: [bool; 2],
    /// The stored word: 0xRRGGBBAA or an index.
    pub colour: Option<u32>,
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
                p.depth.fill(y, x0, x1, z & ODSY_DEPTH24_MAX);
            }
            if let Some(c) = self.colour {
                for (b, rows) in p.colour.iter().enumerate() {
                    if self.targets[b] {
                        rows.fill(y, x0, x1, c);
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
                p.depth.store_word(x, y, z & ODSY_DEPTH24_MAX);
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
                        (c & self.colour_mask) | (d & !self.colour_mask)
                    };
                    rows.store_word(x, y, out);
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

// Packers: an 8:8:8:8 colour word from its components, red highest, and
// back.

/// An RGBA8 word from its 8-bit components.
#[inline(always)]
pub(crate) fn pack_rgba8(c: [u32; 4]) -> u32 {
    (c[0] & 0xff) << 24 | (c[1] & 0xff) << 16 | (c[2] & 0xff) << 8 | (c[3] & 0xff)
}

#[inline(always)]
pub(crate) fn unpack_rgba8(w: u32) -> [u32; 4] {
    [w >> 24, (w >> 16) & 0xff, (w >> 8) & 0xff, w & 0xff]
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
    /// The blend, for the draw's factors and equation (`blend_for`).
    pub blend_fn: BlendFn,
    /// The draw's quad store (`select`); `store_generic` until chosen.
    pub store: QuadStore,
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
            blend_fn: blend_for(w.blend_src(), w.blend_dst(), w.blend_equation()),
            store: store_generic,
        }
    }

    /// The quad store for this state over planes: a monomorphised one where
    /// the state allows, else `store_generic`.
    pub fn select(&self, planes: &DestMem) -> QuadStore {
        let generic: QuadStore = store_generic;
        let blend_ok = !self.blend || blend_index(self.blend_src).is_some() && blend_index(self.blend_dst).is_some();
        // The copy logical operation stores the source, as no operation does.
        let logic_ok = self.logic.is_none_or(|op| op == LOGIC_COPY);

        if self.stencil || !self.rgba || !blend_ok || !logic_ok || (self.colour_mask != 0 && self.colour_mask != 0xf) {
            return generic;
        }
        if self.depth_test && planes.depth.v.kind != Kind::Z24 {
            return generic;
        }
        let op = if self.colour_mask == 0 || !self.targets.iter().any(|&t| t) {
            OP_NONE
        } else if self.blend {
            OP_BLEND
        } else {
            OP_REPLACE
        };
        // Every target's field has the format the store is made for.
        let format = |k: Kind| match k {
            Kind::Rgba8 => Some(C_RGBA8),
            Kind::Coded(Codec::Lanes) => Some(C_LANES),
            Kind::Coded(Codec::Texel { bytes: 4, field: FIELD_PACKED }) => Some(C_PACKED10),
            _ => None,
        };
        let mut c = C_RGBA8;
        if op != OP_NONE {
            let kinds: Vec<Option<u32>> = (0..2).filter(|&b| self.targets[b]).map(|b| format(planes.colour[b].v.kind)).collect();
            match kinds[..] {
                [Some(a)] => c = a,
                [Some(a), Some(b)] if a == b => c = a,
                _ => return generic,
            }
        }
        let z = if self.depth_test { 1 + (compare_code(self.depth_func) << 1 | self.depth_write as u32) } else { 0 };
        STORES[z as usize][op as usize][c as usize]
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
            let stored = planes.depth.load_word(x, y).unwrap_or(0);

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
            planes.depth.store_word(x, y, zq & ODSY_DEPTH24_MAX);
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
                rows.store_word(x, y, pack_rgba8(src.map(|c| c >> 4)));
                continue;
            }
            let Some(dst) = rows.load_word(x, y).map(unpack_rgba8) else {
                continue;
            };

            rows.store_word(x, y, pack_rgba8(self.colour_result(src, dst)));
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

/// The copy logical operation (GL_COPY minus 0x1500).
const LOGIC_COPY: u8 = 3;

/// A quad's store into memory, for one draw's state (`BackendState::select`).
pub(crate) type QuadStore = fn(&BackendState, &DestMem, &BackQuad);

/// The fragments of q, each through `fragment_q`.
pub(crate) fn store_generic(st: &BackendState, m: &DestMem, q: &BackQuad) {
    for i in 0..4usize {
        if q.cov & (1 << i) != 0 {
            st.fragment_q(m, q.x + (i & 1) as i32, q.y + (i >> 1) as i32, q.zq[i], q.rgba[i].map(u32::from));
        }
    }
}

/// The colour operations a monomorphised store makes.
const OP_REPLACE: u32 = 0;
const OP_BLEND: u32 = 1;
const OP_NONE: u32 = 2;
/// The colour field formats it stores: 8:8:8:8 bytes, 12-bit lanes, and
/// 10:10:10:2 in a 32-bit word, R highest.
const C_RGBA8: u32 = 0;
const C_LANES: u32 = 1;
const C_PACKED10: u32 = 2;

/// A comparison's code (Compare::of() reads it back).
fn compare_code(c: Compare) -> u32 {
    match c {
        Compare::Never => 0,
        Compare::Less => 1,
        Compare::Equal => 2,
        Compare::LessEqual => 3,
        Compare::Greater => 4,
        Compare::NotEqual => 5,
        Compare::GreaterEqual => 6,
        Compare::Always => 7,
    }
}

/// A colour field's components, 8 bits each, at byte address a.
#[inline(always)]
fn load_colour<const C: u32>(m: VramPtr, a: u64) -> [u32; 4] {
    if C == C_RGBA8 {
        unpack_rgba8(m.load(a, 4) as u32)
    } else if C == C_PACKED10 {
        // Each component's top 8 of 10 bits; alpha's 2 bits repeated.
        let v = m.load(a, 4) as u32;
        [(v >> 24) & 0xff, (v >> 14) & 0xff, (v >> 4) & 0xff, (v & 3) * 0x55]
    } else {
        let v = m.load(a, 6);
        [36, 24, 12, 0].map(|s| ((v >> s) as u32 & 0xfff) >> 4)
    }
}

/// Store 8-bit components c at byte address a: a word of bytes, 10:10:10:2,
/// or lanes widened by repeating their top bits.
#[inline(always)]
fn store_colour<const C: u32>(m: VramPtr, a: u64, c: [u32; 4]) {
    if C == C_RGBA8 {
        m.store(a, 4, pack_rgba8(c) as u64);
    } else if C == C_PACKED10 {
        // Each component widened to 10 bits by repeating its top bits;
        // alpha's top 2.
        let w10 = |v: u32| (v << 2) | (v >> 6);
        m.store(a, 4, (w10(c[0]) << 22 | w10(c[1]) << 12 | w10(c[2]) << 2 | c[3] >> 6) as u64);
    } else {
        let l = c.map(|v| ((v << 4) | (v >> 4)) as u64 & 0xfff);
        m.store(a, 6, (l[0] << 36) | (l[1] << 24) | (l[2] << 12) | l[3]);
    }
}

/// A quad's store: depth Z (0 none, else 1 + the comparison's code times 2
/// + the write), colour operation OP and colour format C.
fn store_quad<const Z: u32, const OP: u32, const C: u32>(st: &BackendState, m: &DestMem, q: &BackQuad) {
    for i in 0..4usize {
        if q.cov & (1 << i) == 0 {
            continue;
        }
        let (x, y) = (q.x + (i & 1) as i32, q.y + (i >> 1) as i32);
        if Z != 0 {
            let (mem, a) = (m.depth.mem(), m.depth.addr(x, y));
            let zq = q.zq[i];
            let stored = mem.load(a, 3) as u32;
            if !Compare::of((Z - 1) >> 1).test(zq, stored) {
                continue;
            }
            if (Z - 1) & 1 != 0 {
                mem.store(a, 3, (zq & ODSY_DEPTH24_MAX) as u64);
            }
        }
        if OP == OP_NONE {
            continue;
        }
        let src = q.rgba[i].map(u32::from);
        for b in 0..2 {
            if !st.targets[b] {
                continue;
            }
            // The target's colour field, of format C (BackendState::select).
            let (mem, a) = (m.colour[b].mem(), m.colour[b].addr(x, y));
            let out = if OP == OP_REPLACE {
                src.map(|c| c >> 4)
            } else {
                let d = load_colour::<C>(mem, a).map(widen8);
                (st.blend_fn)(src, d, &st.const_colour).map(|c| c >> 4)
            };
            store_colour::<C>(mem, a, out);
        }
    }
}

macro_rules! store_ops {
    ($z:expr) => {
        [
            [store_quad::<{ $z }, OP_REPLACE, C_RGBA8> as QuadStore, store_quad::<{ $z }, OP_REPLACE, C_LANES>, store_quad::<{ $z }, OP_REPLACE, C_PACKED10>],
            [store_quad::<{ $z }, OP_BLEND, C_RGBA8>, store_quad::<{ $z }, OP_BLEND, C_LANES>, store_quad::<{ $z }, OP_BLEND, C_PACKED10>],
            [store_quad::<{ $z }, OP_NONE, C_RGBA8>, store_quad::<{ $z }, OP_NONE, C_LANES>, store_quad::<{ $z }, OP_NONE, C_PACKED10>],
        ]
    };
}

/// The monomorphised stores by depth, colour operation and format.
static STORES: [[[QuadStore; 3]; 3]; 17] = [
    store_ops!(0),
    store_ops!(1), store_ops!(2), store_ops!(3), store_ops!(4), store_ops!(5), store_ops!(6), store_ops!(7), store_ops!(8),
    store_ops!(9), store_ops!(10), store_ops!(11), store_ops!(12), store_ops!(13), store_ops!(14), store_ops!(15), store_ops!(16),
];

/// A blend of a fragment's 12-bit components over the destination's, with
/// the constant colour's.
pub(crate) type BlendFn = fn([u32; 4], [u32; 4], &[u32; 4]) -> [u32; 4];

/// The blend factors in their table order (blend_index()).
const FACTORS: [BlendFactor; 13] = [
    BlendFactor::Zero,
    BlendFactor::One,
    BlendFactor::OtherColour,
    BlendFactor::SrcAlpha,
    BlendFactor::DstAlpha,
    BlendFactor::ConstColour,
    BlendFactor::ConstAlpha,
    BlendFactor::OneMinusOtherColour,
    BlendFactor::OneMinusSrcAlpha,
    BlendFactor::OneMinusDstAlpha,
    BlendFactor::OneMinusConstColour,
    BlendFactor::OneMinusConstAlpha,
    BlendFactor::SrcAlphaSaturate,
];

/// The blend equations in their table order; the others add, as
/// BackendState::blend does.
const EQUATIONS: [BlendEquation; 5] = [BlendEquation::Add, BlendEquation::Min, BlendEquation::Max, BlendEquation::Subtract, BlendEquation::ReverseSubtract];

fn blend_index(f: BlendFactor) -> Option<usize> {
    FACTORS.iter().position(|&g| g == f)
}

/// BackendState::blend for factors FACTORS[SF], FACTORS[DF] and equation
/// EQUATIONS[EQ].
fn blend<const SF: usize, const DF: usize, const EQ: usize>(s: [u32; 4], d: [u32; 4], c: &[u32; 4]) -> [u32; 4] {
    std::array::from_fn(|k| {
        let sf = factor(FACTORS[SF], k, true, &s, &d, c);
        let df = factor(FACTORS[DF], k, false, &s, &d, c);
        let (a, b) = (mul12(s[k], sf) as i32, mul12(d[k], df) as i32);
        let v = match EQUATIONS[EQ] {
            BlendEquation::Min => s[k].min(d[k]) as i32,
            BlendEquation::Max => s[k].max(d[k]) as i32,
            BlendEquation::Subtract => a - b,
            BlendEquation::ReverseSubtract => b - a,
            _ => a + b,
        };
        v.clamp(0, 4095) as u32
    })
}

/// The runtime blend, for a factor the table has no entry for.
fn blend_any(_s: [u32; 4], _d: [u32; 4], _c: &[u32; 4]) -> [u32; 4] {
    [0; 4]
}

macro_rules! blend_eqs {
    ($s:literal, $d:literal) => {
        [blend::<$s, $d, 0> as BlendFn, blend::<$s, $d, 1>, blend::<$s, $d, 2>, blend::<$s, $d, 3>, blend::<$s, $d, 4>]
    };
}
macro_rules! blend_dsts {
    ($s:literal) => {
        [
            blend_eqs!($s, 0), blend_eqs!($s, 1), blend_eqs!($s, 2), blend_eqs!($s, 3), blend_eqs!($s, 4), blend_eqs!($s, 5), blend_eqs!($s, 6),
            blend_eqs!($s, 7), blend_eqs!($s, 8), blend_eqs!($s, 9), blend_eqs!($s, 10), blend_eqs!($s, 11), blend_eqs!($s, 12),
        ]
    };
}

static BLENDS: [[[BlendFn; 5]; 13]; 13] = [
    blend_dsts!(0), blend_dsts!(1), blend_dsts!(2), blend_dsts!(3), blend_dsts!(4), blend_dsts!(5), blend_dsts!(6),
    blend_dsts!(7), blend_dsts!(8), blend_dsts!(9), blend_dsts!(10), blend_dsts!(11), blend_dsts!(12),
];

/// The monomorphised blend for factors src and dst under equation eq;
/// `blend_any` where a factor has none (select() then stores generically).
pub(crate) fn blend_for(src: BlendFactor, dst: BlendFactor, eq: BlendEquation) -> BlendFn {
    let e = EQUATIONS.iter().position(|&q| q == eq).unwrap_or(0);
    match (blend_index(src), blend_index(dst)) {
        (Some(s), Some(d)) => BLENDS[s][d][e],
        _ => blend_any,
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
                    p.depth.store_word(x, y, ODSY_DEPTH24_MAX);
                }
            }
            t
        }

        fn planes(&mut self) -> DestMem {
            DestMem::of(self.mem.ptr(), &self.d)
        }

        /// Colour (b) at (x, y) as its 8-bit components.
        fn colour(&mut self, b: usize, x: u32, y: u32) -> [u8; 4] {
            unpack_rgba8(self.planes().colour[b].load_word(x as i32, y as i32).unwrap()).map(|c| c as u8)
        }

        fn depth(&mut self, x: u32, y: u32) -> u32 {
            self.planes().depth.load_word(x as i32, y as i32).unwrap()
        }
    }

    fn words() -> BackendWords {
        BackendWords { write_mask: write::DEPTH | (0xf << write::COLOUR_SHIFT) | (0xff << write::STENCIL_SHIFT), draw_mask: 1, ..Default::default() }
    }

    #[test]
    fn packers_round_trip_and_lay_red_in_the_high_byte() {
        assert_eq!(pack_rgba8([0x12, 0x34, 0x56, 0x78]), 0x1234_5678);
        assert_eq!(unpack_rgba8(pack_rgba8([1, 2, 3, 4])), [1, 2, 3, 4]);
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
            p.depth.store_word(i, 0, 0x10);
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
