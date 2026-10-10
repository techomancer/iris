//! Raster shader emitters: one Cranelift function per `PipeKey`.
//!
//! Each emitter mirrors an interpreter routine in `rss.rs` / `te1.rs`, and
//! is named after it. They evaluate the same expressions in the same order
//! and precision (f64 for colour and coordinates, f32 for texel filtering,
//! saturating float-to-int conversions), so a shader's output is bit-exact
//! with the interpreter's. Where Rust and Cranelift differ the emitters
//! spell Rust's semantics out:
//!
//! - `f64::round` rounds halves away from zero (`round`); Cranelift's
//!   `nearest` rounds them to even.
//! - `f64::clamp` and `min` let NaN through or prefer the number
//!   (`clamp`, `min_num`); `fmin`/`fmax` propagate NaN.
//! - The level of detail is `te1::lod` emitted inline (sqrt and a cubic
//!   log2, no libm). Shaders make no calls.
//! - Colour, from the iterators through texturing, fog and blending, is
//!   12.16 fixed point (`mgras::fixed`), integer code on both sides.
//!
//! What the key fixes is decided here, at compile time; what the context
//! holds is loaded once, in the entry block, and stays in registers.
//!
//! Pixel memory: a 36-bit word's bits 63:36 are always zero (only
//! `PixMem::put` writes words, through `WORD_MASK`), so a wide store need
//! not merge with the old word.

use std::mem::{offset_of, size_of};

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::{F32, F64, I32, I64, I8};
use cranelift_codegen::ir::{self, AbiParam, Block, InstBuilder, MemFlagsData, Type, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use super::{ClipRect, Draw, PipeKey, Pix, Prim, RasterCtx, ShaderFn, Target, Tex, XFmt};
use crate::dev::ng1::rex3_generic::BAYER_PACKED;
use crate::dev::mgras::pixmem::{PAGES, PAGE_WORDS, TILE_H, TILE_W, WORD_MASK};
use crate::dev::mgras::rss::WIDTH;
use crate::dev::mgras::fixed;
use crate::dev::mgras::te1::{PAGE_NIBBLES, TRAM_NIBBLES};

macro_rules! off {
    ($($f:tt)+) => {
        offset_of!(RasterCtx, $($f)+) as i32
    };
}

pub struct Compiler {
    module: JITModule,
    ctx: Context,
    fctx: FunctionBuilderContext,
    n: u32,
}

impl Compiler {
    pub fn new() -> Self {
        let mut flags = settings::builder();
        flags.set("opt_level", "speed").unwrap();
        flags.set("is_pic", "false").unwrap();
        let isa = cranelift_native::builder()
            .expect("host ISA not supported")
            .finish(settings::Flags::new(flags))
            .unwrap();
        let module = JITModule::new(JITBuilder::with_isa(isa, cranelift_module::default_libcall_names()));
        Compiler { ctx: module.make_context(), module, fctx: FunctionBuilderContext::new(), n: 0 }
    }

    /// Compile the shader for a packed key: its entry and code size.
    pub fn compile(&mut self, packed: u64) -> Option<(ShaderFn, u32)> {
        let key = PipeKey::unpack(packed);
        let ptr = self.module.target_config().pointer_type();
        let name = format!("gr4_{packed:016x}_{}", self.n);
        self.n += 1;
        self.ctx.func.signature.params.clear();
        self.ctx.func.signature.returns.clear();
        self.ctx.func.signature.params.push(AbiParam::new(ptr));
        let id = match self.module.declare_function(&name, Linkage::Local, &self.ctx.func.signature) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("GR4 JIT: declare_function: {e}");
                return None;
            }
        };
        {
            let mut b = FunctionBuilder::new(&mut self.ctx.func, &mut self.fctx);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            b.seal_block(entry);
            let ctx = b.block_params(entry)[0];
            let mut e = E::new(b, key, ctx, ptr);
            match key.prim {
                Prim::Fill => e.fill(),
                Prim::Line => e.line(),
                Prim::Stipple => e.stipple(),
                Prim::Xfer => e.xfer(),
                Prim::Tri => e.triangle(),
                Prim::GlLine => e.gl_line(),
            }
            e.b.ins().return_(&[]);
            e.b.seal_all_blocks();
            let fc = self.module.target_config();
            e.b.finalize(fc);
        }
        let disasm = std::env::var_os("GR4_JIT_DISASM").is_some();
        self.ctx.set_disasm(disasm);
        if let Err(err) = self.module.define_function(id, &mut self.ctx) {
            eprintln!("GR4 JIT: define_function failed for {key}: {err}");
            eprintln!("--- Cranelift IR ---\n{}", self.ctx.func.display());
            self.module.clear_context(&mut self.ctx);
            return None;
        }
        let bytes = self.ctx.compiled_code().map(|c| c.code_buffer().len() as u32).unwrap_or(0);
        if disasm {
            if let Some(v) = self.ctx.compiled_code().and_then(|c| c.vcode.as_ref()) {
                eprintln!("GR4 JIT: {key} ({bytes} bytes)\n{v}");
            }
        }
        self.module.clear_context(&mut self.ctx);
        if let Err(err) = self.module.finalize_definitions() {
            eprintln!("GR4 JIT: finalize_definitions: {err}");
            return None;
        }
        let code = self.module.get_finalized_function(id);
        // SAFETY: the function was built with the ShaderFn signature.
        Some((unsafe { std::mem::transmute::<*const u8, ShaderFn>(code) }, bytes))
    }
}

/// One shader under construction, with the context's invariants loaded.
struct E<'a> {
    b: FunctionBuilder<'a>,
    k: PipeKey,
    ctx: Value,
    /// Pixel memory and the fields a shader writes back.
    m: MemFlagsData,
    /// Context fields and TRAM: never written while a shader runs. (Not
    /// `can_move`: Cranelift sinks those to their uses, inside the loops.
    /// Invariants are loaded once, in the entry block, instead: `inv`.)
    mr: MemFlagsData,
    /// GL invariants loaded in the entry block, by context offset.
    invs: std::collections::HashMap<i32, Value>,
    blk: [Value; 4],
    words: Value,
    cidp: Value,
    tram: Value,
    ox: Value,
    oy: Value,
    ysign: Value,
    cidmatch: Value,
    cidwmask: Value,
    clips: Vec<[Value; 5]>,
    tgt: [[Value; 4]; 2],
    zxtiles: Value,
}

impl<'a> E<'a> {
    fn new(mut b: FunctionBuilder<'a>, k: PipeKey, ctx: Value, ptr: Type) -> Self {
        let m = MemFlagsData::trusted();
        let mr = m.with_readonly();
        let ld = |b: &mut FunctionBuilder, ty: Type, off: i32| b.ins().load(ty, mr, ctx, off);
        // Only what this key uses: every invariant held in a register
        // through the pixel loops costs the loops a register.
        let zero = b.ins().iconst(I32, 0);
        let zero_p = b.ins().iconst(ptr, 0);
        let cid_any = k.cid_test || k.draw == Draw::Cid;
        let zbuf = k.stencil.is_some() || k.z.is_some() || (k.prim == Prim::Xfer && k.xfmt == XFmt::Depth);
        let words = ld(&mut b, ptr, off!(words));
        let cidp = if cid_any { ld(&mut b, ptr, off!(cid)) } else { zero_p };
        let tram = if k.tex.is_some() { ld(&mut b, ptr, off!(tram)) } else { zero_p };
        let ox = ld(&mut b, I32, off!(common.ox));
        let oy = ld(&mut b, I32, off!(common.oy));
        let ysign = ld(&mut b, I32, off!(common.ysign));
        let cidmatch = if k.cid_test { ld(&mut b, I32, off!(common.cidmatch)) } else { zero };
        let cidwmask = if k.draw == Draw::Cid { ld(&mut b, I32, off!(common.cidwmask)) } else { zero };
        let zxtiles = if zbuf { ld(&mut b, I32, off!(common.zxtiles)) } else { zero };
        let clips = (0..k.nclip as i32)
            .map(|n| {
                let base = off!(common.clip) + n * size_of::<ClipRect>() as i32;
                let f = |b: &mut FunctionBuilder, o: usize| b.ins().load(I32, mr, ctx, base + o as i32);
                [
                    f(&mut b, offset_of!(ClipRect, xmin)),
                    f(&mut b, offset_of!(ClipRect, xmax)),
                    f(&mut b, offset_of!(ClipRect, ymin)),
                    f(&mut b, offset_of!(ClipRect, ymax)),
                    f(&mut b, offset_of!(ClipRect, keep_inside)),
                ]
            })
            .collect();
        let mut tgt = [[zero; 4]; 2];
        let ntgt = match k.draw {
            Draw::Cid => 0,
            Draw::Dual => 2,
            _ => 1,
        };
        let masked = k.draw == Draw::Overlay || matches!(k.pix, Pix::Rgba8 | Pix::Ci12) || k.pix.pair();
        for (t, row) in tgt.iter_mut().enumerate().take(ntgt) {
            let base = off!(common.tgt) + (t * size_of::<Target>()) as i32;
            let f = |b: &mut FunctionBuilder, o: usize| b.ins().load(I32, mr, ctx, base + o as i32);
            row[0] = f(&mut b, offset_of!(Target, ptr));
            row[1] = f(&mut b, offset_of!(Target, xtiles));
            if masked {
                row[2] = f(&mut b, offset_of!(Target, mask));
            }
            if k.logic.is_some() {
                row[3] = f(&mut b, offset_of!(Target, lop_width));
            }
        }
        let blk = if matches!(k.prim, Prim::Fill | Prim::Stipple | Prim::Xfer) {
            [ld(&mut b, I32, off!(bxs)), ld(&mut b, I32, off!(bys)), ld(&mut b, I32, off!(bdx)), ld(&mut b, I32, off!(bdy))]
        } else {
            [zero; 4]
        };
        let mut invs = std::collections::HashMap::new();
        if k.prim.is_gl() {
            for o in [off!(gl.smp_w), off!(gl.smp_h)] {
                invs.insert(o, ld(&mut b, F64, o));
            }
            let mut i32s = vec![off!(gl.aref), off!(gl.env_a), off!(gl.sref), off!(gl.scmask), off!(gl.swmask)];
            i32s.extend([off!(gl.smp.max_level), off!(gl.smp.ls), off!(gl.smp.lt)]);
            i32s.extend((0..3).map(|k| off!(gl.env_c) + 4 * k));
            i32s.extend((0..3).map(|k| off!(gl.fog_c) + 4 * k));
            for o in i32s {
                invs.insert(o, ld(&mut b, I32, o));
            }
            invs.insert(off!(gl.zmask), ld(&mut b, I64, off!(gl.zmask)));
            for k in 0..4 {
                let o = off!(gl.smp.border_color) + 4 * k;
                invs.insert(o, ld(&mut b, I32, o));
            }
        }
        E { b, k, ctx, m, mr, invs, blk, words, cidp, tram, ox, oy, ysign, cidmatch, cidwmask, clips, tgt, zxtiles }
    }

    // ── small helpers ────────────────────────────────────────────────────

    fn ld(&mut self, ty: Type, off: i32) -> Value {
        self.b.ins().load(ty, self.mr, self.ctx, off)
    }

    /// A GL invariant (loaded in the entry block).
    fn inv(&self, off: i32) -> Value {
        self.invs[&off]
    }

    /// A field the shader also writes (stipple and line stipple positions).
    fn ld_rw(&mut self, ty: Type, off: i32) -> Value {
        self.b.ins().load(ty, self.m, self.ctx, off)
    }

    fn st(&mut self, v: Value, off: i32) {
        self.b.ins().store(self.m, v, self.ctx, off);
    }

    fn i32c(&mut self, v: i64) -> Value {
        self.b.ins().iconst(I32, v as i32 as u32 as i64)
    }

    fn i64c(&mut self, v: i64) -> Value {
        self.b.ins().iconst(I64, v)
    }

    fn f64c(&mut self, v: f64) -> Value {
        self.b.ins().f64const(v)
    }

    fn f32c(&mut self, v: f32) -> Value {
        self.b.ins().f32const(v)
    }

    fn var(&mut self, ty: Type, init: Value) -> Variable {
        let v = self.b.declare_var(ty);
        self.b.def_var(v, init);
        v
    }

    fn get(&mut self, v: Variable) -> Value {
        self.b.use_var(v)
    }

    fn set(&mut self, v: Variable, x: Value) {
        self.b.def_var(v, x);
    }

    /// The values as parameters of a fresh block, so what computed them
    /// stays here instead of being recomputed in the pixel loop: Cranelift
    /// rematerialises ALU ops with an immediate at every use, and a row's
    /// address arithmetic comes back into the loop with them. A block
    /// parameter is opaque only while its incoming values differ (single
    /// values are folded away), so the block is entered from a branch on
    /// the pixel memory pointer (never null) with the values, and from an
    /// unreachable path with zeros.
    fn pin(&mut self, vals: &[Value]) -> Vec<Value> {
        if vals.is_empty() {
            return Vec::new();
        }
        let blk = self.b.create_block();
        let never = self.b.create_block();
        let params: Vec<Value> = vals
            .iter()
            .map(|&v| {
                let ty = self.b.func.dfg.value_type(v);
                self.b.append_block_param(blk, ty)
            })
            .collect();
        let args: Vec<ir::BlockArg> = vals.iter().map(|&v| ir::BlockArg::Value(v)).collect();
        let live = self.b.ins().icmp_imm_s(IntCC::NotEqual, self.words, 0);
        self.b.ins().brif(live, blk, &args, never, &[]);
        self.b.switch_to_block(never);
        self.b.seal_block(never);
        let zeros: Vec<ir::BlockArg> = vals
            .iter()
            .map(|&v| {
                let z = match self.b.func.dfg.value_type(v) {
                    F64 => self.b.ins().f64const(0.0),
                    F32 => self.b.ins().f32const(0.0),
                    ty => self.b.ins().iconst(ty, 0),
                };
                ir::BlockArg::Value(z)
            })
            .collect();
        self.b.ins().jump(blk, &zeros);
        self.b.switch_to_block(blk);
        self.b.seal_block(blk);
        params
    }

    /// Continue in a fresh block when `cond` holds, else go to `other`.
    fn need(&mut self, cond: Value, other: Block) {
        let next = self.b.create_block();
        self.b.ins().brif(cond, next, &[], other, &[]);
        self.b.switch_to_block(next);
        self.b.seal_block(next);
    }

    /// Go to `other` when `cond` holds, else continue in a fresh block.
    fn bail(&mut self, cond: Value, other: Block) {
        let next = self.b.create_block();
        self.b.ins().brif(cond, other, &[], next, &[]);
        self.b.switch_to_block(next);
        self.b.seal_block(next);
    }

    /// `for v in start..end` (signed i32); the body gets `v` and the block
    /// that continues the loop, and may branch there.
    fn for_range(&mut self, start: Value, end: Value, body: impl FnOnce(&mut Self, Value, Block)) {
        let iv = self.var(I32, start);
        let head = self.b.create_block();
        let bodyb = self.b.create_block();
        let cont = self.b.create_block();
        let exit = self.b.create_block();
        self.b.ins().jump(head, &[]);
        self.b.switch_to_block(head);
        let i = self.get(iv);
        let c = self.b.ins().icmp(IntCC::SignedLessThan, i, end);
        self.b.ins().brif(c, bodyb, &[], exit, &[]);
        self.b.switch_to_block(bodyb);
        self.b.seal_block(bodyb);
        body(self, i, cont);
        self.b.ins().jump(cont, &[]);
        self.b.switch_to_block(cont);
        self.b.seal_block(cont);
        let i = self.get(iv);
        let i1 = self.b.ins().iadd_imm_s(i, 1);
        self.set(iv, i1);
        self.b.ins().jump(head, &[]);
        self.b.seal_block(head);
        self.b.switch_to_block(exit);
        self.b.seal_block(exit);
    }

    /// Run `f` only when `cond` holds; it must leave its block open.
    fn when(&mut self, cond: Value, f: impl FnOnce(&mut Self)) {
        let then = self.b.create_block();
        let join = self.b.create_block();
        self.b.ins().brif(cond, then, &[], join, &[]);
        self.b.switch_to_block(then);
        self.b.seal_block(then);
        f(self);
        self.b.ins().jump(join, &[]);
        self.b.switch_to_block(join);
        self.b.seal_block(join);
    }


    // ── Rust float semantics ─────────────────────────────────────────────

    /// `f64::round`: halves away from zero.
    fn round(&mut self, x: Value) -> Value {
        let t = self.b.ins().trunc(x);
        let d = self.b.ins().fsub(x, t);
        let d = self.b.ins().fabs(d);
        let half = self.f64c(0.5);
        let up = self.b.ins().fcmp(FloatCC::GreaterThanOrEqual, d, half);
        let one = self.f64c(1.0);
        let one = self.b.ins().fcopysign(one, x);
        let r = self.b.ins().fadd(t, one);
        self.b.ins().select(up, r, t)
    }

    /// `f64::round` for x >= +-0 or NaN (what a clamp to [0, n] leaves):
    /// the fraction is then non-negative, so no sign handling. NaN stays
    /// NaN, and converts to 0 as Rust's `as` does.
    fn round_nonneg(&mut self, x: Value) -> Value {
        let t = self.b.ins().trunc(x);
        let d = self.b.ins().fsub(x, t);
        let half = self.f64c(0.5);
        let up = self.b.ins().fcmp(FloatCC::GreaterThanOrEqual, d, half);
        let one = self.f64c(1.0);
        let r = self.b.ins().fadd(t, one);
        self.b.ins().select(up, r, t)
    }

    /// `clamp(lo, hi)`: below lo, lo; above hi, hi; NaN stays.
    fn clamp(&mut self, x: Value, lo: Value, hi: Value) -> Value {
        let c = self.b.ins().fcmp(FloatCC::LessThan, x, lo);
        let x = self.b.ins().select(c, lo, x);
        let c = self.b.ins().fcmp(FloatCC::GreaterThan, x, hi);
        self.b.ins().select(c, hi, x)
    }

    fn clamp01(&mut self, x: Value) -> Value {
        let (lo, hi) = if self.b.func.dfg.value_type(x) == F32 { (self.f32c(0.0), self.f32c(1.0)) } else { (self.f64c(0.0), self.f64c(1.0)) };
        self.clamp(x, lo, hi)
    }

    /// `f64::max`: the number when one side is NaN.
    fn max_num(&mut self, a: Value, b: Value) -> Value {
        let gt = self.b.ins().fcmp(FloatCC::GreaterThan, a, b);
        let bnan = self.b.ins().fcmp(FloatCC::Unordered, b, b);
        let alt = self.b.ins().select(bnan, a, b);
        self.b.ins().select(gt, a, alt)
    }

    /// `f64::min`: the number when one side is NaN.
    fn min_num(&mut self, a: Value, b: Value) -> Value {
        let lt = self.b.ins().fcmp(FloatCC::LessThan, a, b);
        let bnan = self.b.ins().fcmp(FloatCC::Unordered, b, b);
        let alt = self.b.ins().select(bnan, a, b);
        self.b.ins().select(lt, a, alt)
    }

    fn itof(&mut self, x: Value) -> Value {
        self.b.ins().fcvt_from_sint(F64, x)
    }

    /// `rss::compare` on floats (OpenGL order, "a OP b").
    fn fcompare(&mut self, func: u8, a: Value, b: Value) -> Value {
        let cc = match func & 7 {
            0 => return self.b.ins().iconst(I8, 0),
            1 => FloatCC::LessThan,
            2 => FloatCC::Equal,
            3 => FloatCC::LessThanOrEqual,
            4 => FloatCC::GreaterThan,
            5 => FloatCC::NotEqual,
            6 => FloatCC::GreaterThanOrEqual,
            _ => return self.b.ins().iconst(I8, 1),
        };
        self.b.ins().fcmp(cc, a, b)
    }

    /// `rss::compare` on values exact in f64 (stencil, depth).
    fn icompare(&mut self, func: u8, a: Value, b: Value) -> Value {
        let cc = match func & 7 {
            0 => return self.b.ins().iconst(I8, 0),
            1 => IntCC::UnsignedLessThan,
            2 => IntCC::Equal,
            3 => IntCC::UnsignedLessThanOrEqual,
            4 => IntCC::UnsignedGreaterThan,
            5 => IntCC::NotEqual,
            6 => IntCC::UnsignedGreaterThanOrEqual,
            _ => return self.b.ins().iconst(I8, 1),
        };
        self.b.ins().icmp(cc, a, b)
    }

    // ── pixel memory (pixmem) ────────────────────────────────────────────

    /// `Buffer::locate`, the y half: the page of the row's first tile and
    /// the row's offset within its page.
    fn locate_row(&mut self, ptr: Value, xt: Value, y: Value) -> (Value, Value) {
        let ty = self.b.ins().ushr_imm_s(y, 4);
        let row = self.b.ins().imul(ty, xt);
        let page = self.b.ins().iadd(ptr, row);
        let yin = self.b.ins().band_imm_s(y, TILE_H as i64 - 1);
        let yoff = self.b.ins().imul_imm_s(yin, TILE_W as i64);
        (page, yoff)
    }

    /// `Buffer::locate`, the x half: the word of x (0 <= x < 2048) in a row
    /// from `locate_row`, and for the overlay the bit shift.
    fn locate_x(&mut self, row: (Value, Value), x: Value, overlay: bool) -> (Value, Option<Value>) {
        // x / 192 = (x >> 6) / 3 and x / 768 = (x >> 8) / 3, by multiply.
        let sh = if overlay { 8 } else { 6 };
        let tw = if overlay { 4 * TILE_W } else { TILE_W } as i64;
        let xs = self.b.ins().ushr_imm_s(x, sh);
        let q = self.b.ins().imul_imm_s(xs, 43691);
        let tx = self.b.ins().ushr_imm_s(q, 17);
        let txw = self.b.ins().imul_imm_s(tx, tw);
        let xin = self.b.ins().isub(x, txw);
        let page = self.b.ins().iadd(row.0, tx);
        let page = self.b.ins().band_imm_s(page, PAGES as i64 - 1);
        let base = self.b.ins().imul_imm_s(page, PAGE_WORDS as i64);
        let idx = self.b.ins().iadd(base, row.1);
        let col = if overlay { self.b.ins().ushr_imm_s(xin, 2) } else { xin };
        let idx = self.b.ins().iadd(idx, col);
        let idx = self.b.ins().uextend(I64, idx);
        let boff = self.b.ins().ishl_imm_s(idx, 3);
        let addr = self.b.ins().iadd(self.words, boff);
        let shift = overlay.then(|| {
            let x3 = self.b.ins().band_imm_s(x, 3);
            self.b.ins().imul_imm_s(x3, 9)
        });
        (addr, shift)
    }

    /// `PixMem::get`.
    fn get_px(&mut self, addr: Value, shift: Option<Value>) -> Value {
        let w = self.b.ins().load(I64, self.m, addr, 0);
        match shift {
            None => self.b.ins().band_imm_s(w, WORD_MASK as i64),
            Some(s) => {
                let s = self.b.ins().uextend(I64, s);
                let v = self.b.ins().ushr(w, s);
                self.b.ins().band_imm_s(v, 0x1FF)
            }
        }
    }

    /// `PixMem::put`.
    fn put_px(&mut self, addr: Value, shift: Option<Value>, v: Value) {
        match shift {
            None => {
                let v = self.b.ins().band_imm_s(v, WORD_MASK as i64);
                self.b.ins().store(self.m, v, addr, 0);
            }
            Some(s) => {
                let s = self.b.ins().uextend(I64, s);
                let w = self.b.ins().load(I64, self.m, addr, 0);
                let mask = self.i64c(0x1FF);
                let hole = self.b.ins().ishl(mask, s);
                let w = self.b.ins().band_not(w, hole);
                let v = self.b.ins().band_imm_s(v, 0x1FF);
                let v = self.b.ins().ishl(v, s);
                let w = self.b.ins().bor(w, v);
                self.b.ins().store(self.m, w, addr, 0);
            }
        }
    }

    // ── the write path (rss.rs) ──────────────────────────────────────────

    /// `rss::logic_op`.
    fn logic_op(&mut self, op: u8, s: Value, d: Value) -> Value {
        match op & 0xF {
            0x0 => self.b.ins().iconst(I32, 0),
            0x1 => self.b.ins().band(s, d),
            0x2 => self.b.ins().band_not(s, d),
            0x3 => s,
            0x4 => self.b.ins().band_not(d, s),
            0x5 => d,
            0x6 => self.b.ins().bxor(s, d),
            0x7 => self.b.ins().bor(s, d),
            0x8 => {
                let v = self.b.ins().bor(s, d);
                self.b.ins().bnot(v)
            }
            0x9 => self.b.ins().bxor_not(s, d),
            0xA => self.b.ins().bnot(d),
            0xB => self.b.ins().bor_not(s, d),
            0xC => self.b.ins().bnot(s),
            0xD => self.b.ins().bor_not(d, s),
            0xE => {
                let v = self.b.ins().band(s, d);
                self.b.ins().bnot(v)
            }
            _ => self.b.ins().iconst(I32, 0xFFFF_FFFFu32 as i64),
        }
    }

    /// `Rss::put_in` into target `t` (0: the draw target, 1: buffer B of
    /// a dual write) at x in row `r`.
    fn put_in(&mut self, r: &Row, t: usize, x: Value, v: Value) {
        let k = self.k;
        let overlay = t == 0 && k.draw == Draw::Overlay;
        let pair = !overlay && k.pix.pair();
        let masked = overlay || matches!(k.pix, Pix::Rgba8 | Pix::Ci12) || pair;
        let [_, _, mask, lop_width] = self.tgt[t];
        let (addr, shift) = self.locate_x(r.tgt[t], x, overlay);
        let old = (k.logic.is_some() || masked).then(|| {
            let w = self.get_px(addr, shift);
            self.b.ins().ireduce(I32, w)
        });
        let mut v = v;
        if pair {
            // `rss::rgb12_pixel`, in both halves.
            let c = self.rgb12_pixel(r, x, v);
            let hi = self.b.ins().ishl_imm_s(c, 12);
            v = self.b.ins().bor(c, hi);
        }
        if let Some(op) = k.logic {
            let r = self.logic_op(op, v, old.unwrap());
            v = self.b.ins().band(r, lop_width);
        }
        if masked {
            let keep = self.b.ins().band_not(old.unwrap(), mask);
            let new = self.b.ins().band(v, mask);
            v = self.b.ins().bor(keep, new);
        }
        let v = self.b.ins().uextend(I64, v);
        self.put_px(addr, shift, v);
    }

    /// `rss::rgb12_pixel`: an 8-8-8 colour to 12 bits, red in 3:0, through
    /// the REX3 4x4 Bayer matrix when the key dithers
    /// (`rex3_generic::rgb24_to_rgb12_dither`).
    fn rgb12_pixel(&mut self, r: &Row, x: Value, v: Value) -> Value {
        let thr = r.bayer.map(|row| {
            let xi = self.b.ins().band_imm_s(x, 3);
            let idx = self.b.ins().bor(row, xi);
            let sh = self.b.ins().ishl_imm_s(idx, 2);
            let sh = self.b.ins().uextend(I64, sh);
            let m = self.i64c(BAYER_PACKED as i64);
            let t = self.b.ins().ushr(m, sh);
            let t = self.b.ins().ireduce(I32, t);
            self.b.ins().band_imm_s(t, 0xF)
        });
        let mut out = self.i32c(0);
        for c in 0..3 {
            let ch = self.b.ins().ushr_imm_s(v, 8 * c);
            let ch = self.b.ins().band_imm_s(ch, 0xFF);
            let n = match thr {
                Some(t) => {
                    let q = self.b.ins().ushr_imm_s(ch, 4);
                    let s = self.b.ins().isub(ch, q);
                    let d = self.b.ins().ushr_imm_s(s, 4);
                    let f = self.b.ins().band_imm_s(s, 0xF);
                    let up = self.b.ins().icmp(IntCC::UnsignedGreaterThan, f, t);
                    let up = self.b.ins().uextend(I32, up);
                    let d = self.b.ins().iadd(d, up);
                    let max = self.i32c(15);
                    self.b.ins().umin(d, max)
                }
                None => self.b.ins().ushr_imm_s(ch, 4),
            };
            let n = if c > 0 { self.b.ins().ishl_imm_s(n, 4 * c) } else { n };
            out = self.b.ins().bor(out, n);
        }
        out
    }

    /// The parts of `Rss::visible` and of every address that depend on the
    /// framebuffer row alone. A row off screen goes on to `kill`.
    fn row(&mut self, fy: Value, kill: Block) -> Row {
        let yo = self.b.ins().icmp_imm_s(IntCC::UnsignedGreaterThanOrEqual, fy, WIDTH as i64);
        self.bail(yo, kill);
        let cid = (self.k.cid_test || self.k.draw == Draw::Cid).then(|| {
            let rowoff = self.b.ins().imul_imm_s(fy, WIDTH as i64);
            let rowoff = self.b.ins().uextend(I64, rowoff);
            self.b.ins().iadd(self.cidp, rowoff)
        });
        let clip = (0..self.clips.len())
            .map(|n| {
                let [_, _, ymin, ymax, keep] = self.clips[n];
                let c = self.b.ins().icmp(IntCC::SignedGreaterThanOrEqual, fy, ymin);
                let d = self.b.ins().icmp(IntCC::SignedLessThanOrEqual, fy, ymax);
                let iny = self.b.ins().band(c, d);
                let keep = self.b.ins().icmp_imm_s(IntCC::NotEqual, keep, 0);
                (iny, keep)
            })
            .collect();
        let mut tgt = [(fy, fy); 2];
        let ntgt = if self.k.draw == Draw::Dual { 2 } else { 1 };
        for (t, row) in tgt.iter_mut().enumerate().take(ntgt) {
            let [ptr, xt, _, _] = self.tgt[t];
            *row = self.locate_row(ptr, xt, fy);
        }
        let z = (self.k.stencil.is_some() || self.k.z.is_some() || self.k.xfmt == XFmt::Depth).then(|| {
            let zp = self.i32c(crate::dev::mgras::rss::ZST_PAGE as i64);
            self.locate_row(zp, self.zxtiles, fy)
        });
        let bayer = self.k.pix.dither().then(|| {
            let y = self.b.ins().band_imm_s(fy, 3);
            self.b.ins().ishl_imm_s(y, 2)
        });
        let mut r = Row { cid, clip, tgt, z, bayer };
        // Pin it all, so it is computed once per row.
        let mut vals: Vec<Value> = Vec::new();
        vals.extend(r.cid);
        for &(a, b) in &r.clip {
            vals.extend([a, b]);
        }
        for &(a, b) in &r.tgt {
            vals.extend([a, b]);
        }
        if let Some((a, b)) = r.z {
            vals.extend([a, b]);
        }
        vals.extend(r.bayer);
        let mut p = self.pin(&vals).into_iter();
        if r.cid.is_some() {
            r.cid = p.next();
        }
        for c in r.clip.iter_mut() {
            *c = (p.next().unwrap(), p.next().unwrap());
        }
        for t in r.tgt.iter_mut() {
            *t = (p.next().unwrap(), p.next().unwrap());
        }
        if r.z.is_some() {
            r.z = Some((p.next().unwrap(), p.next().unwrap()));
        }
        if r.bayer.is_some() {
            r.bayer = p.next();
        }
        r
    }

    /// The rest of `Rss::visible`, for x in row `r`.
    fn visible_x(&mut self, r: &Row, x: Value, kill: Block) {
        let xo = self.b.ins().icmp_imm_s(IntCC::UnsignedGreaterThanOrEqual, x, WIDTH as i64);
        self.bail(xo, kill);
        if self.k.cid_test {
            let x64 = self.b.ins().uextend(I64, x);
            let a = self.b.ins().iadd(r.cid.unwrap(), x64);
            let id = self.b.ins().uload8(I32, self.m, a, 0);
            let bit = self.b.ins().ushr(self.cidmatch, id);
            let bit = self.b.ins().band_imm_s(bit, 1);
            self.need(bit, kill);
        }
        for n in 0..self.clips.len() {
            let [xmin, xmax, _, _, _] = self.clips[n];
            let (iny, keep) = r.clip[n];
            let a = self.b.ins().icmp(IntCC::SignedGreaterThanOrEqual, x, xmin);
            let b = self.b.ins().icmp(IntCC::SignedLessThanOrEqual, x, xmax);
            let inx = self.b.ins().band(a, b);
            let inside = self.b.ins().band(inx, iny);
            let fail = self.b.ins().bxor(inside, keep);
            self.bail(fail, kill);
        }
    }

    /// `Rss::put`: a 2D pixel at x in row `r`.
    fn put_row(&mut self, r: &Row, x: Value, v: Value, kill: Block) {
        self.visible_x(r, x, kill);
        match self.k.draw {
            Draw::Cid => {
                let x64 = self.b.ins().uextend(I64, x);
                let a = self.b.ins().iadd(r.cid.unwrap(), x64);
                let c = self.b.ins().uload8(I32, self.m, a, 0);
                let c = self.b.ins().band_not(c, self.cidwmask);
                let n = self.b.ins().band(v, self.cidwmask);
                let c = self.b.ins().bor(c, n);
                self.b.ins().istore8(self.m, c, a, 0);
            }
            Draw::Dual => {
                self.put_in(r, 0, x, v);
                self.put_in(r, 1, x, v);
            }
            _ => self.put_in(r, 0, x, v),
        }
    }

    /// `Rss::put` at framebuffer (x, y).
    fn put(&mut self, x: Value, y: Value, v: Value, kill: Block) {
        let r = self.row(y, kill);
        self.put_row(&r, x, v, kill);
    }

    /// `Rss::to_fb`.
    fn to_fb(&mut self, x: Value, y: Value) -> (Value, Value) {
        let fx = self.b.ins().iadd(self.ox, x);
        let sy = self.b.ins().imul(self.ysign, y);
        let fy = self.b.ins().iadd(self.oy, sy);
        (fx, fy)
    }

    /// `Rss::block_px`.
    fn block_px(&mut self, col: Value, row: Value) -> (Value, Value) {
        let [bxs, bys, bdx, bdy] = self.blk;
        let cx = self.b.ins().imul(col, bdx);
        let x = self.b.ins().iadd(bxs, cx);
        let ry = self.b.ins().imul(row, bdy);
        let y = self.b.ins().iadd(bys, ry);
        self.to_fb(x, y)
    }

    // ── 2D primitives ────────────────────────────────────────────────────

    /// `Rss::fill`: per row, the row's share once; per pixel the rest.
    fn fill(&mut self) {
        let rows = self.ld(I32, off!(brows));
        let cols = self.ld(I32, off!(bcols));
        let color = self.ld(I32, off!(color));
        let [bxs, bys, bdx, bdy] = self.blk;
        let fx0 = self.b.ins().iadd(self.ox, bxs);
        let zero = self.i32c(0);
        self.for_range(zero, rows, |e, row, next_row| {
            let ry = e.b.ins().imul(row, bdy);
            let y = e.b.ins().iadd(bys, ry);
            let sy = e.b.ins().imul(e.ysign, y);
            let fy = e.b.ins().iadd(e.oy, sy);
            let r = e.row(fy, next_row);
            let zero = e.i32c(0);
            e.for_range(zero, cols, |e, col, cont| {
                let cx = e.b.ins().imul(col, bdx);
                let fx = e.b.ins().iadd(fx0, cx);
                e.put_row(&r, fx, color, cont);
            });
        });
    }

    /// `Rss::line`: Bresenham, stipple bit `31 - k % 32`.
    fn line(&mut self) {
        let k = self.k;
        let (x0, y0) = (self.ld(I32, off!(lx0)), self.ld(I32, off!(ly0)));
        let (x1, y1) = (self.ld(I32, off!(lx1)), self.ld(I32, off!(ly1)));
        let color = self.ld(I32, off!(color));
        let bg = self.ld(I32, off!(bg));
        let pattern = self.ld(I32, off!(lpattern));
        let ddx = self.b.ins().isub(x1, x0);
        let dx = self.b.ins().iabs(ddx);
        let ddy = self.b.ins().isub(y1, y0);
        let ady = self.b.ins().iabs(ddy);
        let dy = self.b.ins().ineg(ady);
        let one = self.i32c(1);
        let m1 = self.i32c(-1);
        let xl = self.b.ins().icmp(IntCC::SignedLessThan, x0, x1);
        let sx = self.b.ins().select(xl, one, m1);
        let yl = self.b.ins().icmp(IntCC::SignedLessThan, y0, y1);
        let sy = self.b.ins().select(yl, one, m1);
        let err0 = self.b.ins().iadd(dx, dy);
        let zero = self.i32c(0);
        let (xv, yv, ev, kv) = (self.var(I32, x0), self.var(I32, y0), self.var(I32, err0), self.var(I32, zero));
        let head = self.b.create_block();
        let after = self.b.create_block();
        let exit = self.b.create_block();
        self.b.ins().jump(head, &[]);
        self.b.switch_to_block(head);
        let (x, y) = (self.get(xv), self.get(yv));
        let ex = self.b.ins().icmp(IntCC::Equal, x, x1);
        let ey = self.b.ins().icmp(IntCC::Equal, y, y1);
        let at_end = self.b.ins().band(ex, ey);
        if k.skip_last {
            self.bail(at_end, exit);
        }
        let c = if k.stipple {
            let kk = self.get(kv);
            let b = self.b.ins().band_imm_s(kk, 31);
            let c31 = self.i32c(31);
            let sh = self.b.ins().isub(c31, b);
            let bit = self.b.ins().ushr(pattern, sh);
            let lit = self.b.ins().band_imm_s(bit, 1);
            if k.opaque {
                self.b.ins().select(lit, color, bg)
            } else {
                self.need(lit, after);
                color
            }
        } else {
            color
        };
        let (fx, fy) = self.to_fb(x, y);
        self.put(fx, fy, c, after);
        self.b.ins().jump(after, &[]);
        self.b.switch_to_block(after);
        self.b.seal_block(after);
        let (x, y, err) = (self.get(xv), self.get(yv), self.get(ev));
        let ex = self.b.ins().icmp(IntCC::Equal, x, x1);
        let ey = self.b.ins().icmp(IntCC::Equal, y, y1);
        let at_end = self.b.ins().band(ex, ey);
        self.bail(at_end, exit);
        let e2 = self.b.ins().iadd(err, err);
        let stepx = self.b.ins().icmp(IntCC::SignedGreaterThanOrEqual, e2, dy);
        let ex1 = self.b.ins().iadd(err, dy);
        let err = self.b.ins().select(stepx, ex1, err);
        let xs = self.b.ins().iadd(x, sx);
        let x = self.b.ins().select(stepx, xs, x);
        let stepy = self.b.ins().icmp(IntCC::SignedLessThanOrEqual, e2, dx);
        let ey1 = self.b.ins().iadd(err, dx);
        let err = self.b.ins().select(stepy, ey1, err);
        let ys = self.b.ins().iadd(y, sy);
        let y = self.b.ins().select(stepy, ys, y);
        self.set(xv, x);
        self.set(yv, y);
        self.set(ev, err);
        let kk = self.get(kv);
        let kk = self.b.ins().iadd_imm_s(kk, 1);
        self.set(kv, kk);
        self.b.ins().jump(head, &[]);
        self.b.seal_block(head);
        self.b.switch_to_block(exit);
        self.b.seal_block(exit);
    }

    /// `Rss::stipple_bits`, from the check that the block has rows left.
    fn stipple(&mut self) {
        let k = self.k;
        let bits = self.ld(I64, off!(sbits));
        let n = self.ld(I32, off!(sn));
        let cols = self.ld(I32, off!(bcols));
        let color = self.ld(I32, off!(color));
        let bg = self.ld(I32, off!(bg));
        let c0 = self.ld_rw(I32, off!(scol));
        let r0 = self.ld_rw(I32, off!(srow));
        let zero = self.i32c(0);
        let (colv, rowv, iv) = (self.var(I32, c0), self.var(I32, r0), self.var(I32, zero));
        let head = self.b.create_block();
        let adv = self.b.create_block();
        let exit = self.b.create_block();
        self.b.ins().jump(head, &[]);
        self.b.switch_to_block(head);
        let i = self.get(iv);
        let more = self.b.ins().icmp(IntCC::UnsignedLessThan, i, n);
        self.need(more, exit);
        let c63 = self.i32c(63);
        let sh = self.b.ins().isub(c63, i);
        let sh = self.b.ins().uextend(I64, sh);
        let bit = self.b.ins().ushr(bits, sh);
        let bit = self.b.ins().band_imm_s(bit, 1);
        let c = if k.opaque {
            self.b.ins().select(bit, color, bg)
        } else {
            self.need(bit, adv);
            color
        };
        let (col, row) = (self.get(colv), self.get(rowv));
        let (x, y) = self.block_px(col, row);
        self.put(x, y, c, adv);
        self.b.ins().jump(adv, &[]);
        self.b.switch_to_block(adv);
        self.b.seal_block(adv);
        let col = self.get(colv);
        let col = self.b.ins().iadd_imm_s(col, 1);
        self.set(colv, col);
        let i = self.get(iv);
        let i = self.b.ins().iadd_imm_s(i, 1);
        self.set(iv, i);
        let wrap = self.b.ins().icmp(IntCC::SignedGreaterThanOrEqual, col, cols);
        let wrapb = self.b.create_block();
        self.b.ins().brif(wrap, wrapb, &[], head, &[]);
        self.b.seal_block(head);
        self.b.switch_to_block(wrapb);
        self.b.seal_block(wrapb);
        let zero = self.i32c(0);
        self.set(colv, zero);
        let row = self.get(rowv);
        let row = self.b.ins().iadd_imm_s(row, 1);
        self.set(rowv, row);
        self.b.ins().jump(exit, &[]);
        self.b.switch_to_block(exit);
        self.b.seal_block(exit);
        let (col, row) = (self.get(colv), self.get(rowv));
        self.st(col, off!(scol));
        self.st(row, off!(srow));
    }

    /// `rss::from_host` for the key's format, on a pixel's bytes as one
    /// big-endian number.
    fn from_host(&mut self, v: Value) -> Value {
        let pack = |e: &mut Self, r: Value, g: Value, b: Value| {
            let r = e.b.ins().band_imm_s(r, 0xFF);
            let g = e.b.ins().band_imm_s(g, 0xFF);
            let g = e.b.ins().ishl_imm_s(g, 8);
            let b = e.b.ins().band_imm_s(b, 0xFF);
            let b = e.b.ins().ishl_imm_s(b, 16);
            let rg = e.b.ins().bor(r, g);
            e.b.ins().bor(rg, b)
        };
        let comp64 = |e: &mut Self, s: i64| {
            let x = e.b.ins().ushr_imm_s(v, s);
            e.b.ins().ireduce(I32, x)
        };
        match self.k.xfmt {
            XFmt::Rgb16 => {
                let (r, g, b) = (comp64(self, 40), comp64(self, 24), comp64(self, 8));
                pack(self, r, g, b)
            }
            XFmt::Rgba16 => {
                let (r, g, b) = (comp64(self, 56), comp64(self, 40), comp64(self, 24));
                pack(self, r, g, b)
            }
            XFmt::Rgb8 => {
                let (r, g, b) = (comp64(self, 16), comp64(self, 8), comp64(self, 0));
                pack(self, r, g, b)
            }
            XFmt::Depth => {
                let v = self.b.ins().ireduce(I32, v);
                let v = self.b.ins().ushr_imm_s(v, 8);
                self.b.ins().band_imm_s(v, 0xFF_FFFF)
            }
            XFmt::Rgba4 => {
                let v = self.b.ins().ireduce(I32, v);
                let c4 = |e: &mut Self, s: i64| {
                    let x = e.b.ins().ushr_imm_s(v, s);
                    let x = e.b.ins().band_imm_s(x, 0xF);
                    e.b.ins().imul_imm_s(x, 0x11)
                };
                let (r, g, b) = (c4(self, 0), c4(self, 4), c4(self, 8));
                pack(self, r, g, b)
            }
            XFmt::Rgb5 => {
                let v = self.b.ins().ireduce(I32, v);
                let c5 = |e: &mut Self, s: i64| {
                    let x = e.b.ins().ushr_imm_s(v, s);
                    let x = e.b.ins().band_imm_s(x, 0x1F);
                    let hi = e.b.ins().ishl_imm_s(x, 3);
                    let lo = e.b.ins().ushr_imm_s(x, 2);
                    e.b.ins().bor(hi, lo)
                };
                let (r, g, b) = (c5(self, 0), c5(self, 5), c5(self, 10));
                pack(self, r, g, b)
            }
            XFmt::Ci12 => {
                let v = self.b.ins().ireduce(I32, v);
                self.b.ins().band_imm_s(v, 0xFFF)
            }
            XFmt::Raw => self.b.ins().ireduce(I32, v),
        }
    }

    /// `Rss::put_line`: one transfer line into the block (one row).
    fn xfer(&mut self) {
        let width = self.ld(I32, off!(xwidth));
        let line = self.ld(I32, off!(xline));
        let src = self.ld(I64, off!(src));
        let bpp = self.k.xbpp as i64;
        let [bxs, bys, bdx, bdy] = self.blk;
        let ly = self.b.ins().imul(line, bdy);
        let y = self.b.ins().iadd(bys, ly);
        let sy = self.b.ins().imul(self.ysign, y);
        let fy = self.b.ins().iadd(self.oy, sy);
        let fx0 = self.b.ins().iadd(self.ox, bxs);
        let done = self.b.create_block();
        let r = self.row(fy, done);
        let zero = self.i32c(0);
        self.for_range(zero, width, |e, k, cont| {
            let k64 = e.b.ins().uextend(I64, k);
            let o = e.b.ins().imul_imm_s(k64, bpp);
            let p = e.b.ins().iadd(src, o);
            let mut v = e.i64c(0);
            for i in 0..bpp {
                let byte = e.b.ins().uload8(I64, e.mr, p, i as i32);
                let sh = e.b.ins().ishl_imm_s(v, 8);
                v = e.b.ins().bor(sh, byte);
            }
            let kx = e.b.ins().imul(k, bdx);
            let fx = e.b.ins().iadd(fx0, kx);
            if e.k.xfmt == XFmt::Depth {
                // Depth into the depth buffer, its stencil kept.
                e.visible_x(&r, fx, cont);
                let (a, _) = e.locate_x(r.z.unwrap(), fx, false);
                let old = e.get_px(a, None);
                let old = e.b.ins().band_imm_s(old, !0xFF_FFFFi64);
                let d = e.from_host(v);
                let d = e.b.ins().uextend(I64, d);
                let new = e.b.ins().bor(old, d);
                e.put_px(a, None, new);
            } else {
                let c = e.from_host(v);
                e.put_row(&r, fx, c, cont);
            }
        });
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
        self.b.seal_block(done);
    }

    // ── GL primitives ────────────────────────────────────────────────────

    /// A 12.16 plane (three i32s).
    fn iplane(&mut self, base: i32) -> [Value; 3] {
        [self.ld(I32, base), self.ld(I32, base + 4), self.ld(I32, base + 8)]
    }

    /// `Rss::triangle`'s pixel loop over its setup (`ctx.tri`).
    fn triangle(&mut self) {
        let k = self.k;
        macro_rules! t {
            ($f:ident) => {
                self.ld(F64, off!(tri.$f))
            };
        }
        let (x0, x1, x2) = (t!(x0), t!(x1), t!(x2));
        let (ymid, yref, yref2) = (t!(ymid), t!(yref), t!(yref2));
        let (s0, s1, s2, xs) = (t!(s0), t!(s1), t!(s2), t!(xs));
        let cplanes: Vec<[Value; 3]> = (0..4).map(|c| self.iplane(off!(tri.cplanes) + 12 * c)).collect();
        let (xs_i, yref_i) = (self.ld(I32, off!(tri.xs_i)), self.ld(I32, off!(tri.yref_i)));
        let zp = [self.ld(I64, off!(tri.zp)), self.ld(I64, off!(tri.zp) + 8), self.ld(I64, off!(tri.zp) + 16)];
        let tp64 = |e: &mut Self, c: i32| -> [Value; 3] {
            let o = off!(tri.tp) + 24 * c;
            [e.ld(I64, o), e.ld(I64, o + 8), e.ld(I64, o + 16)]
        };
        let tplanes: Vec<[Value; 3]> = if k.tex.is_some() { (0..3).map(|c| tp64(self, c)).collect() } else { Vec::new() };
        let fogp = k.fog.then(|| self.iplane(off!(tri.fogp)));
        let ltor = self.ld(I32, off!(tri.ltor));
        let ltor = self.b.ins().icmp_imm_s(IntCC::NotEqual, ltor, 0);
        let (j0, j1) = (self.ld(I32, off!(tri.j0)), self.ld(I32, off!(tri.j1)));
        let vis: Vec<Value> = (0..4).map(|n| self.ld(I32, off!(common.vis) + 4 * n)).collect();
        let jlo = self.b.ins().smax(j0, vis[2]);
        let jhi = self.b.ins().smin(j1, vis[3]);
        self.for_range(jlo, jhi, |e, j, next_row| {
            let sy = e.b.ins().imul(e.ysign, j);
            let fy = e.b.ins().iadd(e.oy, sy);
            let span = e.row(fy, next_row);
            let jf = e.itof(j);
            let half = e.f64c(0.5);
            let yc = e.b.ins().fadd(jf, half);
            let dy = e.b.ins().fsub(yref, yc);
            // The row's share of the colour, fog and depth planes, once a
            // row: `p0 + p2 (yref - j)` (colours 12.16, depth z.12).
            let dj = e.b.ins().isub(yref_i, j);
            let mut terms: Vec<Value> = cplanes.iter().chain(fogp.iter()).map(|p| {
                let m = e.b.ins().imul(p[2], dj);
                e.b.ins().iadd(p[0], m)
            }).collect();
            let dj64 = e.b.ins().sextend(I64, dj);
            // i64 planes (depth, then S/W, T/W, 1/W).
            for p in std::iter::once(&zp).chain(tplanes.iter()) {
                let m = e.b.ins().imul(p[2], dj64);
                terms.push(e.b.ins().iadd(p[0], m));
            }
            let terms = e.pin(&terms);
            let n32 = 4 + fogp.is_some() as usize;
            let zterm = terms[n32];
            let trows: Vec<Value> = terms[n32 + 1..].to_vec();
            let m = e.b.ins().fmul(s0, dy);
            let major = e.b.ins().fadd(x0, m);
            let m1 = e.b.ins().fmul(s1, dy);
            let up = e.b.ins().fadd(x1, m1);
            let dy2 = e.b.ins().fsub(yref2, yc);
            let m2 = e.b.ins().fmul(s2, dy2);
            let low = e.b.ins().fadd(x2, m2);
            let upper = e.b.ins().fcmp(FloatCC::GreaterThanOrEqual, yc, ymid);
            let minor = e.b.ins().select(upper, up, low);
            let l = e.b.ins().select(ltor, major, minor);
            let r = e.b.ins().select(ltor, minor, major);
            let lh = e.b.ins().fsub(l, half);
            let lc = e.b.ins().ceil(lh);
            let i0 = e.b.ins().fcvt_to_sint_sat(I32, lc);
            let rh = e.b.ins().fsub(r, half);
            let rc = e.b.ins().ceil(rh);
            let i1 = e.b.ins().fcvt_to_sint_sat(I32, rc);
            let ilo = e.b.ins().smax(i0, vis[0]);
            let ihi = e.b.ins().smin(i1, vis[1]);
            let row = k.stipple.then(|| {
                let jm = e.b.ins().band_imm_s(j, 31);
                let jm = e.b.ins().uextend(I64, jm);
                let o = e.b.ins().ishl_imm_s(jm, 2);
                let a = e.b.ins().iadd(e.ctx, o);
                e.b.ins().load(I32, e.m, a, off!(gl.poly))
            });
            e.for_range(ilo, ihi, |e, i, cont| {
                if let Some(row) = row {
                    let im = e.b.ins().band_imm_s(i, 31);
                    let c31 = e.i32c(31);
                    let sh = e.b.ins().isub(c31, im);
                    let bit = e.b.ins().ushr(row, sh);
                    let bit = e.b.ins().band_imm_s(bit, 1);
                    e.need(bit, cont);
                }
                let di = e.b.ins().isub(i, xs_i);
                let di64 = e.b.ins().sextend(I64, di);
                let fixed_at = |e: &mut Self, p: &[Value; 3], row: Value| {
                    let m = e.b.ins().imul(p[1], di);
                    e.b.ins().iadd(row, m)
                };
                let rgba: Vec<Variable> = (0..4)
                    .map(|c| {
                        let v = fixed_at(e, &cplanes[c], terms[c]);
                        e.var(I32, v)
                    })
                    .collect();
                let rgba: [Variable; 4] = rgba.try_into().unwrap();
                if k.tex.is_some() {
                    let v: [Value; 3] = [0, 1, 2].map(|c| {
                        let m = e.b.ins().imul(tplanes[c][1], di64);
                        e.b.ins().iadd(trows[c], m)
                    });
                    let d = [0, 1, 2].map(|c| [tplanes[c][1], tplanes[c][2]]);
                    e.textured(v, d, rgba);
                }
                let mut c: [Value; 4] = rgba.map(|v| e.get(v));
                if let Some(fp) = fogp {
                    let f = fixed_at(e, &fp, terms[4]);
                    c = e.fogged(c, f);
                }
                let zm = e.b.ins().imul(zp[1], di64);
                let z = e.b.ins().iadd(zterm, zm);
                let fx = e.b.ins().iadd(e.ox, i);
                e.fragment(&span, fx, c, z, cont);
            });
        });
    }

    /// `Rss::gl_line`'s pixel loop over its setup (`ctx.gll`).
    fn gl_line(&mut self) {
        let k = self.k;
        macro_rules! g {
            ($t:expr, $($f:tt)+) => {
                self.ld($t, off!(gll.$($f)+))
            };
        }
        let cbase: Vec<Value> = (0..4).map(|c| self.ld(I32, off!(gll.cbase) + 4 * c)).collect();
        let cstep: Vec<Value> = (0..4).map(|c| self.ld(I32, off!(gll.cstep) + 4 * c)).collect();
        let (zbase, zstep) = (g!(I64, zbase), g!(I64, zstep));
        let tbase: Vec<Value> = (0..3).map(|c| self.ld(I64, off!(gll.tbase) + 8 * c)).collect();
        let tstep: Vec<Value> = (0..3).map(|c| self.ld(I64, off!(gll.tstep) + 8 * c)).collect();
        let (fbase, fstep) = (g!(I32, fbase), g!(I32, fstep));
        let (a0, b0, slope) = (g!(F64, a0), g!(F64, b0), g!(F64, slope));
        let (width, dir, p0, p1) = (g!(I32, width), g!(I32, dir), g!(I32, p0), g!(I32, p1));
        let xmajor = g!(I32, xmajor);
        let xmajor = self.b.ins().icmp_imm_s(IntCC::NotEqual, xmajor, 0);
        let (pattern, repeat) = (g!(I32, pattern), g!(I32, repeat));
        let pos0 = self.ld_rw(I32, off!(stipple_pos));
        let zero = self.i32c(0);
        let (pv, posv, kv) = (self.var(I32, p0), self.var(I32, pos0), self.var(I32, zero));
        let head = self.b.create_block();
        let next = self.b.create_block();
        let exit = self.b.create_block();
        self.b.ins().jump(head, &[]);
        self.b.switch_to_block(head);
        let p = self.get(pv);
        let done = self.b.ins().icmp(IntCC::Equal, p, p1);
        self.bail(done, exit);
        let pf = self.itof(p);
        let half = self.f64c(0.5);
        let pc = self.b.ins().fadd(pf, half);
        let d = self.b.ins().fsub(pc, a0);
        let t = self.b.ins().fabs(d);
        if k.stipple {
            let pos = self.get(posv);
            let q = self.b.ins().udiv(pos, repeat);
            let bit = self.b.ins().band_imm_s(q, 15);
            let pos1 = self.b.ins().iadd_imm_s(pos, 1);
            self.set(posv, pos1);
            let c15 = self.i32c(15);
            let sh = self.b.ins().isub(c15, bit);
            let on = self.b.ins().ushr(pattern, sh);
            let on = self.b.ins().band_imm_s(on, 1);
            self.need(on, next);
        }
        let sd = self.b.ins().fmul(slope, d);
        let bb = self.b.ins().fadd(b0, sd);
        let wf = self.itof(width);
        let hw = self.b.ins().fmul(half, wf);
        let q = self.b.ins().fsub(bb, hw);
        let q = self.b.ins().fadd(q, half);
        let q = self.b.ins().floor(q);
        let q0 = self.b.ins().fcvt_to_sint_sat(I32, q);
        // Colours and fog `base + step k`, k pixels along (12.16).
        let kk = self.get(kv);
        let rgba: Vec<Variable> = (0..4)
            .map(|c| {
                let m = self.b.ins().imul(cstep[c], kk);
                let v = self.b.ins().iadd(cbase[c], m);
                self.var(I32, v)
            })
            .collect();
        let rgba: [Variable; 4] = rgba.try_into().unwrap();
        if k.tex.is_some() {
            // Steps along the line, none across it.
            let k64 = self.b.ins().sextend(I64, kk);
            let v: [Value; 3] = [0, 1, 2].map(|c| {
                let m = self.b.ins().imul(tstep[c], k64);
                self.b.ins().iadd(tbase[c], m)
            });
            let zero = self.i64c(0);
            let d = [0, 1, 2].map(|c| [tstep[c], zero]);
            self.textured(v, d, rgba);
        }
        let mut c: [Value; 4] = rgba.map(|v| self.get(v));
        if k.fog {
            let m = self.b.ins().imul(fstep, kk);
            let f = self.b.ins().iadd(fbase, m);
            c = self.fogged(c, f);
        }
        let k64 = self.b.ins().sextend(I64, kk);
        let zm = self.b.ins().imul(zstep, k64);
        let z = self.b.ins().iadd(zbase, zm);
        let qend = self.b.ins().iadd(q0, width);
        self.for_range(q0, qend, |e, q, cont| {
            let wx = e.b.ins().select(xmajor, p, q);
            let wy = e.b.ins().select(xmajor, q, p);
            let (fx, fy) = e.to_fb(wx, wy);
            let r = e.row(fy, cont);
            e.fragment(&r, fx, c, z, cont);
        });
        self.b.ins().jump(next, &[]);
        self.b.switch_to_block(next);
        self.b.seal_block(next);
        let p = self.get(pv);
        let p = self.b.ins().iadd(p, dir);
        self.set(pv, p);
        let kk = self.get(kv);
        let kk = self.b.ins().iadd_imm_s(kk, 1);
        self.set(kv, kk);
        self.b.ins().jump(head, &[]);
        self.b.seal_block(head);
        self.b.switch_to_block(exit);
        self.b.seal_block(exit);
        let pos = self.get(posv);
        self.st(pos, off!(stipple_pos));
    }

    // ── 12.16 colour arithmetic (`mgras::fixed`) ─────────────────────────

    /// `fixed::mul`.
    fn fmul16(&mut self, a: Value, b: Value) -> Value {
        let a = self.b.ins().sextend(I64, a);
        let b = self.b.ins().sextend(I64, b);
        let p = self.b.ins().imul(a, b);
        let p = self.b.ins().sshr_imm_s(p, 28);
        self.b.ins().ireduce(I32, p)
    }

    /// `fixed::clamp`.
    fn fclamp16(&mut self, v: Value) -> Value {
        let zero = self.i32c(0);
        let v = self.b.ins().smax(v, zero);
        let max = self.i32c(fixed::MAX as i64);
        self.b.ins().smin(v, max)
    }

    /// `fixed::ONE - v`.
    fn fone_minus(&mut self, v: Value) -> Value {
        let one = self.i32c(fixed::ONE as i64);
        self.b.ins().isub(one, v)
    }

    /// `Rss::fogged`.
    fn fogged(&mut self, rgba: [Value; 4], f: Value) -> [Value; 4] {
        let zero = self.i32c(0);
        let one = self.i32c(fixed::ONE as i64);
        let f = self.b.ins().smax(f, zero);
        let f = self.b.ins().smin(f, one);
        let nf = self.b.ins().isub(one, f);
        let mut out = rgba;
        for (k, o) in out.iter_mut().enumerate().take(3) {
            let fc = self.inv(off!(gl.fog_c) + 4 * k as i32);
            let a = self.fmul16(f, rgba[k]);
            let b = self.fmul16(nf, fc);
            *o = self.b.ins().iadd(a, b);
        }
        out
    }

    /// `Rss::gl_fragment` at window (wx, wy); a fragment that fails a test
    /// goes on to `kill`.
    fn fragment(&mut self, r: &Row, fx: Value, rgba: [Value; 4], z: Value, kill: Block) {
        let k = self.k;
        self.visible_x(r, fx, kill);
        let rgba = rgba.map(|c| self.fclamp16(c));
        if let Some(func) = k.alpha {
            // Both non-negative 12.16.
            let aref = self.inv(off!(gl.aref));
            let pass = self.icompare(func, rgba[3], aref);
            self.need(pass, kill);
        }
        if k.stencil.is_some() || k.z.is_some() {
            let (za, _) = self.locate_x(r.z.unwrap(), fx, false);
            let zst = self.get_px(za, None);
            let s_old = self.b.ins().ushr_imm_s(zst, 24);
            let s_old = self.b.ins().band_imm_s(s_old, 0xFF);
            let s_old = self.b.ins().ireduce(I32, s_old);
            let zold = self.b.ins().band_imm_s(zst, 0xFF_FFFF);
            // `fixed::zbuf`: z.12 rounded to 24 bits.
            let zr = self.b.ins().iadd_imm_s(z, 1 << (fixed::Z_FRAC - 1));
            let zr = self.b.ins().sshr_imm_s(zr, fixed::Z_FRAC as i64);
            let zero = self.i64c(0);
            let zr = self.b.ins().smax(zr, zero);
            let top = self.i64c(0xFF_FFFF);
            let zi = self.b.ins().smin(zr, top);
            let sref = self.inv(off!(gl.sref));
            let true_ = self.b.ins().iconst(I8, 1);
            let spass = match k.stencil {
                Some(s) => {
                    let cmask = self.inv(off!(gl.scmask));
                    let a = self.b.ins().band(sref, cmask);
                    let b = self.b.ins().band(s_old, cmask);
                    self.icompare(s.func, a, b)
                }
                None => true_,
            };
            let zpass = match k.z {
                Some(f) => {
                    let c = self.icompare(f, zi, zold);
                    self.b.ins().band(spass, c)
                }
                None => spass,
            };
            let mut new = zst;
            if let Some(s) = k.stencil {
                let wmask = self.inv(off!(gl.swmask));
                let op_val = |e: &mut Self, op: u8| -> Value {
                    match op {
                        1 => e.i32c(0),
                        2 => sref,
                        3 => {
                            let inc = e.b.ins().iadd_imm_s(s_old, 1);
                            let m = e.i32c(0xFF);
                            e.b.ins().umin(inc, m)
                        }
                        4 => {
                            let one = e.i32c(1);
                            let d = e.b.ins().isub(s_old, one);
                            let z = e.b.ins().icmp_imm_s(IntCC::Equal, s_old, 0);
                            let zero = e.i32c(0);
                            e.b.ins().select(z, zero, d)
                        }
                        5 => {
                            let n = e.b.ins().bnot(s_old);
                            e.b.ins().band_imm_s(n, 0xFF)
                        }
                        _ => s_old,
                    }
                };
                let (vf, vzf, vzp) = (op_val(self, s.fail), op_val(self, s.zfail), op_val(self, s.zpass));
                let inner = self.b.ins().select(zpass, vzp, vzf);
                let s_new = self.b.ins().select(spass, inner, vf);
                let keep = self.b.ins().band_not(s_old, wmask);
                let put = self.b.ins().band(s_new, wmask);
                let sv = self.b.ins().bor(keep, put);
                let sv = self.b.ins().uextend(I64, sv);
                let sv = self.b.ins().ishl_imm_s(sv, 24);
                let cleared = self.b.ins().band_imm_s(new, !(0xFFi64 << 24));
                new = self.b.ins().bor(cleared, sv);
            }
            if k.z.is_some() {
                let zmask = self.inv(off!(gl.zmask));
                let keep = self.b.ins().band_not(new, zmask);
                let put = self.b.ins().band(zi, zmask);
                let wz = self.b.ins().bor(keep, put);
                new = self.b.ins().select(zpass, wz, new);
            }
            self.put_px(za, None, new);
            self.need(zpass, kill);
        }
        self.fragment_color(r, 0, fx, rgba);
        if k.draw == Draw::Dual {
            self.fragment_color(r, 1, fx, rgba);
        }
    }

    /// A 12-bit pair's half as 8-8-8-8, nibbles repeated, alpha 0xFF
    /// (`Rss::fragment_color`).
    fn rgb12_half(&mut self, w: Value, b: bool) -> Value {
        let c = if b { self.b.ins().ushr_imm_s(w, 12) } else { w };
        let r = self.b.ins().band_imm_s(c, 0xF);
        let g = self.b.ins().band_imm_s(c, 0xF0);
        let g = self.b.ins().ishl_imm_s(g, 4);
        let bl = self.b.ins().band_imm_s(c, 0xF00);
        let bl = self.b.ins().ishl_imm_s(bl, 8);
        let n = self.b.ins().bor(r, g);
        let n = self.b.ins().bor(n, bl);
        let hi = self.b.ins().ishl_imm_s(n, 4);
        let n = self.b.ins().bor(n, hi);
        self.b.ins().bor_imm_u(n, 0xFF00_0000)
    }

    /// `Rss::fragment_color`: blend (12.16) or not, shift down, write.
    fn fragment_color(&mut self, r: &Row, t: usize, x: Value, rgba: [Value; 4]) {
        let k = self.k;
        let src = if k.rgb {
            let c = match k.blend {
                Some(bl) => {
                    let overlay = t == 0 && k.draw == Draw::Overlay;
                    let (a, sh) = self.locate_x(r.tgt[t], x, overlay);
                    let dst = self.get_px(a, sh);
                    let dst = self.b.ins().ireduce(I32, dst);
                    let dst = if k.pix.pair() && !overlay { self.rgb12_half(dst, k.pix.pair_b()) } else { dst };
                    // Destination bytes widened to 12 bits, 12.16.
                    let d: Vec<Value> = (0..4)
                        .map(|c| {
                            let v = self.b.ins().ushr_imm_s(dst, 8 * c);
                            let v = self.b.ins().band_imm_s(v, 0xFF);
                            let hi = self.b.ins().ishl_imm_s(v, 4);
                            let lo = self.b.ins().ushr_imm_s(v, 4);
                            let w = self.b.ins().bor(hi, lo);
                            self.b.ins().ishl_imm_s(w, 16)
                        })
                        .collect();
                    let nd3 = self.fone_minus(d[3]);
                    let sat = self.b.ins().smin(rgba[3], nd3);
                    let mut out = [rgba[0]; 4];
                    for (kk, o) in out.iter_mut().enumerate() {
                        let f = |e: &mut Self, code: u8| -> Value {
                            match code {
                                0 => e.i32c(0),
                                2 => rgba[kk],
                                3 => e.fone_minus(rgba[kk]),
                                4 => rgba[3],
                                5 => e.fone_minus(rgba[3]),
                                6 => d[3],
                                7 => e.fone_minus(d[3]),
                                8 => d[kk],
                                9 => e.fone_minus(d[kk]),
                                10 if kk != 3 => sat,
                                _ => e.i32c(fixed::ONE as i64),
                            }
                        };
                        let fs = f(self, bl.src);
                        let fd = f(self, bl.dst);
                        let a = self.fmul16(rgba[kk], fs);
                        let b = self.fmul16(d[kk], fd);
                        let s = self.b.ins().iadd(a, b);
                        *o = self.fclamp16(s);
                    }
                    out
                }
                None => rgba,
            };
            // Clamped 12.16 shifted down to bytes (`fixed::to_byte`).
            let mut src = self.i32c(0);
            for (kk, v) in c.iter().enumerate() {
                let q = self.b.ins().ushr_imm_s(*v, 20);
                let q = if kk > 0 { self.b.ins().ishl_imm_s(q, 8 * kk as i64) } else { q };
                src = self.b.ins().bor(src, q);
            }
            src
        } else {
            // `fixed::to_index`.
            let q = self.b.ins().ushr_imm_s(rgba[0], 16);
            self.b.ins().band_imm_s(q, 0xFFF)
        };
        self.put_in(r, t, x, src);
    }

    // ── texturing (rss.rs `textured`, te1.rs, `mgras::fixed`) ───────────

    /// i64 `(a * b) >> sh` through 128 bits (`sh` an I64 value), wrapping
    /// back to 64. `b_unsigned`: b is a u64.
    fn mul_shr128(&mut self, a: Value, b: Value, b_unsigned: bool, sh: Value) -> Value {
        let a = self.b.ins().sextend(ir::types::I128, a);
        let b = if b_unsigned { self.b.ins().uextend(ir::types::I128, b) } else { self.b.ins().sextend(ir::types::I128, b) };
        let p = self.b.ins().imul(a, b);
        let p = self.b.ins().sshr(p, sh);
        self.b.ins().ireduce(I64, p)
    }

    /// `63 - leading_zeros(v)` (I64).
    fn top_bit(&mut self, v: Value) -> Value {
        let z = self.b.ins().clz(v);
        let c = self.i64c(63);
        self.b.ins().isub(c, z)
    }

    /// `fixed::recip`: (y, n) for wi > 0.
    fn recip(&mut self, wi: Value) -> (Value, Value) {
        let n = self.top_bit(wi);
        let up = self.b.ins().icmp_imm_s(IntCC::SignedGreaterThanOrEqual, n, 30);
        let c30 = self.i64c(30);
        let r = self.b.ins().isub(n, c30);
        let l = self.b.ins().isub(c30, n);
        let xr = self.b.ins().ushr(wi, r);
        let xl = self.b.ins().ishl(wi, l);
        let x = self.b.ins().select(up, xr, xl);
        let idx = self.b.ins().ushr_imm_s(x, 20);
        let idx = self.b.ins().band_imm_s(idx, 0x3FF);
        let off = self.b.ins().ishl_imm_s(idx, 2);
        let table = self.b.ins().iconst(I64, fixed::RECIP.as_ptr() as i64);
        let a = self.b.ins().iadd(table, off);
        let y0 = self.b.ins().uload32(self.mr, a, 0);
        let e = self.b.ins().imul(x, y0);
        let e = self.b.ins().ushr_imm_s(e, 30);
        let two = self.i64c(1 << 32);
        let d = self.b.ins().isub(two, e);
        let y = self.b.ins().imul(y0, d);
        let y = self.b.ins().ushr_imm_s(y, 31);
        (y, n)
    }

    /// `fixed::log2_q8` of v > 0 (I64) as I32.
    fn log2_q8(&mut self, v: Value) -> Value {
        let n = self.top_bit(v);
        let up = self.b.ins().icmp_imm_s(IntCC::SignedGreaterThanOrEqual, n, 8);
        let c8 = self.i64c(8);
        let r = self.b.ins().isub(n, c8);
        let l = self.b.ins().isub(c8, n);
        let fr = self.b.ins().ushr(v, r);
        let fl = self.b.ins().ishl(v, l);
        let f = self.b.ins().select(up, fr, fl);
        let f = self.b.ins().band_imm_s(f, 0xFF);
        let off = self.b.ins().ishl_imm_s(f, 1);
        let table = self.b.ins().iconst(I64, fixed::log2_table().as_ptr() as i64);
        let a = self.b.ins().iadd(table, off);
        let t = self.b.ins().uload16(I32, self.mr, a, 0);
        let n = self.b.ins().ireduce(I32, n);
        let n = self.b.ins().ishl_imm_s(n, 8);
        self.b.ins().iadd(n, t)
    }

    /// `Rss::textured`: replaces `rgba` with the textured colour unless 1/W
    /// is not positive. `v`: S/W, T/W, 1/W (2^32); `d`: their d/dx and
    /// d/(-y).
    fn textured(&mut self, v: [Value; 3], d: [[Value; 2]; 3], rgba: [Variable; 4]) {
        let tex = self.k.tex.unwrap();
        let [sw, tw, wi] = v;
        let join = self.b.create_block();
        let nonpos = self.b.ins().icmp_imm_s(IntCC::SignedLessThanOrEqual, wi, 0);
        self.bail(nonpos, join);
        let (y, n) = self.recip(wi);
        let s = self.mul_shr128(sw, y, true, n);
        let t = self.mul_shr128(tw, y, true, n);
        // Without mipmaps, and with one filter for both, the level of
        // detail decides nothing.
        let lambda = if !tex.mipmap && tex.mag_linear == tex.min_linear {
            self.i32c(0)
        } else {
            let ls = self.inv(off!(gl.smp.ls));
            let lt = self.inv(off!(gl.smp.lt));
            let c31 = self.i64c(31);
            let scale31 = |e: &mut Self, c: Value, k: Value| e.mul_shr128(c, k, false, c31);
            let dx = |e: &mut Self, q: [Value; 2], c: Value| {
                let m = scale31(e, c, d[2][0]);
                e.b.ins().isub(q[0], m)
            };
            let dy = |e: &mut Self, q: [Value; 2], c: Value| {
                let m = scale31(e, c, d[2][1]);
                e.b.ins().isub(m, q[1])
            };
            // `fixed::deriv`: shift n + 15 - l, clamp to +-2^31.
            let deriv = |e: &mut Self, a: Value, l: Value| {
                let l = e.b.ins().uextend(I64, l);
                let sh = e.b.ins().iadd_imm_s(n, 15);
                let sh = e.b.ins().isub(sh, l);
                let r = e.mul_shr128(a, y, true, sh);
                let lo = e.i64c(-(1 << 31));
                let hi = e.i64c(1 << 31);
                let r = e.b.ins().smax(r, lo);
                e.b.ins().smin(r, hi)
            };
            let a = dx(self, d[0], s);
            let dsx = deriv(self, a, ls);
            let a = dx(self, d[1], t);
            let dtx = deriv(self, a, lt);
            let a = dy(self, d[0], s);
            let dsy = deriv(self, a, ls);
            let a = dy(self, d[1], t);
            let dty = deriv(self, a, lt);
            // `fixed::lod_q8`.
            let sq = |e: &mut Self, a: Value, b: Value| {
                let a2 = e.b.ins().imul(a, a);
                let b2 = e.b.ins().imul(b, b);
                e.b.ins().iadd(a2, b2)
            };
            let ax = sq(self, dsx, dtx);
            let ay = sq(self, dsy, dty);
            let m = self.b.ins().umax(ax, ay);
            let empty = self.b.ins().icmp_imm_s(IntCC::Equal, m, 0);
            let one = self.i64c(1);
            let m = self.b.ins().select(empty, one, m);
            let l2 = self.log2_q8(m);
            let l2 = self.b.ins().iadd_imm_s(l2, -(32 << 8));
            let l2 = self.b.ins().sshr_imm_s(l2, 1);
            let none = self.i32c(fixed::LOD_NONE as i64);
            self.b.ins().select(empty, none, l2)
        };
        let texel = self.sample(&tex, s, t, lambda);
        let f = rgba.map(|v| self.get(v));
        let out = self.tex_env(&tex, f, texel);
        for (v, o) in rgba.iter().zip(out) {
            self.set(*v, o);
        }
        self.b.ins().jump(join, &[]);
        self.b.switch_to_block(join);
        self.b.seal_block(join);
    }

    /// `te1::tex_env`, 12.16.
    fn tex_env(&mut self, tex: &Tex, f: [Value; 4], tx: [Value; 4]) -> [Value; 4] {
        let nc = tex.nc;
        let (ct, at): (Option<[Value; 3]>, Option<Value>) = match tex.class {
            1 => (None, Some(tx[0])),
            2 => (Some([tx[0]; 3]), (nc == 2).then_some(tx[1])),
            3 => (Some([tx[0]; 3]), Some(tx[0])),
            _ if nc >= 3 => (Some([tx[0], tx[1], tx[2]]), (nc == 4).then_some(tx[3])),
            _ => (Some([tx[0]; 3]), (nc == 2).then_some(tx[1])),
        };
        let mut out = f;
        // `a (1 - w) + b w`
        let lerp = |e: &mut Self, a: Value, b: Value, w: Value| {
            let nw = e.fone_minus(w);
            let x = e.fmul16(a, nw);
            let y = e.fmul16(b, w);
            e.b.ins().iadd(x, y)
        };
        match tex.env & 3 {
            1 => {
                if let Some(ct) = ct {
                    let a = at.unwrap_or_else(|| self.i32c(fixed::ONE as i64));
                    for k in 0..3 {
                        out[k] = lerp(self, f[k], ct[k], a);
                    }
                }
            }
            2 => {
                if let Some(ct) = ct {
                    for k in 0..3 {
                        let cc = self.inv(off!(gl.env_c) + 4 * k as i32);
                        out[k] = lerp(self, f[k], cc, ct[k]);
                    }
                }
                if let Some(at) = at {
                    out[3] = if tex.class == 3 {
                        let ac = self.inv(off!(gl.env_a));
                        lerp(self, f[3], ac, at)
                    } else {
                        self.fmul16(f[3], at)
                    };
                }
            }
            3 => {
                if let Some(at) = at {
                    out[3] = self.fmul16(f[3], at);
                }
            }
            _ => {
                if let Some(ct) = ct {
                    for k in 0..3 {
                        out[k] = self.fmul16(f[k], ct[k]);
                    }
                }
                if let Some(at) = at {
                    out[3] = self.fmul16(f[3], at);
                }
            }
        }
        out
    }

    /// `Sampler::sample`: the filtered texel (12.16) at (s, t) (Q31) for
    /// lambda (Q8).
    fn sample(&mut self, tex: &Tex, s: Value, t: Value, lambda: Value) -> [Value; 4] {
        if !tex.mipmap && tex.mag_linear == tex.min_linear {
            let l0 = self.i32c(0);
            return self.level(tex, l0, s, t, tex.mag_linear);
        }
        let zi = self.i32c(0);
        let res: [Variable; 4] = [0, 1, 2, 3].map(|_| self.var(I32, zi));
        let join = self.b.create_block();
        let minify = self.b.ins().icmp_imm_s(IntCC::SignedGreaterThan, lambda, 0);
        let magb = self.b.create_block();
        let minb = self.b.create_block();
        self.b.ins().brif(minify, minb, &[], magb, &[]);
        self.b.switch_to_block(magb);
        self.b.seal_block(magb);
        let l0 = self.i32c(0);
        let r = self.level(tex, l0, s, t, tex.mag_linear);
        self.def_all(&res, r);
        self.b.ins().jump(join, &[]);
        self.b.switch_to_block(minb);
        self.b.seal_block(minb);
        if !tex.mipmap {
            let l0 = self.i32c(0);
            let r = self.level(tex, l0, s, t, tex.min_linear);
            self.def_all(&res, r);
            self.b.ins().jump(join, &[]);
        } else {
            let ml = self.inv(off!(gl.smp.max_level));
            let flat = self.b.ins().icmp_imm_s(IntCC::Equal, ml, 0);
            let flatb = self.b.create_block();
            let mipb = self.b.create_block();
            self.b.ins().brif(flat, flatb, &[], mipb, &[]);
            self.b.switch_to_block(flatb);
            self.b.seal_block(flatb);
            let l0 = self.i32c(0);
            let r = self.level(tex, l0, s, t, tex.min_linear);
            self.def_all(&res, r);
            self.b.ins().jump(join, &[]);
            self.b.switch_to_block(mipb);
            self.b.seal_block(mipb);
            let zero = self.i32c(0);
            let clamp_top = |e: &mut Self, v: Value| {
                let v = e.b.ins().smax(v, zero);
                e.b.ins().smin(v, ml)
            };
            if !tex.mip_linear {
                // ceil(lambda + 0.5) - 1
                let l = self.b.ins().iadd_imm_s(lambda, 128 + 255);
                let l = self.b.ins().sshr_imm_s(l, 8);
                let l = self.b.ins().iadd_imm_s(l, -1);
                let l = clamp_top(self, l);
                let r = self.level(tex, l, s, t, tex.min_linear);
                self.def_all(&res, r);
            } else {
                let l0 = self.b.ins().sshr_imm_s(lambda, 8);
                let l0 = clamp_top(self, l0);
                let l1 = self.b.ins().iadd_imm_s(l0, 1);
                let l1 = self.b.ins().smin(l1, ml);
                let base = self.b.ins().ishl_imm_s(l0, 8);
                let f = self.b.ins().isub(lambda, base);
                let f = self.b.ins().smax(f, zero);
                let c256 = self.i32c(256);
                let f = self.b.ins().smin(f, c256);
                let f = self.b.ins().sextend(I64, f);
                let a = self.level(tex, l0, s, t, tex.min_linear);
                let b = self.level(tex, l1, s, t, tex.min_linear);
                let mut r = a;
                for k in 0..4 {
                    let d = self.b.ins().isub(b[k], a[k]);
                    let d = self.b.ins().sextend(I64, d);
                    let m = self.b.ins().imul(d, f);
                    let m = self.b.ins().sshr_imm_s(m, 8);
                    let m = self.b.ins().ireduce(I32, m);
                    r[k] = self.b.ins().iadd(a[k], m);
                }
                self.def_all(&res, r);
            }
            self.b.ins().jump(join, &[]);
        }
        self.b.switch_to_block(join);
        self.b.seal_block(join);
        res.map(|v| self.get(v))
    }

    fn def_all(&mut self, vars: &[Variable; 4], vals: [Value; 4]) {
        for (v, x) in vars.iter().zip(vals) {
            self.set(*v, x);
        }
    }

    /// `Sampler::level`: one level at (s, t) (Q31), nearest or bilinear,
    /// in 12.16.
    fn level(&mut self, tex: &Tex, level: Value, s: Value, t: Value, linear: bool) -> [Value; 4] {
        let ls = self.inv(off!(gl.smp.ls));
        let lt = self.inv(off!(gl.smp.lt));
        let sat_sub = |e: &mut Self, a: Value| {
            let gt = e.b.ins().icmp(IntCC::UnsignedGreaterThan, a, level);
            let d = e.b.ins().isub(a, level);
            let zero = e.i32c(0);
            e.b.ins().select(gt, d, zero)
        };
        let lw = sat_sub(self, ls);
        let lh = sat_sub(self, lt);
        let one = self.i64c(1);
        let lw64 = self.b.ins().uextend(I64, lw);
        let lh64 = self.b.ins().uextend(I64, lh);
        let wi = self.b.ins().ishl(one, lw64);
        let hi = self.b.ins().ishl(one, lh64);
        let clamp_unit = |e: &mut Self, v: Value| {
            let zero = e.i64c(0);
            let top = e.i64c(1 << 31);
            let v = e.b.ins().smax(v, zero);
            e.b.ins().smin(v, top)
        };
        let s = if tex.gl_clamp && tex.clamp_s { clamp_unit(self, s) } else { s };
        let t = if tex.gl_clamp && tex.clamp_t { clamp_unit(self, t) } else { t };
        // Texels in Q16: s >> (15 - lw).
        let c15 = self.i64c(15);
        let shs = self.b.ins().isub(c15, lw64);
        let sht = self.b.ins().isub(c15, lh64);
        let u = self.b.ins().sshr(s, shs);
        let v = self.b.ins().sshr(t, sht);
        // The level's pages and shared-page offsets, once for its texels.
        let lside = self.b.ins().umax(lw, lh);
        let lside = self.b.ins().uextend(I64, lside);
        let lsx4 = self.b.ins().ishl_imm_s(lside, 2);
        let tab = self.b.ins().iadd(self.ctx, lsx4);
        let lvl = self.b.ins().band_imm_s(level, 15);
        let lvl = self.b.ins().uextend(I64, lvl);
        let lvx4 = self.b.ins().ishl_imm_s(lvl, 2);
        let pt = self.b.ins().iadd(self.ctx, lvx4);
        let ld64 = |e: &mut Self, base: Value, off: i32| {
            let v = e.b.ins().load(I32, e.mr, base, off);
            e.b.ins().uextend(I64, v)
        };
        let page = ld64(self, pt, off!(gl.smp.pages));
        let off = ld64(self, tab, off!(gl.small_off));
        let border = (tex.clamp_s || tex.clamp_t) && !tex.no_border;
        let (bpage, boff) = if border {
            (ld64(self, pt, off!(gl.smp.border_pages)), ld64(self, tab, off!(gl.small_boff)))
        } else {
            (page, off)
        };
        let geo = Geo { wi, hi, page, off, bpage, boff };
        if !linear {
            let cap = |e: &mut Self, x: Value, n: Value, clamp: bool| {
                let i = e.b.ins().sshr_imm_s(x, 16);
                if clamp && tex.gl_clamp {
                    let last = e.b.ins().iadd_imm_s(n, -1);
                    e.b.ins().smin(i, last)
                } else {
                    i
                }
            };
            let i = cap(self, u, wi, tex.clamp_s);
            let j = cap(self, v, hi, tex.clamp_t);
            return self.texel(tex, &geo, i, j).map(|c| self.b.ins().ishl_imm_s(c, 16));
        }
        let u = self.b.ins().iadd_imm_s(u, -0x8000);
        let v = self.b.ins().iadd_imm_s(v, -0x8000);
        let i = self.b.ins().sshr_imm_s(u, 16);
        let j = self.b.ins().sshr_imm_s(v, 16);
        // 8-bit weights: the four products sum to 12.16.
        let a = self.b.ins().ushr_imm_s(u, 8);
        let a = self.b.ins().band_imm_s(a, 0xFF);
        let a = self.b.ins().ireduce(I32, a);
        let b = self.b.ins().ushr_imm_s(v, 8);
        let b = self.b.ins().band_imm_s(b, 0xFF);
        let b = self.b.ins().ireduce(I32, b);
        let i1 = self.b.ins().iadd_imm_s(i, 1);
        let j1 = self.b.ins().iadd_imm_s(j, 1);
        let t00 = self.texel(tex, &geo, i, j);
        let t10 = self.texel(tex, &geo, i1, j);
        let t01 = self.texel(tex, &geo, i, j1);
        let t11 = self.texel(tex, &geo, i1, j1);
        let c256 = self.i32c(256);
        let na = self.b.ins().isub(c256, a);
        let nb = self.b.ins().isub(c256, b);
        let w00 = self.b.ins().imul(na, nb);
        let w10 = self.b.ins().imul(a, nb);
        let w01 = self.b.ins().imul(na, b);
        let w11 = self.b.ins().imul(a, b);
        let mut c = t00;
        for k in 0..4 {
            let x = self.b.ins().imul(w00, t00[k]);
            let y = self.b.ins().imul(w10, t10[k]);
            let s = self.b.ins().iadd(x, y);
            let z = self.b.ins().imul(w01, t01[k]);
            let s = self.b.ins().iadd(s, z);
            let q = self.b.ins().imul(w11, t11[k]);
            c[k] = self.b.ins().iadd(s, q);
        }
        c
    }

    /// `Sampler::texel`: texel (i, j) of a level, wrapped or clamped, from
    /// TRAM, its border, or the border colour. Branch-free: an address
    /// that will not be used still lies in TRAM.
    fn texel(&mut self, tex: &Tex, g: &Geo, i: Value, j: Value) -> [Value; 4] {
        let axis = |e: &mut Self, v: Value, n: Value, clamp: bool| {
            if clamp {
                let m1 = e.i64c(-1);
                let v = e.b.ins().smax(v, m1);
                e.b.ins().smin(v, n)
            } else {
                let m = e.b.ins().iadd_imm_s(n, -1);
                e.b.ins().band(v, m)
            }
        };
        let s = axis(self, i, g.wi, tex.clamp_s);
        let t = axis(self, j, g.hi, tex.clamp_t);
        let out_axis = |e: &mut Self, v: Value, n: Value| {
            let lo = e.b.ins().icmp_imm_s(IntCC::SignedLessThan, v, 0);
            let hi = e.b.ins().icmp(IntCC::SignedGreaterThanOrEqual, v, n);
            e.b.ins().bor(lo, hi)
        };
        let outside = match (tex.clamp_s, tex.clamp_t) {
            (false, false) => None,
            (true, false) => Some(out_axis(self, s, g.wi)),
            (false, true) => Some(out_axis(self, t, g.hi)),
            (true, true) => {
                let a = out_axis(self, s, g.wi);
                let b = out_axis(self, t, g.hi);
                Some(self.b.ins().bor(a, b))
            }
        };
        let d = tex.depth as i64;
        let tn = (4 * d as usize).next_power_of_two() as i64;
        // texel_addr
        let tw = self.b.ins().imul(t, g.wi);
        let idx = self.b.ins().iadd(g.off, tw);
        let idx = self.b.ins().iadd(idx, s);
        let idx = self.b.ins().imul_imm_s(idx, tn);
        let pb = self.b.ins().imul_imm_s(g.page, PAGE_NIBBLES as i64);
        let addr = self.b.ins().iadd(pb, idx);
        let addr = match outside {
            Some(out) if !tex.no_border => {
                // border_addr
                let (bpage, boff) = (g.bpage, g.boff);
                let w2 = self.b.ins().iadd_imm_s(g.wi, 2);
                let bottom = self.b.ins().iadd_imm_s(s, 1);
                let top = self.b.ins().iadd(w2, bottom);
                let w22 = self.b.ins().iadd(w2, w2);
                let left = self.b.ins().iadd(w22, t);
                let right = self.b.ins().iadd(left, g.hi);
                let tlo = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, t, 0);
                let thi = self.b.ins().icmp(IntCC::SignedGreaterThanOrEqual, t, g.hi);
                let slo = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, s, 0);
                let side = self.b.ins().select(slo, left, right);
                let x = self.b.ins().select(thi, top, side);
                let bidx = self.b.ins().select(tlo, bottom, x);
                let bi = self.b.ins().iadd(boff, bidx);
                let bi = self.b.ins().imul_imm_s(bi, tn);
                let bp = self.b.ins().imul_imm_s(bpage, PAGE_NIBBLES as i64);
                let bp = self.b.ins().iadd_imm_s(bp, PAGE_NIBBLES as i64 / 2);
                let baddr = self.b.ins().iadd(bp, bi);
                self.b.ins().select(out, baddr, addr)
            }
            _ => addr,
        };
        let addr = self.b.ins().band_imm_s(addr, TRAM_NIBBLES as i64 - 1);
        let nc = tex.nc as usize;
        let slot = match nc {
            1 => tex.slot as i64 & 3,
            2 => (tex.slot as i64 & 1) * 2,
            _ => 0,
        };
        // A cell is `tn` nibbles at a multiple of `tn` (pages, the border
        // half and cells all are), so it never wraps TRAM and loads whole:
        // nibble j at bits 4j..4j+3 of the little-endian word.
        let byte = self.b.ins().ushr_imm_s(addr, 1);
        let p = self.b.ins().iadd(self.tram, byte);
        let cty = match tn {
            4 => ir::types::I16,
            8 => I32,
            _ => I64,
        };
        let le = self.mr.with_endianness(ir::Endianness::Little);
        let cell = self.b.ins().load(cty, le, p, 0);
        let cell = if cty == I64 { cell } else { self.b.ins().uextend(I64, cell) };
        // Components widened to 12 bits (`fixed::widen`).
        let zero = self.i32c(0);
        let mut c = [zero; 4];
        for (k, o) in c.iter_mut().enumerate().take(nc) {
            let co = (slot + k as i64).min(3) * d;
            let v = self.b.ins().ushr_imm_s(cell, 4 * co);
            let v = self.b.ins().band_imm_s(v, (1i64 << (4 * d)) - 1);
            let v = self.b.ins().ireduce(I32, v);
            *o = match d {
                1 => self.b.ins().imul_imm_s(v, 0x111),
                2 => {
                    let hi = self.b.ins().ishl_imm_s(v, 4);
                    let lo = self.b.ins().ushr_imm_s(v, 4);
                    self.b.ins().bor(hi, lo)
                }
                _ => v,
            };
        }
        if let (Some(out), true) = (outside, tex.no_border) {
            for (k, o) in c.iter_mut().enumerate() {
                let bc = self.inv(off!(gl.smp.border_color) + 4 * k as i32);
                *o = self.b.ins().select(out, bc, *o);
            }
        }
        c
    }
}

/// One framebuffer row's share of visibility and addressing (`E::row`):
/// its clip-ID row, each screen mask's y test and keep flag, and each
/// target's (and the depth buffer's) row page and offset.
struct Row {
    cid: Option<Value>,
    clip: Vec<(Value, Value)>,
    tgt: [(Value, Value); 2],
    z: Option<(Value, Value)>,
    /// Dithering: the Bayer matrix row, (fy & 3) << 2.
    bayer: Option<Value>,
}

/// A level's geometry (i64): its size, its page and offset in it, and its
/// border page and offset.
struct Geo {
    wi: Value,
    hi: Value,
    page: Value,
    off: Value,
    bpage: Value,
    boff: Value,
}
