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
//! - The level of detail (`hypot`, `log2`) is computed by a Rust helper
//!   with the interpreter's expression; component and byte scaling come
//!   from tables filled by the interpreter's divisions.
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
use cranelift_codegen::ir::{self, AbiParam, Block, InstBuilder, MemFlagsData, SigRef, Signature, Type, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use super::{ClipRect, Draw, PipeKey, Pix, Prim, RasterCtx, ShaderFn, Target, Tex, XFmt};
use crate::dev::mgras::pixmem::{PAGES, PAGE_WORDS, TILE_H, TILE_W, WORD_MASK};
use crate::dev::mgras::rss::WIDTH;
use crate::dev::mgras::te1::{PAGE_NIBBLES, TRAM_NIBBLES};

macro_rules! off {
    ($($f:tt)+) => {
        offset_of!(RasterCtx, $($f)+) as i32
    };
}

/// `Rss::textured`'s level of detail from the footprint's scaled
/// derivatives, the interpreter's very expression. One call, not three: a
/// call clobbers every float register the shader holds.
extern "C" fn jit_lambda(sx: f64, tx: f64, sy: f64, ty: f64) -> f64 {
    let rho = sx.hypot(tx).max(sy.hypot(ty));
    if rho > 0.0 { rho.log2() } else { f64::NEG_INFINITY }
}

/// Component values as `Te1::component` computes them, by depth in
/// nibbles: `v as f32 / (2^(4d) - 1) as f32`, the very division, done once.
/// A load instead of a division per component.
fn comp_table(d: usize) -> &'static [f32] {
    static T: std::sync::OnceLock<[Vec<f32>; 3]> = std::sync::OnceLock::new();
    let t = T.get_or_init(|| {
        [1usize, 2, 3].map(|d| {
            let n = 1u32 << (4 * d);
            (0..n).map(|v| v as f32 / (n - 1) as f32).collect()
        })
    });
    &t[d - 1]
}

/// A destination byte as `Rss::fragment_color` reads it: `v as f64 / 255.0`.
fn byte_table() -> &'static [f64; 256] {
    static T: std::sync::OnceLock<[f64; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| std::array::from_fn(|v| v as f64 / 255.0))
}

/// The same, when only `lambda > 0` matters: 1 or -1.
extern "C" fn jit_lambda_sign(sx: f64, tx: f64, sy: f64, ty: f64) -> f64 {
    let rho = sx.hypot(tx).max(sy.hypot(ty));
    if rho > 1.0 { 1.0 } else { -1.0 }
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
            let cc = self.module.target_config().default_call_conv;
            let mut e = E::new(b, key, ctx, ptr, cc);
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

/// Where a plane is evaluated: a triangle fragment (`p0 + p1 dx + p2 dy`)
/// or a point along a GL line (`p0 + p1 t`).
#[derive(Clone, Copy)]
enum At {
    Tri { dx: Value, dy: Value },
    Line { t: Value },
}

/// A plane's per-row term `p[2] * dy`, when the row has computed it.
type RowTerm = Option<Value>;

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
    sig4: SigRef,
}

impl<'a> E<'a> {
    fn new(mut b: FunctionBuilder<'a>, k: PipeKey, ctx: Value, ptr: Type, cc: cranelift_codegen::isa::CallConv) -> Self {
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
        let masked = k.draw == Draw::Overlay || matches!(k.pix, Pix::Rgba8 | Pix::Ci12);
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
        // The host's C convention (the ISA's default), for the helpers.
        let mut s4 = Signature::new(cc);
        s4.params.extend([AbiParam::new(F64); 4]);
        s4.returns.push(AbiParam::new(F64));
        let sig4 = b.import_signature(s4);
        let blk = if matches!(k.prim, Prim::Fill | Prim::Stipple | Prim::Xfer) {
            [ld(&mut b, I32, off!(bxs)), ld(&mut b, I32, off!(bys)), ld(&mut b, I32, off!(bdx)), ld(&mut b, I32, off!(bdy))]
        } else {
            [zero; 4]
        };
        let mut invs = std::collections::HashMap::new();
        if k.prim.is_gl() {
            let mut f64s = vec![off!(gl.aref), off!(gl.smp_w), off!(gl.smp_h), off!(gl.env_a)];
            f64s.extend((0..3).map(|k| off!(gl.env_c) + 8 * k));
            f64s.extend((0..3).map(|k| off!(gl.fog_c) + 8 * k));
            for o in f64s {
                invs.insert(o, ld(&mut b, F64, o));
            }
            for o in [off!(gl.sref), off!(gl.scmask), off!(gl.swmask), off!(gl.smp.max_level), off!(gl.smp.ls), off!(gl.smp.lt)] {
                invs.insert(o, ld(&mut b, I32, o));
            }
            invs.insert(off!(gl.zmask), ld(&mut b, I64, off!(gl.zmask)));
            for k in 0..4 {
                let o = off!(gl.smp.border_color) + 4 * k;
                invs.insert(o, ld(&mut b, F32, o));
            }
        }
        E { b, k, ctx, m, mr, invs, blk, words, cidp, tram, ox, oy, ysign, cidmatch, cidwmask, clips, tgt, zxtiles, sig4 }
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

    fn call4(&mut self, f: extern "C" fn(f64, f64, f64, f64) -> f64, args: [Value; 4]) -> Value {
        let p = self.b.ins().iconst(I64, f as usize as i64);
        let inst = self.b.ins().call_indirect(self.sig4, p, &args);
        self.b.inst_results(inst)[0]
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
        let rep12 = !overlay && k.pix == Pix::Rgb12;
        let masked = overlay || matches!(k.pix, Pix::Rgba8 | Pix::Ci12);
        let [_, _, mask, lop_width] = self.tgt[t];
        let (addr, shift) = self.locate_x(r.tgt[t], x, overlay);
        let old = (k.logic.is_some() || masked).then(|| {
            let w = self.get_px(addr, shift);
            self.b.ins().ireduce(I32, w)
        });
        let mut v = v;
        if let Some(op) = k.logic {
            let r = self.logic_op(op, v, old.unwrap());
            v = self.b.ins().band(r, lop_width);
        }
        if rep12 {
            let n = self.b.ins().band_imm_s(v, 0xF0_F0F0);
            let lo = self.b.ins().ushr_imm_s(n, 4);
            v = self.b.ins().bor(n, lo);
        }
        if masked {
            let keep = self.b.ins().band_not(old.unwrap(), mask);
            let new = self.b.ins().band(v, mask);
            v = self.b.ins().bor(keep, new);
        }
        let v = self.b.ins().uextend(I64, v);
        self.put_px(addr, shift, v);
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
        let mut r = Row { cid, clip, tgt, z };
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

    fn plane(&mut self, base: i32) -> [Value; 3] {
        [self.ld(F64, base), self.ld(F64, base + 8), self.ld(F64, base + 16)]
    }

    fn at(&mut self, p: &[Value; 3], at: At) -> Value {
        self.at_row(p, at, None)
    }

    /// `p0 + p1 dx + p2 dy`, with `p2 dy` from the row when given (the
    /// same product, computed once).
    fn at_row(&mut self, p: &[Value; 3], at: At, row: RowTerm) -> Value {
        match at {
            At::Tri { dx, dy } => {
                let a = self.b.ins().fmul(p[1], dx);
                let s = self.b.ins().fadd(p[0], a);
                let c = row.unwrap_or_else(|| self.b.ins().fmul(p[2], dy));
                self.b.ins().fadd(s, c)
            }
            At::Line { t } => {
                let a = self.b.ins().fmul(p[1], t);
                self.b.ins().fadd(p[0], a)
            }
        }
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
        let planes: Vec<[Value; 3]> = (0..4).map(|c| self.plane(off!(tri.planes) + 24 * c)).collect();
        let zplane = self.plane(off!(tri.zplane));
        let tplanes: Vec<[Value; 3]> =
            if k.tex.is_some() { (0..3).map(|c| self.plane(off!(tri.tplanes) + 24 * c)).collect() } else { Vec::new() };
        let fplane = k.fog.then(|| self.plane(off!(tri.fplane)));
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
            // The rows's share of the colour and depth planes, once a row.
            let terms: Vec<Value> = planes.iter().chain(std::iter::once(&zplane)).map(|p| e.b.ins().fmul(p[2], dy)).collect();
            let terms = e.pin(&terms);
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
                let xf = e.itof(i);
                let half = e.f64c(0.5);
                let xc = e.b.ins().fadd(xf, half);
                let dx = e.b.ins().fsub(xc, xs);
                let at = At::Tri { dx, dy };
                let rgba: Vec<Variable> = (0..4)
                    .map(|c| {
                        let v = e.at_row(&planes[c], at, Some(terms[c]));
                        e.var(F64, v)
                    })
                    .collect();
                let rgba: [Variable; 4] = rgba.try_into().unwrap();
                if k.tex.is_some() {
                    let tp: [[Value; 3]; 3] = [tplanes[0], tplanes[1], tplanes[2]];
                    e.textured(&tp, at, rgba);
                }
                let mut c: [Value; 4] = rgba.map(|v| e.get(v));
                if let Some(fp) = fplane {
                    let f = e.at(&fp, at);
                    c = e.fogged(c, f);
                }
                let z = e.at_row(&zplane, at, Some(terms[4]));
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
        let c0: Vec<Value> = (0..4).map(|c| self.ld(F64, off!(gll.c0) + 8 * c)).collect();
        let dc: Vec<Value> = (0..4).map(|c| self.ld(F64, off!(gll.dc) + 8 * c)).collect();
        let (z0, dz) = (g!(F64, z0), g!(F64, dz));
        let tplanes: Vec<[Value; 3]> =
            if k.tex.is_some() { (0..3).map(|c| self.plane(off!(gll.tplanes) + 24 * c)).collect() } else { Vec::new() };
        let (f0, df) = (g!(F64, f0), g!(F64, df));
        let (a0, b0, slope) = (g!(F64, a0), g!(F64, b0), g!(F64, slope));
        let (width, dir, p0, p1) = (g!(I32, width), g!(I32, dir), g!(I32, p0), g!(I32, p1));
        let xmajor = g!(I32, xmajor);
        let xmajor = self.b.ins().icmp_imm_s(IntCC::NotEqual, xmajor, 0);
        let (pattern, repeat) = (g!(I32, pattern), g!(I32, repeat));
        let pos0 = self.ld_rw(I32, off!(stipple_pos));
        let (pv, posv) = (self.var(I32, p0), self.var(I32, pos0));
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
        let rgba: Vec<Variable> = (0..4)
            .map(|c| {
                let m = self.b.ins().fmul(dc[c], t);
                let v = self.b.ins().fadd(c0[c], m);
                self.var(F64, v)
            })
            .collect();
        let rgba: [Variable; 4] = rgba.try_into().unwrap();
        if k.tex.is_some() {
            let tp: [[Value; 3]; 3] = [tplanes[0], tplanes[1], tplanes[2]];
            self.textured(&tp, At::Line { t }, rgba);
        }
        let mut c: [Value; 4] = rgba.map(|v| self.get(v));
        if k.fog {
            let m = self.b.ins().fmul(df, t);
            let f = self.b.ins().fadd(f0, m);
            c = self.fogged(c, f);
        }
        let m = self.b.ins().fmul(dz, t);
        let z = self.b.ins().fadd(z0, m);
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
        self.b.ins().jump(head, &[]);
        self.b.seal_block(head);
        self.b.switch_to_block(exit);
        self.b.seal_block(exit);
        let pos = self.get(posv);
        self.st(pos, off!(stipple_pos));
    }

    /// `Rss::fogged`.
    fn fogged(&mut self, rgba: [Value; 4], f: Value) -> [Value; 4] {
        let f = self.clamp01(f);
        let one = self.f64c(1.0);
        let nf = self.b.ins().fsub(one, f);
        let mut out = rgba;
        for (k, o) in out.iter_mut().enumerate().take(3) {
            let fc = self.inv(off!(gl.fog_c) + 8 * k as i32);
            let a = self.b.ins().fmul(f, rgba[k]);
            let b = self.b.ins().fmul(nf, fc);
            *o = self.b.ins().fadd(a, b);
        }
        out
    }

    /// `Rss::gl_fragment` at window (wx, wy); a fragment that fails a test
    /// goes on to `kill`.
    fn fragment(&mut self, r: &Row, fx: Value, rgba: [Value; 4], z: Value, kill: Block) {
        let k = self.k;
        self.visible_x(r, fx, kill);
        let rgba = rgba.map(|c| self.clamp01(c));
        if let Some(func) = k.alpha {
            let aref = self.inv(off!(gl.aref));
            let pass = self.fcompare(func, rgba[3], aref);
            self.need(pass, kill);
        }
        if k.stencil.is_some() || k.z.is_some() {
            let (za, _) = self.locate_x(r.z.unwrap(), fx, false);
            let zst = self.get_px(za, None);
            let s_old = self.b.ins().ushr_imm_s(zst, 24);
            let s_old = self.b.ins().band_imm_s(s_old, 0xFF);
            let s_old = self.b.ins().ireduce(I32, s_old);
            let zold = self.b.ins().band_imm_s(zst, 0xFF_FFFF);
            // `z.round().clamp(0, 2^24 - 1) as u64`: clamping first rounds
            // the same (the bounds are integers), and in range a signed
            // conversion is the same and cheaper.
            let lo = self.f64c(0.0);
            let hi = self.f64c(16_777_215.0);
            let zc = self.clamp(z, lo, hi);
            let zr = self.round_nonneg(zc);
            let zi = self.b.ins().fcvt_to_sint_sat(I64, zr);
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

    /// `Rss::fragment_color`.
    fn fragment_color(&mut self, r: &Row, t: usize, x: Value, rgba: [Value; 4]) {
        let k = self.k;
        let src = if k.rgb {
            let c = match k.blend {
                Some(bl) => {
                    let overlay = t == 0 && k.draw == Draw::Overlay;
                    let (a, sh) = self.locate_x(r.tgt[t], x, overlay);
                    let dst = self.get_px(a, sh);
                    let dst = self.b.ins().ireduce(I32, dst);
                    // `dst >> 24` is a byte too: dst is at most 32 bits.
                    let table = self.b.ins().iconst(I64, byte_table().as_ptr() as i64);
                    let d: Vec<Value> = (0..4)
                        .map(|c| {
                            let v = self.b.ins().ushr_imm_s(dst, 8 * c);
                            let v = self.b.ins().band_imm_s(v, 0xFF);
                            let v = self.b.ins().uextend(I64, v);
                            let off = self.b.ins().ishl_imm_s(v, 3);
                            let a = self.b.ins().iadd(table, off);
                            self.b.ins().load(F64, self.mr, a, 0)
                        })
                        .collect();
                    let one = self.f64c(1.0);
                    let nd3 = self.b.ins().fsub(one, d[3]);
                    let sat = self.min_num(rgba[3], nd3);
                    let mut out = [rgba[0]; 4];
                    for (kk, o) in out.iter_mut().enumerate() {
                        let f = |e: &mut Self, code: u8| -> Value {
                            let one = e.f64c(1.0);
                            match code {
                                0 => e.f64c(0.0),
                                1 => one,
                                2 => rgba[kk],
                                3 => e.b.ins().fsub(one, rgba[kk]),
                                4 => rgba[3],
                                5 => e.b.ins().fsub(one, rgba[3]),
                                6 => d[3],
                                7 => e.b.ins().fsub(one, d[3]),
                                8 => d[kk],
                                9 => e.b.ins().fsub(one, d[kk]),
                                10 => {
                                    if kk == 3 {
                                        one
                                    } else {
                                        sat
                                    }
                                }
                                _ => one,
                            }
                        };
                        let fs = f(self, bl.src);
                        let fd = f(self, bl.dst);
                        let a = self.b.ins().fmul(rgba[kk], fs);
                        let b = self.b.ins().fmul(d[kk], fd);
                        let s = self.b.ins().fadd(a, b);
                        *o = self.clamp01(s);
                    }
                    out
                }
                None => rgba,
            };
            let mut src = self.i32c(0);
            for (kk, v) in c.iter().enumerate() {
                // Clamped to [0, 1] (or NaN): non-negative, in i32 range.
                let c255 = self.f64c(255.0);
                let m = self.b.ins().fmul(*v, c255);
                let r = self.round_nonneg(m);
                let q = self.b.ins().fcvt_to_sint_sat(I32, r);
                let q = if kk > 0 { self.b.ins().ishl_imm_s(q, 8 * kk as i64) } else { q };
                src = self.b.ins().bor(src, q);
            }
            src
        } else {
            let c = self.f64c(4095.0);
            let m = self.b.ins().fmul(rgba[0], c);
            let r = self.round_nonneg(m);
            let q = self.b.ins().fcvt_to_sint_sat(I32, r);
            self.b.ins().band_imm_s(q, 0xFFF)
        };
        self.put_in(r, t, x, src);
    }

    // ── texturing (rss.rs `textured`, te1.rs) ────────────────────────────

    /// `Rss::textured`: replaces `rgba` with the textured colour unless 1/W
    /// is zero or not finite.
    fn textured(&mut self, p: &[[Value; 3]; 3], at: At, rgba: [Variable; 4]) {
        let tex = self.k.tex.unwrap();
        let sw = self.at(&p[0], at);
        let tw = self.at(&p[1], at);
        let wi = self.at(&p[2], at);
        let zero = self.f64c(0.0);
        let inf = self.f64c(f64::INFINITY);
        let z = self.b.ins().fcmp(FloatCC::Equal, wi, zero);
        let aw = self.b.ins().fabs(wi);
        // Not finite: !(|wi| < inf). (Booleans are 0/1 bytes: no bnot.)
        let nonfin = self.b.ins().fcmp(FloatCC::UnorderedOrGreaterThanOrEqual, aw, inf);
        let skip = self.b.ins().bor(z, nonfin);
        let join = self.b.create_block();
        self.bail(skip, join);
        let s = self.b.ins().fdiv(sw, wi);
        let t = self.b.ins().fdiv(tw, wi);
        // Without mipmaps, and with one filter for both, the level of
        // detail decides nothing.
        let lambda = if !tex.mipmap && tex.mag_linear == tex.min_linear {
            zero
        } else {
            self.lambda(p, s, t, wi, tex.mipmap)
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

    /// `Rss::textured`'s level of detail: log2 of the larger footprint
    /// axis in level-0 texels, -inf when it is not positive.
    /// Without mipmaps only `lambda > 0` matters, which is `rho > 1`:
    /// then this returns 1 or -1 instead of calling log2.
    fn lambda(&mut self, p: &[[Value; 3]; 3], s: Value, t: Value, wi: Value, need_log: bool) -> Value {
        let d = |e: &mut Self, q: &[Value; 3], v: Value| {
            let a = e.b.ins().fmul(v, p[2][1]);
            let x = e.b.ins().fsub(q[1], a);
            let x = e.b.ins().fdiv(x, wi);
            let b = e.b.ins().fmul(v, p[2][2]);
            let y = e.b.ins().fsub(b, q[2]);
            let y = e.b.ins().fdiv(y, wi);
            (x, y)
        };
        let zero = self.f64c(0.0);
        let (dsx, dsy) = d(self, &p[0], s);
        let (dtx, dty) = d(self, &p[1], t);
        let w = self.inv(off!(gl.smp_w));
        let h = self.inv(off!(gl.smp_h));
        let sx = self.b.ins().fmul(dsx, w);
        let tx = self.b.ins().fmul(dtx, h);
        let sy = self.b.ins().fmul(dsy, w);
        let ty = self.b.ins().fmul(dty, h);
        let _ = zero;
        self.call4(if need_log { jit_lambda } else { jit_lambda_sign }, [sx, tx, sy, ty])
    }

    /// `te1::tex_env`.
    fn tex_env(&mut self, tex: &Tex, f: [Value; 4], texel: [Value; 4]) -> [Value; 4] {
        let tx = texel.map(|c| self.b.ins().fpromote(F64, c));
        let nc = tex.nc;
        let (ct, at): (Option<[Value; 3]>, Option<Value>) = match tex.class {
            1 => (None, Some(tx[0])),
            2 => (Some([tx[0]; 3]), (nc == 2).then_some(tx[1])),
            3 => (Some([tx[0]; 3]), Some(tx[0])),
            _ if nc >= 3 => (Some([tx[0], tx[1], tx[2]]), (nc == 4).then_some(tx[3])),
            _ => (Some([tx[0]; 3]), (nc == 2).then_some(tx[1])),
        };
        let one = self.f64c(1.0);
        let mut out = f;
        match tex.env & 3 {
            1 => {
                if let Some(ct) = ct {
                    let a = at.unwrap_or(one);
                    let na = self.b.ins().fsub(one, a);
                    for k in 0..3 {
                        let x = self.b.ins().fmul(f[k], na);
                        let y = self.b.ins().fmul(ct[k], a);
                        out[k] = self.b.ins().fadd(x, y);
                    }
                }
            }
            2 => {
                if let Some(ct) = ct {
                    for k in 0..3 {
                        let cc = self.inv(off!(gl.env_c) + 8 * k as i32);
                        let n = self.b.ins().fsub(one, ct[k]);
                        let x = self.b.ins().fmul(f[k], n);
                        let y = self.b.ins().fmul(cc, ct[k]);
                        out[k] = self.b.ins().fadd(x, y);
                    }
                }
                if let Some(at) = at {
                    out[3] = if tex.class == 3 {
                        let ac = self.inv(off!(gl.env_a));
                        let n = self.b.ins().fsub(one, at);
                        let x = self.b.ins().fmul(f[3], n);
                        let y = self.b.ins().fmul(ac, at);
                        self.b.ins().fadd(x, y)
                    } else {
                        self.b.ins().fmul(f[3], at)
                    };
                }
            }
            3 => {
                if let Some(at) = at {
                    out[3] = self.b.ins().fmul(f[3], at);
                }
            }
            _ => {
                if let Some(ct) = ct {
                    for k in 0..3 {
                        out[k] = self.b.ins().fmul(f[k], ct[k]);
                    }
                }
                if let Some(at) = at {
                    out[3] = self.b.ins().fmul(f[3], at);
                }
            }
        }
        out
    }

    /// `Sampler::sample`: the filtered texel (f32 components).
    fn sample(&mut self, tex: &Tex, s: Value, t: Value, lambda: Value) -> [Value; 4] {
        if !tex.mipmap && tex.mag_linear == tex.min_linear {
            let l0 = self.i32c(0);
            return self.level(tex, l0, s, t, tex.mag_linear);
        }
        let zf = self.f32c(0.0);
        let res: [Variable; 4] = [0, 1, 2, 3].map(|_| self.var(F32, zf));
        let join = self.b.create_block();
        let zero = self.f64c(0.0);
        let minify = self.b.ins().fcmp(FloatCC::GreaterThan, lambda, zero);
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
            let top = self.b.ins().fcvt_from_sint(F64, ml);
            if !tex.mip_linear {
                let half = self.f64c(0.5);
                let x = self.b.ins().fadd(lambda, half);
                let x = self.b.ins().ceil(x);
                let one = self.f64c(1.0);
                let x = self.b.ins().fsub(x, one);
                let x = self.clamp(x, zero, top);
                let l = self.b.ins().fcvt_to_uint_sat(I32, x);
                let r = self.level(tex, l, s, t, tex.min_linear);
                self.def_all(&res, r);
            } else {
                let fl = self.b.ins().floor(lambda);
                let l0 = self.clamp(fl, zero, top);
                let one = self.f64c(1.0);
                let l1 = self.b.ins().fadd(l0, one);
                let l1 = self.min_num(l1, top);
                let fr = self.b.ins().fsub(lambda, l0);
                let fr = self.clamp01(fr);
                let fr = self.b.ins().fdemote(F32, fr);
                let li0 = self.b.ins().fcvt_to_uint_sat(I32, l0);
                let li1 = self.b.ins().fcvt_to_uint_sat(I32, l1);
                let a = self.level(tex, li0, s, t, tex.min_linear);
                let b = self.level(tex, li1, s, t, tex.min_linear);
                let mut r = a;
                for k in 0..4 {
                    let d = self.b.ins().fsub(b[k], a[k]);
                    let m = self.b.ins().fmul(d, fr);
                    r[k] = self.b.ins().fadd(a[k], m);
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

    /// `Sampler::level`: one level at (s, t), nearest or bilinear.
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
        // Sizes are at most 2^15: a signed conversion is exact and cheap.
        let w = self.b.ins().fcvt_from_sint(F64, wi);
        let h = self.b.ins().fcvt_from_sint(F64, hi);
        let s = if tex.gl_clamp && tex.clamp_s { self.clamp01(s) } else { s };
        let t = if tex.gl_clamp && tex.clamp_t { self.clamp01(t) } else { t };
        let u = self.b.ins().fmul(s, w);
        let v = self.b.ins().fmul(t, h);
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
                let f = e.b.ins().floor(x);
                let i = e.b.ins().fcvt_to_sint_sat(I64, f);
                if clamp && tex.gl_clamp {
                    let last = e.b.ins().iadd_imm_s(n, -1);
                    e.b.ins().smin(i, last)
                } else {
                    i
                }
            };
            let i = cap(self, u, wi, tex.clamp_s);
            let j = cap(self, v, hi, tex.clamp_t);
            return self.texel(tex, &geo, i, j);
        }
        let half = self.f64c(0.5);
        let u = self.b.ins().fsub(u, half);
        let v = self.b.ins().fsub(v, half);
        let fi = self.b.ins().floor(u);
        let fj = self.b.ins().floor(v);
        let a = self.b.ins().fsub(u, fi);
        let a = self.b.ins().fdemote(F32, a);
        let b = self.b.ins().fsub(v, fj);
        let b = self.b.ins().fdemote(F32, b);
        let i = self.b.ins().fcvt_to_sint_sat(I64, fi);
        let j = self.b.ins().fcvt_to_sint_sat(I64, fj);
        let i1 = self.b.ins().iadd_imm_s(i, 1);
        let j1 = self.b.ins().iadd_imm_s(j, 1);
        let t00 = self.texel(tex, &geo, i, j);
        let t10 = self.texel(tex, &geo, i1, j);
        let t01 = self.texel(tex, &geo, i, j1);
        let t11 = self.texel(tex, &geo, i1, j1);
        let one = self.f32c(1.0);
        let na = self.b.ins().fsub(one, a);
        let nb = self.b.ins().fsub(one, b);
        let w00 = self.b.ins().fmul(na, nb);
        let w10 = self.b.ins().fmul(a, nb);
        let w01 = self.b.ins().fmul(na, b);
        let w11 = self.b.ins().fmul(a, b);
        let mut c = t00;
        for k in 0..4 {
            let x = self.b.ins().fmul(w00, t00[k]);
            let y = self.b.ins().fmul(w10, t10[k]);
            let s = self.b.ins().fadd(x, y);
            let z = self.b.ins().fmul(w01, t01[k]);
            let s = self.b.ins().fadd(s, z);
            let q = self.b.ins().fmul(w11, t11[k]);
            c[k] = self.b.ins().fadd(s, q);
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
        let zero = self.f32c(0.0);
        let mut c = [zero; 4];
        let table = self.b.ins().iconst(I64, comp_table(d as usize).as_ptr() as i64);
        for (k, o) in c.iter_mut().enumerate().take(nc) {
            let co = (slot + k as i64).min(3) * d;
            let v = self.b.ins().ushr_imm_s(cell, 4 * co);
            let v = self.b.ins().band_imm_s(v, (1i64 << (4 * d)) - 1);
            let off = self.b.ins().ishl_imm_s(v, 2);
            let a = self.b.ins().iadd(table, off);
            *o = self.b.ins().load(F32, self.mr, a, 0);
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
