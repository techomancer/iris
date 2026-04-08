//! REX3 uber-shader compiler: compiles specialized draw shaders via Cranelift.
//!
//! Each shader is specialized on a (DrawMode0, DrawMode1) pair — constant bits are
//! resolved at compile time so Cranelift can eliminate dead branches and optimize.
//!
//! Shader signature: `extern "C" fn(ctx: *mut Rex3Context, fb_rgb: *mut u32, fb_aux: *mut u32)`
//!
//! The compiled shader covers:
//!  - draw_block and draw_span outer loops (coordinate stepping, stop conditions)
//!  - per-pixel: clipping (smask0-4), pixel processing (fastclear / noblend / blend)
//!  - shade DDA iteration (if SHADE bit set)
//!  - pattern advance (if ENZPATTERN or ENLSPATTERN bit set)
//!
//! compress, expand, blend, and dither are all emitted as inline Cranelift IR —
//! no extern "C" helper calls needed.

use std::mem::offset_of;

use cranelift_codegen::ir::{self, types, AbiParam, InstBuilder, MemFlags, Value};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::rex3::{
    Rex3Context,
    DRAWMODE0_OPCODE_DRAW, DRAWMODE0_ADRMODE_BLOCK, DRAWMODE0_ADRMODE_SPAN,
    DRAWMODE1_PLANES_RGB, DRAWMODE1_PLANES_RGBA,
    DRAWMODE1_PLANES_OLAY, DRAWMODE1_PLANES_PUP, DRAWMODE1_PLANES_CID,
    OCTANT_XDEC, OCTANT_YDEC,
    REX3_COORD_BIAS, REX3_SCREEN_WIDTH, REX3_SCREEN_HEIGHT,
};

// ── Context field offsets (must match #[repr(C)] Rex3Context layout) ─────────

macro_rules! ctx_off {
    ($field:ident) => { offset_of!(Rex3Context, $field) as i32 }
}

// ── Draw mode bit extraction helpers (Rust-side, used during compilation) ────

struct Dm0 { val: u32 }
impl Dm0 {
    fn opcode(&self)      -> u32 { self.val & 3 }
    fn adrmode(&self)     -> u32 { (self.val >> 2) & 7 }
    fn colorhost(&self)   -> bool { self.val & (1 << 6) != 0 }
    fn alphahost(&self)   -> bool { self.val & (1 << 7) != 0 }
    fn stoponx(&self)     -> bool { self.val & (1 << 8) != 0 }
    fn stopony(&self)     -> bool { self.val & (1 << 9) != 0 }
    fn skipfirst(&self)   -> bool { self.val & (1 << 10) != 0 }
    fn skiplast(&self)    -> bool { self.val & (1 << 11) != 0 }
    fn enzpattern(&self)  -> bool { self.val & (1 << 12) != 0 }
    fn enlspattern(&self) -> bool { self.val & (1 << 13) != 0 }
    fn length32(&self)    -> bool { self.val & (1 << 15) != 0 }
    fn zpopaque(&self)    -> bool { self.val & (1 << 16) != 0 }
    fn lsopaque(&self)    -> bool { self.val & (1 << 17) != 0 }
    fn shade(&self)       -> bool { self.val & (1 << 18) != 0 }
    fn lronly(&self)      -> bool { self.val & (1 << 19) != 0 }
    fn xyoffset(&self)    -> bool { self.val & (1 << 20) != 0 }
    fn ciclamp(&self)     -> bool { self.val & (1 << 21) != 0 }
    fn ystride(&self)     -> bool { self.val & (1 << 23) != 0 }
    fn fastclear(&self)   -> bool { false } // fastclear is a dm1 bit
}

struct Dm1 { val: u32 }
impl Dm1 {
    fn planes(&self)    -> u32  { self.val & 7 }
    fn drawdepth(&self) -> u32  { (self.val >> 3) & 3 }
    fn dblsrc(&self)    -> bool { self.val & (1 << 5) != 0 }
    fn yflip(&self)     -> bool { self.val & (1 << 6) != 0 }
    fn rgbmode(&self)   -> bool { self.val & (1 << 15) != 0 }
    fn dither(&self)    -> bool { self.val & (1 << 16) != 0 }
    fn fastclear(&self) -> bool { self.val & (1 << 17) != 0 }
    fn blend(&self)     -> bool { self.val & (1 << 18) != 0 }
    fn sfactor(&self)   -> u32  { (self.val >> 19) & 7 }
    fn dfactor(&self)   -> u32  { (self.val >> 22) & 7 }
    fn backblend(&self) -> bool { self.val & (1 << 25) != 0 }
    fn logicop(&self)   -> u32  { (self.val >> 28) & 0xF }
}

// ── Compiler ──────────────────────────────────────────────────────────────────

pub struct ShaderCompiler {
    jit_module:  JITModule,
    ctx:         Context,
    builder_ctx: FunctionBuilderContext,
    counter:     u32,
}

impl ShaderCompiler {
    pub fn new() -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        flag_builder.set("is_pic", "false").unwrap();

        let isa_builder = cranelift_native::builder().expect("host ISA not supported");
        let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();
        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let jit_module = JITModule::new(jit_builder);

        Self {
            ctx: jit_module.make_context(),
            jit_module,
            builder_ctx: FunctionBuilderContext::new(),
            counter: 0,
        }
    }

    /// Compile a shader for the given (DrawMode0, DrawMode1) pair.
    /// Returns the compiled function entry point, or None if this mode is not JIT-able.
    pub fn compile_shader(&mut self, dm0_val: u32, dm1_val: u32)
        -> Option<unsafe extern "C" fn(*mut Rex3Context, *mut u32, *mut u32)>
    {
        let dm0 = Dm0 { val: dm0_val };

        // Only DRAW opcode, SPAN or BLOCK adrmode (no host FIFO, no lines).
        let opcode = dm0.opcode();
        let adrmode = dm0.adrmode() << 2; // match the <<2 convention from rex3.rs
        if opcode != DRAWMODE0_OPCODE_DRAW { return None; }
        if dm0.colorhost() || dm0.alphahost() { return None; }
        if adrmode != DRAWMODE0_ADRMODE_SPAN && adrmode != DRAWMODE0_ADRMODE_BLOCK { return None; }

        // fastclear takes priority over blend — hardware ignores blend when fastclear=1.
        let dm1_val = if dm1_val & (1 << 17) != 0 { dm1_val & !(1 << 18) } else { dm1_val };
        let dm1 = Dm1 { val: dm1_val };

        let name = format!("rex_shader_{:08x}_{:08x}_{}", dm0_val, dm1_val, self.counter);
        self.counter += 1;

        let ptr_type = self.jit_module.target_config().pointer_type();

        // Clear signature from any prior call before pushing new params.
        self.ctx.func.signature.params.clear();
        self.ctx.func.signature.returns.clear();
        // Shader signature: fn(ctx: *mut Rex3Context, fb_rgb: *mut u32, fb_aux: *mut u32)
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type)); // ctx
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type)); // fb_rgb
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type)); // fb_aux

        let func_id = match self.jit_module
            .declare_function(&name, Linkage::Local, &self.ctx.func.signature)
        {
            Ok(id) => id,
            Err(e) => { eprintln!("REX JIT: declare_function failed: {e}"); return None; }
        };

        let result = emit_shader(
            &mut self.ctx.func,
            &mut self.builder_ctx,
            &dm0, &dm1,
            ptr_type,
        );

        if !result {
            self.ctx.func.clear();
            self.jit_module.clear_context(&mut self.ctx);
            eprintln!("REX JIT: emit_shader returned false for dm0={dm0_val:#010x} dm1={dm1_val:#010x}: {}  {}",
                crate::rex3::decode_dm0(dm0_val), crate::rex3::decode_dm1(dm1_val));
            return None;
        }

        if let Err(e) = self.jit_module.define_function(func_id, &mut self.ctx) {
            eprintln!("REX JIT: define_function failed for dm0={dm0_val:#010x} dm1={dm1_val:#010x}: {}  {}  -- {e}",
                crate::rex3::decode_dm0(dm0_val), crate::rex3::decode_dm1(dm1_val));
            eprintln!("--- Cranelift IR ---\n{}", self.ctx.func.display());
            self.jit_module.clear_context(&mut self.ctx);
            return None;
        }
        self.jit_module.clear_context(&mut self.ctx);
        if let Err(e) = self.jit_module.finalize_definitions() {
            eprintln!("REX JIT: finalize_definitions failed: {e}");
            return None;
        }

        let code_ptr = self.jit_module.get_finalized_function(func_id);
        Some(unsafe {
            std::mem::transmute::<*const u8, unsafe extern "C" fn(*mut Rex3Context, *mut u32, *mut u32)>(code_ptr)
        })
    }
}

// ── Shader IR emission ────────────────────────────────────────────────────────

/// Emit the full shader IR. Returns false if the mode cannot be compiled.
fn emit_shader(
    func: &mut ir::Function,
    builder_ctx: &mut FunctionBuilderContext,
    dm0: &Dm0,
    dm1: &Dm1,
    ptr_type: ir::Type,
) -> bool {
    let mut b = FunctionBuilder::new(func, builder_ctx);
    let mem  = MemFlags::trusted();
    let memv = MemFlags::new(); // for potentially-aliased reads after stores

    // Entry block
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);

    let ctx_ptr  = b.block_params(entry)[0];
    let fb_rgb   = b.block_params(entry)[1];
    let fb_aux   = b.block_params(entry)[2];

    // ── Load loop-invariant ctx fields ────────────────────────────────────────
    macro_rules! ld32 { ($off:expr) => {
        b.ins().load(types::I32, mem, ctx_ptr, ir::immediates::Offset32::new($off as i32))
    }}
    macro_rules! ld8 { ($off:expr) => {
        b.ins().load(types::I8, mem, ctx_ptr, ir::immediates::Offset32::new($off as i32))
    }}
    macro_rules! st_bool { ($off:expr, $val:expr) => {
        b.ins().store(mem, $val, ctx_ptr, ir::immediates::Offset32::new($off as i32))
    }}

    let xend_v    = ld32!(ctx_off!(xend));
    let xsave_v   = ld32!(ctx_off!(xsave));
    let yend_v    = ld32!(ctx_off!(yend));
    let xywin_v   = ld32!(ctx_off!(xywin));
    let xymove_v  = ld32!(ctx_off!(xymove));
    let clipmode_v= ld32!(ctx_off!(clipmode));
    let wrmask_v  = ld32!(ctx_off!(wrmask));
    let colorback_v  = ld32!(ctx_off!(colorback));
    let colorvram_v  = ld32!(ctx_off!(colorvram));

    // smask fields
    let smask0x_v = ld32!(ctx_off!(smask0x));
    let smask0y_v = ld32!(ctx_off!(smask0y));
    let smask1x_v = ld32!(ctx_off!(smask1x));
    let smask1y_v = ld32!(ctx_off!(smask1y));
    let smask2x_v = ld32!(ctx_off!(smask2x));
    let smask2y_v = ld32!(ctx_off!(smask2y));
    let smask3x_v = ld32!(ctx_off!(smask3x));
    let smask3y_v = ld32!(ctx_off!(smask3y));
    let smask4x_v = ld32!(ctx_off!(smask4x));
    let smask4y_v = ld32!(ctx_off!(smask4y));

    // Constants
    let coord_bias = b.ins().iconst(types::I32, REX3_COORD_BIAS as i64);
    let c2048      = b.ins().iconst(types::I32, 2048);
    let c11        = b.ins().iconst(types::I32, 11);
    let c0         = b.ins().iconst(types::I32, 0);
    let c1         = b.ins().iconst(types::I32, 1);
    let c31        = b.ins().iconst(types::I32, 31);
    let c32        = b.ins().iconst(types::I32, 32);

    // ── Loop setup ────────────────────────────────────────────────────────────
    // octant is read from ctx at runtime (direction changes per primitive)
    let bres_v   = ld32!(ctx_off!(bresoctinc1));
    let octant_v = {
        let shifted = b.ins().ushr_imm(bres_v, 24);
        b.ins().band_imm(shifted, 7)
    };
    let x_dec_v = {
        let bit = b.ins().band_imm(octant_v, OCTANT_XDEC as i64);
        b.ins().icmp_imm(IntCC::NotEqual, bit, 0)
    };
    let y_dec_v = {
        let bit = b.ins().band_imm(octant_v, OCTANT_YDEC as i64);
        b.ins().icmp_imm(IntCC::NotEqual, bit, 0)
    };

    // stepx = x_dec ? -(1<<11) : (1<<11)
    let step_pos  = b.ins().iconst(types::I32, 1 << 11);
    let step_neg  = b.ins().iconst(types::I32, -(1i64 << 11));
    let stepx_v   = b.ins().select(x_dec_v, step_neg, step_pos);

    // stepy: ystride means 2 rows per step
    let y_inc_bits: i64 = if dm0.ystride() { 2 << 11 } else { 1 << 11 };
    let y_inc_neg  = b.ins().iconst(types::I32, -y_inc_bits);
    let y_inc_pos  = b.ins().iconst(types::I32, y_inc_bits);
    let stepy_v    = b.ins().select(y_dec_v, y_inc_neg, y_inc_pos);

    let is_block = dm0.adrmode() << 2 == DRAWMODE0_ADRMODE_BLOCK;
    let stopony  = dm0.stopony() && is_block; // span has no stopony

    // ── Loop blocks ───────────────────────────────────────────────────────────
    let loop_header  = b.create_block(); // params: [xstart, ystart, first_flag, colorred, colorgrn, colorblue, coloralpha, zpat_bit, pat_bit, lsrcount_packed]
    let loop_body    = b.create_block();
    let loop_next_x  = b.create_block(); // after pixel, before end-of-x check
    let loop_x_done  = b.create_block(); // x_end_reached branch
    let loop_end     = b.create_block(); // primitive done

    // Block parameters for loop_header (mutable loop state)
    // [xstart: i32, ystart: i32, first: i8]
    // + shade state if shade enabled: [colorred: i32, colorgrn: i32, colorblue: i32, coloralpha: i32]
    // + pattern state: [zpat_bit: i8, pat_bit: i8, lsmode: i32]
    b.append_block_param(loop_header, types::I32); // xstart
    b.append_block_param(loop_header, types::I32); // ystart
    b.append_block_param(loop_header, types::I8);  // first flag
    if dm0.shade() {
        b.append_block_param(loop_header, types::I32); // colorred
        b.append_block_param(loop_header, types::I32); // colorgrn
        b.append_block_param(loop_header, types::I32); // colorblue
        b.append_block_param(loop_header, types::I32); // coloralpha
    }
    if dm0.enzpattern() {
        b.append_block_param(loop_header, types::I8); // zpat_bit
    }
    if dm0.enlspattern() {
        b.append_block_param(loop_header, types::I8);  // pat_bit
        b.append_block_param(loop_header, types::I32); // lsmode
    }

    // Initial values for loop entry
    let xstart_init = ld32!(ctx_off!(xstart));
    let ystart_init = ld32!(ctx_off!(ystart));
    let first_init  = b.ins().iconst(types::I8, 1);

    // length32: stop after 32 pixels if span_len >= 32.
    // xstop = xstart ± (32<<11) depending on direction.
    // Computed once at shader entry; used in cont_x_block to stop the loop early.
    let xstop_v: Option<Value> = if dm0.length32() {
        let c32_11_pos = b.ins().iconst(types::I32,  32 << 11);
        let c32_11_neg = b.ins().iconst(types::I32, -(32i64 << 11));
        let step32 = b.ins().select(x_dec_v, c32_11_neg, c32_11_pos);
        let xstop_raw = b.ins().iadd(xstart_init, step32);
        // Only activate when span_len >= 32: span_len = abs(xend - xstart) >> 11
        let diff = b.ins().isub(xend_v, xstart_init);
        let diff_abs = {
            let neg = b.ins().ineg(diff);
            let is_neg = b.ins().icmp_imm(IntCC::SignedLessThan, diff, 0);
            b.ins().select(is_neg, neg, diff)
        };
        let span_len_v = b.ins().ushr_imm(diff_abs, 11);
        let c32i = b.ins().iconst(types::I32, 32);
        let long_enough = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, span_len_v, c32i);
        // If span < 32: use a sentinel that never triggers (xend ± 1 step beyond)
        let never = b.ins().iadd(xend_v, step32);
        Some(b.ins().select(long_enough, xstop_raw, never))
    } else {
        None
    };

    let mut init_args: Vec<Value> = vec![xstart_init, ystart_init, first_init];
    if dm0.shade() {
        let cr = ld32!(ctx_off!(colorred));
        let cg = ld32!(ctx_off!(colorgrn));
        let cb = ld32!(ctx_off!(colorblue));
        let ca = ld32!(ctx_off!(coloralpha));
        init_args.extend([cr, cg, cb, ca]);
    }
    if dm0.enzpattern() {
        let zpb: Value = ld8!(ctx_off!(zpat_bit));
        init_args.push(zpb);
    }
    if dm0.enlspattern() {
        let pb: Value  = ld8!(ctx_off!(pat_bit));
        let lsm = ld32!(ctx_off!(lsmode));
        init_args.push(pb);
        init_args.push(lsm);
    }

    // Set mid_primitive = 1 before entering loop
    let one8 = b.ins().iconst(types::I8, 1);
    st_bool!(ctx_off!(mid_primitive), one8);

    // LRONLY span mode: draw_span returns early (no pixels, no shade) when x_dec=1.
    // Emit a runtime check here — jump straight to loop_end if x_dec is set.
    if dm0.lronly() && !is_block {
        let done_block = b.create_block();
        b.ins().brif(x_dec_v, loop_end, &[], done_block, &[]);
        b.switch_to_block(done_block);
        b.seal_block(done_block);
    }

    b.ins().jump(loop_header, &init_args);

    // ── loop_header ───────────────────────────────────────────────────────────
    b.switch_to_block(loop_header);
    // DO NOT seal yet — back edge from loop_next_x will be added later

    let hp: Vec<Value> = b.block_params(loop_header).to_vec();
    let xstart_v = hp[0];
    let ystart_v = hp[1];
    let first_v  = hp[2];
    let mut param_idx = 3usize;
    let (colorred_v, colorgrn_v, colorblue_v, coloralpha_v) = if dm0.shade() {
        let r = hp[param_idx]; let g = hp[param_idx+1];
        let bl = hp[param_idx+2]; let a = hp[param_idx+3];
        param_idx += 4;
        (r, g, bl, a)
    } else {
        let z = b.ins().iconst(types::I32, 0);
        (z, z, z, z)
    };
    let zpat_bit_v = if dm0.enzpattern() {
        let v = hp[param_idx]; param_idx += 1; v
    } else {
        b.ins().iconst(types::I8, 0)
    };
    let (pat_bit_v, lsmode_v) = if dm0.enlspattern() {
        let pb = hp[param_idx]; let lsm = hp[param_idx+1]; param_idx += 2; (pb, lsm)
    } else {
        let pb  = b.ins().iconst(types::I8,  0);
        let lsm = b.ins().iconst(types::I32, 0);
        (pb, lsm)
    };

    // x = xstart >> 11, y = ystart >> 11
    let x_v = b.ins().sshr(xstart_v, c11);
    let y_v = b.ins().sshr(ystart_v, c11);

    // Advance xstart
    let xstart_next = b.ins().iadd(xstart_v, stepx_v);

    // x_end_reached = x_dec ? (xstart_next < xend) : (xstart_next > xend)
    let x_past_end_dec = b.ins().icmp(IntCC::SignedLessThan, xstart_next, xend_v);
    let x_past_end_inc = b.ins().icmp(IntCC::SignedGreaterThan, xstart_next, xend_v);
    let x_end_reached  = b.ins().select(x_dec_v, x_past_end_dec, x_past_end_inc);

    // do_pixel: I8 (1 = draw pixel, 0 = skip).
    // icmp returns I8 in Cranelift 0.116; brif treats any non-zero as true.
    // Use icmp_imm Equal/NotEqual to get clean 0/1 rather than bnot (which gives 0xFE).
    // skip if: (skipfirst && first) || (skiplast && x_end_reached)
    //
    // LRONLY (block mode): skip pixel when x_dec=1, but shade/pattern still advance.
    // LRONLY (span mode):  draw_span returns early (no shade either); for the JIT we
    //                      bail out immediately via an early return at shader entry.
    //
    // For span+lronly+x_dec: we handle this with an early-exit at the top of the shader
    // (jump directly to loop_end before any pixel or shade work) so the loop body is
    // never entered at all — matching draw_span's `return` with no side effects.
    // That early-exit is emitted below for the span+lronly case.
    let do_pixel: Value = if dm0.skipfirst() && dm0.skiplast() {
        let first_is_zero   = b.ins().icmp_imm(IntCC::Equal, first_v, 0);     // 1 if not-first
        let not_x_end       = b.ins().icmp_imm(IntCC::Equal, x_end_reached, 0); // 1 if not-last
        b.ins().band(first_is_zero, not_x_end)
    } else if dm0.skipfirst() {
        b.ins().icmp_imm(IntCC::Equal, first_v, 0)    // 1 when first==0 (not the first pixel)
    } else if dm0.skiplast() {
        b.ins().icmp_imm(IntCC::Equal, x_end_reached, 0) // 1 when not at end
    } else {
        b.ins().iconst(types::I8, 1) // always draw
    };

    // LRONLY block mode: AND with !x_dec (skip pixel when x_dec=1, shade still runs).
    let do_pixel = if dm0.lronly() && is_block {
        let not_x_dec = b.ins().icmp_imm(IntCC::Equal, x_dec_v, 0); // 1 when x_dec=0
        b.ins().band(do_pixel, not_x_dec)
    } else {
        do_pixel
    };

    // Jump to loop_body if do_pixel, else skip to loop_next_x
    let pixel_block = b.create_block();
    let skip_block  = b.create_block();
    // Both continue to loop_next_x with the same args
    let continue_block = loop_next_x;

    // Branch: do_pixel is I1 (from bnot/icmp) or I8 constant 1.
    // Use brif directly — it accepts I1 or any integer type.
    b.ins().brif(do_pixel, pixel_block, &[], skip_block, &[]);

    // ── pixel_block: compute address and draw ─────────────────────────────────
    b.switch_to_block(pixel_block);
    b.seal_block(pixel_block);

    // calculate_fb_address inlined:
    // 1. apply xymove if xyoffset
    let (x_curr, y_curr) = if dm0.xyoffset() {
        let xm_raw  = b.ins().ushr_imm(xymove_v, 16);
        let xm16    = b.ins().ireduce(types::I16, xm_raw);
        let xm32    = b.ins().sextend(types::I32, xm16);
        let ym_raw  = b.ins().band_imm(xymove_v, 0xFFFF_i64);
        let ym16    = b.ins().ireduce(types::I16, ym_raw);
        let ym32    = b.ins().sextend(types::I32, ym16);
        let xc      = b.ins().iadd(x_v, xm32);
        let yc      = b.ins().iadd(y_v, ym32);
        (xc, yc)
    } else {
        (x_v, y_v)
    };

    // 2. apply xywin
    let xw_raw  = b.ins().ushr_imm(xywin_v, 16);
    let xw16    = b.ins().ireduce(types::I16, xw_raw);
    let xw32    = b.ins().sextend(types::I32, xw16);
    let yw_raw  = b.ins().band_imm(xywin_v, 0xFFFF_i64);
    let yw16    = b.ins().ireduce(types::I16, yw_raw);
    let yw32    = b.ins().sextend(types::I32, yw16);
    let x_abs = b.ins().iadd(x_curr, xw32);
    let y_abs = b.ins().iadd(y_curr, yw32);

    // 3. clipping (smask0 = window-relative, smasks 1-4 = screen-absolute)
    // Each clipping check that fails jumps to skip_block (no draw).
    // smask0 (window-relative: compare x_curr/y_curr)
    let after_clip = b.create_block();
    {
        let ensmask = b.ins().band_imm(clipmode_v, 0x1F_i64);
        // smask0 (bit 0)
        let bit0 = b.ins().band_imm(ensmask, 1);
        let smask0_active = b.ins().icmp_imm(IntCC::NotEqual, bit0, 0);
        let clip_check0 = b.create_block();
        let pass0 = b.create_block();
        b.ins().brif(smask0_active, clip_check0, &[], pass0, &[]);
        b.switch_to_block(clip_check0); b.seal_block(clip_check0);
        let sm0x_hi  = b.ins().ushr_imm(smask0x_v, 16);
        let sm0x_hi16 = b.ins().ireduce(types::I16, sm0x_hi);
        let min_x0   = b.ins().sextend(types::I32, sm0x_hi16);
        let sm0x_lo16 = b.ins().ireduce(types::I16, smask0x_v);
        let max_x0   = b.ins().sextend(types::I32, sm0x_lo16);
        let sm0y_hi  = b.ins().ushr_imm(smask0y_v, 16);
        let sm0y_hi16 = b.ins().ireduce(types::I16, sm0y_hi);
        let min_y0   = b.ins().sextend(types::I32, sm0y_hi16);
        let sm0y_lo16 = b.ins().ireduce(types::I16, smask0y_v);
        let max_y0   = b.ins().sextend(types::I32, sm0y_lo16);
        let ok0 = and4_range(&mut b, x_curr, y_curr, min_x0, max_x0, min_y0, max_y0);
        b.ins().brif(ok0, pass0, &[], skip_block, &[]);
        b.switch_to_block(pass0); b.seal_block(pass0);

        // smasks 1-4 (screen-absolute: compare x_abs/y_abs)
        // At least one must match if any are enabled.
        let smask_hi = b.ins().band_imm(ensmask, 0x1E_i64);
        let any_smask = b.ins().icmp_imm(IntCC::NotEqual, smask_hi, 0);
        let smask_check = b.create_block();
        b.ins().brif(any_smask, smask_check, &[], after_clip, &[]);
        b.switch_to_block(smask_check); b.seal_block(smask_check);

        let masks = [
            (smask1x_v, smask1y_v, 2i64),
            (smask2x_v, smask2y_v, 4i64),
            (smask3x_v, smask3y_v, 8i64),
            (smask4x_v, smask4y_v, 16i64),
        ];
        // Build OR chain: inside_any = mask1_ok | mask2_ok | mask3_ok | mask4_ok
        // Use I1 values combined with bor
        let mut inside_any: Value = b.ins().iconst(types::I8, 0); // start as I8 zero
        for (sx, sy, bit_mask) in masks {
            let bit      = b.ins().band_imm(ensmask, bit_mask);
            let enabled  = b.ins().icmp_imm(IntCC::NotEqual, bit, 0); // I1
            // Split nested sextend/ireduce/ushr into steps
            let sx_hi    = b.ins().ushr_imm(sx, 16);
            let sx_hi16  = b.ins().ireduce(types::I16, sx_hi);
            let min_x    = b.ins().sextend(types::I32, sx_hi16);
            let sx_lo16  = b.ins().ireduce(types::I16, sx);
            let max_x    = b.ins().sextend(types::I32, sx_lo16);
            let sy_hi    = b.ins().ushr_imm(sy, 16);
            let sy_hi16  = b.ins().ireduce(types::I16, sy_hi);
            let min_y    = b.ins().sextend(types::I32, sy_hi16);
            let sy_lo16  = b.ins().ireduce(types::I16, sy);
            let max_y    = b.ins().sextend(types::I32, sy_lo16);
            let in_range = and4_range(&mut b, x_abs, y_abs, min_x, max_x, min_y, max_y); // I8
            // enabled (I8) & in_range (I8) — band directly
            let contrib  = b.ins().band(enabled, in_range);
            inside_any   = b.ins().bor(inside_any, contrib);
        }
        let inside = b.ins().icmp_imm(IntCC::NotEqual, inside_any, 0);
        b.ins().brif(inside, after_clip, &[], skip_block, &[]);
    }

    b.switch_to_block(after_clip); b.seal_block(after_clip);

    // 4. Physical address: addr = (y_abs - coord_bias) * 2048 + (x_abs - coord_bias)
    //    with yflip: y_phys = 0x23FF - y_abs
    let y_phys = if dm1.yflip() {
        let c23ff = b.ins().iconst(types::I32, 0x23FF);
        b.ins().isub(c23ff, y_abs)
    } else {
        b.ins().isub(y_abs, coord_bias)
    };
    let x_phys = b.ins().isub(x_abs, coord_bias);

    // Bounds check (VRAM clipping)
    let screen_w = b.ins().iconst(types::I32, REX3_SCREEN_WIDTH as i64);
    let screen_h = b.ins().iconst(types::I32, REX3_SCREEN_HEIGHT as i64);
    let x_neg = b.ins().icmp(IntCC::SignedLessThan, x_phys, c0);
    let x_wide = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, x_phys, screen_w);
    let y_neg  = b.ins().icmp(IntCC::SignedLessThan, y_phys, c0);
    let y_tall = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, y_phys, screen_h);
    let oob = {
        let a = b.ins().bor(x_neg, x_wide);
        let c_v = b.ins().bor(y_neg, y_tall);
        b.ins().bor(a, c_v)
    };
    let in_bounds = b.create_block();
    b.ins().brif(oob, skip_block, &[], in_bounds, &[]);
    b.switch_to_block(in_bounds); b.seal_block(in_bounds);

    // addr = y_phys * 2048 + x_phys  (as u32 index into fb_rgb/fb_aux)
    let ym2048  = b.ins().imul(y_phys, c2048);
    let addr_i32 = b.ins().iadd(ym2048, x_phys);
    // byte offset = addr_i32 * 4 (each fb entry is u32)
    let c4       = b.ins().iconst(types::I32, 4);
    let byte_off = b.ins().imul(addr_i32, c4);
    let byte_off64 = b.ins().uextend(types::I64, byte_off);

    // Choose fb pointer: rgb planes → fb_rgb, aux planes → fb_aux
    let use_aux = matches!(dm1.planes(), p if p == DRAWMODE1_PLANES_OLAY || p == DRAWMODE1_PLANES_PUP || p == DRAWMODE1_PLANES_CID);
    let fb_ptr  = if use_aux { fb_aux } else { fb_rgb };
    // fb_ptr is already a native pointer (ptr_type); byte_off64 is I64.
    // On x86-64 ptr_type == I64, so no extend needed.
    let px_ptr = b.ins().iadd(fb_ptr, byte_off64);

    // ── Pixel processing ──────────────────────────────────────────────────────
    // Source color
    let raw_src = if dm0.shade() {
        // combine_host_dda: RGB clamp per component
        let r = clamp_color_component(&mut b, colorred_v);
        let g = clamp_color_component(&mut b, colorgrn_v);
        let bl = clamp_color_component(&mut b, colorblue_v);
        let g8  = b.ins().ishl_imm(g, 8);
        let bl16 = b.ins().ishl_imm(bl, 16);
        let rb  = b.ins().bor(r, g8);
        b.ins().bor(rb, bl16)
    } else {
        // Solid color from colorred (CI) or colorred/grn/blue (RGB)
        if dm1.rgbmode() {
            let r = ld32!(ctx_off!(colorred));
            let g = ld32!(ctx_off!(colorgrn));
            let bl = ld32!(ctx_off!(colorblue));
            let r_c  = clamp_color_component(&mut b, r);
            let g_c  = clamp_color_component(&mut b, g);
            let b_c  = clamp_color_component(&mut b, bl);
            let g8   = b.ins().ishl_imm(g_c, 8);
            let bl16 = b.ins().ishl_imm(b_c, 16);
            let rb   = b.ins().bor(r_c, g8);
            b.ins().bor(rb, bl16)
        } else {
            // CI mode: colorred >> 11
            let cr = ld32!(ctx_off!(colorred));
            b.ins().sshr(cr, c11)
        }
    };

    // Pattern check: may replace raw_src with colorback (zpopaque/lsopaque)
    // or skip pixel entirely. We emit conditional branches.
    // draw_block takes use_bg_flag as a block param so each predecessor can
    // pass the correct value without SSA dominance violations.
    let draw_block = b.create_block();
    b.append_block_param(draw_block, types::I8); // use_bg_flag

    // Current use_bg value from the pre-pattern context (always 0 at entry).
    let mut cur_use_bg: Value = b.ins().iconst(types::I8, 0);

    if dm0.enzpattern() {
        let zp_block = b.create_block();
        let zp_pass  = b.create_block();
        b.append_block_param(zp_pass, types::I8); // use_bg carried through
        let zpattern_v = ld32!(ctx_off!(zpattern));
        let zpat_bit32 = b.ins().uextend(types::I32, zpat_bit_v);
        let zpat_shifted = b.ins().ushr(zpattern_v, zpat_bit32);
        let bit_v = b.ins().band_imm(zpat_shifted, 1);
        let bit_set = b.ins().icmp_imm(IntCC::NotEqual, bit_v, 0);
        b.ins().brif(bit_set, zp_pass, &[cur_use_bg], zp_block, &[]);
        b.switch_to_block(zp_block); b.seal_block(zp_block);
        if dm0.zpopaque() {
            let bg1 = b.ins().iconst(types::I8, 1);
            b.ins().jump(zp_pass, &[bg1]);
        } else {
            b.ins().jump(skip_block, &[]);
        }
        b.switch_to_block(zp_pass); b.seal_block(zp_pass);
        cur_use_bg = b.block_params(zp_pass).to_vec()[0];
    }

    if dm0.enlspattern() {
        let ls_block = b.create_block();
        let ls_pass  = b.create_block();
        b.append_block_param(ls_pass, types::I8); // use_bg carried through
        let lspattern_v = ld32!(ctx_off!(lspattern));
        let pat_bit32 = b.ins().uextend(types::I32, pat_bit_v);
        let lspat_shifted = b.ins().ushr(lspattern_v, pat_bit32);
        let bit_v = b.ins().band_imm(lspat_shifted, 1);
        let bit_set = b.ins().icmp_imm(IntCC::NotEqual, bit_v, 0);
        b.ins().brif(bit_set, ls_pass, &[cur_use_bg], ls_block, &[]);
        b.switch_to_block(ls_block); b.seal_block(ls_block);
        if dm0.lsopaque() {
            let bg1 = b.ins().iconst(types::I8, 1);
            b.ins().jump(ls_pass, &[bg1]);
        } else {
            b.ins().jump(skip_block, &[]);
        }
        b.switch_to_block(ls_pass); b.seal_block(ls_pass);
        cur_use_bg = b.block_params(ls_pass).to_vec()[0];
    }

    b.ins().jump(draw_block, &[cur_use_bg]);
    b.switch_to_block(draw_block); b.seal_block(draw_block);
    let use_bg_flag = b.block_params(draw_block).to_vec()[0];

    // Actual source: use_bg ? colorback : raw_src
    let use_bg_bool = b.ins().icmp_imm(IntCC::NotEqual, use_bg_flag, 0);
    let src_color = b.ins().select(use_bg_bool, colorback_v, raw_src);

    // Read current fb pixel (needed for logic ops that read dst, and for blend)
    let needs_dst = dm1.blend() || dm1.logicop() != 3 /* SRC */;
    let fb_px_raw: Value = if needs_dst {
        // px_ptr is I64; load needs a ptr-type value; use it directly as addr
        b.ins().load(types::I32, memv, px_ptr, ir::immediates::Offset32::new(0))
    } else {
        b.ins().iconst(types::I32, 0)
    };

    // Read/mask for depth.
    // For RGB planes: dblsrc selects bank 0 (low bits) or bank 1 (high bits).
    // For aux planes: bit layout is fixed by plane type (not depth bits).
    // dblsrc_shift is also the amplify shift for RGB (replicate pixel across its slot width).
    let depth_mask: i64 = match dm1.drawdepth() { 0 => 0xF, 1 => 0xFF, 2 => 0xFFF, _ => 0xFFFFFF };
    let dblsrc_shift: i64 = match dm1.drawdepth() { 0 => 4, 1 => 8, 2 => 12, _ => 0 };

    // For aux planes the read offset and mask differ from RGB:
    // OLAY bank0: bits[15:8], bank1: bits[23:16], mask=0xFF
    // CID  bank0: bits[1:0],  bank1: bits[5:4],   mask=0x3
    // PUP  bank0: bits[3:2],  bank1: bits[7:6],   mask=0x3
    let (aux_read_shift0, aux_read_shift1, aux_read_mask): (i64, i64, i64) = match dm1.planes() {
        p if p == DRAWMODE1_PLANES_OLAY => (8,  16, 0xFF),
        p if p == DRAWMODE1_PLANES_CID  => (0,  4,  0x3),
        p if p == DRAWMODE1_PLANES_PUP  => (2,  6,  0x3),
        _                               => (0,  0,  0),
    };

    let dst_plane = if needs_dst {
        if use_aux {
            // Aux plane: extract from fixed bit position
            let read_shift = if dm1.dblsrc() { aux_read_shift1 } else { aux_read_shift0 };
            let extracted = if read_shift > 0 {
                b.ins().ushr_imm(fb_px_raw, read_shift)
            } else {
                fb_px_raw
            };
            b.ins().band_imm(extracted, aux_read_mask)
        } else if dm1.dblsrc() && dblsrc_shift > 0 {
            let shifted = b.ins().ushr_imm(fb_px_raw, dblsrc_shift);
            b.ins().band_imm(shifted, depth_mask)
        } else {
            b.ins().band_imm(fb_px_raw, depth_mask)
        }
    } else {
        b.ins().iconst(types::I32, 0)
    };

    // Result pixel value
    let result_val: Value = if dm1.fastclear() {
        // Fast clear: replicate colorvram into plane slots (matches fastclear_color in rex3.rs)
        match dm1.drawdepth() {
            0 => { // 4bpp: replicate nibble
                let c = b.ins().band_imm(colorvram_v, 0xf);
                let c4 = b.ins().ishl_imm(c, 4);
                let c8 = b.ins().ishl_imm(c, 8);
                let c16 = b.ins().ishl_imm(c, 16);
                let r1 = b.ins().bor(c, c4);
                let r2 = b.ins().bor(r1, c8);
                b.ins().bor(r2, c16)
            }
            1 => { // 8bpp: replicate byte
                let c = b.ins().band_imm(colorvram_v, 0xff);
                let c8  = b.ins().ishl_imm(c, 8);
                let c16 = b.ins().ishl_imm(c, 16);
                let r1 = b.ins().bor(c, c8);
                b.ins().bor(r1, c16)
            }
            2 => { // 12bpp
                if dm1.rgbmode() {
                    let hi = b.ins().band_imm(colorvram_v, 0xf00000i64);
                    let hi_s = b.ins().ushr_imm(hi, 12);
                    let mid = b.ins().band_imm(colorvram_v, 0xf000i64);
                    let mid_s = b.ins().ushr_imm(mid, 8);
                    let lo = b.ins().band_imm(colorvram_v, 0xf0i64);
                    let lo_s = b.ins().ushr_imm(lo, 4);
                    let r1 = b.ins().bor(hi_s, mid_s);
                    let c = b.ins().bor(r1, lo_s);
                    let c12 = b.ins().ishl_imm(c, 12);
                    b.ins().bor(c, c12)
                } else {
                    let c = b.ins().band_imm(colorvram_v, 0xfffi64);
                    let c12 = b.ins().ishl_imm(c, 12);
                    b.ins().bor(c, c12)
                }
            }
            _ => b.ins().band_imm(colorvram_v, 0xffffffi64), // 24bpp
        }
    } else if dm1.blend() {
        // Blend path: expand dst to 24-bit BGR, blend, compress back
        let dst_24 = if dm1.rgbmode() && dm1.drawdepth() != 3 {
            emit_expand_ir(&mut b, dst_plane, dm1.drawdepth())
        } else {
            dst_plane
        };
        let blend_dst = if dm1.backblend() { colorback_v } else { dst_24 };
        let blended = emit_blend_ir(&mut b, src_color, blend_dst, dm1.sfactor(), dm1.dfactor());
        // bayer_pack + compress + amplify
        let packed = bayer_pack_ir(&mut b, blended, x_v, y_v);
        let compressed = if dm1.rgbmode() && dm1.drawdepth() != 3 {
            emit_compress_ir(&mut b, packed, dm1.drawdepth(), dm1.dither())
        } else {
            packed
        };
        // Amplify: replicate into both banks (sub-24bpp only; 24bpp is identity)
        if dblsrc_shift > 0 {
            let shifted = b.ins().ishl_imm(compressed, dblsrc_shift);
            b.ins().bor(compressed, shifted)
        } else {
            compressed
        }
    } else {
        // Logic op path
        // compress(bayer_pack(raw_src, x, y))
        let packed = bayer_pack_ir(&mut b, src_color, x_v, y_v);
        let compressed = if dm1.rgbmode() && dm1.drawdepth() != 3 {
            emit_compress_ir(&mut b, packed, dm1.drawdepth(), dm1.dither())
        } else {
            packed
        };
        // Amplify: replicate pixel into both banks before logic op.
        // For RGB planes: amplify_rgb_N(v) = v | (v << depth_shift) for sub-24bpp.
        // For aux planes: use plane-specific amplify packing.
        // For 24bpp RGB: identity (dblsrc_shift == 0).
        let amp_src = if use_aux {
            amplify_aux_ir(&mut b, compressed, dm1.planes())
        } else if dblsrc_shift > 0 {
            let shifted = b.ins().ishl_imm(compressed, dblsrc_shift);
            b.ins().bor(compressed, shifted)
        } else {
            compressed
        };
        let amp_dst = if use_aux {
            amplify_aux_ir(&mut b, dst_plane, dm1.planes())
        } else if dblsrc_shift > 0 {
            let shifted = b.ins().ishl_imm(dst_plane, dblsrc_shift);
            b.ins().bor(dst_plane, shifted)
        } else {
            dst_plane
        };
        // logic op
        emit_logic_op(&mut b, dm1.logicop(), amp_src, amp_dst)
    };

    // Write result: (fb[addr] & !wrmask) | (result & wrmask)
    // px_ptr is the already-computed I64 byte address
    let old_val     = b.ins().load(types::I32, memv, px_ptr, ir::immediates::Offset32::new(0));
    let inv_mask    = b.ins().bnot(wrmask_v);
    let kept        = b.ins().band(old_val, inv_mask);
    let written     = b.ins().band(result_val, wrmask_v);
    let new_val     = b.ins().bor(kept, written);
    b.ins().store(mem, new_val, px_ptr, ir::immediates::Offset32::new(0));
    b.ins().jump(skip_block, &[]);

    // ── skip_block: shade + pattern advance, then check end-of-x ─────────────
    b.switch_to_block(skip_block);
    b.seal_block(skip_block);

    // Shade DDA step
    let (new_cr, new_cg, new_cb, new_ca) = if dm0.shade() {
        let slopered_v   = ld32!(ctx_off!(slopered));
        let slopegrn_v   = ld32!(ctx_off!(slopegrn));
        let slopeblue_v  = ld32!(ctx_off!(slopeblue));
        let slopealpha_v = ld32!(ctx_off!(slopealpha));

        let ncr = b.ins().iadd(colorred_v,   slopered_v);
        let ncg = b.ins().iadd(colorgrn_v,   slopegrn_v);
        let ncb = b.ins().iadd(colorblue_v,  slopeblue_v);
        let nca = b.ins().iadd(coloralpha_v, slopealpha_v);

        // RGB clamp after add
        let (ncr, ncg, ncb, nca) = if dm1.rgbmode() {
            (clamp_shade(&mut b, ncr), clamp_shade(&mut b, ncg),
             clamp_shade(&mut b, ncb), clamp_shade(&mut b, nca))
        } else if dm0.ciclamp() {
            let depth = dm1.drawdepth();
            let ncr2 = if depth == 1 { // 8bpp: clamp colorred if bit 19 set
                let overflow = b.ins().band_imm(ncr, 1 << 19);
                let ov_set = b.ins().icmp_imm(IntCC::NotEqual, overflow, 0);
                let max8 = b.ins().iconst(types::I32, 0x0007_FFFFi64);
                b.ins().select(ov_set, max8, ncr)
            } else if depth == 2 { // 12bpp: clamp colorred if bit 21 set
                let overflow = b.ins().band_imm(ncr, 1 << 21);
                let ov_set = b.ins().icmp_imm(IntCC::NotEqual, overflow, 0);
                let max12 = b.ins().iconst(types::I32, 0x001F_FFFFi64);
                b.ins().select(ov_set, max12, ncr)
            } else { ncr };
            (ncr2, ncg, ncb, nca)
        } else {
            (ncr, ncg, ncb, nca)
        };
        (ncr, ncg, ncb, nca)
    } else {
        (colorred_v, colorgrn_v, colorblue_v, coloralpha_v)
    };

    // Pattern advance
    let new_zpat_bit = if dm0.enzpattern() {
        // zpat_bit = (zpat_bit - 1) & 31
        let c1_i8 = b.ins().iconst(types::I8, 1);
        let dec = b.ins().isub(zpat_bit_v, c1_i8);
        b.ins().band_imm(dec, 31)
    } else {
        zpat_bit_v
    };

    let (new_pat_bit, new_lsmode) = if dm0.enlspattern() {
        // Advance lspattern: decrement lsrcount; on zero, advance pat_bit and reload
        // lsrcount = lsmode[7:0], lsrepeat = lsmode[15:8], lsrcntsave = lsmode[23:16], lslength = lsmode[27:24]
        let lsrcount = b.ins().band_imm(lsmode_v, 0xFF);
        let lsm_shr8 = b.ins().ushr_imm(lsmode_v, 8);
        let lsrepeat_raw = b.ins().band_imm(lsm_shr8, 0xFF);
        let lsrepeat = {
            let is_zero = b.ins().icmp_imm(IntCC::Equal, lsrepeat_raw, 0);
            b.ins().select(is_zero, c1, lsrepeat_raw)
        };
        let is_zero = b.ins().icmp_imm(IntCC::Equal, lsrcount, 0);
        let reload_count = b.ins().isub(lsrepeat, c1);
        let lsrcount_dec = b.ins().isub(lsrcount, c1);
        let new_count = b.ins().select(is_zero, reload_count, lsrcount_dec);
        // If lsrcount was 0: advance pat_bit
        let lsm_shr24 = b.ins().ushr_imm(lsmode_v, 24);
        let lslength = b.ins().band_imm(lsm_shr24, 0xF);
        let length_bits = b.ins().iadd_imm(lslength, 17); // 17..=32
        let wrap_point = b.ins().isub(c32, length_bits);
        let pat_bit32 = b.ins().uextend(types::I32, pat_bit_v);
        let at_wrap = b.ins().icmp(IntCC::Equal, pat_bit32, wrap_point);
        let pat_dec = b.ins().isub(pat_bit32, c1);
        let dec_pat = b.ins().band_imm(pat_dec, 31);
        let new_pat32 = b.ins().select(at_wrap, c31, dec_pat);
        let new_pat32_masked = b.ins().select(is_zero, new_pat32, pat_bit32);
        let new_pb = b.ins().ireduce(types::I8, new_pat32_masked);
        // Update lsmode: set lsrcount field (bits 7:0)
        let lsm_cleared = b.ins().band_imm(lsmode_v, !0xFF_i64);
        let new_lsm = b.ins().bor(lsm_cleared, new_count);
        (new_pb, new_lsm)
    } else {
        (pat_bit_v, lsmode_v)
    };

    // x_end_reached branch
    let end_x_block = b.create_block();
    let cont_x_block = b.create_block(); // not x_end, check stoponx/xstop

    // x_end_reached check (re-use computed value from loop_header context)
    // Actually we need to pass it through. Simplest: recompute here.
    let x_end_reached2 = {
        let past_dec = b.ins().icmp(IntCC::SignedLessThan, xstart_next, xend_v);
        let past_inc = b.ins().icmp(IntCC::SignedGreaterThan, xstart_next, xend_v);
        b.ins().select(x_dec_v, past_dec, past_inc)
    };
    b.ins().brif(x_end_reached2, end_x_block, &[], cont_x_block, &[]);

    // ── end_x_block: advance y, check stopony ─────────────────────────────────
    b.switch_to_block(end_x_block);
    b.seal_block(end_x_block);

    let ystart_next  = b.ins().iadd(ystart_v, stepy_v);
    let pat_bit_reset = b.ins().iconst(types::I8, 31);
    let zpat_bit_reset = b.ins().iconst(types::I8, 31);

    if !stopony {
        // Span / block without stopony: primitive done after one row
        let zero8 = b.ins().iconst(types::I8, 0);
        st_bool!(ctx_off!(mid_primitive), zero8);
        // Write back state
        emit_writeback(&mut b, ctx_ptr, &mem, xsave_v, ystart_next,
            dm0, new_cr, new_cg, new_cb, new_ca, zpat_bit_reset, pat_bit_reset, new_lsmode);
        b.ins().jump(loop_end, &[]);
    } else {
        // Block with stopony: advance y and check y_end
        let y_past_dec = b.ins().icmp(IntCC::SignedLessThan, ystart_next, yend_v);
        let y_past_inc = b.ins().icmp(IntCC::SignedGreaterThan, ystart_next, yend_v);
        let y_end_reached = b.ins().select(y_dec_v, y_past_dec, y_past_inc);

        let y_done_block = b.create_block();
        let y_cont_block = b.create_block();
        b.ins().brif(y_end_reached, y_done_block, &[], y_cont_block, &[]);

        b.switch_to_block(y_done_block); b.seal_block(y_done_block);
        let zero8 = b.ins().iconst(types::I8, 0);
        st_bool!(ctx_off!(mid_primitive), zero8);
        emit_writeback(&mut b, ctx_ptr, &mem, xsave_v, ystart_next,
            dm0, new_cr, new_cg, new_cb, new_ca, zpat_bit_reset, pat_bit_reset, new_lsmode);
        b.ins().jump(loop_end, &[]);

        b.switch_to_block(y_cont_block); b.seal_block(y_cont_block);
        // Continue loop: reset x to xsave, new row
        let new_first = b.ins().iconst(types::I8, 1);
        let mut back_args: Vec<Value> = vec![xsave_v, ystart_next, new_first];
        if dm0.shade() { back_args.extend([new_cr, new_cg, new_cb, new_ca]); }
        if dm0.enzpattern() { back_args.push(zpat_bit_reset); }
        if dm0.enlspattern() { back_args.push(pat_bit_reset); back_args.push(new_lsmode); }
        b.ins().jump(loop_header, &back_args);
    }

    // ── cont_x_block: not x_end — check stoponx ───────────────────────────────
    b.switch_to_block(cont_x_block);
    b.seal_block(cont_x_block);

    if !dm0.stoponx() {
        // stoponx=0: after one x step we stop (span primitive, next GO continues)
        emit_writeback(&mut b, ctx_ptr, &mem, xstart_next, ystart_v,
            dm0, new_cr, new_cg, new_cb, new_ca, new_zpat_bit, new_pat_bit, new_lsmode);
        b.ins().jump(loop_end, &[]);
    } else {
        // stoponx=1: continue x loop — jump back to loop_header.
        // But first check length32 xstop: if xstart_next >= xstop, stop the primitive.
        let new_first = b.ins().iconst(types::I8, 0);
        let mut back_args: Vec<Value> = vec![xstart_next, ystart_v, new_first];
        if dm0.shade() { back_args.extend([new_cr, new_cg, new_cb, new_ca]); }
        if dm0.enzpattern() { back_args.push(new_zpat_bit); }
        if dm0.enlspattern() { back_args.push(new_pat_bit); back_args.push(new_lsmode); }

        if let Some(xstop) = xstop_v {
            // length32: stop if xstart_next >= xstop (x_inc) or <= xstop (x_dec)
            let at_xstop_inc = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, xstart_next, xstop);
            let at_xstop_dec = b.ins().icmp(IntCC::SignedLessThanOrEqual,    xstart_next, xstop);
            let at_xstop = b.ins().select(x_dec_v, at_xstop_dec, at_xstop_inc);
            let xstop_block = b.create_block();
            let keep_going  = b.create_block();
            b.ins().brif(at_xstop, xstop_block, &[], keep_going, &[]);

            b.switch_to_block(xstop_block); b.seal_block(xstop_block);
            emit_writeback(&mut b, ctx_ptr, &mem, xstart_next, ystart_v,
                dm0, new_cr, new_cg, new_cb, new_ca, new_zpat_bit, new_pat_bit, new_lsmode);
            b.ins().jump(loop_end, &[]);

            b.switch_to_block(keep_going); b.seal_block(keep_going);
            b.ins().jump(loop_header, &back_args);
        } else {
            b.ins().jump(loop_header, &back_args);
        }
    }

    // ── loop_end ──────────────────────────────────────────────────────────────
    b.switch_to_block(loop_end);
    b.seal_block(loop_end);
    b.seal_block(loop_header); // now all predecessors are known
    b.ins().return_(&[]);
    b.finalize();

    true
}

// ── IR helpers ────────────────────────────────────────────────────────────────

/// Emit `bayer_pack(color, x, y)`: pack bayer index into bits 27:24 of color.
fn bayer_pack_ir(b: &mut FunctionBuilder, color: Value, x: Value, y: Value) -> Value {
    let x3       = b.ins().band_imm(x, 3);
    let y3       = b.ins().band_imm(y, 3);
    let y3_shift = b.ins().ishl_imm(y3, 2);
    let idx      = b.ins().bor(y3_shift, x3);
    let color24  = b.ins().band_imm(color, 0x00FF_FFFFi64);
    let idx_shifted = b.ins().ishl_imm(idx, 24);
    b.ins().bor(color24, idx_shifted)
}

/// clamp_color_component for shade output: extract bits[22:11] (integer), return 8-bit value.
fn clamp_color_component(b: &mut FunctionBuilder, c: Value) -> Value {
    // integer part = (c >> 11) & 0x1FF
    let c_shr11 = b.ins().ushr_imm(c, 11);
    let val = b.ins().band_imm(c_shr11, 0x1FF);
    // if bit31 set or val >= 0x180: 0
    // if val > 0xFF: 0xFF
    // else: val & 0xFF
    let bit31 = b.ins().band_imm(c, 1i64 << 31);
    let neg = b.ins().icmp_imm(IntCC::NotEqual, bit31, 0);
    let overflow = b.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, val, 0x180);
    let clamped_zero = b.ins().bor(neg, overflow);
    let max_255 = b.ins().icmp_imm(IntCC::UnsignedGreaterThan, val, 0xFF);
    let c255 = b.ins().iconst(types::I32, 0xFF);
    let c0   = b.ins().iconst(types::I32, 0);
    let val_masked = b.ins().band_imm(val, 0xFF);
    let clamped_255 = b.ins().select(max_255, c255, val_masked);
    b.ins().select(clamped_zero, c0, clamped_255)
}

/// Clamp a shade DDA value (o12.11): keep as-is for unclamped, or apply RGB clamp.
fn clamp_shade(b: &mut FunctionBuilder, c: Value) -> Value {
    let c_shr11 = b.ins().ushr_imm(c, 11);
    let val = b.ins().band_imm(c_shr11, 0x1FF);
    let bit31 = b.ins().band_imm(c, 1i64 << 31);
    let neg = b.ins().icmp_imm(IntCC::NotEqual, bit31, 0);
    let overflow = b.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, val, 0x180);
    let clamped_zero = b.ins().bor(neg, overflow);
    let max_7ffff = b.ins().icmp_imm(IntCC::UnsignedGreaterThan, val, 0xFF);
    let c7ffff = b.ins().iconst(types::I32, 0x0007_FFFFi64);
    let c0     = b.ins().iconst(types::I32, 0);
    let clamped_max = b.ins().select(max_7ffff, c7ffff, c);
    b.ins().select(clamped_zero, c0, clamped_max)
}

/// Emit a 4-way range check: (x >= min_x && x <= max_x && y >= min_y && y <= max_y) as I1.
fn and4_range(b: &mut FunctionBuilder, x: Value, y: Value,
    min_x: Value, max_x: Value, min_y: Value, max_y: Value) -> Value
{
    // icmp returns I8 in Cranelift 0.116; band them directly.
    let ok_x_lo = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, x, min_x);
    let ok_x_hi = b.ins().icmp(IntCC::SignedLessThanOrEqual,    x, max_x);
    let ok_y_lo = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, y, min_y);
    let ok_y_hi = b.ins().icmp(IntCC::SignedLessThanOrEqual,    y, max_y);
    let ok_x = b.ins().band(ok_x_lo, ok_x_hi);
    let ok_y = b.ins().band(ok_y_lo, ok_y_hi);
    b.ins().band(ok_x, ok_y)
}

/// Amplify an aux-plane pixel value into its packed fb_aux bit position.
/// Mirrors Rex3::amplify_olay/cid/pup:
///   OLAY: (val << 8) | (val << 16)  — stores 8-bit value in bits[23:8]
///   CID:  val | (val << 4)           — stores 2-bit value in bits[5:0]
///   PUP:  (val << 2) | (val << 6)   — stores 2-bit value in bits[7:2]
fn amplify_aux_ir(b: &mut FunctionBuilder, val: Value, planes: u32) -> Value {
    match planes {
        p if p == DRAWMODE1_PLANES_OLAY => {
            let v8  = b.ins().ishl_imm(val, 8);
            let v16 = b.ins().ishl_imm(val, 16);
            b.ins().bor(v8, v16)
        }
        p if p == DRAWMODE1_PLANES_CID => {
            let v4 = b.ins().ishl_imm(val, 4);
            b.ins().bor(val, v4)
        }
        p if p == DRAWMODE1_PLANES_PUP => {
            let v2 = b.ins().ishl_imm(val, 2);
            let v6 = b.ins().ishl_imm(val, 6);
            b.ins().bor(v2, v6)
        }
        _ => val,
    }
}

/// Emit a 16-opcode logic operation.
fn emit_logic_op(b: &mut FunctionBuilder, logicop: u32, src: Value, dst: Value) -> Value {
    match logicop {
        0  => b.ins().iconst(types::I32, 0),
        1  => b.ins().band(src, dst),
        2  => { let nd = b.ins().bnot(dst); b.ins().band(src, nd) }
        3  => src,
        4  => { let ns = b.ins().bnot(src); b.ins().band(ns, dst) }
        5  => dst,
        6  => b.ins().bxor(src, dst),
        7  => b.ins().bor(src, dst),
        8  => { let o = b.ins().bor(src, dst); b.ins().bnot(o) }
        9  => { let x = b.ins().bxor(src, dst); b.ins().bnot(x) }
        10 => b.ins().bnot(dst),
        11 => { let nd = b.ins().bnot(dst); b.ins().bor(src, nd) }
        12 => b.ins().bnot(src),
        13 => { let ns = b.ins().bnot(src); b.ins().bor(ns, dst) }
        14 => { let a = b.ins().band(src, dst); b.ins().bnot(a) }
        15 => b.ins().iconst(types::I32, -1),
        _  => src,
    }
}

/// Write back all mutable per-draw state from IR values to ctx.
fn emit_writeback(
    b: &mut FunctionBuilder,
    ctx_ptr: Value,
    mem: &MemFlags,
    xstart_new: Value,
    ystart_new: Value,
    dm0: &Dm0,
    colorred: Value, colorgrn: Value, colorblue: Value, coloralpha: Value,
    zpat_bit: Value, pat_bit: Value, lsmode: Value,
) {
    macro_rules! st32w { ($off:expr, $val:expr) => {
        b.ins().store(*mem, $val, ctx_ptr, ir::immediates::Offset32::new($off as i32));
    }}
    macro_rules! st8w { ($off:expr, $val:expr) => {
        b.ins().store(*mem, $val, ctx_ptr, ir::immediates::Offset32::new($off as i32));
    }}
    st32w!(ctx_off!(xstart), xstart_new);
    st32w!(ctx_off!(ystart), ystart_new);
    if dm0.shade() {
        st32w!(ctx_off!(colorred),   colorred);
        st32w!(ctx_off!(colorgrn),   colorgrn);
        st32w!(ctx_off!(colorblue),  colorblue);
        st32w!(ctx_off!(coloralpha), coloralpha);
    }
    if dm0.enzpattern() {
        st8w!(ctx_off!(zpat_bit), zpat_bit);
    }
    if dm0.enlspattern() {
        st8w!(ctx_off!(pat_bit), pat_bit);
        st32w!(ctx_off!(lsmode), lsmode);
    }
}

// ── Inline IR: compress / expand / blend ─────────────────────────────────────
//
// All three are specialized on compile-time constants (drawdepth, dither, sfactor, dfactor)
// so Cranelift sees only straight-line operations — no branches, no call overhead.

/// Compress 24-bit RGB (BGR byte-packed, bits[23:0]) to packed drawdepth format.
/// Input `val` has bayer index in bits[31:24] (packed by bayer_pack_ir).
/// Mirrors helper_rgb24_to_rgb{4,8,12}[_dither].
/// Bayer 4x4 ordered dither table lookup: index 0..15 → threshold 0..15.
/// Table: [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
/// Packed as u64: nibble i = BAYER[i], so lookup = (packed >> (idx*4)) & 0xF.
fn emit_bayer_lookup_ir(b: &mut FunctionBuilder, idx: Value) -> Value {
    // packed: nibble 0 = 0, nibble 1 = 8, nibble 2 = 2, ... nibble 15 = 5
    const BAYER_PACKED: i64 = 0x5D7F91B36E4CA280_u64 as i64;
    let packed  = b.ins().iconst(types::I64, BAYER_PACKED);
    let idx64   = b.ins().uextend(types::I64, idx);
    let shift   = b.ins().ishl_imm(idx64, 2); // idx * 4
    let shifted = b.ins().ushr(packed, shift);
    let nibble  = b.ins().band_imm(shifted, 0xF);
    b.ins().ireduce(types::I32, nibble)
}

fn emit_compress_ir(b: &mut FunctionBuilder, val: Value, drawdepth: u32, dither: bool) -> Value {
    if dither {
        // Bayer threshold: val bits[31:24] is the raw bayer index, look it up in the table
        let idx   = b.ins().ushr_imm(val, 24);
        let bayer = emit_bayer_lookup_ir(b, idx);
        // Extract r/g/bv channels (bits[7:0], [15:8], [23:16])
        let r = b.ins().band_imm(val, 0xFF);
        let t8  = b.ins().ushr_imm(val, 8);
        let g   = b.ins().band_imm(t8, 0xFF);
        let t16 = b.ins().ushr_imm(val, 16);
        let bv  = b.ins().band_imm(t16, 0xFF);
        match drawdepth {
            0 => {
                // rgb24→rgb4 dither: sr = (r>>3)-(r>>4), etc.
                let r3 = b.ins().ushr_imm(r, 3);
                let r4 = b.ins().ushr_imm(r, 4);
                let sr = b.ins().isub(r3, r4);
                let g2 = b.ins().ushr_imm(g, 2);
                let g4 = b.ins().ushr_imm(g, 4);
                let sg = b.ins().isub(g2, g4);
                let b3 = b.ins().ushr_imm(bv, 3);
                let b4 = b.ins().ushr_imm(bv, 4);
                let sb = b.ins().isub(b3, b4);
                let sr4_v = b.ins().ushr_imm(sr, 4);
                let sg4_v = b.ins().ushr_imm(sg, 4);
                let sb4_v = b.ins().ushr_imm(sb, 4);
                let mut dr = b.ins().band_imm(sr4_v, 1);
                let mut dg = b.ins().band_imm(sg4_v, 3);
                let mut db = b.ins().band_imm(sb4_v, 1);
                let sr_lo = b.ins().band_imm(sr, 0xF);
                let sg_lo = b.ins().band_imm(sg, 0xF);
                let sb_lo = b.ins().band_imm(sb, 0xF);
                let cond_r = b.ins().icmp(IntCC::UnsignedGreaterThan, sr_lo, bayer);
                let cond_g = b.ins().icmp(IntCC::UnsignedGreaterThan, sg_lo, bayer);
                let cond_b = b.ins().icmp(IntCC::UnsignedGreaterThan, sb_lo, bayer);
                let c1i = b.ins().iconst(types::I32, 1);
                let c3i = b.ins().iconst(types::I32, 3);
                let dr1 = b.ins().iadd(dr, c1i);
                let dg1 = b.ins().iadd(dg, c1i);
                let db1 = b.ins().iadd(db, c1i);
                let dr1_min = b.ins().umin(dr1, c1i);
                let dg1_min = b.ins().umin(dg1, c3i);
                let db1_min = b.ins().umin(db1, c1i);
                dr = b.ins().select(cond_r, dr1_min, dr);
                dg = b.ins().select(cond_g, dg1_min, dg);
                db = b.ins().select(cond_b, db1_min, db);
                let db3s = b.ins().ishl_imm(db, 3);
                let dg1s = b.ins().ishl_imm(dg, 1);
                let t = b.ins().bor(db3s, dg1s);
                b.ins().bor(t, dr)
            }
            1 => {
                // rgb24→rgb8 dither: sr = (r>>1)-(r>>4), etc.
                let r1 = b.ins().ushr_imm(r, 1);
                let r4 = b.ins().ushr_imm(r, 4);
                let sr = b.ins().isub(r1, r4);
                let g1 = b.ins().ushr_imm(g, 1);
                let g4 = b.ins().ushr_imm(g, 4);
                let sg = b.ins().isub(g1, g4);
                let b2 = b.ins().ushr_imm(bv, 2);
                let b4 = b.ins().ushr_imm(bv, 4);
                let sb = b.ins().isub(b2, b4);
                let sr4_v = b.ins().ushr_imm(sr, 4);
                let sg4_v = b.ins().ushr_imm(sg, 4);
                let sb4_v = b.ins().ushr_imm(sb, 4);
                let mut dr = b.ins().band_imm(sr4_v, 7);
                let mut dg = b.ins().band_imm(sg4_v, 7);
                let mut db = b.ins().band_imm(sb4_v, 3);
                let sr_lo = b.ins().band_imm(sr, 0xF);
                let sg_lo = b.ins().band_imm(sg, 0xF);
                let sb_lo = b.ins().band_imm(sb, 0xF);
                let cond_r = b.ins().icmp(IntCC::UnsignedGreaterThan, sr_lo, bayer);
                let cond_g = b.ins().icmp(IntCC::UnsignedGreaterThan, sg_lo, bayer);
                let cond_b = b.ins().icmp(IntCC::UnsignedGreaterThan, sb_lo, bayer);
                let c1i = b.ins().iconst(types::I32, 1);
                let c7i = b.ins().iconst(types::I32, 7);
                let c3i = b.ins().iconst(types::I32, 3);
                let dr1 = b.ins().iadd(dr, c1i);
                let dg1 = b.ins().iadd(dg, c1i);
                let db1 = b.ins().iadd(db, c1i);
                let dr1_min = b.ins().umin(dr1, c7i);
                let dg1_min = b.ins().umin(dg1, c7i);
                let db1_min = b.ins().umin(db1, c3i);
                dr = b.ins().select(cond_r, dr1_min, dr);
                dg = b.ins().select(cond_g, dg1_min, dg);
                db = b.ins().select(cond_b, db1_min, db);
                let db6 = b.ins().ishl_imm(db, 6);
                let dg3 = b.ins().ishl_imm(dg, 3);
                let t = b.ins().bor(db6, dg3);
                b.ins().bor(t, dr)
            }
            _ => {
                // rgb24→rgb12 dither: sr = r - (r>>4), etc.
                let r4 = b.ins().ushr_imm(r, 4);
                let sr = b.ins().isub(r, r4);
                let g4 = b.ins().ushr_imm(g, 4);
                let sg = b.ins().isub(g, g4);
                let b4 = b.ins().ushr_imm(bv, 4);
                let sb = b.ins().isub(bv, b4);
                let sr4_v = b.ins().ushr_imm(sr, 4);
                let sg4_v = b.ins().ushr_imm(sg, 4);
                let sb4_v = b.ins().ushr_imm(sb, 4);
                let mut dr = b.ins().band_imm(sr4_v, 15);
                let mut dg = b.ins().band_imm(sg4_v, 15);
                let mut db = b.ins().band_imm(sb4_v, 15);
                let sr_lo = b.ins().band_imm(sr, 0xF);
                let sg_lo = b.ins().band_imm(sg, 0xF);
                let sb_lo = b.ins().band_imm(sb, 0xF);
                let cond_r = b.ins().icmp(IntCC::UnsignedGreaterThan, sr_lo, bayer);
                let cond_g = b.ins().icmp(IntCC::UnsignedGreaterThan, sg_lo, bayer);
                let cond_b = b.ins().icmp(IntCC::UnsignedGreaterThan, sb_lo, bayer);
                let c1i  = b.ins().iconst(types::I32, 1);
                let c15i = b.ins().iconst(types::I32, 15);
                let dr1 = b.ins().iadd(dr, c1i);
                let dg1 = b.ins().iadd(dg, c1i);
                let db1 = b.ins().iadd(db, c1i);
                let dr1_min = b.ins().umin(dr1, c15i);
                let dg1_min = b.ins().umin(dg1, c15i);
                let db1_min = b.ins().umin(db1, c15i);
                dr = b.ins().select(cond_r, dr1_min, dr);
                dg = b.ins().select(cond_g, dg1_min, dg);
                db = b.ins().select(cond_b, db1_min, db);
                let db8 = b.ins().ishl_imm(db, 8);
                let dg4 = b.ins().ishl_imm(dg, 4);
                let t = b.ins().bor(db8, dg4);
                b.ins().bor(t, dr)
            }
        }
    } else {
        // No dither — straight compress
        match drawdepth {
            0 => {
                // rgb24→rgb4: r=bit7, g=bits[15:14], b=bit23
                let rs = b.ins().ushr_imm(val,  7);
                let r  = b.ins().band_imm(rs, 1);
                let gs = b.ins().ushr_imm(val, 14);
                let g  = b.ins().band_imm(gs, 3);
                let bs = b.ins().ushr_imm(val, 23);
                let bv = b.ins().band_imm(bs, 1);
                let b3 = b.ins().ishl_imm(bv, 3);
                let g1 = b.ins().ishl_imm(g,  1);
                let t  = b.ins().bor(b3, g1);
                b.ins().bor(t, r)
            }
            1 => {
                // rgb24→rgb8: r=bits[7:5], g=bits[15:13], b=bits[23:22]
                let rs = b.ins().ushr_imm(val,  5);
                let r  = b.ins().band_imm(rs, 7);
                let gs = b.ins().ushr_imm(val, 13);
                let g  = b.ins().band_imm(gs, 7);
                let bs = b.ins().ushr_imm(val, 22);
                let bv = b.ins().band_imm(bs, 3);
                let b6 = b.ins().ishl_imm(bv, 6);
                let g3 = b.ins().ishl_imm(g,  3);
                let t  = b.ins().bor(b6, g3);
                b.ins().bor(t, r)
            }
            _ => {
                // rgb24→rgb12: r=bits[7:4], g=bits[15:12], b=bits[23:20]
                let rs = b.ins().ushr_imm(val,  4);
                let r  = b.ins().band_imm(rs, 0xF);
                let gs = b.ins().ushr_imm(val, 12);
                let g  = b.ins().band_imm(gs, 0xF);
                let bs = b.ins().ushr_imm(val, 20);
                let bv = b.ins().band_imm(bs, 0xF);
                let b8 = b.ins().ishl_imm(bv, 8);
                let g4 = b.ins().ishl_imm(g,  4);
                let t  = b.ins().bor(b8, g4);
                b.ins().bor(t, r)
            }
        }
    }
}

/// Expand packed drawdepth format to 24-bit RGB (BGR byte-packed, bits[23:0]).
/// Mirrors helper_rgb{4,8,12}_to_rgb24.
fn emit_expand_ir(b: &mut FunctionBuilder, val: Value, drawdepth: u32) -> Value {
    match drawdepth {
        0 => {
            // rgb4→rgb24: r=bit0 → 0 or 0xFF; g=bits[2:1] → replicated; b=bit3 → 0 or 0xFF
            let r_bit  = b.ins().band_imm(val, 1);
            let r_set  = b.ins().icmp_imm(IntCC::NotEqual, r_bit, 0);
            let c255   = b.ins().iconst(types::I32, 0xFF);
            let c0     = b.ins().iconst(types::I32, 0);
            let r      = b.ins().select(r_set, c255, c0);
            let gs     = b.ins().ushr_imm(val, 1);
            let g_raw  = b.ins().band_imm(gs, 3);
            let g6 = b.ins().ishl_imm(g_raw, 6);
            let g4 = b.ins().ishl_imm(g_raw, 4);
            let g2 = b.ins().ishl_imm(g_raw, 2);
            let t1 = b.ins().bor(g6, g4);
            let t2 = b.ins().bor(g2, g_raw);
            let g  = b.ins().bor(t1, t2);
            let b_shr  = b.ins().ushr_imm(val, 3);
            let b_bit  = b.ins().band_imm(b_shr, 1);
            let b_set  = b.ins().icmp_imm(IntCC::NotEqual, b_bit, 0);
            let bv     = b.ins().select(b_set, c255, c0);
            let g8  = b.ins().ishl_imm(g,  8);
            let b16 = b.ins().ishl_imm(bv, 16);
            let t   = b.ins().bor(b16, g8);
            b.ins().bor(t, r)
        }
        1 => {
            // rgb8→rgb24: r=bits[2:0], g=bits[5:3], b=bits[7:6]
            let r_raw = b.ins().band_imm(val, 7);
            let r5 = b.ins().ishl_imm(r_raw, 5);
            let r2 = b.ins().ishl_imm(r_raw, 2);
            let r1 = b.ins().ushr_imm(r_raw, 1);
            let t1 = b.ins().bor(r5, r2);
            let r  = b.ins().bor(t1, r1);
            let gs    = b.ins().ushr_imm(val, 3);
            let g_raw = b.ins().band_imm(gs, 7);
            let g5 = b.ins().ishl_imm(g_raw, 5);
            let g2 = b.ins().ishl_imm(g_raw, 2);
            let g1 = b.ins().ushr_imm(g_raw, 1);
            let t2 = b.ins().bor(g5, g2);
            let g  = b.ins().bor(t2, g1);
            let bs    = b.ins().ushr_imm(val, 6);
            let b_raw = b.ins().band_imm(bs, 3);
            let b6 = b.ins().ishl_imm(b_raw, 6);
            let b4 = b.ins().ishl_imm(b_raw, 4);
            let b2 = b.ins().ishl_imm(b_raw, 2);
            let t3 = b.ins().bor(b6, b4);
            let t4 = b.ins().bor(b2, b_raw);
            let bv = b.ins().bor(t3, t4);
            let g8  = b.ins().ishl_imm(g,  8);
            let b16 = b.ins().ishl_imm(bv, 16);
            let t   = b.ins().bor(b16, g8);
            b.ins().bor(t, r)
        }
        _ => {
            // rgb12→rgb24: r=bits[3:0], g=bits[7:4], b=bits[11:8]
            let r_raw = b.ins().band_imm(val, 0xF);
            let r4  = b.ins().ishl_imm(r_raw, 4);
            let r   = b.ins().bor(r4, r_raw);
            let gs    = b.ins().ushr_imm(val, 4);
            let g_raw = b.ins().band_imm(gs, 0xF);
            let g4  = b.ins().ishl_imm(g_raw, 4);
            let g   = b.ins().bor(g4, g_raw);
            let bs    = b.ins().ushr_imm(val, 8);
            let b_raw = b.ins().band_imm(bs, 0xF);
            let bb4 = b.ins().ishl_imm(b_raw, 4);
            let bv  = b.ins().bor(bb4, b_raw);
            let g8  = b.ins().ishl_imm(g,  8);
            let b16 = b.ins().ishl_imm(bv, 16);
            let t   = b.ins().bor(b16, g8);
            b.ins().bor(t, r)
        }
    }
}

/// Inline blend: src and dst are 24-bit BGR (alpha in bits[31:24] of src).
/// sfactor/dfactor are compile-time constants (0..=5).
/// Mirrors helper_blend but specialized — constant sfactor/dfactor let Cranelift
/// fold all the factor-selection branches away.
fn emit_blend_ir(b: &mut FunctionBuilder, src: Value, dst: Value, sfactor: u32, dfactor: u32) -> Value {
    let sa   = b.ins().ushr_imm(src, 24); // alpha from src bits[31:24]
    let c255 = b.ins().iconst(types::I32, 255);

    // Extract each 8-bit channel (no nesting)
    let sr   = b.ins().band_imm(src, 0xFF);
    let src8 = b.ins().ushr_imm(src,  8);
    let sg   = b.ins().band_imm(src8, 0xFF);
    let src16= b.ins().ushr_imm(src, 16);
    let sb   = b.ins().band_imm(src16, 0xFF);
    let dr   = b.ins().band_imm(dst, 0xFF);
    let dst8 = b.ins().ushr_imm(dst,  8);
    let dg   = b.ins().band_imm(dst8, 0xFF);
    let dst16= b.ins().ushr_imm(dst, 16);
    let db   = b.ins().band_imm(dst16, 0xFF);
    let dst24= b.ins().ushr_imm(dst, 24);
    let da   = b.ins().band_imm(dst24, 0xFF);

    // get_factor_ir: sel is compile-time constant; comp_other and alpha are runtime Values
    // Returns IR Value for the factor (0..255).
    fn get_factor_ir(b: &mut FunctionBuilder, sel: u32, comp_other: Value, alpha: Value, c255: Value) -> Value {
        match sel {
            0 => b.ins().iconst(types::I32, 0),
            1 => b.ins().iconst(types::I32, 255),
            2 => comp_other,
            3 => b.ins().isub(c255, comp_other),
            4 => alpha,
            5 => b.ins().isub(c255, alpha),
            _ => b.ins().iconst(types::I32, 0),
        }
    }

    // Blend one channel: (sc*sf + dc*df)/255, clamped to 255, shifted
    macro_rules! blend_ch {
        ($sc:expr, $dc:expr, $shift:literal) => {{
            let sf = get_factor_ir(b, sfactor, $dc, sa, c255);
            let df = get_factor_ir(b, dfactor, $sc, sa, c255);
            let sc_sf = b.ins().imul($sc, sf);
            let dc_df = b.ins().imul($dc, df);
            let num   = b.ins().iadd(sc_sf, dc_df);
            let val   = b.ins().udiv(num, c255);
            let cl    = b.ins().umin(val, c255);
            b.ins().ishl_imm(cl, $shift)
        }}
    }

    let r_out = blend_ch!(sr, dr, 0);
    let g_out = blend_ch!(sg, dg, 8);
    let b_out = blend_ch!(sb, db, 16);
    let a_out = blend_ch!(sa, da, 24);

    let t1 = b.ins().bor(r_out, g_out);
    let t2 = b.ins().bor(t1, b_out);
    b.ins().bor(t2, a_out)
}
