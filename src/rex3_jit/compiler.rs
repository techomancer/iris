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

use cranelift_codegen::ir::{self, types, AbiParam, InstBuilder, MemFlagsData, Value};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

/// Wrap a plain-`Value` block-argument list as `BlockArg`s for `jump`/`brif`
/// — cranelift 0.134 gave block arguments their own type (`ir::BlockArg`,
/// supporting `try_call`'s multi-return/exception-payload args alongside
/// plain SSA values) instead of accepting bare `Value`s directly. All our
/// jump targets only ever pass ordinary values, never a `try_call` result,
/// so this is always `BlockArg::Value(v)`.
fn block_args(vals: &[Value]) -> Vec<ir::BlockArg> {
    vals.iter().map(|&v| ir::BlockArg::Value(v)).collect()
}

use crate::rex3::{
    Rex3Context,
    DRAWMODE0_OPCODE_DRAW, DRAWMODE0_OPCODE_SCR2SCR, DRAWMODE0_OPCODE_READ,
    DRAWMODE0_ADRMODE_BLOCK, DRAWMODE0_ADRMODE_SPAN,
    DRAWMODE0_ADRMODE_I_LINE, DRAWMODE0_ADRMODE_F_LINE, DRAWMODE0_ADRMODE_A_LINE,
    DRAWMODE1_PLANES_RGB, DRAWMODE1_PLANES_RGBA,
    DRAWMODE1_PLANES_OLAY, DRAWMODE1_PLANES_PUP, DRAWMODE1_PLANES_CID,
    OCTANT_XDEC, OCTANT_YDEC, OCTANT_XMAJOR,
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
    fn planes(&self)      -> u32  { self.val & 7 }
    fn drawdepth(&self)   -> u32  { (self.val >> 3) & 3 }
    fn dblsrc(&self)      -> bool { self.val & (1 << 5) != 0 }
    fn yflip(&self)       -> bool { self.val & (1 << 6) != 0 }
    fn rwpacked(&self)    -> bool { self.val & (1 << 7) != 0 }
    fn hostdepth(&self)   -> u32  { (self.val >> 8) & 3 }
    fn rwdouble(&self)    -> bool { self.val & (1 << 10) != 0 }
    fn swapendian(&self)  -> bool { self.val & (1 << 11) != 0 }
    fn compare(&self)     -> u32  { (self.val >> 12) & 7 }
    fn rgbmode(&self)     -> bool { self.val & (1 << 15) != 0 }
    fn dither(&self)      -> bool { self.val & (1 << 16) != 0 }
    fn fastclear(&self)   -> bool { self.val & (1 << 17) != 0 }
    fn blend(&self)       -> bool { self.val & (1 << 18) != 0 }
    fn sfactor(&self)     -> u32  { (self.val >> 19) & 7 }
    fn dfactor(&self)     -> u32  { (self.val >> 22) & 7 }
    fn backblend(&self)   -> bool { self.val & (1 << 25) != 0 }
    fn logicop(&self)     -> u32  { (self.val >> 28) & 0xF }

    /// Compute compile-time host pixel count (pixels per host word) from dm1 fields.
    /// Mirrors Rex3::host_setup() count logic.
    fn host_count(&self) -> u32 {
        if !self.rwpacked() { return 1; }
        let depth = self.hostdepth();
        if self.rwdouble() {
            match depth { 0 | 1 => 8, 2 => 4, 3 => 2, _ => 1 }
        } else {
            match depth { 0 | 1 => 4, 2 => 2, 3 => 1, _ => 1 }
        }
    }

    /// Compute compile-time host shift amount (bits per slot).
    /// Mirrors Rex3::host_setup() shift logic.
    fn host_shift(&self) -> u32 {
        if !self.rwpacked() { return 0; }
        let depth = self.hostdepth();
        match depth { 0 | 1 => 8, 2 => 16, 3 => 32, _ => 0 }
    }
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

    /// Compile a shader for the given (DrawMode0, DrawMode1, clipmode_key) triple.
    /// `clipmode_key` = `ctx.clipmode & CLIPMODE_JIT_KEY_MASK` (ensmask bits + cidmatch bits).
    /// Returns `(entry_fn, code_bytes)`, or None if this mode is not JIT-able.
    pub fn compile_shader(&mut self, dm0_val: u32, dm1_val: u32, clipmode_key: u32)
        -> Option<(unsafe extern "C" fn(*mut Rex3Context, *mut u32, *mut u32), u32)>
    {
        let dm0 = Dm0 { val: dm0_val };

        // DRAW/SCR2SCR/READ opcodes; SPAN, BLOCK, or I/F/A_LINE adrmode.
        // Lines do not support host modes (READ/colorhost) — guard below.
        let opcode = dm0.opcode();
        let adrmode = dm0.adrmode() << 2; // match the <<2 convention from rex3.rs
        let is_scr2scr  = opcode == DRAWMODE0_OPCODE_SCR2SCR;
        let is_hostr    = opcode == DRAWMODE0_OPCODE_READ;
        let is_hostw    = opcode == DRAWMODE0_OPCODE_DRAW
            && (dm0.colorhost() || dm0.alphahost());
        let is_line = adrmode == DRAWMODE0_ADRMODE_I_LINE
            || adrmode == DRAWMODE0_ADRMODE_F_LINE
            || adrmode == DRAWMODE0_ADRMODE_A_LINE;
        if opcode != DRAWMODE0_OPCODE_DRAW && !is_scr2scr && !is_hostr {
            eprintln!("REX JIT: not jittable (opcode={opcode}) for dm0={dm0_val:#010x}");
            return None;
        }
        if is_line && (is_hostr || is_hostw) {
            eprintln!("REX JIT: not jittable (line+host) for dm0={dm0_val:#010x}");
            return None;
        }
        if !is_line && adrmode != DRAWMODE0_ADRMODE_SPAN && adrmode != DRAWMODE0_ADRMODE_BLOCK {
            eprintln!("REX JIT: not jittable (adrmode={adrmode:#x}) for dm0={dm0_val:#010x}");
            return None;
        }

        // fastclear takes priority over blend — hardware ignores blend when fastclear=1.
        // SCR2SCR copies already-quantized pixels — dithering would corrupt them (matches execute_go).
        let dm1_val = if dm1_val & (1 << 17) != 0 { dm1_val & !(1 << 18) }
                      else if is_scr2scr          { dm1_val & !(1 << 16) } // clear DITHER
                      else                        { dm1_val };
        let dm1 = Dm1 { val: dm1_val };

        let name = format!("rex_shader_{:08x}_{:08x}_{:08x}_{}", dm0_val, dm1_val, clipmode_key, self.counter);
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

        let target_config = self.jit_module.target_config();
        let result = if is_line {
            emit_draw_iline(
                &mut self.ctx.func,
                &mut self.builder_ctx,
                &dm0, &dm1,
                clipmode_key,
                ptr_type,
                target_config,
            )
        } else {
            emit_shader(
                &mut self.ctx.func,
                &mut self.builder_ctx,
                &dm0, &dm1,
                is_scr2scr, is_hostw, is_hostr,
                clipmode_key,
                ptr_type,
                target_config,
            )
        };

        if !result {
            self.ctx.func.clear();
            self.jit_module.clear_context(&mut self.ctx);
            eprintln!("REX JIT: emit failed for dm0={dm0_val:#010x} dm1={dm1_val:#010x} cm={clipmode_key:#010x}: {}  {}",
                crate::rex3::decode_dm0(dm0_val), crate::rex3::decode_dm1(dm1_val));
            return None;
        }

        if let Err(e) = self.jit_module.define_function(func_id, &mut self.ctx) {
            eprintln!("REX JIT: define_function failed for dm0={dm0_val:#010x} dm1={dm1_val:#010x} cm={clipmode_key:#010x}: {}  {}  -- {e}",
                crate::rex3::decode_dm0(dm0_val), crate::rex3::decode_dm1(dm1_val));
            eprintln!("--- Cranelift IR ---\n{}", self.ctx.func.display());
            self.jit_module.clear_context(&mut self.ctx);
            return None;
        }
        // Read code size before clearing context (compiled_code is cleared by clear_context).
        let code_bytes = self.ctx.compiled_code()
            .map(|cc| cc.code_buffer().len() as u32)
            .unwrap_or(0);
        self.jit_module.clear_context(&mut self.ctx);
        if let Err(e) = self.jit_module.finalize_definitions() {
            eprintln!("REX JIT: finalize_definitions failed: {e}");
            return None;
        }

        let code_ptr = self.jit_module.get_finalized_function(func_id);
        let entry = unsafe {
            std::mem::transmute::<*const u8, unsafe extern "C" fn(*mut Rex3Context, *mut u32, *mut u32)>(code_ptr)
        };
        Some((entry, code_bytes))
    }
}

// ── Shader IR emission ────────────────────────────────────────────────────────

/// Loop-invariant ctx values needed by address calculation and pixel write.
struct PixelCtx {
    xywin_v:     Value, // packed: hi=xwin, lo=ywin
    xymove_v:    Value,
    clipmode_v:  Value,
    wrmask_v:    Value,
    colorback_v: Value,
    colorvram_v: Value,
    smask0x_v:   Value,
    smask0y_v:   Value,
    smask1x_v:   Value,
    smask1y_v:   Value,
    smask2x_v:   Value,
    smask2y_v:   Value,
    smask3x_v:   Value,
    smask3y_v:   Value,
    smask4x_v:   Value,
    smask4y_v:   Value,
    fb_rgb:      Value, // ptr
    fb_aux:      Value, // ptr
}

/// Mirrors Rex3::calculate_src_address.
/// Source address for scr2scr: just xywin applied (no xymove, no smask clipping).
/// On OOB: branches to `src_oob` block (caller puts 0 color there).
/// On hit: falls through and returns `src_px_ptr`.
/// `use_aux`: when true, pointer is into fb_aux (planes=OLAY/PUP/CID).
fn emit_calculate_src_address(
    b:          &mut FunctionBuilder,
    x_v:        Value,
    y_v:        Value,
    pctx:       &PixelCtx,
    src_oob:    ir::Block,
    dm1:        &Dm1,
    use_aux:    bool,
    coord_bias: Value,
    c0:         Value,
    c2048:      Value,
) -> Value { // src_px_ptr
    // Source: (x, y) + xywin — no xymove, no smask (matches calculate_src_address).
    let xw_raw = b.ins().ushr_imm_s(pctx.xywin_v, 16);
    let xw16   = b.ins().ireduce(types::I16, xw_raw);
    let xw32   = b.ins().sextend(types::I32, xw16);
    let yw_raw = b.ins().band_imm_s(pctx.xywin_v, 0xFFFF_i64);
    let yw16   = b.ins().ireduce(types::I16, yw_raw);
    let yw32   = b.ins().sextend(types::I32, yw16);
    let x_abs  = b.ins().iadd(x_v, xw32);
    let y_abs  = b.ins().iadd(y_v, yw32);

    let x_phys = b.ins().isub(x_abs, coord_bias);
    let y_phys = if dm1.yflip() {
        let c23ff = b.ins().iconst(types::I32, 0x23FF);
        b.ins().isub(c23ff, y_abs)
    } else {
        b.ins().isub(y_abs, coord_bias)
    };

    let screen_w = b.ins().iconst(types::I32, REX3_SCREEN_WIDTH as i64);
    let screen_h = b.ins().iconst(types::I32, REX3_SCREEN_HEIGHT as i64);
    let xn = b.ins().icmp(IntCC::SignedLessThan, x_phys, c0);
    let xw = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, x_phys, screen_w);
    let yn = b.ins().icmp(IntCC::SignedLessThan, y_phys, c0);
    let yt = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, y_phys, screen_h);
    let ab  = b.ins().bor(xn, xw);
    let cd  = b.ins().bor(yn, yt);
    let oob = b.ins().bor(ab, cd);
    let in_bounds = b.create_block();
    b.ins().brif(oob, src_oob, &[], in_bounds, &[]);
    b.switch_to_block(in_bounds); b.seal_block(in_bounds);

    let ym2048    = b.ins().imul(y_phys, c2048);
    let addr_i32  = b.ins().iadd(ym2048, x_phys);
    let c4        = b.ins().iconst(types::I32, 4);
    let byte_off  = b.ins().imul(addr_i32, c4);
    let byte_off64 = b.ins().uextend(types::I64, byte_off);
    // Use fb_aux for aux planes (OLAY/PUP/CID), fb_rgb for RGB planes.
    let fb_ptr = if use_aux { pctx.fb_aux } else { pctx.fb_rgb };
    b.ins().iadd(fb_ptr, byte_off64)
}

/// Mirrors Rex3::calculate_fb_address.
/// Emits xymove (unconditional for scr2scr, conditional on xyoffset for draw),
/// xywin, smask clipping, bounds check.
/// `ensmask_key`: compile-time ensmask bits — used to skip dead smask checks.
/// On clip/OOB miss: branches to `skip_block` with `skip_args`.
/// On hit: falls through to a new `in_bounds` block and returns `(px_ptr, x_bayer, y_bayer)`.
/// `x_bayer`/`y_bayer` are the window-relative coords (before xywin bias) — used for dither packing.
fn emit_calculate_fb_address(
    b:            &mut FunctionBuilder,
    x_v:          Value,
    y_v:          Value,
    pctx:         &PixelCtx,
    skip_block:   ir::Block,
    skip_args:    &[Value],
    dm0:          &Dm0,
    dm1:          &Dm1,
    is_scr2scr:   bool,
    ensmask_key:  u32,
    coord_bias:   Value,
    c0:           Value,
    c2048:        Value,
    ptr_type:     ir::Type,
) -> (Value, Value, Value) { // (px_ptr, x_bayer, y_bayer)
    // 1. Apply xymove: always in scr2scr (destination offset), conditional on xyoffset in draw.
    //    Mirrors: apply_xymove = is_scr2scr || ctx.drawmode0.xyoffset()
    let apply_xymove = is_scr2scr || dm0.xyoffset();
    let (x_curr, y_curr) = if apply_xymove {
        let xm_raw = b.ins().ushr_imm_s(pctx.xymove_v, 16);
        let xm16   = b.ins().ireduce(types::I16, xm_raw);
        let xm32   = b.ins().sextend(types::I32, xm16);
        let ym_raw = b.ins().band_imm_s(pctx.xymove_v, 0xFFFF_i64);
        let ym16   = b.ins().ireduce(types::I16, ym_raw);
        let ym32   = b.ins().sextend(types::I32, ym16);
        (b.ins().iadd(x_v, xm32), b.ins().iadd(y_v, ym32))
    } else {
        (x_v, y_v)
    };

    // 2. Apply xywin to get screen-absolute coords
    let xw_raw = b.ins().ushr_imm_s(pctx.xywin_v, 16);
    let xw16   = b.ins().ireduce(types::I16, xw_raw);
    let xw32   = b.ins().sextend(types::I32, xw16);
    let yw_raw = b.ins().band_imm_s(pctx.xywin_v, 0xFFFF_i64);
    let yw16   = b.ins().ireduce(types::I16, yw_raw);
    let yw32   = b.ins().sextend(types::I32, yw16);
    let x_abs  = b.ins().iadd(x_curr, xw32);
    let y_abs  = b.ins().iadd(y_curr, yw32);

    // 3. Scissor / smask clipping — baked from ensmask_key (compile-time constant).
    let after_clip = b.create_block();
    {
        // smask0: window-relative clip rect (bit 0 of ensmask)
        if ensmask_key & 1 != 0 {
            let sm0x_hi    = b.ins().ushr_imm_s(pctx.smask0x_v, 16);
            let sm0x_hi16  = b.ins().ireduce(types::I16, sm0x_hi);
            let min_x0     = b.ins().sextend(types::I32, sm0x_hi16);
            let sm0x_lo16  = b.ins().ireduce(types::I16, pctx.smask0x_v);
            let max_x0     = b.ins().sextend(types::I32, sm0x_lo16);
            let sm0y_hi    = b.ins().ushr_imm_s(pctx.smask0y_v, 16);
            let sm0y_hi16  = b.ins().ireduce(types::I16, sm0y_hi);
            let min_y0     = b.ins().sextend(types::I32, sm0y_hi16);
            let sm0y_lo16  = b.ins().ireduce(types::I16, pctx.smask0y_v);
            let max_y0     = b.ins().sextend(types::I32, sm0y_lo16);
            let ok0 = and4_range(b, x_curr, y_curr, min_x0, max_x0, min_y0, max_y0);
            let pass0 = b.create_block();
            b.ins().brif(ok0, pass0, &[], skip_block, &block_args(skip_args));
            b.switch_to_block(pass0); b.seal_block(pass0);
        }

        // smasks 1-4: screen-absolute, OR-combined (any enabled mask passes the pixel).
        let active14: &[(Value, Value, u32)] = &[
            (pctx.smask1x_v, pctx.smask1y_v, 2),
            (pctx.smask2x_v, pctx.smask2y_v, 4),
            (pctx.smask3x_v, pctx.smask3y_v, 8),
            (pctx.smask4x_v, pctx.smask4y_v, 16),
        ];
        let enabled14: Vec<_> = active14.iter().filter(|(_, _, bit)| ensmask_key & bit != 0).collect();
        if !enabled14.is_empty() {
            let mut inside_any: Value = b.ins().iconst(types::I8, 0);
            for (sx, sy, _) in &enabled14 {
                let sx_hi   = b.ins().ushr_imm_s(*sx, 16);
                let sx_hi16 = b.ins().ireduce(types::I16, sx_hi);
                let min_x   = b.ins().sextend(types::I32, sx_hi16);
                let sx_lo16 = b.ins().ireduce(types::I16, *sx);
                let max_x   = b.ins().sextend(types::I32, sx_lo16);
                let sy_hi   = b.ins().ushr_imm_s(*sy, 16);
                let sy_hi16 = b.ins().ireduce(types::I16, sy_hi);
                let min_y   = b.ins().sextend(types::I32, sy_hi16);
                let sy_lo16 = b.ins().ireduce(types::I16, *sy);
                let max_y   = b.ins().sextend(types::I32, sy_lo16);
                let in_range = and4_range(b, x_abs, y_abs, min_x, max_x, min_y, max_y);
                inside_any   = b.ins().bor(inside_any, in_range);
            }
            let inside = b.ins().icmp_imm_s(IntCC::NotEqual, inside_any, 0);
            b.ins().brif(inside, after_clip, &[], skip_block, &block_args(skip_args));
        } else {
            b.ins().jump(after_clip, &[]);
        }
    }
    b.switch_to_block(after_clip); b.seal_block(after_clip);

    // 4. Physical address
    let y_phys = if dm1.yflip() {
        let c23ff = b.ins().iconst(types::I32, 0x23FF);
        b.ins().isub(c23ff, y_abs)
    } else {
        b.ins().isub(y_abs, coord_bias)
    };
    let x_phys = b.ins().isub(x_abs, coord_bias);

    // Bounds check
    let screen_w = b.ins().iconst(types::I32, REX3_SCREEN_WIDTH as i64);
    let screen_h = b.ins().iconst(types::I32, REX3_SCREEN_HEIGHT as i64);
    let oob = {
        let xn = b.ins().icmp(IntCC::SignedLessThan, x_phys, c0);
        let xw = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, x_phys, screen_w);
        let yn = b.ins().icmp(IntCC::SignedLessThan, y_phys, c0);
        let yt = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, y_phys, screen_h);
        let ab = b.ins().bor(xn, xw);
        let cd = b.ins().bor(yn, yt);
        b.ins().bor(ab, cd)
    };
    let in_bounds = b.create_block();
    b.ins().brif(oob, skip_block, &block_args(skip_args), in_bounds, &[]);
    b.switch_to_block(in_bounds); b.seal_block(in_bounds);

    // Compute byte pointer into framebuffer
    let ym2048    = b.ins().imul(y_phys, c2048);
    let addr_i32  = b.ins().iadd(ym2048, x_phys);
    let c4        = b.ins().iconst(types::I32, 4);
    let byte_off  = b.ins().imul(addr_i32, c4);
    let byte_off64 = b.ins().uextend(types::I64, byte_off);
    let use_aux   = matches!(dm1.planes(),
        p if p == DRAWMODE1_PLANES_OLAY || p == DRAWMODE1_PLANES_PUP || p == DRAWMODE1_PLANES_CID);
    let fb_ptr    = if use_aux { pctx.fb_aux } else { pctx.fb_rgb };
    let px_ptr    = b.ins().iadd(fb_ptr, byte_off64);
    let _ptr_type = ptr_type; // consumed by caller if needed

    (px_ptr, x_curr, y_curr)
}

/// Mirrors the inner body of Rex3::process_pixel_draw / process_pixel_fastclear
/// after `calculate_fb_address` succeeds.
/// `src_color`: already-resolved source (colorback substitution applied by caller).
/// `x_bayer`/`y_bayer`: window-relative coords for dither index.
/// `is_hostw`: when true, src_color is already from host — colorhost beats fastclear
/// Emits the fb read → fastclear/blend/logicop result → wrmask → store.
fn emit_pixel_write(
    b:           &mut FunctionBuilder,
    px_ptr:      Value,
    x_bayer:     Value,
    y_bayer:     Value,
    src_color:   Value,
    pctx:        &PixelCtx,
    mem:         &MemFlagsData,
    memv:        &MemFlagsData,
    dm1:         &Dm1,
    is_hostw:    bool,
) {
    let use_aux     = matches!(dm1.planes(),
        p if p == DRAWMODE1_PLANES_OLAY || p == DRAWMODE1_PLANES_PUP || p == DRAWMODE1_PLANES_CID);
    let depth_mask: i64  = match dm1.drawdepth() { 0 => 0xF, 1 => 0xFF, 2 => 0xFFF, _ => 0xFFFFFF };
    let dblsrc_shift: i64 = match dm1.drawdepth() { 0 => 4, 1 => 8, 2 => 12, _ => 0 };
    let (aux_read_shift0, aux_read_shift1, aux_read_mask): (i64, i64, i64) = match dm1.planes() {
        p if p == DRAWMODE1_PLANES_OLAY => (8,  16, 0xFF),
        p if p == DRAWMODE1_PLANES_CID  => (0,  4,  0x3),
        p if p == DRAWMODE1_PLANES_PUP  => (2,  6,  0x3),
        _                               => (0,  0,  0),
    };

    let needs_dst = dm1.blend() || dm1.logicop() != 3;
    let fb_px_raw: Value = if needs_dst {
        b.ins().load(types::I32, *memv, px_ptr, ir::immediates::Offset32::new(0))
    } else {
        b.ins().iconst(types::I32, 0)
    };

    let dst_plane: Value = if needs_dst {
        if use_aux {
            let read_shift = if dm1.dblsrc() { aux_read_shift1 } else { aux_read_shift0 };
            let extracted  = if read_shift > 0 { b.ins().ushr_imm_s(fb_px_raw, read_shift) } else { fb_px_raw };
            b.ins().band_imm_s(extracted, aux_read_mask)
        } else if dm1.dblsrc() && dblsrc_shift > 0 {
            let shifted = b.ins().ushr_imm_s(fb_px_raw, dblsrc_shift);
            b.ins().band_imm_s(shifted, depth_mask)
        } else {
            b.ins().band_imm_s(fb_px_raw, depth_mask)
        }
    } else {
        b.ins().iconst(types::I32, 0)
    };

    // Mirrors fastclear_color / blend / logic_op paths in process_pixel_draw.
    // colorhost beats fastclear
    let result_val: Value = if dm1.fastclear() && !is_hostw {
        // fastclear_color: replicate colorvram nibble/byte/etc into all plane slots
        match dm1.drawdepth() {
            0 => {
                let c   = b.ins().band_imm_s(pctx.colorvram_v, 0xf);
                let c4  = b.ins().ishl_imm_s(c, 4);
                let c8  = b.ins().ishl_imm_s(c, 8);
                let c16 = b.ins().ishl_imm_s(c, 16);
                let r1 = b.ins().bor(c, c4);
                let r2 = b.ins().bor(c8, c16);
                b.ins().bor(r1, r2)
            }
            1 => {
                let c   = b.ins().band_imm_s(pctx.colorvram_v, 0xff);
                let c8  = b.ins().ishl_imm_s(c, 8);
                let c16 = b.ins().ishl_imm_s(c, 16);
                let r1 = b.ins().bor(c, c8);
                b.ins().bor(r1, c16)
            }
            2 => {
                if dm1.rgbmode() {
                    let hi  = b.ins().band_imm_s(pctx.colorvram_v, 0xf00000i64);
                    let his = b.ins().ushr_imm_s(hi, 12);
                    let mid = b.ins().band_imm_s(pctx.colorvram_v, 0xf000i64);
                    let mids= b.ins().ushr_imm_s(mid, 8);
                    let lo  = b.ins().band_imm_s(pctx.colorvram_v, 0xf0i64);
                    let los = b.ins().ushr_imm_s(lo, 4);
                    let r1  = b.ins().bor(his, mids);
                    let c   = b.ins().bor(r1, los);
                    let c12 = b.ins().ishl_imm_s(c, 12);
                    b.ins().bor(c, c12)
                } else {
                    let c   = b.ins().band_imm_s(pctx.colorvram_v, 0xfffi64);
                    let c12 = b.ins().ishl_imm_s(c, 12);
                    b.ins().bor(c, c12)
                }
            }
            _ => b.ins().band_imm_s(pctx.colorvram_v, 0xffffffi64),
        }
    } else if dm1.blend() {
        // blend path: expand dst → blend → compress + amplify
        let dst_24 = if dm1.rgbmode() && dm1.drawdepth() != 3 {
            emit_expand_ir(b, dst_plane, dm1.drawdepth())
        } else {
            dst_plane
        };
        let blend_dst = if dm1.backblend() { pctx.colorback_v } else { dst_24 };
        let blended   = emit_blend_ir(b, src_color, blend_dst, dm1.sfactor(), dm1.dfactor());
        let compressed = if dm1.rgbmode() && dm1.drawdepth() != 3 {
            let packed = bayer_pack_ir(b, blended, x_bayer, y_bayer);
            emit_compress_ir(b, packed, dm1.drawdepth(), dm1.dither())
        } else { blended };
        if dblsrc_shift > 0 {
            let shifted = b.ins().ishl_imm_s(compressed, dblsrc_shift);
            b.ins().bor(compressed, shifted)
        } else { compressed }
    } else {
        // logic op path: compress → amplify src, amplify dst, logic op
        let compressed = if dm1.rgbmode() && dm1.drawdepth() != 3 {
            let packed = bayer_pack_ir(b, src_color, x_bayer, y_bayer);
            emit_compress_ir(b, packed, dm1.drawdepth(), dm1.dither())
        } else { src_color };
        let amp_src = if use_aux {
            amplify_aux_ir(b, compressed, dm1.planes())
        } else if dblsrc_shift > 0 {
            let shifted = b.ins().ishl_imm_s(compressed, dblsrc_shift);
            b.ins().bor(compressed, shifted)
        } else { compressed };
        let amp_dst = if use_aux {
            amplify_aux_ir(b, dst_plane, dm1.planes())
        } else if dblsrc_shift > 0 {
            let shifted = b.ins().ishl_imm_s(dst_plane, dblsrc_shift);
            b.ins().bor(dst_plane, shifted)
        } else { dst_plane };
        emit_logic_op(b, dm1.logicop(), amp_src, amp_dst)
    };

    // Write: (fb[addr] & !wrmask) | (result & wrmask)
    let old_val  = b.ins().load(types::I32, *memv, px_ptr, ir::immediates::Offset32::new(0));
    let inv_mask = b.ins().bnot(pctx.wrmask_v);
    let kept     = b.ins().band(old_val, inv_mask);
    let written  = b.ins().band(result_val, pctx.wrmask_v);
    let new_val = b.ins().bor(kept, written);
    b.ins().store(*mem, new_val, px_ptr, ir::immediates::Offset32::new(0));
}

/// Emit the full shader IR. Returns false if the mode cannot be compiled.
fn emit_shader(
    func: &mut ir::Function,
    builder_ctx: &mut FunctionBuilderContext,
    dm0: &Dm0,
    dm1: &Dm1,
    is_scr2scr: bool,
    is_hostw: bool,    // DRAW + colorhost/alphahost: read pixels from ctx.hostrw
    is_hostr: bool,    // READ opcode: pack fb pixels into ctx.hostrw
    clipmode_key: u32,
    ptr_type: ir::Type,
    target_config: cranelift_codegen::isa::TargetFrontendConfig,
) -> bool {
    let ensmask_key = clipmode_key & 0x1F;
    let cidmatch    = (clipmode_key >> 9) & 0xF; // 0xF = disabled
    let mut b = FunctionBuilder::new(func, builder_ctx);
    let mem  = MemFlagsData::trusted();
    let memv = MemFlagsData::new(); // for potentially-aliased reads after stores

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
        let shifted = b.ins().ushr_imm_s(bres_v, 24);
        b.ins().band_imm_s(shifted, 7)
    };
    let x_dec_v = {
        let bit = b.ins().band_imm_s(octant_v, OCTANT_XDEC as i64);
        b.ins().icmp_imm_s(IntCC::NotEqual, bit, 0)
    };
    let y_dec_v = {
        let bit = b.ins().band_imm_s(octant_v, OCTANT_YDEC as i64);
        b.ins().icmp_imm_s(IntCC::NotEqual, bit, 0)
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
    // + host state if hostw/hostr: [host_shifter: i64]
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
    if is_hostw || is_hostr {
        b.append_block_param(loop_header, types::I64); // host_shifter
        b.append_block_param(loop_header, types::I32); // hostcnt
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
            let is_neg = b.ins().icmp_imm_s(IntCC::SignedLessThan, diff, 0);
            b.ins().select(is_neg, neg, diff)
        };
        let span_len_v = b.ins().ushr_imm_s(diff_abs, 11);
        let c32i = b.ins().iconst(types::I32, 32);
        let long_enough = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, span_len_v, c32i);
        // If span < 32: use a sentinel that never triggers (xend ± 1 step beyond)
        let never = b.ins().iadd(xend_v, step32);
        Some(b.ins().select(long_enough, xstop_raw, never))
    } else {
        None
    };

    // host_count: compile-time number of pixels per host word (from dm1 fields).
    // For host mode the loop stops after host_count pixels (one word), like stop_on_word.
    // We compute an xstop analogous to length32 for this.
    let host_count = if is_hostw || is_hostr { dm1.host_count() } else { 0 };
    let host_xstop_v: Option<Value> = if (is_hostw || is_hostr) && host_count > 1 {
        // Stop after host_count pixels: xstop = xstart + host_count * stepx
        let offset = b.ins().iconst(types::I32, (host_count as i64) << 11);
        let offset_neg = b.ins().iconst(types::I32, -((host_count as i64) << 11));
        let step_hc = b.ins().select(x_dec_v, offset_neg, offset);
        Some(b.ins().iadd(xstart_init, step_hc))
    } else if (is_hostw || is_hostr) && host_count == 1 {
        // Single pixel per word: loop runs once then x_end_reached exits naturally,
        // but we also need stoponx to be true (below). Use xstop set to xstart+stepx
        // (same as next pixel) so cont_x_block immediately stops.
        Some(b.ins().iadd(xstart_init, stepx_v))
    } else {
        None
    };

    // Initial host shifter: for HOSTW load ctx.hostrw; for HOSTR start at 0.
    // Apply swapendian at shader entry if needed (compile-time constant from dm1).
    let host_shifter_init: Value = if is_hostw {
        let raw = {
            let hi = b.ins().load(types::I64, mem, ctx_ptr,
                ir::immediates::Offset32::new(ctx_off!(hostrw) as i32));
            hi
        };
        if dm1.swapendian() {
            if dm1.rwdouble() {
                // swap all 8 bytes
                emit_bswap64(&mut b, raw)
            } else {
                // swap only the high 32-bit word (32-bit write lands in [63:32])
                let hi32 = b.ins().ushr_imm_s(raw, 32);
                let hi32_i32 = b.ins().ireduce(types::I32, hi32);
                let hi32_swap = emit_bswap32(&mut b, hi32_i32);
                let hi32_ext = b.ins().uextend(types::I64, hi32_swap);
                let lo32 = b.ins().band_imm_s(raw, 0xFFFF_FFFFi64);
                let hi32_shifted = b.ins().ishl_imm_s(hi32_ext, 32);
                b.ins().bor(hi32_shifted, lo32)
            }
        } else { raw }
    } else if is_hostr {
        b.ins().iconst(types::I64, 0)
    } else {
        b.ins().iconst(types::I64, 0)
    };

    // Mirrors Rex3Context::hostcnt: pixel slots still unfilled in the word
    // currently being packed/consumed. Loaded fresh from ctx at shader entry
    // (same persisted per-context state the interpreter reads/writes), so a
    // primitive whose GOs are split across interpreter and JIT dispatch stays
    // in sync. Reset to host_count at the start of a new word (loop_header,
    // when the previous GO fully drained/filled its word), decremented once
    // per pixel processed, written back to ctx.hostcnt at every exit.
    let hostcnt_init: Value = if is_hostw || is_hostr {
        ld32!(ctx_off!(hostcnt))
    } else {
        c0
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
    if is_hostw || is_hostr {
        init_args.push(host_shifter_init);
        init_args.push(hostcnt_init);
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

    b.ins().jump(loop_header, &block_args(&init_args));

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
    let (host_shifter_v, hostcnt_v): (Value, Value) = if is_hostw || is_hostr {
        let s = hp[param_idx]; let c = hp[param_idx+1]; param_idx += 2; (s, c)
    } else {
        (b.ins().iconst(types::I64, 0), c0)
    };
    let _ = param_idx;

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
        let first_is_zero   = b.ins().icmp_imm_s(IntCC::Equal, first_v, 0);     // 1 if not-first
        let not_x_end       = b.ins().icmp_imm_s(IntCC::Equal, x_end_reached, 0); // 1 if not-last
        b.ins().band(first_is_zero, not_x_end)
    } else if dm0.skipfirst() {
        b.ins().icmp_imm_s(IntCC::Equal, first_v, 0)    // 1 when first==0 (not the first pixel)
    } else if dm0.skiplast() {
        b.ins().icmp_imm_s(IntCC::Equal, x_end_reached, 0) // 1 when not at end
    } else {
        b.ins().iconst(types::I8, 1) // always draw
    };

    // LRONLY block mode: AND with !x_dec (skip pixel when x_dec=1, shade still runs).
    let do_pixel = if dm0.lronly() && is_block {
        let not_x_dec = b.ins().icmp_imm_s(IntCC::Equal, x_dec_v, 0); // 1 when x_dec=0
        b.ins().band(do_pixel, not_x_dec)
    } else {
        do_pixel
    };

    // skip_block carries the updated host_shifter/hostcnt when in host mode.
    // This is necessary because clipped pixels still consume a host shifter slot
    // (fetch_host_pixel is called before calculate_fb_address in the interpreter).
    let pixel_block = b.create_block();
    let skip_block  = b.create_block();
    if is_hostw || is_hostr {
        b.append_block_param(skip_block, types::I64); // host_shifter (updated)
        b.append_block_param(skip_block, types::I32); // hostcnt (updated)
    }

    // Branch: do_pixel is I1 (from bnot/icmp) or I8 constant 1.
    // On skip (skipfirst/skiplast): host shifter/hostcnt are NOT advanced (proc_fn is not called).
    let skip_init_args_buf: [Value; 2] = [host_shifter_v, hostcnt_v];
    let skip_init_args: &[Value] = if is_hostw || is_hostr {
        &skip_init_args_buf[..]
    } else { &[] };
    b.ins().brif(do_pixel, pixel_block, &[], skip_block, &block_args(skip_init_args));

    // ── pixel_block: fetch host pixel (if any), compute address, draw ─────────
    b.switch_to_block(pixel_block);
    b.seal_block(pixel_block);

    let pctx = PixelCtx {
        xywin_v, xymove_v, clipmode_v, wrmask_v, colorback_v, colorvram_v,
        smask0x_v, smask0y_v, smask1x_v, smask1y_v,
        smask2x_v, smask2y_v, smask3x_v, smask3y_v, smask4x_v, smask4y_v,
        fb_rgb, fb_aux,
    };

    // Host pixel is fetched only when the pattern passes and we're drawing from host (not colorback).
    // We pre-compute the fetched shifter here so it's available after pattern checks, but
    // clip-skips and pattern-skips pass host_shifter_v (un-fetched) to skip_block.
    let (host_pixel_v, host_shifter_after_fetch) = if is_hostw {
        emit_fetch_host_pixel_ir(&mut b, host_shifter_v, dm1)
    } else {
        let z = b.ins().iconst(types::I32, 0);
        (z, host_shifter_v)
    };

    // hostcnt after consuming one slot: mirrors fetch_host_pixel/store_host_pixel
    // (rex3.rs) — reload to host_count on a fresh word (hostcnt==0), then -1.
    // Used at every "this pixel consumed a host word slot" jump into skip_block;
    // sites that skip without consuming (skipfirst/skiplast, pattern-miss-not-
    // opaque) pass hostcnt_v unchanged instead.
    let hostcnt_after_consume: Value = if is_hostw || is_hostr {
        let hc = dm1.host_count() as i64;
        let host_count_c = b.ins().iconst(types::I32, hc);
        let is_fresh = b.ins().icmp_imm_s(IntCC::Equal, hostcnt_v, 0);
        let reloaded = b.ins().select(is_fresh, host_count_c, hostcnt_v);
        b.ins().isub(reloaded, c1)
    } else {
        c0
    };

    // Clip/pattern-skip: host pixel NOT consumed (pass un-fetched shifter).
    let no_fetch_skip_args_buf: [Value; 2] = [host_shifter_v, hostcnt_v];
    let clip_skip_args: &[Value] = if is_hostw || is_hostr {
        &no_fetch_skip_args_buf[..]
    } else { &[] };
    // After a successful draw from host: pass the fetched (advanced) shifter/hostcnt.
    let clip_skip_args_buf: [Value; 2] = [host_shifter_after_fetch, hostcnt_after_consume];

    let (px_ptr, x_bayer, y_bayer) = emit_calculate_fb_address(
        &mut b, x_v, y_v, &pctx, skip_block, clip_skip_args, dm0, dm1, is_scr2scr,
        ensmask_key, coord_bias, c0, c2048, ptr_type,
    );

    // CID mask check: skip write if fb_aux[addr] & 0xF != cidmatch.
    // Only emitted when cidmatch != 0xF (0xF = disabled).
    // Does not apply to HOSTR (READ) since that reads from fb, not writes to it.
    if cidmatch != 0xF && !is_hostr {
        let aux_ptr = {
            // Re-derive the fb_aux byte offset from px_ptr (which is already byte-offset into fb_rgb).
            // fb_rgb and fb_aux share the same stride/layout (2048 u32 entries per row),
            // so the byte offset into fb_aux is the same as into fb_rgb.
            let fb_rgb_base = pctx.fb_rgb;
            let byte_off64  = b.ins().isub(px_ptr, fb_rgb_base);
            b.ins().iadd(pctx.fb_aux, byte_off64)
        };
        let aux_raw = b.ins().load(types::I32, memv, aux_ptr, ir::immediates::Offset32::new(0));
        let aux_lo4 = b.ins().band_imm_s(aux_raw, 0xF_i64);
        let cid_matches = b.ins().icmp_imm_s(IntCC::Equal, aux_lo4, cidmatch as i64);
        let cid_ok = b.create_block();
        let cid_skip_args: &[Value] = if is_hostw { &clip_skip_args_buf[..] } else { &[] };
        b.ins().brif(cid_matches, cid_ok, &[], skip_block, &block_args(cid_skip_args));
        b.switch_to_block(cid_ok); b.seal_block(cid_ok);
    }

    // depth_mask used by scr2scr src read and by emit_pixel_write.
    let depth_mask: i64 = match dm1.drawdepth() { 0 => 0xF, 1 => 0xFF, 2 => 0xFFF, _ => 0xFFFFFF };

    // ── HOSTR (READ opcode): read fb pixel, pack into host shifter, jump to skip ──
    // Mirrors process_pixel_read: calculate_fb_address + expand + store_host_pixel.
    // No fb write; jump to skip_block with updated shifter.
    let new_shifter_after_pixel: Value = if is_hostr {
        let fb_raw = b.ins().load(types::I32, memv, px_ptr, ir::immediates::Offset32::new(0));
        let depth_mask_i: i64 = match dm1.drawdepth() { 0 => 0xF, 1 => 0xFF, 2 => 0xFFF, _ => 0xFFFFFF };
        let masked = b.ins().band_imm_s(fb_raw, depth_mask_i);
        let expanded = if dm1.rgbmode() && dm1.drawdepth() != 3 {
            emit_expand_ir(&mut b, masked, dm1.drawdepth())
        } else { masked };
        let new_s = emit_pack_host_pixel_ir(&mut b, host_shifter_after_fetch, expanded, dm1);
        b.ins().jump(skip_block, &[ir::BlockArg::Value(new_s), ir::BlockArg::Value(hostcnt_after_consume)]);
        new_s // unreachable but satisfies type checker
    } else {
        // ── Source color ──────────────────────────────────────────────────────
        // draw_use_bg_bool: set in the DRAW (non-scr2scr) path, used after src_color is resolved
        // to decide whether we consumed a host pixel (false) or drew colorback (true).
        let mut draw_use_bg_bool: Value = b.ins().iconst(types::I8, 0); // default: drew from host/DDA
        let src_color = if is_scr2scr {
            let use_aux = matches!(dm1.planes(),
                p if p == DRAWMODE1_PLANES_OLAY || p == DRAWMODE1_PLANES_PUP || p == DRAWMODE1_PLANES_CID);
            let (aux_read_shift0, aux_read_shift1, aux_read_mask): (i64, i64, i64) = match dm1.planes() {
                p if p == DRAWMODE1_PLANES_OLAY => (8,  16, 0xFF),
                p if p == DRAWMODE1_PLANES_CID  => (0,  4,  0x3),
                p if p == DRAWMODE1_PLANES_PUP  => (2,  6,  0x3),
                _                               => (0,  0,  0),
            };
            let src_oob    = b.create_block();
            let src_valid  = b.create_block();
            b.append_block_param(src_valid, types::I32);
            let src_ptr = emit_calculate_src_address(
                &mut b, x_v, y_v, &pctx, src_oob, dm1, use_aux, coord_bias, c0, c2048,
            );
            let src_raw = b.ins().load(types::I32, memv, src_ptr, ir::immediates::Offset32::new(0));
            let src_expanded = if use_aux {
                // Aux plane: extract the plane bits with shift+mask (mirrors rd_fn in interpreter).
                let read_shift = if dm1.dblsrc() { aux_read_shift1 } else { aux_read_shift0 };
                let extracted  = if read_shift > 0 { b.ins().ushr_imm_s(src_raw, read_shift) } else { src_raw };
                b.ins().band_imm_s(extracted, aux_read_mask)
            } else {
                // RGB plane: mask to depth bits then expand to 24-bit if needed.
                let src_masked = b.ins().band_imm_s(src_raw, depth_mask as i64);
                if dm1.rgbmode() && dm1.drawdepth() != 3 {
                    emit_expand_ir(&mut b, src_masked, dm1.drawdepth())
                } else { src_masked }
            };
            b.ins().jump(src_valid, &[ir::BlockArg::Value(src_expanded)]);
            b.switch_to_block(src_oob); b.seal_block(src_oob);
            let zero = b.ins().iconst(types::I32, 0);
            b.ins().jump(src_valid, &[ir::BlockArg::Value(zero)]);
            b.switch_to_block(src_valid); b.seal_block(src_valid);
            b.block_params(src_valid).to_vec()[0]
        } else {
            // DRAW path (possibly HOSTW): resolve source color from DDA / host pixel.
            let raw_src = if dm0.shade() {
                let r  = clamp_color_component(&mut b, colorred_v);
                let g  = clamp_color_component(&mut b, colorgrn_v);
                let bl = clamp_color_component(&mut b, colorblue_v);
                let g8   = b.ins().ishl_imm_s(g, 8);
                let bl16 = b.ins().ishl_imm_s(bl, 16);
                let rb   = b.ins().bor(r, g8);
                b.ins().bor(rb, bl16)
            } else if is_hostw && dm0.colorhost() {
                host_pixel_v
            } else if dm1.rgbmode() {
                let r  = ld32!(ctx_off!(colorred));
                let g  = ld32!(ctx_off!(colorgrn));
                let bl = ld32!(ctx_off!(colorblue));
                let r_c  = clamp_color_component(&mut b, r);
                let g_c  = clamp_color_component(&mut b, g);
                let b_c  = clamp_color_component(&mut b, bl);
                let g8   = b.ins().ishl_imm_s(g_c, 8);
                let bl16 = b.ins().ishl_imm_s(b_c, 16);
                let rb   = b.ins().bor(r_c, g8);
                b.ins().bor(rb, bl16)
            } else {
                let cr = ld32!(ctx_off!(colorred));
                b.ins().sshr(cr, c11)
            };

            // Afunction's source alpha is "either DDA or host" (selected by ALPHAHOST),
            // independent of RGB/CI plane format — mirrors combine_host_dda in rex3.rs.
            // Always overlay bits 31:24 so px_afunc/afunc_cmp below see the right value;
            // CI write paths mask down to the plane-depth index and never see these bits.
            let alpha_byte = if dm0.alphahost() {
                b.ins().band_imm_s(host_pixel_v, 0xFF00_0000u64 as i64)
            } else {
                let ca = ld32!(ctx_off!(coloralpha));
                let ca_c = clamp_color_component(&mut b, ca);
                b.ins().ishl_imm_s(ca_c, 24)
            };
            let color24 = b.ins().band_imm_s(raw_src, 0x00FF_FFFFi64);
            let raw_src = b.ins().bor(color24, alpha_byte);

            // Pattern checks (zpopaque/lsopaque can override src with colorback)
            let draw_block = b.create_block();
            b.append_block_param(draw_block, types::I8);
            let mut cur_use_bg: Value = b.ins().iconst(types::I8, 0);

            if dm0.enzpattern() {
                let zp_block = b.create_block();
                let zp_pass  = b.create_block();
                b.append_block_param(zp_pass, types::I8);
                let zpattern_v   = ld32!(ctx_off!(zpattern));
                let zpat_bit32   = b.ins().uextend(types::I32, zpat_bit_v);
                let zpat_shifted = b.ins().ushr(zpattern_v, zpat_bit32);
                let bit_v  = b.ins().band_imm_s(zpat_shifted, 1);
                let bit_set = b.ins().icmp_imm_s(IntCC::NotEqual, bit_v, 0);
                b.ins().brif(bit_set, zp_pass, &[ir::BlockArg::Value(cur_use_bg)], zp_block, &[]);
                b.switch_to_block(zp_block); b.seal_block(zp_block);
                if dm0.zpopaque() {
                    let bg1 = b.ins().iconst(types::I8, 1);
                    b.ins().jump(zp_pass, &[ir::BlockArg::Value(bg1)]);
                } else {
                    // Pattern miss, not opaque: skip without consuming host pixel.
                    let skip_args_here: &[Value] = if is_hostw { &no_fetch_skip_args_buf[..] } else { &[] };
                    b.ins().jump(skip_block, &block_args(skip_args_here));
                }
                b.switch_to_block(zp_pass); b.seal_block(zp_pass);
                cur_use_bg = b.block_params(zp_pass).to_vec()[0];
            }

            if dm0.enlspattern() {
                let ls_block = b.create_block();
                let ls_pass  = b.create_block();
                b.append_block_param(ls_pass, types::I8);
                let lspattern_v  = ld32!(ctx_off!(lspattern));
                let pat_bit32    = b.ins().uextend(types::I32, pat_bit_v);
                let lspat_shifted = b.ins().ushr(lspattern_v, pat_bit32);
                let bit_v  = b.ins().band_imm_s(lspat_shifted, 1);
                let bit_set = b.ins().icmp_imm_s(IntCC::NotEqual, bit_v, 0);
                b.ins().brif(bit_set, ls_pass, &[ir::BlockArg::Value(cur_use_bg)], ls_block, &[]);
                b.switch_to_block(ls_block); b.seal_block(ls_block);
                if dm0.lsopaque() {
                    let bg1 = b.ins().iconst(types::I8, 1);
                    b.ins().jump(ls_pass, &[ir::BlockArg::Value(bg1)]);
                } else {
                    // Pattern miss, not opaque: skip without consuming host pixel.
                    let skip_args_here: &[Value] = if is_hostw { &no_fetch_skip_args_buf[..] } else { &[] };
                    b.ins().jump(skip_block, &block_args(skip_args_here));
                }
                b.switch_to_block(ls_pass); b.seal_block(ls_pass);
                cur_use_bg = b.block_params(ls_pass).to_vec()[0];
            }

            b.ins().jump(draw_block, &[ir::BlockArg::Value(cur_use_bg)]);
            b.switch_to_block(draw_block); b.seal_block(draw_block);
            let use_bg_flag = b.block_params(draw_block).to_vec()[0];
            let use_bg_bool = b.ins().icmp_imm_s(IntCC::NotEqual, use_bg_flag, 0);
            draw_use_bg_bool = use_bg_bool;
            b.ins().select(use_bg_bool, colorback_v, raw_src)
        };

        // Afunction: DRAWMODE1 COMPARE tests source alpha (src_color bits 31:24) against
        // ALPHAREF and inhibits the write on failure. Applies in both RGB and CI mode
        // (in CI/DDA mode raw_src bits 31:24 are 0, so e.g. afunc_eq degenerates to
        // "alpharef==0" — that's correct, not a special case). Mirrors the interpreter's
        // px_afunc (afunc_{always,never,lt,eq,le,gt,ne,ge}). cmp==7 (disabled) is the
        // common case — resolved at compile time so no check is emitted at all.
        // FASTCLEAR bypasses afunction entirely, matching process_pixel_fastclear in
        // the interpreter (a dispatch-level bypass, not routed through px_afunc at all).
        let afunc_cmp = if dm1.fastclear() && !is_hostw { 7 } else { dm1.compare() };
        if afunc_cmp == 7 {
            emit_pixel_write(&mut b, px_ptr, x_bayer, y_bayer, src_color, &pctx, &mem, &memv, dm1, is_hostw);
        } else if afunc_cmp == 0 {
            // Always-kill: still walks the loop (shade/pattern/shifter bookkeeping below),
            // just never writes — same shape as a pattern-miss, not-opaque skip.
        } else {
            let sa   = b.ins().ushr_imm_s(src_color, 24);
            let sa8  = b.ins().band_imm_s(sa, 0xFF);
            let aref_v  = ld32!(ctx_off!(alpharef));
            let aref8   = b.ins().band_imm_s(aref_v, 0xFF);
            let cc = match afunc_cmp {
                1 => IntCC::UnsignedLessThan,
                2 => IntCC::Equal,
                3 => IntCC::UnsignedLessThanOrEqual,
                4 => IntCC::UnsignedGreaterThan,
                5 => IntCC::NotEqual,
                6 => IntCC::UnsignedGreaterThanOrEqual,
                _ => unreachable!("afunc_cmp 0 and 7 handled above"),
            };
            let pass = b.ins().icmp(cc, sa8, aref8);
            let write_block = b.create_block();
            let after_write = b.create_block();
            b.ins().brif(pass, write_block, &[], after_write, &[]);
            b.switch_to_block(write_block); b.seal_block(write_block);
            emit_pixel_write(&mut b, px_ptr, x_bayer, y_bayer, src_color, &pctx, &mem, &memv, dm1, is_hostw);
            b.ins().jump(after_write, &[]);
            b.switch_to_block(after_write); b.seal_block(after_write);
        }
        // Advance the host shifter/hostcnt only when we drew from host, not from colorback.
        let (final_shifter, final_hostcnt): (Value, Value) = if is_hostw && (dm0.colorhost() || dm0.alphahost()) {
            let s = b.ins().select(draw_use_bg_bool, no_fetch_skip_args_buf[0], host_shifter_after_fetch);
            let c = b.ins().select(draw_use_bg_bool, hostcnt_v, hostcnt_after_consume);
            (s, c)
        } else {
            (host_shifter_v, hostcnt_v)
        };
        let final_shifter_buf: [Value; 2] = [final_shifter, final_hostcnt];
        let skip_args_after_write: &[Value] = if is_hostw { &final_shifter_buf[..] } else { &[] };
        b.ins().jump(skip_block, &block_args(skip_args_after_write));
        host_shifter_after_fetch
    };
    let _ = new_shifter_after_pixel; // value flows through skip_block param (current_shifter)

    // ── skip_block: shade + pattern advance, then check end-of-x ─────────────
    b.switch_to_block(skip_block);
    b.seal_block(skip_block);

    // Extract updated host_shifter/hostcnt from skip_block params (carries through clip/skip paths).
    let (current_shifter, current_hostcnt): (Value, Value) = if is_hostw || is_hostr {
        let p = b.block_params(skip_block).to_vec();
        (p[0], p[1])
    } else {
        (b.ins().iconst(types::I64, 0), c0)
    };

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
                let overflow = b.ins().band_imm_s(ncr, 1 << 19);
                let ov_set = b.ins().icmp_imm_s(IntCC::NotEqual, overflow, 0);
                let max8 = b.ins().iconst(types::I32, 0x0007_FFFFi64);
                b.ins().select(ov_set, max8, ncr)
            } else if depth == 2 { // 12bpp: clamp colorred if bit 21 set
                let overflow = b.ins().band_imm_s(ncr, 1 << 21);
                let ov_set = b.ins().icmp_imm_s(IntCC::NotEqual, overflow, 0);
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

    // Pattern advance — mirrors advance_zpat/advance_lspat (rex3.rs): decrement,
    // reset to 31 on wrap (pre-f2d0bff behavior; f2d0bff's increment+lspattern-
    // rotate scheme broke PROM text rendering and was reverted in rex3.rs, so
    // the JIT is reverted to match here too — do not reintroduce the rotation).
    let new_zpat_bit = if dm0.enzpattern() {
        let c1_i8 = b.ins().iconst(types::I8, 1);
        let dec = b.ins().isub(zpat_bit_v, c1_i8);
        b.ins().band_imm_s(dec, 31)
    } else {
        zpat_bit_v
    };

    let (new_pat_bit, new_lsmode) = if dm0.enlspattern() {
        // Advance lspattern: decrement lsrcount; on zero, advance pat_bit and reload
        // lsrcount = lsmode[7:0], lsrepeat = lsmode[15:8], lsrcntsave = lsmode[23:16], lslength = lsmode[27:24]
        let lsrcount = b.ins().band_imm_s(lsmode_v, 0xFF);
        let lsm_shr8 = b.ins().ushr_imm_s(lsmode_v, 8);
        let lsrepeat_raw = b.ins().band_imm_s(lsm_shr8, 0xFF);
        let lsrepeat = {
            let is_zero = b.ins().icmp_imm_s(IntCC::Equal, lsrepeat_raw, 0);
            b.ins().select(is_zero, c1, lsrepeat_raw)
        };
        let is_zero = b.ins().icmp_imm_s(IntCC::Equal, lsrcount, 0);
        let reload_count = b.ins().isub(lsrepeat, c1);
        let lsrcount_dec = b.ins().isub(lsrcount, c1);
        let new_count = b.ins().select(is_zero, reload_count, lsrcount_dec);
        // If lsrcount was 0: advance pat_bit
        let lsm_shr24 = b.ins().ushr_imm_s(lsmode_v, 24);
        let lslength = b.ins().band_imm_s(lsm_shr24, 0xF);
        let length_bits = b.ins().iadd_imm_s(lslength, 17); // 17..=32
        let wrap_point = b.ins().isub(c32, length_bits);
        let pat_bit32 = b.ins().uextend(types::I32, pat_bit_v);
        let at_wrap = b.ins().icmp(IntCC::Equal, pat_bit32, wrap_point);
        let pat_dec = b.ins().isub(pat_bit32, c1);
        let dec_pat = b.ins().band_imm_s(pat_dec, 31);
        let new_pat32 = b.ins().select(at_wrap, c31, dec_pat);
        let new_pat32_masked = b.ins().select(is_zero, new_pat32, pat_bit32);
        let new_pb = b.ins().ireduce(types::I8, new_pat32_masked);
        // Update lsmode: set lsrcount field (bits 7:0)
        let lsm_cleared = b.ins().band_imm_s(lsmode_v, !0xFF_i64);
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

    // Host shifter/hostcnt reset at row boundary (host word spans one row, not across rows).
    // For HOSTR at row end: emit send_host_word writeback then reset shifter.
    let host_shifter_row_reset = b.ins().iconst(types::I64, 0);
    let hostcnt_row_reset = c0;

    // Row width isn't always a multiple of host_count, so a row can end mid-word
    // (e.g. a 6px-wide row packed 4px/word needs a second word with only 2 valid
    // pixels). Mirrors Rex3::flush_host_pixel: pad the accumulated shifter left by
    // (unfilled slot count * host_shift) so the valid pixels land at the MSB end,
    // matching how store_host_pixel packs LSB-to-MSB as each slot fills.
    // current_hostcnt is exactly ctx.hostcnt's counterpart — unfilled slot count —
    // so this reads the same way flush_host_pixel does, no extra derivation needed.
    let flushed_shifter = |b: &mut FunctionBuilder, shifter: Value| -> Value {
        let host_shift = dm1.host_shift();
        if !is_hostr || host_count <= 1 || host_shift == 0 {
            return shifter;
        }
        let unfilled64 = b.ins().uextend(types::I64, current_hostcnt);
        let pad_bits = b.ins().imul_imm_s(unfilled64, host_shift as i64);
        b.ins().ishl(shifter, pad_bits)
    };

    if !stopony {
        // Span / block without stopony: primitive done after one row
        let zero8 = b.ins().iconst(types::I8, 0);
        st_bool!(ctx_off!(mid_primitive), zero8);
        // For HOSTR: flush accumulated shifter to ctx.hostrw before returning.
        if is_hostr {
            let padded = flushed_shifter(&mut b, current_shifter);
            emit_store_hostrw(&mut b, ctx_ptr, &mem, padded, dm1);
        }
        if is_hostw || is_hostr {
            b.ins().store(mem, hostcnt_row_reset, ctx_ptr, ir::immediates::Offset32::new(ctx_off!(hostcnt) as i32));
        }
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
        if is_hostr {
            let padded = flushed_shifter(&mut b, current_shifter);
            emit_store_hostrw(&mut b, ctx_ptr, &mem, padded, dm1);
        }
        if is_hostw || is_hostr {
            b.ins().store(mem, hostcnt_row_reset, ctx_ptr, ir::immediates::Offset32::new(ctx_off!(hostcnt) as i32));
        }
        emit_writeback(&mut b, ctx_ptr, &mem, xsave_v, ystart_next,
            dm0, new_cr, new_cg, new_cb, new_ca, zpat_bit_reset, pat_bit_reset, new_lsmode);
        b.ins().jump(loop_end, &[]);

        b.switch_to_block(y_cont_block); b.seal_block(y_cont_block);
        if is_hostw || is_hostr {
            // Host mode: hardware processes exactly one word per GO. host_xstop_v
            // (the "word done" threshold) is computed once at shader entry from the
            // row the GO started on, so it is only valid for that row — continuing
            // the loop internally past a row wrap would keep running with a stale
            // threshold and silently skip the word boundary the caller is waiting
            // on. Treat the row wrap itself as a forced word boundary instead:
            // flush whatever's accumulated (padded, since a row can end mid-word —
            // same as the block-done path above) and return to the caller, who
            // re-enters via a fresh GO/entry() call with a correctly recomputed
            // host_xstop_v for the new row.
            if is_hostr {
                let padded = flushed_shifter(&mut b, current_shifter);
                emit_store_hostrw(&mut b, ctx_ptr, &mem, padded, dm1);
            }
            b.ins().store(mem, hostcnt_row_reset, ctx_ptr, ir::immediates::Offset32::new(ctx_off!(hostcnt) as i32));
            emit_writeback(&mut b, ctx_ptr, &mem, xsave_v, ystart_next,
                dm0, new_cr, new_cg, new_cb, new_ca, zpat_bit_reset, pat_bit_reset, new_lsmode);
            b.ins().jump(loop_end, &[]);
        } else {
            // Continue loop: reset x to xsave, new row
            let new_first = b.ins().iconst(types::I8, 1);
            let mut back_args: Vec<Value> = vec![xsave_v, ystart_next, new_first];
            if dm0.shade() { back_args.extend([new_cr, new_cg, new_cb, new_ca]); }
            if dm0.enzpattern() { back_args.push(zpat_bit_reset); }
            if dm0.enlspattern() { back_args.push(pat_bit_reset); back_args.push(new_lsmode); }
            b.ins().jump(loop_header, &block_args(&back_args));
        }
    }

    // ── cont_x_block: not x_end — check stoponx / host_xstop ─────────────────
    b.switch_to_block(cont_x_block);
    b.seal_block(cont_x_block);

    macro_rules! make_back_args {
        ($new_x:expr) => {{
            let new_first_v = b.ins().iconst(types::I8, 0);
            let mut args = vec![$new_x, ystart_v, new_first_v];
            if dm0.shade() { args.extend([new_cr, new_cg, new_cb, new_ca]); }
            if dm0.enzpattern() { args.push(new_zpat_bit); }
            if dm0.enlspattern() { args.push(new_pat_bit); args.push(new_lsmode); }
            if is_hostw || is_hostr { args.push(current_shifter); args.push(current_hostcnt); }
            args
        }};
    }

    // Mirrors interpreter: stoponx is forced true in host mode (stop_on_word || stoponx).
    let effective_stoponx = dm0.stoponx() || is_hostw || is_hostr;

    if !effective_stoponx {
        // stoponx=0 and not host mode: after one x step we stop (span primitive, next GO continues).
        emit_writeback(&mut b, ctx_ptr, &mem, xstart_next, ystart_v,
            dm0, new_cr, new_cg, new_cb, new_ca, new_zpat_bit, new_pat_bit, new_lsmode);
        b.ins().jump(loop_end, &[]);
    } else {
        // stoponx=1: continue x loop — jump back to loop_header.
        // Check host_xstop first (host_count pixels per word), then length32 xstop.
        let back_args = make_back_args!(xstart_next);

        // Collect all xstop checks into a chain: host_xstop takes priority, then length32.
        // On stop: write back and jump to loop_end.
        // On no stop: jump back to loop_header.
        let mut stop_checks: Vec<Value> = Vec::new();
        if let Some(hxstop) = host_xstop_v {
            let at_inc = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, xstart_next, hxstop);
            let at_dec = b.ins().icmp(IntCC::SignedLessThanOrEqual,    xstart_next, hxstop);
            stop_checks.push(b.ins().select(x_dec_v, at_dec, at_inc));
        }
        if let Some(xstop) = xstop_v {
            let at_inc = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, xstart_next, xstop);
            let at_dec = b.ins().icmp(IntCC::SignedLessThanOrEqual,    xstart_next, xstop);
            stop_checks.push(b.ins().select(x_dec_v, at_dec, at_inc));
        }

        if stop_checks.is_empty() {
            b.ins().jump(loop_header, &block_args(&back_args));
        } else {
            // Combine all stop conditions with OR
            let any_stop = stop_checks.into_iter().reduce(|a, c| b.ins().bor(a, c)).unwrap();
            let stop_block = b.create_block();
            let keep_going = b.create_block();
            b.ins().brif(any_stop, stop_block, &[], keep_going, &[]);

            b.switch_to_block(stop_block); b.seal_block(stop_block);
            if is_hostr {
                // host_xstop always triggers at a full word (host_count pixels
                // packed) — current_hostcnt is 0 here, no flush padding needed.
                emit_store_hostrw(&mut b, ctx_ptr, &mem, current_shifter, dm1);
            }
            if is_hostw || is_hostr {
                b.ins().store(mem, current_hostcnt, ctx_ptr, ir::immediates::Offset32::new(ctx_off!(hostcnt) as i32));
            }
            emit_writeback(&mut b, ctx_ptr, &mem, xstart_next, ystart_v,
                dm0, new_cr, new_cg, new_cb, new_ca, new_zpat_bit, new_pat_bit, new_lsmode);
            b.ins().jump(loop_end, &[]);

            b.switch_to_block(keep_going); b.seal_block(keep_going);
            b.ins().jump(loop_header, &block_args(&back_args));
        }
    }

    // ── loop_end ──────────────────────────────────────────────────────────────
    b.switch_to_block(loop_end);
    b.seal_block(loop_end);
    b.seal_block(loop_header); // now all predecessors are known
    b.ins().return_(&[]);
    b.finalize(target_config);

    true
}

/// Emit the Bresenham line shader IR. Mirrors draw_iline exactly.
/// Handles I_LINE, F_LINE, and A_LINE (F_LINE/A_LINE re-use the same Bresenham loop).
/// Returns false if the mode cannot be compiled.
fn emit_draw_iline(
    func:         &mut ir::Function,
    builder_ctx:  &mut FunctionBuilderContext,
    dm0:          &Dm0,
    dm1:          &Dm1,
    clipmode_key: u32,
    ptr_type:     ir::Type,
    target_config: cranelift_codegen::isa::TargetFrontendConfig,
) -> bool {
    let ensmask_key = clipmode_key & 0x1F;
    let cidmatch    = (clipmode_key >> 9) & 0xF;
    // Bresenham octant table (mirrors BRES in draw_iline).
    // Fields: (incrx1, incrx2, incry1, incry2, y_major)
    // Note: MAME applies y as `y -= incry`, so positive incry moves y in negative direction.
    #[rustfmt::skip]
    const BRES: [(i32, i32, i32, i32, bool); 8] = [
        ( 0,  1, -1, -1, true ),  // octant 0
        ( 0,  1,  1,  1, true ),  // octant 1
        ( 0, -1, -1, -1, true ),  // octant 2
        ( 0, -1,  1,  1, true ),  // octant 3
        ( 1,  1,  0, -1, false),  // octant 4
        ( 1,  1,  0,  1, false),  // octant 5
        (-1, -1,  0, -1, false),  // octant 6
        (-1, -1,  0,  1, false),  // octant 7
    ];

    let mut b = FunctionBuilder::new(func, builder_ctx);
    let mem  = MemFlagsData::trusted();
    let memv = MemFlagsData::new();

    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);

    let ctx_ptr = b.block_params(entry)[0];
    let fb_rgb  = b.block_params(entry)[1];
    let fb_aux  = b.block_params(entry)[2];

    macro_rules! ld32 { ($off:expr) => {
        b.ins().load(types::I32, mem, ctx_ptr, ir::immediates::Offset32::new($off as i32))
    }}
    macro_rules! ld8 { ($off:expr) => {
        b.ins().load(types::I8, mem, ctx_ptr, ir::immediates::Offset32::new($off as i32))
    }}

    // ── Load ctx fields ───────────────────────────────────────────────────────
    let xstart_v  = ld32!(ctx_off!(xstart));
    let ystart_v  = ld32!(ctx_off!(ystart));
    let xend_v    = ld32!(ctx_off!(xend));
    let yend_v    = ld32!(ctx_off!(yend));
    let xywin_v   = ld32!(ctx_off!(xywin));
    let xymove_v  = ld32!(ctx_off!(xymove));
    let clipmode_v= ld32!(ctx_off!(clipmode));
    let wrmask_v  = ld32!(ctx_off!(wrmask));
    let colorback_v  = ld32!(ctx_off!(colorback));
    let colorvram_v  = ld32!(ctx_off!(colorvram));
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

    // Bresenham state from registers
    let bres_v     = ld32!(ctx_off!(bresoctinc1));
    let bres2_v    = ld32!(ctx_off!(bresrndinc2));
    let bresd_raw  = ld32!(ctx_off!(bresd));

    // octant = bresoctinc1[26:24]
    let octant_v = {
        let shifted = b.ins().ushr_imm_s(bres_v, 24);
        b.ins().band_imm_s(shifted, 7)
    };

    // incr1: 20-bit, always positive
    let incr1_v = b.ins().band_imm_s(bres_v, 0xFFFFF);

    // incr2: 21-bit signed — sign-extend from bit 20
    let incr2_v = {
        let raw = b.ins().band_imm_s(bres2_v, 0x1FFFFF);
        let bit20 = b.ins().band_imm_s(raw, 1 << 20);
        let is_neg = b.ins().icmp_imm_s(IntCC::NotEqual, bit20, 0);
        let sign_ext = b.ins().bor_imm_s(raw, 0xFFE0_0000u64 as i64);
        b.ins().select(is_neg, sign_ext, raw)
    };

    // d: 27-bit signed — sign-extend from bit 26
    let d_init_v = {
        let raw = b.ins().band_imm_s(bresd_raw, 0x7FF_FFFF);
        let bit26 = b.ins().band_imm_s(raw, 1 << 26);
        let is_neg = b.ins().icmp_imm_s(IntCC::NotEqual, bit26, 0);
        let sign_ext = b.ins().bor_imm_s(raw, 0xF800_0000u64 as i64);
        b.ins().select(is_neg, sign_ext, raw)
    };

    // x2 = xend >> 11, y2 = yend >> 11
    let c11       = b.ins().iconst(types::I32, 11);
    let c0        = b.ins().iconst(types::I32, 0);
    let c1        = b.ins().iconst(types::I32, 1);
    let c31       = b.ins().iconst(types::I32, 31);
    let c32       = b.ins().iconst(types::I32, 32);
    let coord_bias = b.ins().iconst(types::I32, REX3_COORD_BIAS as i64);
    let c2048     = b.ins().iconst(types::I32, 2048);

    let x2_v = b.ins().sshr(xend_v, c11);
    let y2_v = b.ins().sshr(yend_v, c11);
    let x_init_v = b.ins().sshr(xstart_v, c11);
    let y_init_v = b.ins().sshr(ystart_v, c11);

    // y_major = (octant & OCTANT_XMAJOR) == 0  (OCTANT_XMAJOR = 4)
    let xmajor_bit = b.ins().band_imm_s(octant_v, OCTANT_XMAJOR as i64);
    let y_major_v  = b.ins().icmp_imm_s(IntCC::Equal, xmajor_bit, 0);

    // major = max(|dx|,|dy|); pixel_count = major + 1 (mirrors draw_line_bresenham).
    let dx_abs = {
        let d = b.ins().isub(x2_v, x_init_v);
        let neg = b.ins().ineg(d);
        let is_neg = b.ins().icmp_imm_s(IntCC::SignedLessThan, d, 0);
        b.ins().select(is_neg, neg, d)
    };
    let dy_abs = {
        let d = b.ins().isub(y2_v, y_init_v);
        let neg = b.ins().ineg(d);
        let is_neg = b.ins().icmp_imm_s(IntCC::SignedLessThan, d, 0);
        b.ins().select(is_neg, neg, d)
    };
    let dx_gt_dy = b.ins().icmp(IntCC::SignedGreaterThan, dx_abs, dy_abs);
    let major_v = b.ins().select(dx_gt_dy, dx_abs, dy_abs);
    let pixel_count_v = b.ins().iadd_imm_s(major_v, 1);

    // iterate_one = !stoponx && !stopony (step-mode: always draw exactly 1 pixel)
    let iterate_one = !dm0.stoponx() && !dm0.stopony();

    // In step mode: pixel_count=1, skip_first=false, skip_last=false
    let pixel_count_v = if iterate_one {
        b.ins().iconst(types::I32, 1)
    } else if dm0.length32() {
        // length32: cap at 32
        let long = b.ins().icmp(IntCC::SignedGreaterThan, pixel_count_v, c32);
        b.ins().select(long, c32, pixel_count_v)
    } else {
        pixel_count_v
    };

    // For shade/pattern state we use the same block-param approach as emit_shader.
    // Loop params: [x: i32, y: i32, d: i32, i: i32,  ...shade..., ...pattern...]
    let loop_header = b.create_block();
    let loop_end    = b.create_block();

    b.append_block_param(loop_header, types::I32); // x
    b.append_block_param(loop_header, types::I32); // y
    b.append_block_param(loop_header, types::I32); // d
    b.append_block_param(loop_header, types::I32); // i (pixel index 0..pixel_count)
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

    // Initial shade state
    let mut init_args: Vec<Value> = vec![x_init_v, y_init_v, d_init_v,
        b.ins().iconst(types::I32, 0)]; // i=0
    if dm0.shade() {
        init_args.extend([
            ld32!(ctx_off!(colorred)),
            ld32!(ctx_off!(colorgrn)),
            ld32!(ctx_off!(colorblue)),
            ld32!(ctx_off!(coloralpha)),
        ]);
    }
    if dm0.enzpattern() {
        init_args.push(ld8!(ctx_off!(zpat_bit)));
    }
    if dm0.enlspattern() {
        init_args.push(ld8!(ctx_off!(pat_bit)));
        init_args.push(ld32!(ctx_off!(lsmode)));
    }

    b.ins().jump(loop_header, &block_args(&init_args));

    // ── loop_header ───────────────────────────────────────────────────────────
    b.switch_to_block(loop_header);
    // DO NOT seal yet — back edge will be added later

    let hp: Vec<Value> = b.block_params(loop_header).to_vec();
    let x_v       = hp[0];
    let y_v       = hp[1];
    let d_v       = hp[2];
    let i_v       = hp[3];
    let mut pidx  = 4usize;
    let (colorred_v, colorgrn_v, colorblue_v, coloralpha_v) = if dm0.shade() {
        let r = hp[pidx]; let g = hp[pidx+1]; let bl = hp[pidx+2]; let a = hp[pidx+3];
        pidx += 4;
        (r, g, bl, a)
    } else {
        let z = b.ins().iconst(types::I32, 0);
        (z, z, z, z)
    };
    let zpat_bit_v = if dm0.enzpattern() {
        let v = hp[pidx]; pidx += 1; v
    } else { b.ins().iconst(types::I8, 0) };
    let (pat_bit_v, lsmode_v) = if dm0.enlspattern() {
        let pb = hp[pidx]; let lsm = hp[pidx+1]; pidx += 2; (pb, lsm)
    } else {
        (b.ins().iconst(types::I8, 0), b.ins().iconst(types::I32, 0))
    };
    let _ = pidx; // suppress unused warning

    // is_last = (i == pixel_count - 1)
    let last_idx = b.ins().isub(pixel_count_v, c1);
    let is_last_v  = b.ins().icmp(IntCC::Equal, i_v, last_idx);
    let is_first_v = b.ins().icmp_imm_s(IntCC::Equal, i_v, 0);

    // draw = (!is_first || !skipfirst) && (!is_last || !skiplast)
    // iterate_one overrides both skip flags (draws the single step unconditionally)
    let do_pixel: Value = if iterate_one {
        b.ins().iconst(types::I8, 1)
    } else if dm0.skipfirst() && dm0.skiplast() {
        let not_first = b.ins().icmp_imm_s(IntCC::Equal, is_first_v, 0);
        let not_last  = b.ins().icmp_imm_s(IntCC::Equal, is_last_v,  0);
        b.ins().band(not_first, not_last)
    } else if dm0.skipfirst() {
        b.ins().icmp_imm_s(IntCC::Equal, is_first_v, 0)
    } else if dm0.skiplast() {
        b.ins().icmp_imm_s(IntCC::Equal, is_last_v, 0)
    } else {
        b.ins().iconst(types::I8, 1)
    };

    // ── pixel block ───────────────────────────────────────────────────────────
    let pixel_block = b.create_block();
    let skip_block  = b.create_block();

    b.ins().brif(do_pixel, pixel_block, &[], skip_block, &[]);

    b.switch_to_block(pixel_block);
    b.seal_block(pixel_block);

    let pctx = PixelCtx {
        xywin_v, xymove_v, clipmode_v, wrmask_v, colorback_v, colorvram_v,
        smask0x_v, smask0y_v, smask1x_v, smask1y_v,
        smask2x_v, smask2y_v, smask3x_v, smask3y_v, smask4x_v, smask4y_v,
        fb_rgb, fb_aux,
    };

    // Lines never use scr2scr or host mode (guarded in compile_shader)
    let (px_ptr, x_bayer, y_bayer) = emit_calculate_fb_address(
        &mut b, x_v, y_v, &pctx, skip_block, &[], dm0, dm1, /*is_scr2scr=*/false,
        ensmask_key, coord_bias, c0, c2048, ptr_type,
    );

    // CID mask check for lines (same logic as emit_shader).
    if cidmatch != 0xF {
        let aux_ptr = {
            let fb_rgb_base = pctx.fb_rgb;
            let byte_off64  = b.ins().isub(px_ptr, fb_rgb_base);
            b.ins().iadd(pctx.fb_aux, byte_off64)
        };
        let aux_raw = b.ins().load(types::I32, memv, aux_ptr, ir::immediates::Offset32::new(0));
        let aux_lo4 = b.ins().band_imm_s(aux_raw, 0xF_i64);
        let cid_matches = b.ins().icmp_imm_s(IntCC::Equal, aux_lo4, cidmatch as i64);
        let cid_ok = b.create_block();
        b.ins().brif(cid_matches, cid_ok, &[], skip_block, &[]);
        b.switch_to_block(cid_ok); b.seal_block(cid_ok);
    }
    let _ = cidmatch; // used above (may be 0xF, in which case guard is skipped)

    // Source color (same as emit_shader draw path — no scr2scr for lines)
    let depth_mask: i64 = match dm1.drawdepth() { 0 => 0xF, 1 => 0xFF, 2 => 0xFFF, _ => 0xFFFFFF };
    let _ = depth_mask; // used indirectly via emit_pixel_write

    let src_color: Value = {
        let raw_src = if dm0.shade() {
            let r  = clamp_color_component(&mut b, colorred_v);
            let g  = clamp_color_component(&mut b, colorgrn_v);
            let bl = clamp_color_component(&mut b, colorblue_v);
            let g8   = b.ins().ishl_imm_s(g, 8);
            let bl16 = b.ins().ishl_imm_s(bl, 16);
            let rb   = b.ins().bor(r, g8);
            b.ins().bor(rb, bl16)
        } else {
            if dm1.rgbmode() {
                let r  = ld32!(ctx_off!(colorred));
                let g  = ld32!(ctx_off!(colorgrn));
                let bl = ld32!(ctx_off!(colorblue));
                let r_c  = clamp_color_component(&mut b, r);
                let g_c  = clamp_color_component(&mut b, g);
                let b_c  = clamp_color_component(&mut b, bl);
                let g8   = b.ins().ishl_imm_s(g_c, 8);
                let bl16 = b.ins().ishl_imm_s(b_c, 16);
                let rb   = b.ins().bor(r_c, g8);
                b.ins().bor(rb, bl16)
            } else {
                let cr = ld32!(ctx_off!(colorred));
                b.ins().sshr(cr, c11)
            }
        };

        // Afunction source alpha for lines is always DDA (lines never use host mode —
        // see compile_shader guard), independent of RGB/CI plane format. Mirrors the
        // block/span shader's alpha_byte overlay.
        let ca = ld32!(ctx_off!(coloralpha));
        let ca_c = clamp_color_component(&mut b, ca);
        let alpha_byte = b.ins().ishl_imm_s(ca_c, 24);
        let color24 = b.ins().band_imm_s(raw_src, 0x00FF_FFFFi64);
        let raw_src = b.ins().bor(color24, alpha_byte);

        let draw_block2 = b.create_block();
        b.append_block_param(draw_block2, types::I8);
        let mut cur_use_bg: Value = b.ins().iconst(types::I8, 0);

        if dm0.enzpattern() {
            let zp_block = b.create_block();
            let zp_pass  = b.create_block();
            b.append_block_param(zp_pass, types::I8);
            let zpattern_v   = ld32!(ctx_off!(zpattern));
            let zpat_bit32   = b.ins().uextend(types::I32, zpat_bit_v);
            let zpat_shifted = b.ins().ushr(zpattern_v, zpat_bit32);
            let bit_v  = b.ins().band_imm_s(zpat_shifted, 1);
            let bit_set = b.ins().icmp_imm_s(IntCC::NotEqual, bit_v, 0);
            b.ins().brif(bit_set, zp_pass, &[ir::BlockArg::Value(cur_use_bg)], zp_block, &[]);
            b.switch_to_block(zp_block); b.seal_block(zp_block);
            if dm0.zpopaque() {
                let bg1 = b.ins().iconst(types::I8, 1);
                b.ins().jump(zp_pass, &[ir::BlockArg::Value(bg1)]);
            } else {
                b.ins().jump(skip_block, &[]);
            }
            b.switch_to_block(zp_pass); b.seal_block(zp_pass);
            cur_use_bg = b.block_params(zp_pass).to_vec()[0];
        }

        if dm0.enlspattern() {
            let ls_block = b.create_block();
            let ls_pass  = b.create_block();
            b.append_block_param(ls_pass, types::I8);
            let lspattern_v  = ld32!(ctx_off!(lspattern));
            let pat_bit32    = b.ins().uextend(types::I32, pat_bit_v);
            let lspat_shifted = b.ins().ushr(lspattern_v, pat_bit32);
            let bit_v  = b.ins().band_imm_s(lspat_shifted, 1);
            let bit_set = b.ins().icmp_imm_s(IntCC::NotEqual, bit_v, 0);
            b.ins().brif(bit_set, ls_pass, &[ir::BlockArg::Value(cur_use_bg)], ls_block, &[]);
            b.switch_to_block(ls_block); b.seal_block(ls_block);
            if dm0.lsopaque() {
                let bg1 = b.ins().iconst(types::I8, 1);
                b.ins().jump(ls_pass, &[ir::BlockArg::Value(bg1)]);
            } else {
                b.ins().jump(skip_block, &[]);
            }
            b.switch_to_block(ls_pass); b.seal_block(ls_pass);
            cur_use_bg = b.block_params(ls_pass).to_vec()[0];
        }

        b.ins().jump(draw_block2, &[ir::BlockArg::Value(cur_use_bg)]);
        b.switch_to_block(draw_block2); b.seal_block(draw_block2);
        let use_bg_flag = b.block_params(draw_block2).to_vec()[0];
        let use_bg_bool = b.ins().icmp_imm_s(IntCC::NotEqual, use_bg_flag, 0);
        b.ins().select(use_bg_bool, colorback_v, raw_src)
    };

    // Afunction: see the block/span shader for details. Applies in both RGB and CI
    // mode. cmp==7 (disabled) is resolved at compile time so no check is emitted;
    // cmp==0 always kills without a runtime test.
    let afunc_cmp = dm1.compare();
    if afunc_cmp == 7 {
        emit_pixel_write(&mut b, px_ptr, x_bayer, y_bayer, src_color, &pctx, &mem, &memv, dm1, false);
    } else if afunc_cmp == 0 {
        // Always-kill: no write, still falls through to skip_block below.
    } else {
        let sa   = b.ins().ushr_imm_s(src_color, 24);
        let sa8  = b.ins().band_imm_s(sa, 0xFF);
        let aref_v  = ld32!(ctx_off!(alpharef));
        let aref8   = b.ins().band_imm_s(aref_v, 0xFF);
        let cc = match afunc_cmp {
            1 => IntCC::UnsignedLessThan,
            2 => IntCC::Equal,
            3 => IntCC::UnsignedLessThanOrEqual,
            4 => IntCC::UnsignedGreaterThan,
            5 => IntCC::NotEqual,
            6 => IntCC::UnsignedGreaterThanOrEqual,
            _ => unreachable!("afunc_cmp 0 and 7 handled above"),
        };
        let pass = b.ins().icmp(cc, sa8, aref8);
        let write_block = b.create_block();
        let after_write = b.create_block();
        b.ins().brif(pass, write_block, &[], after_write, &[]);
        b.switch_to_block(write_block); b.seal_block(write_block);
        emit_pixel_write(&mut b, px_ptr, x_bayer, y_bayer, src_color, &pctx, &mem, &memv, dm1, false);
        b.ins().jump(after_write, &[]);
        b.switch_to_block(after_write); b.seal_block(after_write);
    }
    b.ins().jump(skip_block, &[]);

    // ── skip_block: shade + pattern advance ──────────────────────────────────
    b.switch_to_block(skip_block);
    b.seal_block(skip_block);

    // Shade DDA step (mirrors draw_iline calling shade_fn)
    let (new_cr, new_cg, new_cb, new_ca) = if dm0.shade() {
        let slopered_v   = ld32!(ctx_off!(slopered));
        let slopegrn_v   = ld32!(ctx_off!(slopegrn));
        let slopeblue_v  = ld32!(ctx_off!(slopeblue));
        let slopealpha_v = ld32!(ctx_off!(slopealpha));
        let ncr = b.ins().iadd(colorred_v,   slopered_v);
        let ncg = b.ins().iadd(colorgrn_v,   slopegrn_v);
        let ncb = b.ins().iadd(colorblue_v,  slopeblue_v);
        let nca = b.ins().iadd(coloralpha_v, slopealpha_v);
        let (ncr, ncg, ncb, nca) = if dm1.rgbmode() {
            (clamp_shade(&mut b, ncr), clamp_shade(&mut b, ncg),
             clamp_shade(&mut b, ncb), clamp_shade(&mut b, nca))
        } else if dm0.ciclamp() {
            let depth = dm1.drawdepth();
            let ncr2 = if depth == 1 {
                let overflow = b.ins().band_imm_s(ncr, 1 << 19);
                let ov_set = b.ins().icmp_imm_s(IntCC::NotEqual, overflow, 0);
                let max8 = b.ins().iconst(types::I32, 0x0007_FFFFi64);
                b.ins().select(ov_set, max8, ncr)
            } else if depth == 2 {
                let overflow = b.ins().band_imm_s(ncr, 1 << 21);
                let ov_set = b.ins().icmp_imm_s(IntCC::NotEqual, overflow, 0);
                let max12 = b.ins().iconst(types::I32, 0x001F_FFFFi64);
                b.ins().select(ov_set, max12, ncr)
            } else { ncr };
            (ncr2, ncg, ncb, nca)
        } else { (ncr, ncg, ncb, nca) };
        (ncr, ncg, ncb, nca)
    } else {
        (colorred_v, colorgrn_v, colorblue_v, coloralpha_v)
    };

    // Pattern advance (mirrors pattern_fn in draw_iline) — decrement, reset to
    // 31 on wrap (see emit_shader's identical block for the full rationale).
    let new_zpat_bit = if dm0.enzpattern() {
        let c1_i8 = b.ins().iconst(types::I8, 1);
        let dec = b.ins().isub(zpat_bit_v, c1_i8);
        b.ins().band_imm_s(dec, 31)
    } else { zpat_bit_v };

    let (new_pat_bit, new_lsmode) = if dm0.enlspattern() {
        let lsrcount = b.ins().band_imm_s(lsmode_v, 0xFF);
        let lsm_shr8 = b.ins().ushr_imm_s(lsmode_v, 8);
        let lsrepeat_raw = b.ins().band_imm_s(lsm_shr8, 0xFF);
        let lsrepeat = {
            let is_zero = b.ins().icmp_imm_s(IntCC::Equal, lsrepeat_raw, 0);
            b.ins().select(is_zero, c1, lsrepeat_raw)
        };
        let is_zero = b.ins().icmp_imm_s(IntCC::Equal, lsrcount, 0);
        let reload_count = b.ins().isub(lsrepeat, c1);
        let lsrcount_dec = b.ins().isub(lsrcount, c1);
        let new_count = b.ins().select(is_zero, reload_count, lsrcount_dec);
        let lsm_shr24 = b.ins().ushr_imm_s(lsmode_v, 24);
        let lslength = b.ins().band_imm_s(lsm_shr24, 0xF);
        let length_bits = b.ins().iadd_imm_s(lslength, 17);
        let wrap_point = b.ins().isub(c32, length_bits);
        let pat_bit32 = b.ins().uextend(types::I32, pat_bit_v);
        let at_wrap = b.ins().icmp(IntCC::Equal, pat_bit32, wrap_point);
        let pat_dec = b.ins().isub(pat_bit32, c1);
        let dec_pat = b.ins().band_imm_s(pat_dec, 31);
        let new_pat32 = b.ins().select(at_wrap, c31, dec_pat);
        let new_pat32_masked = b.ins().select(is_zero, new_pat32, pat_bit32);
        let new_pb = b.ins().ireduce(types::I8, new_pat32_masked);
        let lsm_cleared = b.ins().band_imm_s(lsmode_v, !0xFF_i64);
        let new_lsm = b.ins().bor(lsm_cleared, new_count);
        (new_pb, new_lsm)
    } else { (pat_bit_v, lsmode_v) };

    // ── Bresenham step (mirrors bres_step! macro in draw_iline) ──────────────
    // Step depends on both octant and d — both are runtime values.
    // We emit an 8-way table lookup for (incrx1, incrx2, incry1, incry2) and
    // a runtime branch on d < 0 to select which step to apply.
    //
    // The step emitted here is used when !is_last || iterate_one.
    // When is_last && !iterate_one: skip step, write back x,y unchanged.
    let step_block  = b.create_block(); // emit step
    let noste_block = b.create_block(); // skip step (last pixel, full-line mode)
    let after_step  = b.create_block(); // params: [x_new, y_new, d_new]
    b.append_block_param(after_step, types::I32); // x_new
    b.append_block_param(after_step, types::I32); // y_new
    b.append_block_param(after_step, types::I32); // d_new

    // In iterate_one mode: always step.
    // In full-line mode: step unless is_last.
    if iterate_one {
        b.ins().jump(step_block, &[]);
    } else {
        b.ins().brif(is_last_v, noste_block, &[], step_block, &[]);
    }

    // step_block: apply Bresenham step
    b.switch_to_block(step_block); b.seal_block(step_block);
    {
        // For each octant, emit a table of (incrx1, incrx2, incry1, incry2).
        // We branch on octant at runtime to select the right step offsets.
        // Then branch on d < 0 to select step 1 (straight) vs step 2 (diagonal).
        //
        // Build two arrays indexed by octant[2:0]:
        //   incrx_straight[oct], incrx_diag[oct], incry_straight[oct], incry_diag[oct]
        // Use a chain of select instructions to do the table lookup.

        // Instead of a full 8-way table, use the octant bits directly:
        //   incrx1 = f(octant), incrx2 = g(octant), ...
        // From the BRES table:
        //   oct 0: ( 0,  1, -1, -1, y_major)
        //   oct 1: ( 0,  1,  1,  1, y_major)
        //   oct 2: ( 0, -1, -1, -1, y_major)
        //   oct 3: ( 0, -1,  1,  1, y_major)
        //   oct 4: ( 1,  1,  0, -1, x_major)
        //   oct 5: ( 1,  1,  0,  1, x_major)
        //   oct 6: (-1, -1,  0, -1, x_major)
        //   oct 7: (-1, -1,  0,  1, x_major)
        //
        // Patterns:
        //   incrx1: 0 for oct 0-3, +1 for oct 4-5, -1 for oct 6-7
        //   incrx2: +1 for oct 0-1,4-5; -1 for oct 2-3,6-7
        //   incry1: -1 for oct 0,2; +1 for oct 1,3; 0 for oct 4-7
        //   incry2: -1 for oct 0,2,4,6; +1 for oct 1,3,5,7

        // Emit the 8 octant combos using a chain of select on octant bits.
        // octant bits: bit0=YDEC, bit1=XDEC, bit2=XMAJOR

        let cm1i = b.ins().iconst(types::I32, -1i64);
        let cp1i = b.ins().iconst(types::I32,  1i64);
        let c0i  = b.ins().iconst(types::I32,  0i64);

        // XMAJOR bit (bit 2)
        let xmajor_v = b.ins().icmp_imm_s(IntCC::NotEqual, xmajor_bit, 0);
        // XDEC bit (bit 1)
        let xdec_bit = b.ins().band_imm_s(octant_v, OCTANT_XDEC as i64);
        let xdec_v   = b.ins().icmp_imm_s(IntCC::NotEqual, xdec_bit, 0);
        // YDEC bit (bit 0)
        let ydec_bit = b.ins().band_imm_s(octant_v, OCTANT_YDEC as i64);
        let ydec_v   = b.ins().icmp_imm_s(IntCC::NotEqual, ydec_bit, 0);

        // incrx1: 0 for y-major, ±1 for x-major (sign = XDEC)
        let incrx1_xmaj = b.ins().select(xdec_v, cm1i, cp1i);
        let incrx1 = b.ins().select(xmajor_v, incrx1_xmaj, c0i);

        // incrx2: ±1 based on XDEC (same for y-major and x-major)
        let incrx2 = b.ins().select(xdec_v, cm1i, cp1i);

        // incry1: 0 for x-major; ±1 for y-major (sign = YDEC, but inverted: YDEC→+1 per BRES)
        // Oct 0 (YDEC=0): incry1 = -1; Oct 1 (YDEC=1): incry1 = +1
        // Oct 2 (YDEC=0): incry1 = -1; Oct 3 (YDEC=1): incry1 = +1
        // So incry1 (y-major) = YDEC ? +1 : -1
        let incry1_ymaj = b.ins().select(ydec_v, cp1i, cm1i);
        let incry1 = b.ins().select(xmajor_v, c0i, incry1_ymaj);

        // incry2: ±1 based on YDEC (inverted: YDEC=0→-1, YDEC=1→+1)
        let incry2 = b.ins().select(ydec_v, cp1i, cm1i);

        // Choose step 1 or step 2 based on d < 0
        let d_neg   = b.ins().icmp_imm_s(IntCC::SignedLessThan, d_v, 0);
        let dx_step = b.ins().select(d_neg, incrx1, incrx2);
        let dy_step = b.ins().select(d_neg, incry1, incry2);
        let d_incr  = b.ins().select(d_neg, incr1_v, incr2_v);

        // x += dx_step,  y -= dy_step  (MAME convention: y -= incry)
        let x_new = b.ins().iadd(x_v, dx_step);
        let y_neg_step = b.ins().ineg(dy_step);
        let y_new = b.ins().iadd(y_v, y_neg_step);
        let d_new = b.ins().iadd(d_v, d_incr);

        b.ins().jump(after_step, &[ir::BlockArg::Value(x_new), ir::BlockArg::Value(y_new), ir::BlockArg::Value(d_new)]);
    }

    // noste_block: no step (last pixel, full-line mode) — pass x,y,d unchanged
    if !iterate_one {
        b.switch_to_block(noste_block); b.seal_block(noste_block);
        b.ins().jump(after_step, &[ir::BlockArg::Value(x_v), ir::BlockArg::Value(y_v), ir::BlockArg::Value(d_v)]);
    }

    // after_step: write back ctx and check if done
    b.switch_to_block(after_step); b.seal_block(after_step);
    let x_new = b.block_params(after_step)[0];
    let y_new = b.block_params(after_step)[1];
    let d_new = b.block_params(after_step)[2];

    // i_next = i + 1; done = (i_next >= pixel_count)
    let i_next = b.ins().iadd_imm_s(i_v, 1);
    let done_v = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, i_next, pixel_count_v);

    let cont_block = b.create_block();
    b.ins().brif(done_v, loop_end, &[], cont_block, &[]);
    b.switch_to_block(cont_block); b.seal_block(cont_block);

    // Loop back
    let mut back_args: Vec<Value> = vec![x_new, y_new, d_new, i_next];
    if dm0.shade() { back_args.extend([new_cr, new_cg, new_cb, new_ca]); }
    if dm0.enzpattern() { back_args.push(new_zpat_bit); }
    if dm0.enlspattern() { back_args.push(new_pat_bit); back_args.push(new_lsmode); }
    b.ins().jump(loop_header, &block_args(&back_args));

    // ── loop_end: write back ctx state ───────────────────────────────────────
    b.switch_to_block(loop_end);
    b.seal_block(loop_end);
    b.seal_block(loop_header);

    // Write back xstart = x_new<<11, ystart = y_new<<11, bresd = d_new & mask
    // (matches ctx.xstart = x<<11; ctx.ystart = y<<11; ctx.bresd = (d as u32) & 0x7FF_FFFF)
    macro_rules! st32e { ($off:expr, $val:expr) => {
        b.ins().store(mem, $val, ctx_ptr, ir::immediates::Offset32::new($off as i32));
    }}
    macro_rules! st8e { ($off:expr, $val:expr) => {
        b.ins().store(mem, $val, ctx_ptr, ir::immediates::Offset32::new($off as i32));
    }}

    let c11_wb   = b.ins().iconst(types::I32, 11);
    let xstart_wb = b.ins().ishl(x_new, c11_wb);
    let ystart_wb = b.ins().ishl(y_new, c11_wb);
    let bresd_wb  = b.ins().band_imm_s(d_new, 0x7FF_FFFFi64);
    st32e!(ctx_off!(xstart), xstart_wb);
    st32e!(ctx_off!(ystart), ystart_wb);
    st32e!(ctx_off!(bresd),  bresd_wb);
    if dm0.shade() {
        st32e!(ctx_off!(colorred),   new_cr);
        st32e!(ctx_off!(colorgrn),   new_cg);
        st32e!(ctx_off!(colorblue),  new_cb);
        st32e!(ctx_off!(coloralpha), new_ca);
    }
    if dm0.enzpattern() {
        st8e!(ctx_off!(zpat_bit), new_zpat_bit);
    }
    if dm0.enlspattern() {
        st8e!(ctx_off!(pat_bit), new_pat_bit);
        st32e!(ctx_off!(lsmode), new_lsmode);
    }

    b.ins().return_(&[]);
    b.finalize(target_config);
    true
}

// ── IR helpers ────────────────────────────────────────────────────────────────

/// Emit `bayer_pack(color, x, y)`: pack bayer index into bits 27:24 of color.
fn bayer_pack_ir(b: &mut FunctionBuilder, color: Value, x: Value, y: Value) -> Value {
    let x3       = b.ins().band_imm_s(x, 3);
    let y3       = b.ins().band_imm_s(y, 3);
    let y3_shift = b.ins().ishl_imm_s(y3, 2);
    let idx      = b.ins().bor(y3_shift, x3);
    let color24  = b.ins().band_imm_s(color, 0x00FF_FFFFi64);
    let idx_shifted = b.ins().ishl_imm_s(idx, 24);
    b.ins().bor(color24, idx_shifted)
}

/// clamp_color_component for shade output: extract bits[22:11] (integer), return 8-bit value.
fn clamp_color_component(b: &mut FunctionBuilder, c: Value) -> Value {
    // integer part = (c >> 11) & 0x1FF
    let c_shr11 = b.ins().ushr_imm_s(c, 11);
    let val = b.ins().band_imm_s(c_shr11, 0x1FF);
    // if bit31 set or val >= 0x180: 0
    // if val > 0xFF: 0xFF
    // else: val & 0xFF
    let bit31 = b.ins().band_imm_s(c, 1i64 << 31);
    let neg = b.ins().icmp_imm_s(IntCC::NotEqual, bit31, 0);
    let overflow = b.ins().icmp_imm_s(IntCC::UnsignedGreaterThanOrEqual, val, 0x180);
    let clamped_zero = b.ins().bor(neg, overflow);
    let max_255 = b.ins().icmp_imm_s(IntCC::UnsignedGreaterThan, val, 0xFF);
    let c255 = b.ins().iconst(types::I32, 0xFF);
    let c0   = b.ins().iconst(types::I32, 0);
    let val_masked = b.ins().band_imm_s(val, 0xFF);
    let clamped_255 = b.ins().select(max_255, c255, val_masked);
    b.ins().select(clamped_zero, c0, clamped_255)
}

/// Clamp a shade DDA value (o12.11): keep as-is for unclamped, or apply RGB clamp.
fn clamp_shade(b: &mut FunctionBuilder, c: Value) -> Value {
    let c_shr11 = b.ins().ushr_imm_s(c, 11);
    let val = b.ins().band_imm_s(c_shr11, 0x1FF);
    let bit31 = b.ins().band_imm_s(c, 1i64 << 31);
    let neg = b.ins().icmp_imm_s(IntCC::NotEqual, bit31, 0);
    let overflow = b.ins().icmp_imm_s(IntCC::UnsignedGreaterThanOrEqual, val, 0x180);
    let clamped_zero = b.ins().bor(neg, overflow);
    let max_7ffff = b.ins().icmp_imm_s(IntCC::UnsignedGreaterThan, val, 0xFF);
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
            let v8  = b.ins().ishl_imm_s(val, 8);
            let v16 = b.ins().ishl_imm_s(val, 16);
            b.ins().bor(v8, v16)
        }
        p if p == DRAWMODE1_PLANES_CID => {
            let v4 = b.ins().ishl_imm_s(val, 4);
            b.ins().bor(val, v4)
        }
        p if p == DRAWMODE1_PLANES_PUP => {
            let v2 = b.ins().ishl_imm_s(val, 2);
            let v6 = b.ins().ishl_imm_s(val, 6);
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
    mem: &MemFlagsData,
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
    let shift   = b.ins().ishl_imm_s(idx64, 2); // idx * 4
    let shifted = b.ins().ushr(packed, shift);
    let nibble  = b.ins().band_imm_s(shifted, 0xF);
    b.ins().ireduce(types::I32, nibble)
}

fn emit_compress_ir(b: &mut FunctionBuilder, val: Value, drawdepth: u32, dither: bool) -> Value {
    if dither {
        // Bayer threshold: val bits[31:24] is the raw bayer index, look it up in the table
        let idx   = b.ins().ushr_imm_s(val, 24);
        let bayer = emit_bayer_lookup_ir(b, idx);
        // Extract r/g/bv channels (bits[7:0], [15:8], [23:16])
        let r = b.ins().band_imm_s(val, 0xFF);
        let t8  = b.ins().ushr_imm_s(val, 8);
        let g   = b.ins().band_imm_s(t8, 0xFF);
        let t16 = b.ins().ushr_imm_s(val, 16);
        let bv  = b.ins().band_imm_s(t16, 0xFF);
        match drawdepth {
            0 => {
                // rgb24→rgb4 dither: sr = (r>>3)-(r>>4), etc.
                let r3 = b.ins().ushr_imm_s(r, 3);
                let r4 = b.ins().ushr_imm_s(r, 4);
                let sr = b.ins().isub(r3, r4);
                let g2 = b.ins().ushr_imm_s(g, 2);
                let g4 = b.ins().ushr_imm_s(g, 4);
                let sg = b.ins().isub(g2, g4);
                let b3 = b.ins().ushr_imm_s(bv, 3);
                let b4 = b.ins().ushr_imm_s(bv, 4);
                let sb = b.ins().isub(b3, b4);
                let sr4_v = b.ins().ushr_imm_s(sr, 4);
                let sg4_v = b.ins().ushr_imm_s(sg, 4);
                let sb4_v = b.ins().ushr_imm_s(sb, 4);
                let mut dr = b.ins().band_imm_s(sr4_v, 1);
                let mut dg = b.ins().band_imm_s(sg4_v, 3);
                let mut db = b.ins().band_imm_s(sb4_v, 1);
                let sr_lo = b.ins().band_imm_s(sr, 0xF);
                let sg_lo = b.ins().band_imm_s(sg, 0xF);
                let sb_lo = b.ins().band_imm_s(sb, 0xF);
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
                let db3s = b.ins().ishl_imm_s(db, 3);
                let dg1s = b.ins().ishl_imm_s(dg, 1);
                let t = b.ins().bor(db3s, dg1s);
                b.ins().bor(t, dr)
            }
            1 => {
                // rgb24→rgb8 dither: sr = (r>>1)-(r>>4), etc.
                let r1 = b.ins().ushr_imm_s(r, 1);
                let r4 = b.ins().ushr_imm_s(r, 4);
                let sr = b.ins().isub(r1, r4);
                let g1 = b.ins().ushr_imm_s(g, 1);
                let g4 = b.ins().ushr_imm_s(g, 4);
                let sg = b.ins().isub(g1, g4);
                let b2 = b.ins().ushr_imm_s(bv, 2);
                let b4 = b.ins().ushr_imm_s(bv, 4);
                let sb = b.ins().isub(b2, b4);
                let sr4_v = b.ins().ushr_imm_s(sr, 4);
                let sg4_v = b.ins().ushr_imm_s(sg, 4);
                let sb4_v = b.ins().ushr_imm_s(sb, 4);
                let mut dr = b.ins().band_imm_s(sr4_v, 7);
                let mut dg = b.ins().band_imm_s(sg4_v, 7);
                let mut db = b.ins().band_imm_s(sb4_v, 3);
                let sr_lo = b.ins().band_imm_s(sr, 0xF);
                let sg_lo = b.ins().band_imm_s(sg, 0xF);
                let sb_lo = b.ins().band_imm_s(sb, 0xF);
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
                let db6 = b.ins().ishl_imm_s(db, 6);
                let dg3 = b.ins().ishl_imm_s(dg, 3);
                let t = b.ins().bor(db6, dg3);
                b.ins().bor(t, dr)
            }
            _ => {
                // rgb24→rgb12 dither: sr = r - (r>>4), etc.
                let r4 = b.ins().ushr_imm_s(r, 4);
                let sr = b.ins().isub(r, r4);
                let g4 = b.ins().ushr_imm_s(g, 4);
                let sg = b.ins().isub(g, g4);
                let b4 = b.ins().ushr_imm_s(bv, 4);
                let sb = b.ins().isub(bv, b4);
                let sr4_v = b.ins().ushr_imm_s(sr, 4);
                let sg4_v = b.ins().ushr_imm_s(sg, 4);
                let sb4_v = b.ins().ushr_imm_s(sb, 4);
                let mut dr = b.ins().band_imm_s(sr4_v, 15);
                let mut dg = b.ins().band_imm_s(sg4_v, 15);
                let mut db = b.ins().band_imm_s(sb4_v, 15);
                let sr_lo = b.ins().band_imm_s(sr, 0xF);
                let sg_lo = b.ins().band_imm_s(sg, 0xF);
                let sb_lo = b.ins().band_imm_s(sb, 0xF);
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
                let db8 = b.ins().ishl_imm_s(db, 8);
                let dg4 = b.ins().ishl_imm_s(dg, 4);
                let t = b.ins().bor(db8, dg4);
                b.ins().bor(t, dr)
            }
        }
    } else {
        // No dither — straight compress
        match drawdepth {
            0 => {
                // rgb24→rgb4: r=bit7, g=bits[15:14], b=bit23
                let rs = b.ins().ushr_imm_s(val,  7);
                let r  = b.ins().band_imm_s(rs, 1);
                let gs = b.ins().ushr_imm_s(val, 14);
                let g  = b.ins().band_imm_s(gs, 3);
                let bs = b.ins().ushr_imm_s(val, 23);
                let bv = b.ins().band_imm_s(bs, 1);
                let b3 = b.ins().ishl_imm_s(bv, 3);
                let g1 = b.ins().ishl_imm_s(g,  1);
                let t  = b.ins().bor(b3, g1);
                b.ins().bor(t, r)
            }
            1 => {
                // rgb24→rgb8: r=bits[7:5], g=bits[15:13], b=bits[23:22]
                let rs = b.ins().ushr_imm_s(val,  5);
                let r  = b.ins().band_imm_s(rs, 7);
                let gs = b.ins().ushr_imm_s(val, 13);
                let g  = b.ins().band_imm_s(gs, 7);
                let bs = b.ins().ushr_imm_s(val, 22);
                let bv = b.ins().band_imm_s(bs, 3);
                let b6 = b.ins().ishl_imm_s(bv, 6);
                let g3 = b.ins().ishl_imm_s(g,  3);
                let t  = b.ins().bor(b6, g3);
                b.ins().bor(t, r)
            }
            _ => {
                // rgb24→rgb12: r=bits[7:4], g=bits[15:12], b=bits[23:20]
                let rs = b.ins().ushr_imm_s(val,  4);
                let r  = b.ins().band_imm_s(rs, 0xF);
                let gs = b.ins().ushr_imm_s(val, 12);
                let g  = b.ins().band_imm_s(gs, 0xF);
                let bs = b.ins().ushr_imm_s(val, 20);
                let bv = b.ins().band_imm_s(bs, 0xF);
                let b8 = b.ins().ishl_imm_s(bv, 8);
                let g4 = b.ins().ishl_imm_s(g,  4);
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
            let r_bit  = b.ins().band_imm_s(val, 1);
            let r_set  = b.ins().icmp_imm_s(IntCC::NotEqual, r_bit, 0);
            let c255   = b.ins().iconst(types::I32, 0xFF);
            let c0     = b.ins().iconst(types::I32, 0);
            let r      = b.ins().select(r_set, c255, c0);
            let gs     = b.ins().ushr_imm_s(val, 1);
            let g_raw  = b.ins().band_imm_s(gs, 3);
            let g6 = b.ins().ishl_imm_s(g_raw, 6);
            let g4 = b.ins().ishl_imm_s(g_raw, 4);
            let g2 = b.ins().ishl_imm_s(g_raw, 2);
            let t1 = b.ins().bor(g6, g4);
            let t2 = b.ins().bor(g2, g_raw);
            let g  = b.ins().bor(t1, t2);
            let b_shr  = b.ins().ushr_imm_s(val, 3);
            let b_bit  = b.ins().band_imm_s(b_shr, 1);
            let b_set  = b.ins().icmp_imm_s(IntCC::NotEqual, b_bit, 0);
            let bv     = b.ins().select(b_set, c255, c0);
            let g8  = b.ins().ishl_imm_s(g,  8);
            let b16 = b.ins().ishl_imm_s(bv, 16);
            let t   = b.ins().bor(b16, g8);
            b.ins().bor(t, r)
        }
        1 => {
            // rgb8→rgb24: r=bits[2:0], g=bits[5:3], b=bits[7:6]
            let r_raw = b.ins().band_imm_s(val, 7);
            let r5 = b.ins().ishl_imm_s(r_raw, 5);
            let r2 = b.ins().ishl_imm_s(r_raw, 2);
            let r1 = b.ins().ushr_imm_s(r_raw, 1);
            let t1 = b.ins().bor(r5, r2);
            let r  = b.ins().bor(t1, r1);
            let gs    = b.ins().ushr_imm_s(val, 3);
            let g_raw = b.ins().band_imm_s(gs, 7);
            let g5 = b.ins().ishl_imm_s(g_raw, 5);
            let g2 = b.ins().ishl_imm_s(g_raw, 2);
            let g1 = b.ins().ushr_imm_s(g_raw, 1);
            let t2 = b.ins().bor(g5, g2);
            let g  = b.ins().bor(t2, g1);
            let bs    = b.ins().ushr_imm_s(val, 6);
            let b_raw = b.ins().band_imm_s(bs, 3);
            let b6 = b.ins().ishl_imm_s(b_raw, 6);
            let b4 = b.ins().ishl_imm_s(b_raw, 4);
            let b2 = b.ins().ishl_imm_s(b_raw, 2);
            let t3 = b.ins().bor(b6, b4);
            let t4 = b.ins().bor(b2, b_raw);
            let bv = b.ins().bor(t3, t4);
            let g8  = b.ins().ishl_imm_s(g,  8);
            let b16 = b.ins().ishl_imm_s(bv, 16);
            let t   = b.ins().bor(b16, g8);
            b.ins().bor(t, r)
        }
        _ => {
            // rgb12→rgb24: r=bits[3:0], g=bits[7:4], b=bits[11:8]
            let r_raw = b.ins().band_imm_s(val, 0xF);
            let r4  = b.ins().ishl_imm_s(r_raw, 4);
            let r   = b.ins().bor(r4, r_raw);
            let gs    = b.ins().ushr_imm_s(val, 4);
            let g_raw = b.ins().band_imm_s(gs, 0xF);
            let g4  = b.ins().ishl_imm_s(g_raw, 4);
            let g   = b.ins().bor(g4, g_raw);
            let bs    = b.ins().ushr_imm_s(val, 8);
            let b_raw = b.ins().band_imm_s(bs, 0xF);
            let bb4 = b.ins().ishl_imm_s(b_raw, 4);
            let bv  = b.ins().bor(bb4, b_raw);
            let g8  = b.ins().ishl_imm_s(g,  8);
            let b16 = b.ins().ishl_imm_s(bv, 16);
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
    let sa   = b.ins().ushr_imm_s(src, 24); // alpha from src bits[31:24]
    let c255 = b.ins().iconst(types::I32, 255);

    // Extract each 8-bit channel (no nesting)
    let sr   = b.ins().band_imm_s(src, 0xFF);
    let src8 = b.ins().ushr_imm_s(src,  8);
    let sg   = b.ins().band_imm_s(src8, 0xFF);
    let src16= b.ins().ushr_imm_s(src, 16);
    let sb   = b.ins().band_imm_s(src16, 0xFF);
    let dr   = b.ins().band_imm_s(dst, 0xFF);
    let dst8 = b.ins().ushr_imm_s(dst,  8);
    let dg   = b.ins().band_imm_s(dst8, 0xFF);
    let dst16= b.ins().ushr_imm_s(dst, 16);
    let db   = b.ins().band_imm_s(dst16, 0xFF);
    let dst24= b.ins().ushr_imm_s(dst, 24);
    let da   = b.ins().band_imm_s(dst24, 0xFF);

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
            b.ins().ishl_imm_s(cl, $shift)
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

// ── Host FIFO IR helpers ──────────────────────────────────────────────────────

/// Emit a byte-swap of a 32-bit value (rev8 semantics).
/// Mirrors u32::swap_bytes().
fn emit_bswap32(b: &mut FunctionBuilder, val: Value) -> Value {
    // Swap: byte0 and byte3, byte1 and byte2.
    // result = ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v >> 8) & 0xFF00) | ((v >> 24) & 0xFF)
    let b0   = b.ins().band_imm_s(val, 0xFF);
    let b0s  = b.ins().ishl_imm_s(b0, 24);
    let b1   = b.ins().ushr_imm_s(val, 8);
    let b1m  = b.ins().band_imm_s(b1, 0xFF);
    let b1s  = b.ins().ishl_imm_s(b1m, 16);
    let b2   = b.ins().ushr_imm_s(val, 16);
    let b2m  = b.ins().band_imm_s(b2, 0xFF);
    let b2s  = b.ins().ishl_imm_s(b2m, 8);
    let b3   = b.ins().ushr_imm_s(val, 24);
    let b3m  = b.ins().band_imm_s(b3, 0xFF);
    let r1   = b.ins().bor(b0s, b1s);
    let r2   = b.ins().bor(b2s, b3m);
    b.ins().bor(r1, r2)
}

/// Emit a byte-swap of a 64-bit value (rev8 semantics).
/// Mirrors u64::swap_bytes().
fn emit_bswap64(b: &mut FunctionBuilder, val: Value) -> Value {
    // Swap each byte: bytes 0..7 → 7..0.
    let mask = b.ins().iconst(types::I64, 0xFF);
    let mut bytes = [b.ins().iconst(types::I64, 0); 8];
    for i in 0u32..8 {
        let shifted = if i == 0 { val } else { b.ins().ushr_imm_s(val, (i * 8) as i64) };
        bytes[i as usize] = b.ins().band(shifted, mask);
    }
    // Reassemble in reversed order
    let mut result = bytes[7]; // byte7 → position 0 (shift 0)
    for i in (0u32..7).rev() {
        let dest_shift = (7 - i) * 8;
        let placed = b.ins().ishl_imm_s(bytes[i as usize], dest_shift as i64);
        result = b.ins().bor(result, placed);
    }
    result
}

/// Emit IR to extract one pixel from the MSB of the host shifter and advance it.
/// Mirrors Rex3::fetch_host_pixel (after initial load+swapendian, which is done at shader entry).
///
/// Returns `(pixel_24bit: Value, new_shifter: Value)`.
/// - `pixel_24bit`: 24-bit BGR value (expanded from hostdepth format).
/// - `new_shifter`:  shifter after shifting left by host_shift bits.
///
/// hostdepth meaning:
///   0 = 4bpp nibble in bits[63:60] (8-bit slot, shift=8, mask=0xF)
///   1 = 8bpp byte  in bits[63:56] (8-bit slot, shift=8, mask=0xFF)
///   2 = 12bpp word in bits[63:48] (16-bit slot, shift=16, mask=0xFFF)
///   3 = 32bpp word in bits[63:32] (32-bit slot, shift=32, mask=0xFFFFFFFF)
fn emit_fetch_host_pixel_ir(
    b:   &mut FunctionBuilder,
    shifter: Value, // I64
    dm1: &Dm1,
) -> (Value, Value) {
    let host_shift = dm1.host_shift();
    let hostdepth  = dm1.hostdepth();

    // Extract pixel from MSB of shifter based on hostdepth.
    // Both packed and non-packed use the same top-of-shifter extraction — the difference is
    // only how many pixels fit per word and whether the shifter advances.
    // Non-packed: one pixel per GO, shifter loaded fresh each time.
    //   where m_host_shift = 64 - s_host_shifts[hostdepth]:
    //     4bpp  → shift=56, mask=0xF
    //     8bpp  → shift=56, mask=0xFF
    //     12bpp → shift=48, mask=0xFFF
    //     32bpp → shift=32, mask=0xFFFFFFFF
    let (pixel_raw, new_shifter) = if !dm1.rwpacked() {
        // Non-packed: extract pixel from the top of the shifter using same logic as packed.
        let pixel_raw: Value = match hostdepth {
            0 => {
                let hi = b.ins().ushr_imm_s(shifter, 56);
                let r  = b.ins().ireduce(types::I32, hi);
                b.ins().band_imm_s(r, 0xF)
            }
            1 => {
                let hi = b.ins().ushr_imm_s(shifter, 56);
                let r  = b.ins().ireduce(types::I32, hi);
                b.ins().band_imm_s(r, 0xFF)
            }
            2 => {
                let hi = b.ins().ushr_imm_s(shifter, 48);
                let r  = b.ins().ireduce(types::I32, hi);
                b.ins().band_imm_s(r, 0xFFF)
            }
            _ => {
                // 32bpp: full top 32 bits
                let hi = b.ins().ushr_imm_s(shifter, 32);
                b.ins().ireduce(types::I32, hi)
            }
        };
        // Shifter is reloaded each GO in non-packed mode; advance is a no-op placeholder.
        let shifted = b.ins().ishl_imm_s(shifter, 32);
        (pixel_raw, shifted)
    } else {
        let pixel_raw: Value = match hostdepth {
            0 => {
                // 4bpp: bits[63:60] → 4-bit value
                let hi = b.ins().ushr_imm_s(shifter, 60);
                b.ins().ireduce(types::I32, hi)
            }
            1 => {
                // 8bpp: bits[63:56] → 8-bit value
                let hi = b.ins().ushr_imm_s(shifter, 56);
                b.ins().ireduce(types::I32, hi)
            }
            2 => {
                // 12bpp: bits[63:48] → 12-bit value
                let hi = b.ins().ushr_imm_s(shifter, 48);
                let r  = b.ins().ireduce(types::I32, hi);
                b.ins().band_imm_s(r, 0xFFF)
            }
            _ => {
                // 32bpp: bits[63:32]
                let hi = b.ins().ushr_imm_s(shifter, 32);
                b.ins().ireduce(types::I32, hi)
            }
        };
        let shift_amt = b.ins().iconst(types::I64, host_shift as i64);
        let shifted = b.ins().ishl(shifter, shift_amt);
        (pixel_raw, shifted)
    };

    // Expand from host depth to 24-bit BGR if rgbmode, else pass through as CI index.
    let pixel_24 = if dm1.rgbmode() {
        match hostdepth {
            0 => emit_expand_ir(b, pixel_raw, 0),  // 4bpp → 24bit
            1 => emit_expand_ir(b, pixel_raw, 1),  // 8bpp → 24bit
            2 => emit_expand_ir(b, pixel_raw, 2),  // 12bpp → 24bit
            _ => pixel_raw,                         // 32bpp: already 24-bit (high byte = alpha)
        }
    } else {
        pixel_raw // CI index: return as-is
    };

    (pixel_24, new_shifter)
}

/// Emit IR to compress one pixel and pack it into the accumulator shifter.
/// Mirrors Rex3::store_host_pixel (the pack step only; send_host_word is emitted separately at loop end).
///
/// Returns `new_shifter: Value` (I64).
///
/// Pack: `acc = (acc << host_shift) | compressed_pixel`
fn emit_pack_host_pixel_ir(
    b:       &mut FunctionBuilder,
    acc:     Value, // I64
    pixel:   Value, // I32, 24-bit BGR (for rgbmode) or CI index
    dm1:     &Dm1,
) -> Value {
    let host_shift = dm1.host_shift();
    let hostdepth  = dm1.hostdepth();

    // Compress pixel to hostdepth format (rgbmode → compress, CI → mask).
    let compressed_i32: Value = if dm1.rgbmode() {
        match hostdepth {
            0 => emit_compress_ir(b, pixel, 0, false),  // 24bit → 4bpp (no dither for host)
            1 => emit_compress_ir(b, pixel, 1, false),  // 24bit → 8bpp
            2 => emit_compress_ir(b, pixel, 2, false),  // 24bit → 12bpp
            _ => b.ins().bor_imm_s(pixel, 0xFF000000u32 as i64), // 32bpp: set alpha=0xFF (compress_32_rgb)
        }
    } else {
        // CI mode: mask to depth
        let mask: i64 = match hostdepth { 0 => 0xF, 1 => 0xFF, 2 => 0xFFF, _ => 0xFFFFFFFF };
        b.ins().band_imm_s(pixel, mask)
    };

    // Pack: acc = (acc << shift) | compressed
    let compressed_i64 = b.ins().uextend(types::I64, compressed_i32);
    if !dm1.rwpacked() || host_shift == 0 {
        // Non-packed: single pixel, just put in low 32 bits.
        compressed_i64
    } else {
        let shift_amt = b.ins().iconst(types::I64, host_shift as i64);
        let acc_shifted = b.ins().ishl(acc, shift_amt);
        b.ins().bor(acc_shifted, compressed_i64)
    }
}

/// Emit IR to apply send_host_word semantics and store to ctx.hostrw.
/// Mirrors Rex3::send_host_word: if swapendian → swap_bytes; else if !rwdouble → shift left 32.
/// Used at loop exit for HOSTR mode.
fn emit_store_hostrw(
    b:       &mut FunctionBuilder,
    ctx_ptr: Value,
    mem:     &MemFlagsData,
    shifter: Value, // I64: accumulated packed pixels
    dm1:     &Dm1,
) {
    let val = if dm1.swapendian() {
        emit_bswap64(b, shifter)
    } else if !dm1.rwdouble() {
        // 32-bit mode: shift left 32 to align data to high half
        b.ins().ishl_imm_s(shifter, 32)
    } else {
        shifter
    };
    b.ins().store(*mem, val, ctx_ptr,
        ir::immediates::Offset32::new(ctx_off!(hostrw) as i32));
}
