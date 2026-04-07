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
//! Complex pure functions (compress, expand, dither, blend) are called as `extern "C"`
//! helpers — they are stateless and trivially safe to call from JIT code.

use std::mem::offset_of;

use cranelift_codegen::ir::{self, types, AbiParam, InstBuilder, MemFlags, Value};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

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

// ── Helper functions callable from JIT (pure, no Rex3 pointer needed) ────────

pub extern "C" fn helper_rgb24_to_rgb4(val: u32) -> u32 {
    let r = (val >> 7) & 1;
    let g = (val >> 14) & 3;
    let b = (val >> 23) & 1;
    (b << 3) | (g << 1) | r
}
pub extern "C" fn helper_rgb24_to_rgb8(val: u32) -> u32 {
    let r = (val >> 5) & 7;
    let g = (val >> 13) & 7;
    let b = (val >> 22) & 3;
    (b << 6) | (g << 3) | r
}
pub extern "C" fn helper_rgb24_to_rgb12(val: u32) -> u32 {
    let r = (val >> 4) & 0xF;
    let g = (val >> 12) & 0xF;
    let b = (val >> 20) & 0xF;
    (b << 8) | (g << 4) | r
}
pub extern "C" fn helper_rgb4_to_rgb24(val: u32) -> u32 {
    let r = if (val & 1) != 0 { 0xFF } else { 0 };
    let g_raw = (val >> 1) & 3;
    let g = (g_raw << 6) | (g_raw << 4) | (g_raw << 2) | g_raw;
    let b = if (val & 8) != 0 { 0xFF } else { 0 };
    (b << 16) | (g << 8) | r
}
pub extern "C" fn helper_rgb8_to_rgb24(val: u32) -> u32 {
    let r_raw = val & 7;
    let r = (r_raw << 5) | (r_raw << 2) | (r_raw >> 1);
    let g_raw = (val >> 3) & 7;
    let g = (g_raw << 5) | (g_raw << 2) | (g_raw >> 1);
    let b_raw = (val >> 6) & 3;
    let b = (b_raw << 6) | (b_raw << 4) | (b_raw << 2) | b_raw;
    (b << 16) | (g << 8) | r
}
pub extern "C" fn helper_rgb12_to_rgb24(val: u32) -> u32 {
    let r_raw = val & 0xF;
    let r = (r_raw << 4) | r_raw;
    let g_raw = (val >> 4) & 0xF;
    let g = (g_raw << 4) | g_raw;
    let b_raw = (val >> 8) & 0xF;
    let b = (b_raw << 4) | b_raw;
    (b << 16) | (g << 8) | r
}

const BAYER: [u8; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

pub extern "C" fn helper_rgb24_to_rgb4_dither(val: u32) -> u32 {
    let bayer = BAYER[(val >> 24) as usize];
    let r = (val & 0xFF) as u8;
    let g = ((val >> 8) & 0xFF) as u8;
    let b = ((val >> 16) & 0xFF) as u8;
    let sr = (r >> 3).wrapping_sub(r >> 4);
    let sg = (g >> 2).wrapping_sub(g >> 4);
    let sb = (b >> 3).wrapping_sub(b >> 4);
    let mut dr = (sr >> 4) & 1;
    let mut dg = (sg >> 4) & 3;
    let mut db = (sb >> 4) & 1;
    if (sr & 0xf) > bayer { dr = (dr + 1).min(1); }
    if (sg & 0xf) > bayer { dg = (dg + 1).min(3); }
    if (sb & 0xf) > bayer { db = (db + 1).min(1); }
    ((db << 3) | (dg << 1) | dr) as u32
}
pub extern "C" fn helper_rgb24_to_rgb8_dither(val: u32) -> u32 {
    let bayer = BAYER[(val >> 24) as usize];
    let r = (val & 0xFF) as u8;
    let g = ((val >> 8) & 0xFF) as u8;
    let b = ((val >> 16) & 0xFF) as u8;
    let sr = (r >> 1).wrapping_sub(r >> 4);
    let sg = (g >> 1).wrapping_sub(g >> 4);
    let sb = (b >> 2).wrapping_sub(b >> 4);
    let mut dr = (sr >> 4) & 7;
    let mut dg = (sg >> 4) & 7;
    let mut db = (sb >> 4) & 3;
    if (sr & 0xf) > bayer { dr = (dr + 1).min(7); }
    if (sg & 0xf) > bayer { dg = (dg + 1).min(7); }
    if (sb & 0xf) > bayer { db = (db + 1).min(3); }
    ((db << 6) | (dg << 3) | dr) as u32
}
pub extern "C" fn helper_rgb24_to_rgb12_dither(val: u32) -> u32 {
    let bayer = BAYER[(val >> 24) as usize];
    let r = (val & 0xFF) as u32;
    let g = ((val >> 8) & 0xFF) as u32;
    let b = ((val >> 16) & 0xFF) as u32;
    let sr = r - (r >> 4);
    let sg = g - (g >> 4);
    let sb = b - (b >> 4);
    let mut dr = (sr >> 4) & 15;
    let mut dg = (sg >> 4) & 15;
    let mut db = (sb >> 4) & 15;
    if (sr & 0xf) > bayer as u32 { dr = (dr + 1).min(15); }
    if (sg & 0xf) > bayer as u32 { dg = (dg + 1).min(15); }
    if (sb & 0xf) > bayer as u32 { db = (db + 1).min(15); }
    (db << 8) | (dg << 4) | dr
}

/// Blend: src and dst are 32-bit RGBA (A in bits[31:24]), sfactor/dfactor are 3-bit.
pub extern "C" fn helper_blend(src: u32, dst: u32, sfactor: u32, dfactor: u32) -> u32 {
    let sa = (src >> 24) & 0xFF;
    let get_factor = |sel: u32, c: u32, a: u32| -> u32 {
        match sel {
            0 => 0,
            1 => 255,
            2 => c,
            3 => 255 - c,
            4 => a,
            5 => 255 - a,
            _ => 0,
        }
    };
    let mut res = 0u32;
    for i in 0..4 {
        let shift = i * 8;
        let s_c = (src >> shift) & 0xFF;
        let d_c = (dst >> shift) & 0xFF;
        let sf = get_factor(sfactor, d_c, sa);
        let df = get_factor(dfactor, s_c, sa);
        let val = (s_c * sf + d_c * df) / 255;
        res |= val.min(255) << shift;
    }
    res
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

// ── Helper IDs bundled together ───────────────────────────────────────────────

struct Helpers {
    compress:    FuncId, // fn(u32) -> u32
    expand:      FuncId, // fn(u32) -> u32
    blend:       FuncId, // fn(u32, u32, u32, u32) -> u32
}

// ── Compiler ──────────────────────────────────────────────────────────────────

pub struct ShaderCompiler {
    jit_module:  JITModule,
    ctx:         Context,
    builder_ctx: FunctionBuilderContext,
    counter:     u32,

    // Helper FuncIds declared once, referenced in each compiled function.
    fn_rgb24_to_rgb4:        FuncId,
    fn_rgb24_to_rgb8:        FuncId,
    fn_rgb24_to_rgb12:       FuncId,
    fn_rgb4_to_rgb24:        FuncId,
    fn_rgb8_to_rgb24:        FuncId,
    fn_rgb12_to_rgb24:       FuncId,
    fn_rgb24_to_rgb4_dither: FuncId,
    fn_rgb24_to_rgb8_dither: FuncId,
    fn_rgb24_to_rgb12_dither:FuncId,
    fn_blend:                FuncId,
}

impl ShaderCompiler {
    pub fn new() -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        flag_builder.set("is_pic", "false").unwrap();

        let isa_builder = cranelift_native::builder().expect("host ISA not supported");
        let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();
        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register helper symbols
        macro_rules! sym {
            ($b:ident, $name:expr, $fn:expr) => {
                $b.symbol($name, $fn as *const u8);
            }
        }
        sym!(jit_builder, "rex_rgb24_to_rgb4",         helper_rgb24_to_rgb4         as extern "C" fn(u32) -> u32);
        sym!(jit_builder, "rex_rgb24_to_rgb8",         helper_rgb24_to_rgb8         as extern "C" fn(u32) -> u32);
        sym!(jit_builder, "rex_rgb24_to_rgb12",        helper_rgb24_to_rgb12        as extern "C" fn(u32) -> u32);
        sym!(jit_builder, "rex_rgb4_to_rgb24",         helper_rgb4_to_rgb24         as extern "C" fn(u32) -> u32);
        sym!(jit_builder, "rex_rgb8_to_rgb24",         helper_rgb8_to_rgb24         as extern "C" fn(u32) -> u32);
        sym!(jit_builder, "rex_rgb12_to_rgb24",        helper_rgb12_to_rgb24        as extern "C" fn(u32) -> u32);
        sym!(jit_builder, "rex_rgb24_to_rgb4_dither",  helper_rgb24_to_rgb4_dither  as extern "C" fn(u32) -> u32);
        sym!(jit_builder, "rex_rgb24_to_rgb8_dither",  helper_rgb24_to_rgb8_dither  as extern "C" fn(u32) -> u32);
        sym!(jit_builder, "rex_rgb24_to_rgb12_dither", helper_rgb24_to_rgb12_dither as extern "C" fn(u32) -> u32);
        sym!(jit_builder, "rex_blend",                 helper_blend as extern "C" fn(u32, u32, u32, u32) -> u32);

        let mut jit_module = JITModule::new(jit_builder);
        let ptr_type = jit_module.target_config().pointer_type();

        // u32→u32 helper signature
        let mut sig_u32_u32 = jit_module.make_signature();
        sig_u32_u32.params.push(AbiParam::new(types::I32));
        sig_u32_u32.returns.push(AbiParam::new(types::I32));

        // blend signature: (u32, u32, u32, u32) -> u32
        let mut sig_blend = jit_module.make_signature();
        for _ in 0..4 { sig_blend.params.push(AbiParam::new(types::I32)); }
        sig_blend.returns.push(AbiParam::new(types::I32));

        let _ = ptr_type; // may be unused in some configurations

        macro_rules! decl {
            ($m:ident, $name:literal, $sig:ident) => {
                $m.declare_function($name, Linkage::Import, &$sig).unwrap()
            }
        }
        let fn_rgb24_to_rgb4         = decl!(jit_module, "rex_rgb24_to_rgb4",         sig_u32_u32);
        let fn_rgb24_to_rgb8         = decl!(jit_module, "rex_rgb24_to_rgb8",         sig_u32_u32);
        let fn_rgb24_to_rgb12        = decl!(jit_module, "rex_rgb24_to_rgb12",        sig_u32_u32);
        let fn_rgb4_to_rgb24         = decl!(jit_module, "rex_rgb4_to_rgb24",         sig_u32_u32);
        let fn_rgb8_to_rgb24         = decl!(jit_module, "rex_rgb8_to_rgb24",         sig_u32_u32);
        let fn_rgb12_to_rgb24        = decl!(jit_module, "rex_rgb12_to_rgb24",        sig_u32_u32);
        let fn_rgb24_to_rgb4_dither  = decl!(jit_module, "rex_rgb24_to_rgb4_dither",  sig_u32_u32);
        let fn_rgb24_to_rgb8_dither  = decl!(jit_module, "rex_rgb24_to_rgb8_dither",  sig_u32_u32);
        let fn_rgb24_to_rgb12_dither = decl!(jit_module, "rex_rgb24_to_rgb12_dither", sig_u32_u32);
        let fn_blend                 = decl!(jit_module, "rex_blend",                 sig_blend);

        Self {
            ctx: jit_module.make_context(),
            jit_module,
            builder_ctx: FunctionBuilderContext::new(),
            counter: 0,
            fn_rgb24_to_rgb4, fn_rgb24_to_rgb8, fn_rgb24_to_rgb12,
            fn_rgb4_to_rgb24, fn_rgb8_to_rgb24, fn_rgb12_to_rgb24,
            fn_rgb24_to_rgb4_dither, fn_rgb24_to_rgb8_dither, fn_rgb24_to_rgb12_dither,
            fn_blend,
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

        let helpers = Helpers {
            compress: self.compress_func_id(&dm1),
            expand:   self.expand_func_id(&dm1),
            blend:    self.fn_blend,
        };

        let result = emit_shader(
            &mut self.ctx.func,
            &mut self.builder_ctx,
            &mut self.jit_module,
            &helpers,
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

    fn compress_func_id(&self, dm1: &Dm1) -> FuncId {
        if !dm1.rgbmode() {
            // CI mode: identity — no compress func needed. We'll inline identity.
            // Return a placeholder; caller checks rgbmode before using.
            self.fn_rgb24_to_rgb8
        } else if dm1.dither() {
            match dm1.drawdepth() {
                0 => self.fn_rgb24_to_rgb4_dither,
                1 => self.fn_rgb24_to_rgb8_dither,
                2 => self.fn_rgb24_to_rgb12_dither,
                _ => self.fn_rgb24_to_rgb8, // 24bpp: identity (not called)
            }
        } else {
            match dm1.drawdepth() {
                0 => self.fn_rgb24_to_rgb4,
                1 => self.fn_rgb24_to_rgb8,
                2 => self.fn_rgb24_to_rgb12,
                _ => self.fn_rgb24_to_rgb8, // 24bpp: identity (not called)
            }
        }
    }

    fn expand_func_id(&self, dm1: &Dm1) -> FuncId {
        if !dm1.rgbmode() {
            self.fn_rgb8_to_rgb24 // placeholder; not called in CI mode
        } else {
            match dm1.drawdepth() {
                0 => self.fn_rgb4_to_rgb24,
                1 => self.fn_rgb8_to_rgb24,
                2 => self.fn_rgb12_to_rgb24,
                _ => self.fn_rgb8_to_rgb24, // 24bpp: identity
            }
        }
    }
}

// ── Shader IR emission ────────────────────────────────────────────────────────

/// Emit the full shader IR. Returns false if the mode cannot be compiled.
fn emit_shader(
    func: &mut ir::Function,
    builder_ctx: &mut FunctionBuilderContext,
    module: &mut JITModule,
    helpers: &Helpers,
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

    // Declare helper refs for this function
    let compress_ref = module.declare_func_in_func(helpers.compress, b.func);
    let expand_ref   = module.declare_func_in_func(helpers.expand,   b.func);
    let blend_ref    = module.declare_func_in_func(helpers.blend,    b.func);

    // ── Load loop-invariant ctx fields ────────────────────────────────────────
    macro_rules! ld32 { ($off:expr) => {
        b.ins().load(types::I32, mem, ctx_ptr, ir::immediates::Offset32::new($off as i32))
    }}
    macro_rules! st32 { ($off:expr, $val:expr) => {
        b.ins().store(mem, $val, ctx_ptr, ir::immediates::Offset32::new($off as i32))
    }}
    macro_rules! ld8 { ($off:expr) => {
        b.ins().load(types::I8, mem, ctx_ptr, ir::immediates::Offset32::new($off as i32))
    }}
    macro_rules! st8 { ($off:expr, $val:expr) => {
        b.ins().store(mem, $val, ctx_ptr, ir::immediates::Offset32::new($off as i32))
    }}
    macro_rules! ld_bool { ($off:expr) => {{
        let v: Value = b.ins().load(types::I8, mem, ctx_ptr, ir::immediates::Offset32::new($off as i32));
        v
    }}}
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

    // lronly: skip if x_dec (right-to-left direction), handled by skipping entire primitive
    // Actually lronly skips if x_dec — we check at entry to draw_span/draw_block.
    // For JIT: if lronly and x_dec we don't emit pixels (whole draw is a no-op).
    // This is handled in execute_go before calling the shader, so we don't special-case here.

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
    let draw_block = b.create_block();
    let mut use_bg_flag: Value = b.ins().iconst(types::I8, 0);

    if dm0.enzpattern() {
        let zp_block = b.create_block();
        let zp_pass  = b.create_block();
        // bit = (zpattern >> zpat_bit) & 1
        let zpattern_v = ld32!(ctx_off!(zpattern));
        let zpat_bit32 = b.ins().uextend(types::I32, zpat_bit_v);
        let zpat_shifted = b.ins().ushr(zpattern_v, zpat_bit32);
        let bit_v = b.ins().band_imm(zpat_shifted, 1);
        let bit_set = b.ins().icmp_imm(IntCC::NotEqual, bit_v, 0);
        b.ins().brif(bit_set, zp_pass, &[], zp_block, &[]);
        b.switch_to_block(zp_block); b.seal_block(zp_block);
        if dm0.zpopaque() {
            use_bg_flag = b.ins().iconst(types::I8, 1);
            b.ins().jump(zp_pass, &[]);
        } else {
            b.ins().jump(skip_block, &[]);
        }
        b.switch_to_block(zp_pass); b.seal_block(zp_pass);
    }

    if dm0.enlspattern() {
        let ls_block = b.create_block();
        let ls_pass  = b.create_block();
        let lspattern_v = ld32!(ctx_off!(lspattern));
        let pat_bit32 = b.ins().uextend(types::I32, pat_bit_v);
        let lspat_shifted = b.ins().ushr(lspattern_v, pat_bit32);
        let bit_v = b.ins().band_imm(lspat_shifted, 1);
        let bit_set = b.ins().icmp_imm(IntCC::NotEqual, bit_v, 0);
        b.ins().brif(bit_set, ls_pass, &[], ls_block, &[]);
        b.switch_to_block(ls_block); b.seal_block(ls_block);
        if dm0.lsopaque() {
            use_bg_flag = b.ins().iconst(types::I8, 1);
            b.ins().jump(ls_pass, &[]);
        } else {
            b.ins().jump(skip_block, &[]);
        }
        b.switch_to_block(ls_pass); b.seal_block(ls_pass);
    }

    b.ins().jump(draw_block, &[]);
    b.switch_to_block(draw_block); b.seal_block(draw_block);

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

    // Read/mask for depth:
    let depth_mask: i64 = match dm1.drawdepth() { 0 => 0xF, 1 => 0xFF, 2 => 0xFFF, _ => 0xFFFFFF };
    let dblsrc_shift: i64 = match dm1.drawdepth() { 0 => 4, 1 => 8, 2 => 12, _ => 0 };

    let dst_plane = if needs_dst {
        if dm1.dblsrc() && dblsrc_shift > 0 {
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
            let inst = b.ins().call(expand_ref, &[dst_plane]);
            b.inst_results(inst)[0]
        } else {
            dst_plane
        };
        let blend_dst = if dm1.backblend() { colorback_v } else { dst_24 };
        let sf_v = b.ins().iconst(types::I32, dm1.sfactor() as i64);
        let df_v = b.ins().iconst(types::I32, dm1.dfactor() as i64);
        let blended = {
            let inst = b.ins().call(blend_ref, &[src_color, blend_dst, sf_v, df_v]);
            b.inst_results(inst)[0]
        };
        // bayer_pack + compress
        let packed = bayer_pack_ir(&mut b, blended, x_v, y_v);
        if dm1.rgbmode() && dm1.drawdepth() != 3 {
            let inst = b.ins().call(compress_ref, &[packed]);
            b.inst_results(inst)[0]
        } else {
            packed
        }
    } else {
        // Logic op path
        // compress(bayer_pack(raw_src, x, y))
        let packed = bayer_pack_ir(&mut b, src_color, x_v, y_v);
        let compressed = if dm1.rgbmode() && dm1.drawdepth() != 3 {
            let inst = b.ins().call(compress_ref, &[packed]);
            b.inst_results(inst)[0]
        } else {
            packed
        };
        // amplify for dblsrc
        let amp_src = if dm1.dblsrc() && dblsrc_shift > 0 {
            let shifted = b.ins().ishl_imm(compressed, dblsrc_shift);
            b.ins().bor(compressed, shifted)
        } else {
            compressed
        };
        let amp_dst = if dm1.dblsrc() && dblsrc_shift > 0 {
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
        // stoponx=1: continue x loop — jump back to loop_header
        let new_first = b.ins().iconst(types::I8, 0);
        let mut back_args: Vec<Value> = vec![xstart_next, ystart_v, new_first];
        if dm0.shade() { back_args.extend([new_cr, new_cg, new_cb, new_ca]); }
        if dm0.enzpattern() { back_args.push(new_zpat_bit); }
        if dm0.enlspattern() { back_args.push(new_pat_bit); back_args.push(new_lsmode); }
        b.ins().jump(loop_header, &back_args);
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
