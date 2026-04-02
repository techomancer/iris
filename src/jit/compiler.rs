//! Block compiler: translates MIPS basic blocks to native code via Cranelift.

use cranelift_codegen::ir::{self, types, AbiParam, InstBuilder, MemFlags, Value};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::{self, Context};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::mips_exec::DecodedInstr;
use crate::mips_isa::*;

use super::cache::CompiledBlock;
use super::context::{JitContext, EXIT_NORMAL, EXIT_INTERPRET};

pub struct BlockCompiler {
    jit_module: JITModule,
    ctx: Context,
    builder_ctx: FunctionBuilderContext,
    func_id_counter: u32,
}

impl BlockCompiler {
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
            func_id_counter: 0,
        }
    }

    /// Compile a block of MIPS instructions to native code.
    /// `instrs` is a slice of (raw_word, DecodedInstr) for each instruction in the block.
    /// `block_pc` is the virtual PC of the first instruction.
    /// Returns None if the block is empty or compilation fails.
    pub fn compile_block(
        &mut self,
        instrs: &[(u32, DecodedInstr)],
        block_pc: u64,
    ) -> Option<CompiledBlock> {
        if instrs.is_empty() {
            return None;
        }

        let num_instrs = instrs.len() as u32;

        // Create a unique function name
        let name = format!("jit_block_{:x}_{}", block_pc, self.func_id_counter);
        self.func_id_counter += 1;

        // Declare function signature: extern "C" fn(*mut JitContext)
        let ptr_type = self.jit_module.target_config().pointer_type();
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
        self.ctx.func.signature.call_conv = cranelift_codegen::isa::CallConv::SystemV;

        let func_id = self.jit_module
            .declare_function(&name, Linkage::Local, &self.ctx.func.signature)
            .unwrap();

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let ctx_ptr = builder.block_params(entry_block)[0];
        let mem = MemFlags::trusted();

        // Load GPRs 1-31 from JitContext (gpr[0] is always 0)
        let mut gpr = [builder.ins().iconst(types::I64, 0); 32];
        for i in 1..32usize {
            gpr[i] = builder.ins().load(
                types::I64, mem, ctx_ptr,
                ir::immediates::Offset32::new(JitContext::gpr_offset(i)),
            );
        }

        // Load hi/lo
        let mut hi = builder.ins().load(types::I64, mem, ctx_ptr,
            ir::immediates::Offset32::new(JitContext::hi_offset()));
        let mut lo = builder.ins().load(types::I64, mem, ctx_ptr,
            ir::immediates::Offset32::new(JitContext::lo_offset()));

        // Emit IR for each instruction
        let mut compiled_count = 0u32;
        for (_, d) in instrs {
            if !emit_instruction(&mut builder, ctx_ptr, &mut gpr, &mut hi, &mut lo, d) {
                break;
            }
            compiled_count += 1;
        }

        if compiled_count == 0 {
            // Nothing was compilable — clean up and return None
            builder.ins().return_(&[]);
            builder.finalize();
            self.ctx.clear();
            return None;
        }

        // Store GPRs back (skip r0)
        for i in 1..32usize {
            builder.ins().store(mem, gpr[i], ctx_ptr,
                ir::immediates::Offset32::new(JitContext::gpr_offset(i)));
        }

        // Store hi/lo back
        builder.ins().store(mem, hi, ctx_ptr,
            ir::immediates::Offset32::new(JitContext::hi_offset()));
        builder.ins().store(mem, lo, ctx_ptr,
            ir::immediates::Offset32::new(JitContext::lo_offset()));

        // Set exit PC = block_pc + 4 * compiled_count
        let exit_pc = block_pc.wrapping_add(compiled_count as u64 * 4);
        let exit_pc_val = builder.ins().iconst(types::I64, exit_pc as i64);
        builder.ins().store(mem, exit_pc_val, ctx_ptr,
            ir::immediates::Offset32::new(JitContext::pc_offset()));

        // Set exit_reason = EXIT_NORMAL
        let exit_val = builder.ins().iconst(types::I32, EXIT_NORMAL as i64);
        builder.ins().store(mem, exit_val, ctx_ptr,
            ir::immediates::Offset32::new(JitContext::exit_reason_offset()));

        // Set block_instrs_executed
        let count_val = builder.ins().iconst(types::I32, compiled_count as i64);
        builder.ins().store(mem, count_val, ctx_ptr,
            ir::immediates::Offset32::new(JitContext::block_instrs_offset()));

        builder.ins().return_(&[]);
        builder.finalize();

        // Compile to native code
        self.jit_module.define_function(func_id, &mut self.ctx).unwrap();
        self.jit_module.clear_context(&mut self.ctx);
        self.jit_module.finalize_definitions().unwrap();

        let code_ptr = self.jit_module.get_finalized_function(func_id);
        let code_size = 0u32; // JITModule doesn't expose size easily; not critical

        Some(CompiledBlock {
            entry: code_ptr,
            phys_addr: 0, // filled in by caller
            virt_addr: block_pc,
            len_mips: compiled_count,
            len_native: code_size,
        })
    }
}

/// Emit Cranelift IR for a single MIPS instruction.
/// Returns true if the instruction was compiled, false if it should terminate the block.
fn emit_instruction(
    builder: &mut FunctionBuilder,
    ctx_ptr: Value,
    gpr: &mut [Value; 32],
    hi: &mut Value,
    lo: &mut Value,
    d: &DecodedInstr,
) -> bool {
    let op = d.op as u32;
    let rs = d.rs as usize;
    let rt = d.rt as usize;
    let rd = d.rd as usize;
    let sa = d.sa as u32;
    let funct = d.funct as u32;

    match op {
        OP_SPECIAL => emit_special(builder, gpr, hi, lo, d, rs, rt, rd, sa, funct),
        OP_ADDIU  => { emit_addiu(builder, gpr, rs, rt, d); true }
        OP_DADDIU => { emit_daddiu(builder, gpr, rs, rt, d); true }
        OP_SLTI   => { emit_slti(builder, gpr, rs, rt, d); true }
        OP_SLTIU  => { emit_sltiu(builder, gpr, rs, rt, d); true }
        OP_ANDI   => { emit_andi(builder, gpr, rs, rt, d); true }
        OP_ORI    => { emit_ori(builder, gpr, rs, rt, d); true }
        OP_XORI   => { emit_xori(builder, gpr, rs, rt, d); true }
        OP_LUI    => { emit_lui(builder, gpr, rt, d); true }
        _ => false, // Non-ALU instruction — terminate block
    }
}

fn emit_special(
    builder: &mut FunctionBuilder,
    gpr: &mut [Value; 32],
    hi: &mut Value,
    lo: &mut Value,
    d: &DecodedInstr,
    rs: usize, rt: usize, rd: usize, sa: u32, funct: u32,
) -> bool {
    match funct {
        // --- Shifts (immediate) ---
        FUNCT_SLL  => { emit_sll(builder, gpr, rt, rd, sa); true }
        FUNCT_SRL  => { emit_srl(builder, gpr, rt, rd, sa); true }
        FUNCT_SRA  => { emit_sra(builder, gpr, rt, rd, sa); true }

        // --- Shifts (variable) ---
        FUNCT_SLLV => { emit_sllv(builder, gpr, rs, rt, rd); true }
        FUNCT_SRLV => { emit_srlv(builder, gpr, rs, rt, rd); true }
        FUNCT_SRAV => { emit_srav(builder, gpr, rs, rt, rd); true }

        // --- 64-bit shifts (immediate) ---
        FUNCT_DSLL   => { emit_dsll(builder, gpr, rt, rd, sa); true }
        FUNCT_DSRL   => { emit_dsrl(builder, gpr, rt, rd, sa); true }
        FUNCT_DSRA   => { emit_dsra(builder, gpr, rt, rd, sa); true }
        FUNCT_DSLL32 => { emit_dsll(builder, gpr, rt, rd, sa + 32); true }
        FUNCT_DSRL32 => { emit_dsrl(builder, gpr, rt, rd, sa + 32); true }
        FUNCT_DSRA32 => { emit_dsra(builder, gpr, rt, rd, sa + 32); true }

        // --- 64-bit shifts (variable) ---
        FUNCT_DSLLV => { emit_dsllv(builder, gpr, rs, rt, rd); true }
        FUNCT_DSRLV => { emit_dsrlv(builder, gpr, rs, rt, rd); true }
        FUNCT_DSRAV => { emit_dsrav(builder, gpr, rs, rt, rd); true }

        // --- ALU register ops ---
        FUNCT_ADDU => { emit_addu(builder, gpr, rs, rt, rd); true }
        FUNCT_SUBU => { emit_subu(builder, gpr, rs, rt, rd); true }
        FUNCT_AND  => { emit_and(builder, gpr, rs, rt, rd); true }
        FUNCT_OR   => { emit_or(builder, gpr, rs, rt, rd); true }
        FUNCT_XOR  => { emit_xor(builder, gpr, rs, rt, rd); true }
        FUNCT_NOR  => { emit_nor(builder, gpr, rs, rt, rd); true }
        FUNCT_SLT  => { emit_slt(builder, gpr, rs, rt, rd); true }
        FUNCT_SLTU => { emit_sltu(builder, gpr, rs, rt, rd); true }

        // --- 64-bit ALU ---
        FUNCT_DADDU => { emit_daddu(builder, gpr, rs, rt, rd); true }
        FUNCT_DSUBU => { emit_dsubu(builder, gpr, rs, rt, rd); true }

        // --- Multiply/Divide ---
        FUNCT_MULT  => { emit_mult(builder, gpr, hi, lo, rs, rt); true }
        FUNCT_MULTU => { emit_multu(builder, gpr, hi, lo, rs, rt); true }
        FUNCT_DIV   => { emit_div(builder, gpr, hi, lo, rs, rt); true }
        FUNCT_DIVU  => { emit_divu(builder, gpr, hi, lo, rs, rt); true }
        FUNCT_DMULT  => { emit_dmult(builder, gpr, hi, lo, rs, rt); true }
        FUNCT_DMULTU => { emit_dmultu(builder, gpr, hi, lo, rs, rt); true }
        FUNCT_DDIV   => { emit_ddiv(builder, gpr, hi, lo, rs, rt); true }
        FUNCT_DDIVU  => { emit_ddivu(builder, gpr, hi, lo, rs, rt); true }

        // --- HI/LO moves ---
        FUNCT_MFHI => { gpr[rd] = *hi; true }
        FUNCT_MTHI => { *hi = gpr[rs]; true }
        FUNCT_MFLO => { gpr[rd] = *lo; true }
        FUNCT_MTLO => { *lo = gpr[rs]; true }

        // --- Conditional moves ---
        FUNCT_MOVZ => { emit_movz(builder, gpr, rs, rt, rd); true }
        FUNCT_MOVN => { emit_movn(builder, gpr, rs, rt, rd); true }

        // --- SYNC (barrier, NOP for JIT) ---
        FUNCT_SYNC => true,

        // Everything else terminates the block
        _ => false,
    }
}

// ─── Helper: sign-extend i32 result to i64 ──────────────────────────────────

/// Truncate a 64-bit value to 32-bit, then sign-extend back to 64-bit.
/// Matches the interpreter pattern: `val as u32 as i32 as i64 as u64`.
fn sext32(builder: &mut FunctionBuilder, val: Value) -> Value {
    let narrow = builder.ins().ireduce(types::I32, val);
    builder.ins().sextend(types::I64, narrow)
}

// ─── Immediate ALU ops ───────────────────────────────────────────────────────

fn emit_addiu(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, d: &DecodedInstr) {
    // (rs as u32).wrapping_add(imm as u32) → sign-extend to 64
    let rs32 = builder.ins().ireduce(types::I32, gpr[rs]);
    let imm = builder.ins().iconst(types::I32, d.imm as i32 as i64);
    let sum = builder.ins().iadd(rs32, imm);
    gpr[rt] = builder.ins().sextend(types::I64, sum);
}

fn emit_daddiu(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, d: &DecodedInstr) {
    let imm = builder.ins().iconst(types::I64, d.imm as i32 as i64);
    gpr[rt] = builder.ins().iadd(gpr[rs], imm);
}

fn emit_slti(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, d: &DecodedInstr) {
    let imm = builder.ins().iconst(types::I64, d.imm as i32 as i64);
    let cmp = builder.ins().icmp(IntCC::SignedLessThan, gpr[rs], imm);
    gpr[rt] = builder.ins().uextend(types::I64, cmp);
}

fn emit_sltiu(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, d: &DecodedInstr) {
    // imm is sign-extended then compared as unsigned
    let imm = builder.ins().iconst(types::I64, d.imm as i32 as i64);
    let cmp = builder.ins().icmp(IntCC::UnsignedLessThan, gpr[rs], imm);
    gpr[rt] = builder.ins().uextend(types::I64, cmp);
}

fn emit_andi(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, d: &DecodedInstr) {
    // zero-extended immediate
    let imm = builder.ins().iconst(types::I64, (d.imm & 0xFFFF) as i64);
    gpr[rt] = builder.ins().band(gpr[rs], imm);
}

fn emit_ori(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, d: &DecodedInstr) {
    let imm = builder.ins().iconst(types::I64, (d.imm & 0xFFFF) as i64);
    gpr[rt] = builder.ins().bor(gpr[rs], imm);
}

fn emit_xori(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, d: &DecodedInstr) {
    let imm = builder.ins().iconst(types::I64, (d.imm & 0xFFFF) as i64);
    gpr[rt] = builder.ins().bxor(gpr[rs], imm);
}

fn emit_lui(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rt: usize, d: &DecodedInstr) {
    // imm is already shifted left 16 by decode (set_imm_lui)
    // sign-extend from 32 to 64
    gpr[rt] = builder.ins().iconst(types::I64, d.imm as i32 as i64);
}

// ─── Register ALU ops ────────────────────────────────────────────────────────

fn emit_addu(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    let a = builder.ins().ireduce(types::I32, gpr[rs]);
    let b = builder.ins().ireduce(types::I32, gpr[rt]);
    let sum = builder.ins().iadd(a, b);
    gpr[rd] = builder.ins().sextend(types::I64, sum);
}

fn emit_subu(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    let a = builder.ins().ireduce(types::I32, gpr[rs]);
    let b = builder.ins().ireduce(types::I32, gpr[rt]);
    let diff = builder.ins().isub(a, b);
    gpr[rd] = builder.ins().sextend(types::I64, diff);
}

fn emit_and(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    gpr[rd] = builder.ins().band(gpr[rs], gpr[rt]);
}

fn emit_or(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    gpr[rd] = builder.ins().bor(gpr[rs], gpr[rt]);
}

fn emit_xor(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    gpr[rd] = builder.ins().bxor(gpr[rs], gpr[rt]);
}

fn emit_nor(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    let or_val = builder.ins().bor(gpr[rs], gpr[rt]);
    gpr[rd] = builder.ins().bnot(or_val);
}

fn emit_slt(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    let cmp = builder.ins().icmp(IntCC::SignedLessThan, gpr[rs], gpr[rt]);
    gpr[rd] = builder.ins().uextend(types::I64, cmp);
}

fn emit_sltu(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    let cmp = builder.ins().icmp(IntCC::UnsignedLessThan, gpr[rs], gpr[rt]);
    gpr[rd] = builder.ins().uextend(types::I64, cmp);
}

// ─── 64-bit ALU ops ──────────────────────────────────────────────────────────

fn emit_daddu(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    gpr[rd] = builder.ins().iadd(gpr[rs], gpr[rt]);
}

fn emit_dsubu(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    gpr[rd] = builder.ins().isub(gpr[rs], gpr[rt]);
}

// ─── 32-bit Shift ops ───────────────────────────────────────────────────────

fn emit_sll(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rt: usize, rd: usize, sa: u32) {
    let rt32 = builder.ins().ireduce(types::I32, gpr[rt]);
    let shift = builder.ins().iconst(types::I32, sa as i64);
    let result = builder.ins().ishl(rt32, shift);
    gpr[rd] = builder.ins().sextend(types::I64, result);
}

fn emit_srl(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rt: usize, rd: usize, sa: u32) {
    let rt32 = builder.ins().ireduce(types::I32, gpr[rt]);
    let shift = builder.ins().iconst(types::I32, sa as i64);
    let result = builder.ins().ushr(rt32, shift);
    // SRL: logical shift, but result is still sign-extended to 64 (MIPS spec)
    gpr[rd] = builder.ins().sextend(types::I64, result);
}

fn emit_sra(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rt: usize, rd: usize, sa: u32) {
    let rt32 = builder.ins().ireduce(types::I32, gpr[rt]);
    let shift = builder.ins().iconst(types::I32, sa as i64);
    let result = builder.ins().sshr(rt32, shift);
    gpr[rd] = builder.ins().sextend(types::I64, result);
}

fn emit_sllv(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    let rt32 = builder.ins().ireduce(types::I32, gpr[rt]);
    let rs32 = builder.ins().ireduce(types::I32, gpr[rs]);
    let mask = builder.ins().iconst(types::I32, 0x1F);
    let sa = builder.ins().band(rs32, mask);
    let result = builder.ins().ishl(rt32, sa);
    gpr[rd] = builder.ins().sextend(types::I64, result);
}

fn emit_srlv(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    let rt32 = builder.ins().ireduce(types::I32, gpr[rt]);
    let rs32 = builder.ins().ireduce(types::I32, gpr[rs]);
    let mask = builder.ins().iconst(types::I32, 0x1F);
    let sa = builder.ins().band(rs32, mask);
    let result = builder.ins().ushr(rt32, sa);
    gpr[rd] = builder.ins().sextend(types::I64, result);
}

fn emit_srav(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    let rt32 = builder.ins().ireduce(types::I32, gpr[rt]);
    let rs32 = builder.ins().ireduce(types::I32, gpr[rs]);
    let mask = builder.ins().iconst(types::I32, 0x1F);
    let sa = builder.ins().band(rs32, mask);
    let result = builder.ins().sshr(rt32, sa);
    gpr[rd] = builder.ins().sextend(types::I64, result);
}

// ─── 64-bit Shift ops ───────────────────────────────────────────────────────

fn emit_dsll(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rt: usize, rd: usize, sa: u32) {
    let shift = builder.ins().iconst(types::I64, sa as i64);
    gpr[rd] = builder.ins().ishl(gpr[rt], shift);
}

fn emit_dsrl(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rt: usize, rd: usize, sa: u32) {
    let shift = builder.ins().iconst(types::I64, sa as i64);
    gpr[rd] = builder.ins().ushr(gpr[rt], shift);
}

fn emit_dsra(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rt: usize, rd: usize, sa: u32) {
    let shift = builder.ins().iconst(types::I64, sa as i64);
    gpr[rd] = builder.ins().sshr(gpr[rt], shift);
}

fn emit_dsllv(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    let mask = builder.ins().iconst(types::I64, 0x3F);
    let sa = builder.ins().band(gpr[rs], mask);
    gpr[rd] = builder.ins().ishl(gpr[rt], sa);
}

fn emit_dsrlv(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    let mask = builder.ins().iconst(types::I64, 0x3F);
    let sa = builder.ins().band(gpr[rs], mask);
    gpr[rd] = builder.ins().ushr(gpr[rt], sa);
}

fn emit_dsrav(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    let mask = builder.ins().iconst(types::I64, 0x3F);
    let sa = builder.ins().band(gpr[rs], mask);
    gpr[rd] = builder.ins().sshr(gpr[rt], sa);
}

// ─── Multiply/Divide ─────────────────────────────────────────────────────────

fn emit_mult(builder: &mut FunctionBuilder, gpr: &[Value; 32], hi: &mut Value, lo: &mut Value, rs: usize, rt: usize) {
    // Signed 32×32 → 64-bit result
    let a32 = builder.ins().ireduce(types::I32, gpr[rs]);
    let a = builder.ins().sextend(types::I64, a32);
    let b32 = builder.ins().ireduce(types::I32, gpr[rt]);
    let b = builder.ins().sextend(types::I64, b32);
    let product = builder.ins().imul(a, b);
    // lo = sign-extend low 32 bits; hi = sign-extend high 32 bits
    *lo = sext32(builder, product);
    let shifted = builder.ins().sshr_imm(product, 32);
    *hi = sext32(builder, shifted);
}

fn emit_multu(builder: &mut FunctionBuilder, gpr: &[Value; 32], hi: &mut Value, lo: &mut Value, rs: usize, rt: usize) {
    let a32 = builder.ins().ireduce(types::I32, gpr[rs]);
    let a = builder.ins().uextend(types::I64, a32);
    let b32 = builder.ins().ireduce(types::I32, gpr[rt]);
    let b = builder.ins().uextend(types::I64, b32);
    let product = builder.ins().imul(a, b);
    *lo = sext32(builder, product);
    let shifted = builder.ins().ushr_imm(product, 32);
    *hi = sext32(builder, shifted);
}

fn emit_div(builder: &mut FunctionBuilder, gpr: &[Value; 32], hi: &mut Value, lo: &mut Value, rs: usize, rt: usize) {
    let a = builder.ins().ireduce(types::I32, gpr[rs]);
    let b = builder.ins().ireduce(types::I32, gpr[rt]);
    let zero = builder.ins().iconst(types::I32, 0);
    let one = builder.ins().iconst(types::I32, 1);
    let is_nonzero = builder.ins().icmp(IntCC::NotEqual, b, zero);
    let safe_b = builder.ins().select(is_nonzero, b, one);
    let q = builder.ins().sdiv(a, safe_b);
    let r = builder.ins().srem(a, safe_b);
    *lo = builder.ins().sextend(types::I64, q);
    *hi = builder.ins().sextend(types::I64, r);
}

fn emit_divu(builder: &mut FunctionBuilder, gpr: &[Value; 32], hi: &mut Value, lo: &mut Value, rs: usize, rt: usize) {
    let a = builder.ins().ireduce(types::I32, gpr[rs]);
    let b = builder.ins().ireduce(types::I32, gpr[rt]);
    let zero = builder.ins().iconst(types::I32, 0);
    let one = builder.ins().iconst(types::I32, 1);
    let is_nonzero = builder.ins().icmp(IntCC::NotEqual, b, zero);
    let safe_b = builder.ins().select(is_nonzero, b, one);
    let q = builder.ins().udiv(a, safe_b);
    let r = builder.ins().urem(a, safe_b);
    *lo = builder.ins().sextend(types::I64, q);
    *hi = builder.ins().sextend(types::I64, r);
}

fn emit_dmult(builder: &mut FunctionBuilder, gpr: &[Value; 32], hi: &mut Value, lo: &mut Value, rs: usize, rt: usize) {
    // Signed 64×64: lo = low 64, hi = high 64
    *lo = builder.ins().imul(gpr[rs], gpr[rt]);
    *hi = builder.ins().smulhi(gpr[rs], gpr[rt]);
}

fn emit_dmultu(builder: &mut FunctionBuilder, gpr: &[Value; 32], hi: &mut Value, lo: &mut Value, rs: usize, rt: usize) {
    *lo = builder.ins().imul(gpr[rs], gpr[rt]);
    *hi = builder.ins().umulhi(gpr[rs], gpr[rt]);
}

fn emit_ddiv(builder: &mut FunctionBuilder, gpr: &[Value; 32], hi: &mut Value, lo: &mut Value, rs: usize, rt: usize) {
    let zero = builder.ins().iconst(types::I64, 0);
    let one = builder.ins().iconst(types::I64, 1);
    let is_nonzero = builder.ins().icmp(IntCC::NotEqual, gpr[rt], zero);
    let safe_b = builder.ins().select(is_nonzero, gpr[rt], one);
    *lo = builder.ins().sdiv(gpr[rs], safe_b);
    *hi = builder.ins().srem(gpr[rs], safe_b);
}

fn emit_ddivu(builder: &mut FunctionBuilder, gpr: &[Value; 32], hi: &mut Value, lo: &mut Value, rs: usize, rt: usize) {
    let zero = builder.ins().iconst(types::I64, 0);
    let one = builder.ins().iconst(types::I64, 1);
    let is_nonzero = builder.ins().icmp(IntCC::NotEqual, gpr[rt], zero);
    let safe_b = builder.ins().select(is_nonzero, gpr[rt], one);
    *lo = builder.ins().udiv(gpr[rs], safe_b);
    *hi = builder.ins().urem(gpr[rs], safe_b);
}

// ─── Conditional moves ───────────────────────────────────────────────────────

fn emit_movz(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    let zero = builder.ins().iconst(types::I64, 0);
    let is_zero = builder.ins().icmp(IntCC::Equal, gpr[rt], zero);
    gpr[rd] = builder.ins().select(is_zero, gpr[rs], gpr[rd]);
}

fn emit_movn(builder: &mut FunctionBuilder, gpr: &mut [Value; 32], rs: usize, rt: usize, rd: usize) {
    let zero = builder.ins().iconst(types::I64, 0);
    let is_nonzero = builder.ins().icmp(IntCC::NotEqual, gpr[rt], zero);
    gpr[rd] = builder.ins().select(is_nonzero, gpr[rs], gpr[rd]);
}
