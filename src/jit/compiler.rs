//! Block compiler: translates MIPS basic blocks to native code via Cranelift.

use cranelift_codegen::ir::{self, types, AbiParam, InstBuilder, MemFlags, Value, FuncRef};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::{self, Context};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, FuncId};

use crate::mips_exec::DecodedInstr;
use crate::mips_isa::*;

use super::cache::{BlockTier, CompiledBlock};
use super::context::{JitContext, EXIT_NORMAL, EXIT_INTERPRET, EXIT_EXCEPTION};
use super::helpers::HelperPtrs;

pub struct BlockCompiler {
    jit_module: JITModule,
    ctx: Context,
    builder_ctx: FunctionBuilderContext,
    func_id_counter: u32,
    // Declared function IDs for memory helpers (registered as imports)
    fn_read_u8: FuncId,
    fn_read_u16: FuncId,
    fn_read_u32: FuncId,
    fn_read_u64: FuncId,
    fn_write_u8: FuncId,
    fn_write_u16: FuncId,
    fn_write_u32: FuncId,
    fn_write_u64: FuncId,
    fn_interp_step: FuncId,
    fn_mfc0: FuncId,
    fn_dmfc0: FuncId,
    fn_mtc0: FuncId,
    fn_dmtc0: FuncId,
}

impl BlockCompiler {
    pub fn new(helpers: &HelperPtrs) -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        flag_builder.set("is_pic", "false").unwrap();

        let isa_builder = cranelift_native::builder().expect("host ISA not supported");
        let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register helper function symbols
        jit_builder.symbol("jit_read_u8", helpers.read_u8);
        jit_builder.symbol("jit_read_u16", helpers.read_u16);
        jit_builder.symbol("jit_read_u32", helpers.read_u32);
        jit_builder.symbol("jit_read_u64", helpers.read_u64);
        jit_builder.symbol("jit_write_u8", helpers.write_u8);
        jit_builder.symbol("jit_write_u16", helpers.write_u16);
        jit_builder.symbol("jit_write_u32", helpers.write_u32);
        jit_builder.symbol("jit_write_u64", helpers.write_u64);
        jit_builder.symbol("jit_interp_step", helpers.interp_step);
        jit_builder.symbol("jit_mfc0", helpers.mfc0);
        jit_builder.symbol("jit_dmfc0", helpers.dmfc0);
        jit_builder.symbol("jit_mtc0", helpers.mtc0);
        jit_builder.symbol("jit_dmtc0", helpers.dmtc0);

        let mut jit_module = JITModule::new(jit_builder);

        // Declare helper function signatures: read(ctx_ptr, exec_ptr, virt_addr) -> u64
        let ptr_type = jit_module.target_config().pointer_type();
        let mut read_sig = jit_module.make_signature();
        read_sig.params.push(AbiParam::new(ptr_type)); // ctx_ptr
        read_sig.params.push(AbiParam::new(ptr_type)); // exec_ptr
        read_sig.params.push(AbiParam::new(types::I64)); // virt_addr
        read_sig.returns.push(AbiParam::new(types::I64)); // value
        // Use the ISA's default calling convention (AppleAarch64 on macOS, SystemV on Linux)

        // write(ctx_ptr, exec_ptr, virt_addr, value) -> u64
        let mut write_sig = jit_module.make_signature();
        write_sig.params.push(AbiParam::new(ptr_type));
        write_sig.params.push(AbiParam::new(ptr_type));
        write_sig.params.push(AbiParam::new(types::I64));
        write_sig.params.push(AbiParam::new(types::I64)); // value
        write_sig.returns.push(AbiParam::new(types::I64));
        // Use default calling convention

        let fn_read_u8  = jit_module.declare_function("jit_read_u8",  Linkage::Import, &read_sig).unwrap();
        let fn_read_u16 = jit_module.declare_function("jit_read_u16", Linkage::Import, &read_sig).unwrap();
        let fn_read_u32 = jit_module.declare_function("jit_read_u32", Linkage::Import, &read_sig).unwrap();
        let fn_read_u64 = jit_module.declare_function("jit_read_u64", Linkage::Import, &read_sig).unwrap();
        let fn_write_u8  = jit_module.declare_function("jit_write_u8",  Linkage::Import, &write_sig).unwrap();
        let fn_write_u16 = jit_module.declare_function("jit_write_u16", Linkage::Import, &write_sig).unwrap();
        let fn_write_u32 = jit_module.declare_function("jit_write_u32", Linkage::Import, &write_sig).unwrap();
        let fn_write_u64 = jit_module.declare_function("jit_write_u64", Linkage::Import, &write_sig).unwrap();

        // interp_step(ctx_ptr, exec_ptr) -> u64
        let mut step_sig = jit_module.make_signature();
        step_sig.params.push(AbiParam::new(ptr_type)); // ctx_ptr
        step_sig.params.push(AbiParam::new(ptr_type)); // exec_ptr
        step_sig.returns.push(AbiParam::new(types::I64));
        let fn_interp_step = jit_module.declare_function("jit_interp_step", Linkage::Import, &step_sig).unwrap();

        // mfc0/dmfc0(ctx_ptr, exec_ptr, rd) -> u64 — same shape as a read
        let fn_mfc0 = jit_module.declare_function("jit_mfc0", Linkage::Import, &read_sig).unwrap();
        let fn_dmfc0 = jit_module.declare_function("jit_dmfc0", Linkage::Import, &read_sig).unwrap();
        // mtc0/dmtc0(ctx_ptr, exec_ptr, rd, value) -> u64 — same shape as a write
        let fn_mtc0 = jit_module.declare_function("jit_mtc0", Linkage::Import, &write_sig).unwrap();
        let fn_dmtc0 = jit_module.declare_function("jit_dmtc0", Linkage::Import, &write_sig).unwrap();

        Self {
            ctx: jit_module.make_context(),
            jit_module,
            builder_ctx: FunctionBuilderContext::new(),
            func_id_counter: 0,
            fn_read_u8, fn_read_u16, fn_read_u32, fn_read_u64,
            fn_write_u8, fn_write_u16, fn_write_u32, fn_write_u64,
            fn_interp_step,
            fn_mfc0, fn_dmfc0,
            fn_mtc0, fn_dmtc0,
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
        tier: BlockTier,
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
        // Use default calling convention (matches extern "C" on host)

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

        // Load executor pointer from JitContext
        let exec_ptr = builder.ins().load(
            ptr_type, mem, ctx_ptr,
            ir::immediates::Offset32::new(JitContext::executor_ptr_offset()),
        );

        // Declare helper function references for this function
        let helpers = EmitHelpers {
            read_u8:  self.jit_module.declare_func_in_func(self.fn_read_u8,  &mut builder.func),
            read_u16: self.jit_module.declare_func_in_func(self.fn_read_u16, &mut builder.func),
            read_u32: self.jit_module.declare_func_in_func(self.fn_read_u32, &mut builder.func),
            read_u64: self.jit_module.declare_func_in_func(self.fn_read_u64, &mut builder.func),
            write_u8:  self.jit_module.declare_func_in_func(self.fn_write_u8,  &mut builder.func),
            write_u16: self.jit_module.declare_func_in_func(self.fn_write_u16, &mut builder.func),
            write_u32: self.jit_module.declare_func_in_func(self.fn_write_u32, &mut builder.func),
            write_u64: self.jit_module.declare_func_in_func(self.fn_write_u64, &mut builder.func),
            interp_step: self.jit_module.declare_func_in_func(self.fn_interp_step, &mut builder.func),
            mfc0:  self.jit_module.declare_func_in_func(self.fn_mfc0,  &mut builder.func),
            dmfc0: self.jit_module.declare_func_in_func(self.fn_dmfc0, &mut builder.func),
            mtc0:  self.jit_module.declare_func_in_func(self.fn_mtc0,  &mut builder.func),
            dmtc0: self.jit_module.declare_func_in_func(self.fn_dmtc0, &mut builder.func),
        };

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

        // Bitmask of GPRs modified so far (bits 1-31); used to flush before helper calls
        let mut modified_gprs: u32 = 0;

        // Emit IR for each instruction
        let mut compiled_count = 0u32;
        let mut branch_exit_pc: Option<Value> = None;

        let mut idx = 0;
        while idx < instrs.len() {
            let (_, d) = &instrs[idx];
            let instr_pc = block_pc.wrapping_add(idx as u64 * 4);
            let result = emit_instruction(
                &mut builder, ctx_ptr, exec_ptr, &helpers,
                &mut gpr, &mut hi, &mut lo, &mut modified_gprs, d, instr_pc, tier,
            );
            match result {
                EmitResult::Ok => { compiled_count += 1; idx += 1; }
                EmitResult::Branch(target_val) => {
                    compiled_count += 1;
                    idx += 1;
                    // Emit the delay slot instruction (next in the list, if present)
                    if idx < instrs.len() {
                        let (_, delay_d) = &instrs[idx];
                        let delay_pc = block_pc.wrapping_add(idx as u64 * 4);
                        let delay_result = emit_instruction(
                            &mut builder, ctx_ptr, exec_ptr, &helpers,
                            &mut gpr, &mut hi, &mut lo, &mut modified_gprs, delay_d, delay_pc, tier,
                        );
                        match delay_result {
                            EmitResult::Ok => { compiled_count += 1; }
                            EmitResult::Stop => {
                                // Delay slot can't be compiled at this tier — interpreter fallback.
                                // Flush all modified GPRs to ctx so interpreter sees current state.
                                flush_modified_gprs(&mut builder, &gpr, ctx_ptr, &mut modified_gprs);
                                builder.ins().store(mem, hi, ctx_ptr,
                                    ir::immediates::Offset32::new(JitContext::hi_offset()));
                                builder.ins().store(mem, lo, ctx_ptr,
                                    ir::immediates::Offset32::new(JitContext::lo_offset()));
                                // Store delay slot PC so interpreter executes the right instruction
                                let delay_pc_val = builder.ins().iconst(types::I64, delay_pc as i64);
                                builder.ins().store(mem, delay_pc_val, ctx_ptr,
                                    ir::immediates::Offset32::new(JitContext::pc_offset()));
                                // Call interpreter: syncs ctx→exec, step(), syncs exec→ctx
                                builder.ins().call(helpers.interp_step, &[ctx_ptr, exec_ptr]);
                                // Reload GPRs from ctx (interpreter may have modified any register)
                                for i in 1..32usize {
                                    gpr[i] = builder.ins().load(
                                        types::I64, mem, ctx_ptr,
                                        ir::immediates::Offset32::new(JitContext::gpr_offset(i)),
                                    );
                                }
                                hi = builder.ins().load(types::I64, mem, ctx_ptr,
                                    ir::immediates::Offset32::new(JitContext::hi_offset()));
                                lo = builder.ins().load(types::I64, mem, ctx_ptr,
                                    ir::immediates::Offset32::new(JitContext::lo_offset()));
                                compiled_count += 1;
                            }
                            _ => {} // Branch in delay slot — shouldn't happen
                        }
                    }
                    branch_exit_pc = Some(target_val);
                    break;
                }
                EmitResult::Stop => break,
            }
        }

        if compiled_count == 0 {
            builder.ins().return_(&[]);
            builder.finalize();
            self.ctx.clear();
            return None;
        }

        // Store all GPRs that may have changed. Use a full bitmask to ensure completeness.
        let mut all_modified: u32 = 0xFFFFFFFE; // bits 1-31 set (skip r0)
        flush_modified_gprs(&mut builder, &gpr, ctx_ptr, &mut all_modified);

        // Store hi/lo back
        builder.ins().store(mem, hi, ctx_ptr,
            ir::immediates::Offset32::new(JitContext::hi_offset()));
        builder.ins().store(mem, lo, ctx_ptr,
            ir::immediates::Offset32::new(JitContext::lo_offset()));

        // Set exit PC
        let exit_pc_val = if let Some(target) = branch_exit_pc {
            target
        } else {
            let fallthrough_pc = block_pc.wrapping_add(compiled_count as u64 * 4);
            builder.ins().iconst(types::I64, fallthrough_pc as i64)
        };
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
            tier,
            // Full-tier blocks contain stores that modify memory. Speculative
            // rollback restores CPU/TLB state but NOT memory, so read-modify-write
            // sequences get double-applied on rollback. Non-speculative blocks skip
            // snapshot/rollback — on exception, the store emitter's flushed GPRs and
            // faulting PC (already in executor via sync_to) are used directly.
            speculative: tier != BlockTier::Full,
            hit_count: 0,
            exception_count: 0,
            stable_hits: 0,
        })
    }
}

/// Helper function references for memory operations within a compiled function.
struct EmitHelpers {
    read_u8: FuncRef, read_u16: FuncRef, read_u32: FuncRef, read_u64: FuncRef,
    write_u8: FuncRef, write_u16: FuncRef, write_u32: FuncRef, write_u64: FuncRef,
    interp_step: FuncRef,
    mfc0: FuncRef, dmfc0: FuncRef,
    mtc0: FuncRef, dmtc0: FuncRef,
}

/// Result of emitting a single instruction.
enum EmitResult {
    /// Instruction compiled normally.
    Ok,
    /// Instruction is a branch; the Value is the computed target PC.
    Branch(Value),
    /// Instruction is not compilable — terminate block before it.
    Stop,
}

/// Emit Cranelift IR for a single MIPS instruction.
fn emit_instruction(
    builder: &mut FunctionBuilder,
    ctx_ptr: Value,
    exec_ptr: Value,
    helpers: &EmitHelpers,
    gpr: &mut [Value; 32],
    hi: &mut Value,
    lo: &mut Value,
    modified_gprs: &mut u32,
    d: &DecodedInstr,
    instr_pc: u64,
    tier: BlockTier,
) -> EmitResult {
    let op = d.op as u32;
    let rs = d.rs as usize;
    let rt = d.rt as usize;
    let rd = d.rd as usize;
    let sa = d.sa as u32;
    let funct = d.funct as u32;

    match op {
        OP_SPECIAL => {
            let result = emit_special(builder, gpr, hi, lo, d, rs, rt, rd, sa, funct, instr_pc);
            // Conservative: mark rd modified for all SPECIAL ops that return Ok.
            // Harmless for ops that don't write rd (JR, MTHI, MTLO) since flush
            // will simply store the still-valid value that was loaded at block entry.
            if matches!(result, EmitResult::Ok) {
                *modified_gprs |= 1u32 << rd;
            }
            result
        }
        OP_ADDIU  => { emit_addiu(builder, gpr, rs, rt, d);  *modified_gprs |= 1 << rt; EmitResult::Ok }
        OP_DADDIU => { emit_daddiu(builder, gpr, rs, rt, d); *modified_gprs |= 1 << rt; EmitResult::Ok }
        OP_SLTI   => { emit_slti(builder, gpr, rs, rt, d);   *modified_gprs |= 1 << rt; EmitResult::Ok }
        OP_SLTIU  => { emit_sltiu(builder, gpr, rs, rt, d);  *modified_gprs |= 1 << rt; EmitResult::Ok }
        OP_ANDI   => { emit_andi(builder, gpr, rs, rt, d);   *modified_gprs |= 1 << rt; EmitResult::Ok }
        OP_ORI    => { emit_ori(builder, gpr, rs, rt, d);    *modified_gprs |= 1 << rt; EmitResult::Ok }
        OP_XORI   => { emit_xori(builder, gpr, rs, rt, d);   *modified_gprs |= 1 << rt; EmitResult::Ok }
        OP_LUI    => { emit_lui(builder, gpr, rt, d);         *modified_gprs |= 1 << rt; EmitResult::Ok }

        // --- Loads (tier-gated) ---
        OP_LB | OP_LBU | OP_LH | OP_LHU | OP_LW | OP_LWU | OP_LD => {
            if tier == BlockTier::Alu { return EmitResult::Stop; }
            match op {
                OP_LB  => emit_load(builder, ctx_ptr, exec_ptr, helpers.read_u8,  gpr, rs, rt, d, LoadWidth::Byte,   true,  instr_pc, modified_gprs),
                OP_LBU => emit_load(builder, ctx_ptr, exec_ptr, helpers.read_u8,  gpr, rs, rt, d, LoadWidth::Byte,   false, instr_pc, modified_gprs),
                OP_LH  => emit_load(builder, ctx_ptr, exec_ptr, helpers.read_u16, gpr, rs, rt, d, LoadWidth::Half,   true,  instr_pc, modified_gprs),
                OP_LHU => emit_load(builder, ctx_ptr, exec_ptr, helpers.read_u16, gpr, rs, rt, d, LoadWidth::Half,   false, instr_pc, modified_gprs),
                OP_LW  => emit_load(builder, ctx_ptr, exec_ptr, helpers.read_u32, gpr, rs, rt, d, LoadWidth::Word,   true,  instr_pc, modified_gprs),
                OP_LWU => emit_load(builder, ctx_ptr, exec_ptr, helpers.read_u32, gpr, rs, rt, d, LoadWidth::Word,   false, instr_pc, modified_gprs),
                OP_LD  => emit_load(builder, ctx_ptr, exec_ptr, helpers.read_u64, gpr, rs, rt, d, LoadWidth::Double, false, instr_pc, modified_gprs),
                _ => unreachable!(),
            }
        }

        // --- Stores (tier-gated) ---
        OP_SB | OP_SH | OP_SW | OP_SD => {
            if tier == BlockTier::Alu || tier == BlockTier::Loads { return EmitResult::Stop; }
            match op {
                OP_SB => emit_store(builder, ctx_ptr, exec_ptr, helpers.write_u8,  gpr, rs, rt, d, instr_pc, modified_gprs),
                OP_SH => emit_store(builder, ctx_ptr, exec_ptr, helpers.write_u16, gpr, rs, rt, d, instr_pc, modified_gprs),
                OP_SW => emit_store(builder, ctx_ptr, exec_ptr, helpers.write_u32, gpr, rs, rt, d, instr_pc, modified_gprs),
                OP_SD => emit_store(builder, ctx_ptr, exec_ptr, helpers.write_u64, gpr, rs, rt, d, instr_pc, modified_gprs),
                _ => unreachable!(),
            }
        }

        // --- Branches ---
        OP_BEQ   => emit_beq(builder, gpr, rs, rt, d, instr_pc, false),
        OP_BNE   => emit_bne(builder, gpr, rs, rt, d, instr_pc, false),
        OP_BLEZ  => emit_blez(builder, gpr, rs, d, instr_pc, false),
        OP_BGTZ  => emit_bgtz(builder, gpr, rs, d, instr_pc, false),

        // --- REGIMM: BLTZ / BGEZ / BLTZAL / BGEZAL ---
        OP_REGIMM => {
            let rt_code = d.rt as u32;
            match rt_code {
                RT_BLTZ => emit_bltz(builder, gpr, rs, d, instr_pc),
                RT_BGEZ => emit_bgez(builder, gpr, rs, d, instr_pc),
                RT_BLTZAL => {
                    // Link address written to $ra ($31) regardless of taken.
                    let link = builder.ins().iconst(types::I64, instr_pc.wrapping_add(8) as i64);
                    gpr[31] = link;
                    *modified_gprs |= 1u32 << 31;
                    emit_bltz(builder, gpr, rs, d, instr_pc)
                }
                RT_BGEZAL => {
                    let link = builder.ins().iconst(types::I64, instr_pc.wrapping_add(8) as i64);
                    gpr[31] = link;
                    *modified_gprs |= 1u32 << 31;
                    emit_bgez(builder, gpr, rs, d, instr_pc)
                }
                _ => EmitResult::Stop,
            }
        }

        // --- Jumps ---
        OP_J   => emit_j(builder, gpr, d, instr_pc),
        OP_JAL => { *modified_gprs |= 1 << 31; emit_jal(builder, gpr, d, instr_pc) }

        // --- COP0: MFC0 / DMFC0 / MTC0 / DMTC0 ---
        // CFC0/CTC0/TLB*/ERET still fall through to Stop.
        OP_COP0 => {
            let sub = rs as u32;  // rs field encodes the COP0 operation
            match sub {
                RS_MFC0 | RS_DMFC0 => {
                    let helper = if sub == RS_MFC0 { helpers.mfc0 } else { helpers.dmfc0 };
                    flush_modified_gprs(builder, gpr, ctx_ptr, modified_gprs);
                    let rd_val = builder.ins().iconst(types::I64, rd as i64);
                    let call = builder.ins().call(helper, &[ctx_ptr, exec_ptr, rd_val]);
                    let result = builder.inst_results(call)[0];
                    gpr[rt] = result;
                    *modified_gprs |= 1u32 << rt;
                    EmitResult::Ok
                }
                RS_MTC0 | RS_DMTC0 => {
                    let helper = if sub == RS_MTC0 { helpers.mtc0 } else { helpers.dmtc0 };
                    // Flush dirty GPRs so write_cp0 side effects (which may re-
                    // derive translation state from full register context) see
                    // a consistent picture.
                    flush_modified_gprs(builder, gpr, ctx_ptr, modified_gprs);
                    let rd_val = builder.ins().iconst(types::I64, rd as i64);
                    let value = gpr[rt];
                    let _ = builder.ins().call(helper, &[ctx_ptr, exec_ptr, rd_val, value]);
                    EmitResult::Ok
                }
                _ => EmitResult::Stop,
            }
        }

        _ => EmitResult::Stop,
    }
}

fn emit_special(
    builder: &mut FunctionBuilder,
    gpr: &mut [Value; 32],
    hi: &mut Value,
    lo: &mut Value,
    d: &DecodedInstr,
    rs: usize, rt: usize, rd: usize, sa: u32, funct: u32,
    instr_pc: u64,
) -> EmitResult {
    match funct {
        // --- Shifts (immediate) ---
        FUNCT_SLL  => { emit_sll(builder, gpr, rt, rd, sa); EmitResult::Ok }
        FUNCT_SRL  => { emit_srl(builder, gpr, rt, rd, sa); EmitResult::Ok }
        FUNCT_SRA  => { emit_sra(builder, gpr, rt, rd, sa); EmitResult::Ok }

        // --- Shifts (variable) ---
        FUNCT_SLLV => { emit_sllv(builder, gpr, rs, rt, rd); EmitResult::Ok }
        FUNCT_SRLV => { emit_srlv(builder, gpr, rs, rt, rd); EmitResult::Ok }
        FUNCT_SRAV => { emit_srav(builder, gpr, rs, rt, rd); EmitResult::Ok }

        // --- 64-bit shifts (immediate) ---
        FUNCT_DSLL   => { emit_dsll(builder, gpr, rt, rd, sa); EmitResult::Ok }
        FUNCT_DSRL   => { emit_dsrl(builder, gpr, rt, rd, sa); EmitResult::Ok }
        FUNCT_DSRA   => { emit_dsra(builder, gpr, rt, rd, sa); EmitResult::Ok }
        FUNCT_DSLL32 => { emit_dsll(builder, gpr, rt, rd, sa + 32); EmitResult::Ok }
        FUNCT_DSRL32 => { emit_dsrl(builder, gpr, rt, rd, sa + 32); EmitResult::Ok }
        FUNCT_DSRA32 => { emit_dsra(builder, gpr, rt, rd, sa + 32); EmitResult::Ok }

        // --- 64-bit shifts (variable) ---
        FUNCT_DSLLV => { emit_dsllv(builder, gpr, rs, rt, rd); EmitResult::Ok }
        FUNCT_DSRLV => { emit_dsrlv(builder, gpr, rs, rt, rd); EmitResult::Ok }
        FUNCT_DSRAV => { emit_dsrav(builder, gpr, rs, rt, rd); EmitResult::Ok }

        // --- ALU register ops ---
        FUNCT_ADDU => { emit_addu(builder, gpr, rs, rt, rd); EmitResult::Ok }
        FUNCT_SUBU => { emit_subu(builder, gpr, rs, rt, rd); EmitResult::Ok }
        FUNCT_AND  => { emit_and(builder, gpr, rs, rt, rd); EmitResult::Ok }
        FUNCT_OR   => { emit_or(builder, gpr, rs, rt, rd); EmitResult::Ok }
        FUNCT_XOR  => { emit_xor(builder, gpr, rs, rt, rd); EmitResult::Ok }
        FUNCT_NOR  => { emit_nor(builder, gpr, rs, rt, rd); EmitResult::Ok }
        FUNCT_SLT  => { emit_slt(builder, gpr, rs, rt, rd); EmitResult::Ok }
        FUNCT_SLTU => { emit_sltu(builder, gpr, rs, rt, rd); EmitResult::Ok }

        // --- 64-bit ALU ---
        FUNCT_DADDU => { emit_daddu(builder, gpr, rs, rt, rd); EmitResult::Ok }
        FUNCT_DSUBU => { emit_dsubu(builder, gpr, rs, rt, rd); EmitResult::Ok }

        // --- Multiply/Divide ---
        FUNCT_MULT  => { emit_mult(builder, gpr, hi, lo, rs, rt); EmitResult::Ok }
        FUNCT_MULTU => { emit_multu(builder, gpr, hi, lo, rs, rt); EmitResult::Ok }
        FUNCT_DIV   => { emit_div(builder, gpr, hi, lo, rs, rt); EmitResult::Ok }
        FUNCT_DIVU  => { emit_divu(builder, gpr, hi, lo, rs, rt); EmitResult::Ok }
        FUNCT_DMULT  => { emit_dmult(builder, gpr, hi, lo, rs, rt); EmitResult::Ok }
        FUNCT_DMULTU => { emit_dmultu(builder, gpr, hi, lo, rs, rt); EmitResult::Ok }
        FUNCT_DDIV   => { emit_ddiv(builder, gpr, hi, lo, rs, rt); EmitResult::Ok }
        FUNCT_DDIVU  => { emit_ddivu(builder, gpr, hi, lo, rs, rt); EmitResult::Ok }

        // --- HI/LO moves ---
        FUNCT_MFHI => { gpr[rd] = *hi; EmitResult::Ok }
        FUNCT_MTHI => { *hi = gpr[rs]; EmitResult::Ok }
        FUNCT_MFLO => { gpr[rd] = *lo; EmitResult::Ok }
        FUNCT_MTLO => { *lo = gpr[rs]; EmitResult::Ok }

        // --- Conditional moves ---
        FUNCT_MOVZ => { emit_movz(builder, gpr, rs, rt, rd); EmitResult::Ok }
        FUNCT_MOVN => { emit_movn(builder, gpr, rs, rt, rd); EmitResult::Ok }

        // --- JR / JALR ---
        FUNCT_JR   => { let target = gpr[rs]; EmitResult::Branch(target) }
        FUNCT_JALR => {
            // JALR rd, rs: jump to gpr[rs], link PC+8 into gpr[rd].
            // rd defaults to $ra ($31) when not specified in assembly.
            let target = gpr[rs];
            let link = builder.ins().iconst(types::I64, instr_pc.wrapping_add(8) as i64);
            gpr[rd] = link;
            EmitResult::Branch(target)
        }

        // --- SYNC (barrier, NOP for JIT) ---
        FUNCT_SYNC => EmitResult::Ok,

        // Everything else terminates the block
        _ => EmitResult::Stop,
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

// ─── GPR flush helper ────────────────────────────────────────────────────────

/// Flush modified GPRs from SSA values to JitContext memory.
/// Called immediately BEFORE each `builder.ins().call(helper, ...)`.
/// After flushing, `*modified` is reset to 0.
/// This eliminates cross-block SSA live value pressure on x86_64 (the "35+ live I64" spill bug).
fn flush_modified_gprs(
    builder: &mut FunctionBuilder,
    gpr: &[Value; 32],
    ctx_ptr: Value,
    modified: &mut u32,
) {
    let mem = MemFlags::trusted();
    for i in 1..32usize {
        if (*modified >> i) & 1 != 0 {
            builder.ins().store(
                mem, gpr[i], ctx_ptr,
                ir::immediates::Offset32::new(JitContext::gpr_offset(i)),
            );
        }
    }
    *modified = 0;
}

// ─── Load/Store emitters ─────────────────────────────────────────────────────

/// Load width tag passed to emit_load so it applies the correct sign extension.
#[derive(Clone, Copy)]
enum LoadWidth { Byte, Half, Word, Double }

/// Emit a load instruction. Calls the helper function, checks for exception,
/// sign/zero-extends the result into the destination GPR.
fn emit_load(
    builder: &mut FunctionBuilder,
    ctx_ptr: Value, exec_ptr: Value,
    helper: FuncRef,
    gpr: &mut [Value; 32],
    rs: usize, rt: usize,
    d: &DecodedInstr,
    width: LoadWidth,
    sign_extend: bool,
    instr_pc: u64,
    modified_gprs: &mut u32,
) -> EmitResult {
    let base = gpr[rs];
    let offset = builder.ins().iconst(types::I64, d.imm as i32 as i64);
    let virt_addr = builder.ins().iadd(base, offset);

    // Flush all GPRs modified so far — prevents cross-block SSA live value pressure
    flush_modified_gprs(builder, gpr, ctx_ptr, modified_gprs);

    // Store faulting PC to ctx BEFORE the helper call, so the dispatch loop
    // knows which instruction caused the exception if one occurs.
    let instr_pc_val = builder.ins().iconst(types::I64, instr_pc as i64);
    builder.ins().store(MemFlags::trusted(), instr_pc_val, ctx_ptr,
        ir::immediates::Offset32::new(JitContext::pc_offset()));

    // Call helper: result = helper(ctx_ptr, exec_ptr, virt_addr)
    let call = builder.ins().call(helper, &[ctx_ptr, exec_ptr, virt_addr]);
    let raw_val = builder.inst_results(call)[0];

    // Check ctx.exit_reason for exception.
    // MUST use MemFlags::new() — helper may have written exit_reason through ctx_ptr.
    let exit_reason = builder.ins().load(types::I32, MemFlags::new(), ctx_ptr,
        ir::immediates::Offset32::new(JitContext::exit_reason_offset()));
    let zero_i32 = builder.ins().iconst(types::I32, 0);
    let is_exception = builder.ins().icmp(IntCC::NotEqual, exit_reason, zero_i32);

    let ok_block = builder.create_block();
    builder.append_block_param(ok_block, types::I64);
    let exc_block = builder.create_block();
    builder.ins().brif(is_exception, exc_block, &[], ok_block, &[raw_val]);

    // Exception path: GPRs already flushed before the helper call — just return
    builder.switch_to_block(exc_block);
    builder.seal_block(exc_block);
    builder.ins().return_(&[]);

    // Normal path — raw_val comes through as a block parameter
    builder.switch_to_block(ok_block);
    builder.seal_block(ok_block);
    let val = builder.block_params(ok_block)[0];

    // Apply correct sign/zero extension based on load width
    gpr[rt] = match (width, sign_extend) {
        (LoadWidth::Byte, true) => {
            // i8 → i64: truncate to 8 bits, sign-extend
            let narrow = builder.ins().ireduce(types::I8, val);
            builder.ins().sextend(types::I64, narrow)
        }
        (LoadWidth::Half, true) => {
            // i16 → i64: truncate to 16 bits, sign-extend
            let narrow = builder.ins().ireduce(types::I16, val);
            builder.ins().sextend(types::I64, narrow)
        }
        (LoadWidth::Word, true) => {
            // i32 → i64: truncate to 32 bits, sign-extend
            sext32(builder, val)
        }
        (_, false) | (LoadWidth::Double, _) => {
            // Zero-extend or 64-bit: raw value is already correct
            val
        }
    };
    *modified_gprs |= 1u32 << rt;

    EmitResult::Ok
}

/// Emit a store instruction. Calls the helper function, checks for exception.
fn emit_store(
    builder: &mut FunctionBuilder,
    ctx_ptr: Value, exec_ptr: Value,
    helper: FuncRef,
    gpr: &[Value; 32],
    rs: usize, rt: usize,
    d: &DecodedInstr,
    instr_pc: u64,
    modified_gprs: &mut u32,
) -> EmitResult {
    let base = gpr[rs];
    let offset = builder.ins().iconst(types::I64, d.imm as i32 as i64);
    let virt_addr = builder.ins().iadd(base, offset);
    let value = gpr[rt];

    // Flush all GPRs modified so far — prevents cross-block SSA live value pressure
    flush_modified_gprs(builder, gpr, ctx_ptr, modified_gprs);

    // Store faulting PC before helper call
    let instr_pc_val = builder.ins().iconst(types::I64, instr_pc as i64);
    builder.ins().store(MemFlags::trusted(), instr_pc_val, ctx_ptr,
        ir::immediates::Offset32::new(JitContext::pc_offset()));

    let _call = builder.ins().call(helper, &[ctx_ptr, exec_ptr, virt_addr, value]);

    // Check ctx.exit_reason — MUST use MemFlags::new()
    let exit_reason = builder.ins().load(types::I32, MemFlags::new(), ctx_ptr,
        ir::immediates::Offset32::new(JitContext::exit_reason_offset()));
    let zero = builder.ins().iconst(types::I32, 0);
    let is_exception = builder.ins().icmp(IntCC::NotEqual, exit_reason, zero);

    let ok_block = builder.create_block();
    let exc_block = builder.create_block();
    builder.ins().brif(is_exception, exc_block, &[], ok_block, &[]);

    // Exception path: GPRs already flushed before the helper call — just return
    builder.switch_to_block(exc_block);
    builder.seal_block(exc_block);
    builder.ins().return_(&[]);

    builder.switch_to_block(ok_block);
    builder.seal_block(ok_block);

    EmitResult::Ok
}

// ─── Branch emitters ─────────────────────────────────────────────────────────
// Branches compute the target PC and return EmitResult::Branch(target_value).
// The compiled block stores this PC and returns. Delay slots are handled by
// the dispatch loop (the next instruction after the branch is interpreted).

fn emit_beq(
    builder: &mut FunctionBuilder, gpr: &[Value; 32],
    rs: usize, rt: usize, d: &DecodedInstr, instr_pc: u64, _likely: bool,
) -> EmitResult {
    let taken_pc = instr_pc.wrapping_add(4).wrapping_add(d.imm as i32 as i64 as u64);
    let not_taken_pc = instr_pc.wrapping_add(8); // skip delay slot
    let taken = builder.ins().iconst(types::I64, taken_pc as i64);
    let not_taken = builder.ins().iconst(types::I64, not_taken_pc as i64);
    let cond = builder.ins().icmp(IntCC::Equal, gpr[rs], gpr[rt]);
    let target = builder.ins().select(cond, taken, not_taken);
    EmitResult::Branch(target)
}

fn emit_bne(
    builder: &mut FunctionBuilder, gpr: &[Value; 32],
    rs: usize, rt: usize, d: &DecodedInstr, instr_pc: u64, _likely: bool,
) -> EmitResult {
    let taken_pc = instr_pc.wrapping_add(4).wrapping_add(d.imm as i32 as i64 as u64);
    let not_taken_pc = instr_pc.wrapping_add(8);
    let taken = builder.ins().iconst(types::I64, taken_pc as i64);
    let not_taken = builder.ins().iconst(types::I64, not_taken_pc as i64);
    let cond = builder.ins().icmp(IntCC::NotEqual, gpr[rs], gpr[rt]);
    let target = builder.ins().select(cond, taken, not_taken);
    EmitResult::Branch(target)
}

fn emit_blez(
    builder: &mut FunctionBuilder, gpr: &[Value; 32],
    rs: usize, d: &DecodedInstr, instr_pc: u64, _likely: bool,
) -> EmitResult {
    let taken_pc = instr_pc.wrapping_add(4).wrapping_add(d.imm as i32 as i64 as u64);
    let not_taken_pc = instr_pc.wrapping_add(8);
    let taken = builder.ins().iconst(types::I64, taken_pc as i64);
    let not_taken = builder.ins().iconst(types::I64, not_taken_pc as i64);
    let zero = builder.ins().iconst(types::I64, 0);
    let cond = builder.ins().icmp(IntCC::SignedLessThanOrEqual, gpr[rs], zero);
    let target = builder.ins().select(cond, taken, not_taken);
    EmitResult::Branch(target)
}

fn emit_bgtz(
    builder: &mut FunctionBuilder, gpr: &[Value; 32],
    rs: usize, d: &DecodedInstr, instr_pc: u64, _likely: bool,
) -> EmitResult {
    let taken_pc = instr_pc.wrapping_add(4).wrapping_add(d.imm as i32 as i64 as u64);
    let not_taken_pc = instr_pc.wrapping_add(8);
    let taken = builder.ins().iconst(types::I64, taken_pc as i64);
    let not_taken = builder.ins().iconst(types::I64, not_taken_pc as i64);
    let zero = builder.ins().iconst(types::I64, 0);
    let cond = builder.ins().icmp(IntCC::SignedGreaterThan, gpr[rs], zero);
    let target = builder.ins().select(cond, taken, not_taken);
    EmitResult::Branch(target)
}

fn emit_bltz(
    builder: &mut FunctionBuilder, gpr: &[Value; 32],
    rs: usize, d: &DecodedInstr, instr_pc: u64,
) -> EmitResult {
    let taken_pc = instr_pc.wrapping_add(4).wrapping_add(d.imm as i32 as i64 as u64);
    let not_taken_pc = instr_pc.wrapping_add(8);
    let taken = builder.ins().iconst(types::I64, taken_pc as i64);
    let not_taken = builder.ins().iconst(types::I64, not_taken_pc as i64);
    let zero = builder.ins().iconst(types::I64, 0);
    let cond = builder.ins().icmp(IntCC::SignedLessThan, gpr[rs], zero);
    let target = builder.ins().select(cond, taken, not_taken);
    EmitResult::Branch(target)
}

fn emit_bgez(
    builder: &mut FunctionBuilder, gpr: &[Value; 32],
    rs: usize, d: &DecodedInstr, instr_pc: u64,
) -> EmitResult {
    let taken_pc = instr_pc.wrapping_add(4).wrapping_add(d.imm as i32 as i64 as u64);
    let not_taken_pc = instr_pc.wrapping_add(8);
    let taken = builder.ins().iconst(types::I64, taken_pc as i64);
    let not_taken = builder.ins().iconst(types::I64, not_taken_pc as i64);
    let zero = builder.ins().iconst(types::I64, 0);
    let cond = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, gpr[rs], zero);
    let target = builder.ins().select(cond, taken, not_taken);
    EmitResult::Branch(target)
}

fn emit_j(
    builder: &mut FunctionBuilder, _gpr: &[Value; 32],
    d: &DecodedInstr, instr_pc: u64,
) -> EmitResult {
    // Target = (PC+4)[63:28] | (target26 << 2) — but imm already has target26<<2 from decode
    let region = instr_pc.wrapping_add(4) & 0xFFFF_FFFF_F000_0000;
    let target_pc = region | (d.imm as u64);
    let target = builder.ins().iconst(types::I64, target_pc as i64);
    EmitResult::Branch(target)
}

fn emit_jal(
    builder: &mut FunctionBuilder, gpr: &mut [Value; 32],
    d: &DecodedInstr, instr_pc: u64,
) -> EmitResult {
    // JAL: $ra = PC + 8 (return address past delay slot)
    let return_addr = instr_pc.wrapping_add(8);
    gpr[31] = builder.ins().iconst(types::I64, return_addr as i64);

    let region = instr_pc.wrapping_add(4) & 0xFFFF_FFFF_F000_0000;
    let target_pc = region | (d.imm as u64);
    let target = builder.ins().iconst(types::I64, target_pc as i64);
    EmitResult::Branch(target)
}
