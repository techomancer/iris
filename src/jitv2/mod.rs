//! JIT v2: physical-page PIC region compiler. See `rules/jitv2/jit-v2-design.md`.

pub mod jitv2;
pub mod comp;
pub mod opcode_support;
pub mod analyzer;
pub mod codegen;
pub mod paged_memory;
#[cfg(feature = "j2wp")]
pub mod pcp_dump;
pub mod equiv_test;

#[cfg(not(feature = "j2wp"))]
pub use jitv2::JitEntry;
pub use jitv2::{
    CompileQueue, CompileRequest, JitFn, JitStats, Jitv2, PageSlot, Pfn, PhysicalCodePage,
    ARENA_RESERVE_SIZE, BITMAP_WORDS, CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES, COMPILE_QUEUE_CAPACITY,
    ENTRIES_PER_PAGE, JITV2_INITIAL_PAGE_CAPACITY, PAGE_SIZE,
    min_calls_before_compile, set_min_calls_before_compile,
};
pub use paged_memory::{PagedArenaMemoryProvider, PagedArenaState};
#[cfg(feature = "developer")]
pub use jitv2::{BatchFlushReason, CodeSizeBucket, RejectReason, REJECT_REASON_COUNT};

/// The jitv2 dirty-page probe — see `rules/jitv2/dirty-cache-page-probe.md`.
/// Absent under `tcache`, which closes that blind spot by construction.
#[cfg(not(feature = "tcache"))]
pub use jitv2::{install_jit_page_probe, clear_jit_page_probe, jit_page_has_dirty_lines};

#[cfg(test)]
mod zz_corpus {
    use crate::jitv2::analyzer::Analyzer;
    use crate::jitv2::codegen::Codegen;
    use crate::jitv2::JitFn;
    use crate::jitv2::jitv2::ENTRIES_PER_PAGE;

    /// Compile every corpus page named in IRIS_CORPUS_LIST and report the
    /// total emitted bytes. Same input set for both builds = an apples-to-
    /// apples measure of emitted code volume on real guest code.
    #[test]
    fn zz_corpus_sizes() {
        let list = match std::env::var("IRIS_CORPUS_LIST") { Ok(v)=>v, Err(_)=>return };
        // Match the real emulator: opt_level is a process-wide static that
        // defaults to `none` under `developer`. Production runs `speed`.
        let speed = std::env::var("IRIS_OPT_SPEED").is_ok();
        Codegen::set_opt_level_speed(speed);
        println!("OPTLEVEL speed={}", speed);
        let names = std::fs::read_to_string(&list).expect("list file");
        // A real, hook-installed core for the constants to point at. Leaked
        // deliberately: compiled code bakes its address, so it must outlive
        // every region this test compiles.
        let consts = {
            use crate::mips_core::MipsCore;
            let core: &'static mut MipsCore = Box::leak(Box::new(MipsCore::new()));
            // The hook fields still hold their not-installed panic sentinels,
            // which is fine — nothing here *runs* the compiled code, and the
            // sentinels are real function addresses, so the emitted shape
            // (baked immediate vs load) is identical to production.
            crate::jitv2::codegen::JitConsts {
                core: core::num::NonZeroUsize::new(core as *mut MipsCore as usize),
            }
        };
        // R4400 L1-D: 16 KiB direct-mapped, 32-byte lines; 1 MiB L2 with
        // 128-byte lines. Matches `CpuCache::jit_dc_geometry` for the Indy.
        let geom = crate::mips_cache_v2::JitDcGeometry {
            supported: true,
            line_shift: 5,
            num_lines_mask: (16 * 1024 / 32) - 1,
            data_mask: 16 * 1024 - 1,
            has_l2: true,
            l2_line_shift: 7,
            l2_num_lines_mask: (1024 * 1024 / 128) - 1,
        };
        let mut total: u64 = 0;
        let mut n_ok = 0u64;
        let mut n_decl = 0u64;
        for name in names.lines() {
            let name = name.trim();
            if name.is_empty() { continue; }
            let bytes = match std::fs::read(name) { Ok(b)=>b, Err(_)=>continue };
            if bytes.len() < ENTRIES_PER_PAGE*4 { continue; }
            let mut pw = [0u32; ENTRIES_PER_PAGE];
            for i in 0..ENTRIES_PER_PAGE {
                pw[i] = u32::from_ne_bytes([bytes[i*4],bytes[i*4+1],bytes[i*4+2],bytes[i*4+3]]);
            }
            // offset encoded in filename: ..._off_XXXX.bin
            let off = name.rsplit("_off_").next()
                .and_then(|t| t.strip_suffix(".bin"))
                .and_then(|t| u16::from_str_radix(t,16).ok())
                .unwrap_or(0);
            if off as usize >= ENTRIES_PER_PAGE { continue; }
            let mut an = Analyzer::new();
            let (walked, ok) = an.walk_bounded(&pw, off, 0x8000_0000u32, usize::MAX);
            if !ok { continue; }
            let mut ins = *walked;
            let mut cg = Codegen::new();
            // Stamp the same compile-time constants a live worker would get,
            // so this measures the code the emulator actually runs. Without
            // it `hook_addr` returns `None` and every callee falls back to a
            // register load — which silently made this benchmark blind to the
            // whole constant-baking change.
            cg.jit_consts = consts;
            // Real L1-D geometry, so `emit_inline_mem_guard` actually emits
            // the inline fast path. With the default `unsupported()` the
            // guard is skipped entirely and this benchmark silently measures
            // callout-only code — invisible to any change in the inline path.
            cg.dc_geometry = geom;
            let f: Option<JitFn> = cg.compile_region(&mut ins, off, true, false);
            // `last_code_size` is `developer`-gated, but this test must be
            // runnable WITHOUT `developer`: that feature also flips
            // `opt_level` to `none` and injects a per-instruction
            // `emit_dev_trace_bp` callout, so a developer build measures
            // code that production never emits (71% of all callouts in one
            // measurement were the trace hook alone). Report 0 bytes there
            // rather than refusing to build — the ok/declined counts and
            // `IRIS_JIT_DISASM=1` output are still the useful part.
            #[cfg(feature = "developer")]
            let sz = cg.last_code_size() as u64;
            #[cfg(not(feature = "developer"))]
            let sz = 0u64;
            if f.is_some() { total += sz; n_ok += 1; }
            else { n_decl += 1; }
            std::mem::forget(cg);
        }
        println!("CORPUS ok={} declined={} total_bytes={}", n_ok, n_decl, total);
    }
}

/// Scratch: how does Cranelift lower a constant base address used many times?
/// `IRIS_CONST_MODE` selects the shape. Answers whether a baked constant can
/// ever match `disp(%reg)` density for struct-field access.
#[cfg(test)]
mod zz_constdedup {
    #[test]
    fn zz_const_dedup() {
        use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlagsData};
        use cranelift_codegen::settings::{self, Configurable};
        use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
        use cranelift_codegen::Context;
        let Some(mode) = std::env::var_os("IRIS_CONST_MODE") else { return };
        let mode = mode.to_string_lossy().to_string();

        let mut fb = settings::builder();
        fb.set("opt_level", "speed").unwrap();
        let isa = cranelift_native::builder().unwrap()
            .finish(settings::Flags::new(fb)).unwrap();

        let mut ctx = Context::new();
        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.returns.push(AbiParam::new(types::I64));
        let mut bctx = FunctionBuilderContext::new();
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut bctx);
            let blk = b.create_block();
            b.append_block_params_for_function_params(blk);
            b.switch_to_block(blk);
            b.seal_block(blk);
            const ADDR: i64 = 0x7f_abcd_1234_5678u64 as i64;
            let mut acc = b.ins().iconst(types::I64, 0);
            let shared = b.ins().iconst(types::I64, ADDR);
            let param  = b.block_params(blk)[0];
            for i in 0..40 {
                let off = (i * 8) as i32;
                let v = match mode.as_str() {
                    // baked constant re-materialized per use
                    "iconst"  => { let base = b.ins().iconst(types::I64, ADDR);
                                   b.ins().load(types::I64, MemFlagsData::trusted(), base, off) }
                    // one SSA constant, reused
                    "shared"  => b.ins().load(types::I64, MemFlagsData::trusted(), shared, off),
                    // const + explicit iadd_imm, then load at 0  (struct-ish)
                    "addimm"  => { let p = b.ins().iadd_imm(shared, off as i64);
                                   b.ins().load(types::I64, MemFlagsData::trusted(), p, 0) }
                    // base arrives in a register (today's design)
                    "param"   => b.ins().load(types::I64, MemFlagsData::trusted(), param, off),
                    // const forced through an opaque use first
                    "opaque"  => { let p = b.ins().bor_imm(shared, 0);
                                   b.ins().load(types::I64, MemFlagsData::trusted(), p, off) }
                    // Call through a baked function-pointer constant vs a
                    // pointer loaded from the struct: the case where baking
                    // should actually win (no load, no indirect predictor slot).
                    "callconst" => {
                        let mut sig = cranelift_codegen::ir::Signature::new(
                            isa.default_call_conv());
                        sig.params.push(AbiParam::new(types::I64));
                        sig.returns.push(AbiParam::new(types::I64));
                        let sr = b.import_signature(sig);
                        let callee = b.ins().iconst(types::I64, ADDR);
                        let c = b.ins().call_indirect(sr, callee, &[param]);
                        b.inst_results(c)[0]
                    }
                    "callload" => {
                        let mut sig = cranelift_codegen::ir::Signature::new(
                            isa.default_call_conv());
                        sig.params.push(AbiParam::new(types::I64));
                        sig.returns.push(AbiParam::new(types::I64));
                        let sr = b.import_signature(sig);
                        let callee = b.ins().load(types::I64, MemFlagsData::trusted(), param, off);
                        let c = b.ins().call_indirect(sr, callee, &[param]);
                        b.inst_results(c)[0]
                    }
                    _ => panic!("bad mode"),
                };
                acc = b.ins().iadd(acc, v);
            }
            b.ins().return_(&[acc]);
            b.finalize(isa.frontend_config());
        }
        ctx.set_disasm(true);
        let _ = ctx.compile(&*isa, &mut Default::default()).unwrap();
        let cc = ctx.compiled_code().unwrap();
        let vc = cc.vcode.as_ref().unwrap();
        let movabs = vc.lines().filter(|l| l.contains("movabsq")).count();
        println!("CONSTMODE {:8} movabs={:3} size={}", mode, movabs, cc.code_info().total_size);
        if std::env::var_os("IRIS_CONST_DUMP").is_some() {
            for l in vc.lines().filter(|l| l.contains("call") || l.contains("movabs")).take(8) { println!("  | {}", l); }
        }
    }
}

#[cfg(test)]
mod zz_offsets {
    #[test]
    fn zz_print_offsets() {
        if std::env::var("IRIS_PRINT_OFFSETS").is_err() { return; }
        use crate::mips_core::MipsCore;
        println!("OFF gpr    = {:#x}", std::mem::offset_of!(MipsCore, gpr));
        println!("OFF pc     = {:#x}", std::mem::offset_of!(MipsCore, pc));
        println!("OFF hot    = {:#x}", std::mem::offset_of!(MipsCore, hot));
        println!("OFF fpr    = {:#x}", std::mem::offset_of!(MipsCore, fpr));
        println!("OFF nutlb  = {:#x}", std::mem::offset_of!(MipsCore, nutlb));
        println!("SIZE core  = {:#x}", std::mem::size_of::<MipsCore>());
    }
}
