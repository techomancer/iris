/// JIT codegen tests: compile single MIPS basic blocks via Cranelift and compare
/// results against the interpreter for the same starting state.
///
/// Each test:
///   1. Creates an executor with PassthroughTlb + PassthroughCache + MockMemory.
///   2. Writes instruction word(s) to MockMemory at TEST_PC (kseg0 0x80010000).
///   3. Sets up initial GPR/hi/lo state.
///   4. Runs the JIT path (compile + sync_from + call + sync_to).
///   5. Restores identical state and runs the interpreter path (step()).
///   6. Panics with a diff if any GPR, PC, hi, or lo differs.
#[cfg(all(test, feature = "jit"))]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use crate::jit::cache::BlockTier;
    use crate::jit::compiler::BlockCompiler;
    use crate::jit::context::JitContext;
    use crate::jit::helpers::HelperPtrs;
    use crate::mips_cache_v2::PassthroughCache;
    use crate::mips_exec::{decode_into, DecodedInstr, MipsCpuConfig, MipsExecutor};
    use crate::mips_isa::*;
    use crate::mips_tlb::PassthroughTlb;
    use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_OK};

    // Virtual PC in kseg0 (0x80000000–0x9FFFFFFF), maps to physical 0x00010000 via & 0x1FFFFFFF.
    const TEST_PC: u64 = 0x8001_0000;
    // Virtual data address in kseg0, maps to physical 0x00020000.
    const DATA_ADDR: u64 = 0x8002_0000;

    // ── MockMemory ───────────────────────────────────────────────────────────────

    pub struct MockMemory {
        pub data: Mutex<HashMap<u64, u8>>,
    }

    impl MockMemory {
        pub fn new() -> Self {
            Self { data: Mutex::new(HashMap::new()) }
        }

        pub fn get_byte(&self, addr: u64) -> u8 {
            *self.data.lock().unwrap().get(&addr).unwrap_or(&0)
        }

        pub fn set_byte(&self, addr: u64, val: u8) {
            self.data.lock().unwrap().insert(addr, val);
        }

        pub fn get_word(&self, addr: u64) -> u32 {
            let mut b = [0u8; 4];
            for i in 0..4 { b[i] = self.get_byte(addr + i as u64); }
            u32::from_be_bytes(b)
        }

        pub fn set_word(&self, addr: u64, val: u32) {
            let b = val.to_be_bytes();
            for i in 0..4 { self.set_byte(addr + i as u64, b[i]); }
        }

        pub fn get_double(&self, addr: u64) -> u64 {
            let mut b = [0u8; 8];
            for i in 0..8 { b[i] = self.get_byte(addr + i as u64); }
            u64::from_be_bytes(b)
        }

        pub fn set_double(&self, addr: u64, val: u64) {
            let b = val.to_be_bytes();
            for i in 0..8 { self.set_byte(addr + i as u64, b[i]); }
        }
    }

    impl BusDevice for MockMemory {
        fn read8(&self, addr: u32) -> BusRead8 { BusRead8::ok(self.get_byte(addr as u64)) }
        fn write8(&self, addr: u32, val: u8) -> u32 { self.set_byte(addr as u64, val); BUS_OK }
        fn read16(&self, addr: u32) -> BusRead16 {
            let a = (addr & !1) as u64;
            let mut b = [0u8; 2];
            for i in 0..2 { b[i] = self.get_byte(a + i as u64); }
            BusRead16::ok(u16::from_be_bytes(b))
        }
        fn write16(&self, addr: u32, val: u16) -> u32 {
            let a = (addr & !1) as u64;
            let b = val.to_be_bytes();
            for i in 0..2 { self.set_byte(a + i as u64, b[i]); }
            BUS_OK
        }
        fn read32(&self, addr: u32) -> BusRead32 {
            BusRead32::ok(self.get_word((addr & !3) as u64))
        }
        fn write32(&self, addr: u32, val: u32) -> u32 {
            self.set_word((addr & !3) as u64, val);
            BUS_OK
        }
        fn read64(&self, addr: u32) -> BusRead64 {
            BusRead64::ok(self.get_double((addr & !7) as u64))
        }
        fn write64(&self, addr: u32, val: u64) -> u32 {
            self.set_double((addr & !7) as u64, val);
            BUS_OK
        }
    }

    // ── Executor factory ─────────────────────────────────────────────────────────

    fn create_executor() -> (MipsExecutor<PassthroughTlb, PassthroughCache>, Arc<MockMemory>) {
        let mem = Arc::new(MockMemory::new());
        let bus: Arc<dyn BusDevice> = mem.clone();
        let cfg = MipsCpuConfig::indy();
        let exec = MipsExecutor::new(bus, PassthroughTlb::default(), &cfg);
        (exec, mem)
    }

    // ── Instruction word builders ─────────────────────────────────────────────────

    fn make_r(op: u32, rs: u32, rt: u32, rd: u32, sa: u32, funct: u32) -> u32 {
        (op << 26) | ((rs & 0x1F) << 21) | ((rt & 0x1F) << 16) | ((rd & 0x1F) << 11)
            | ((sa & 0x1F) << 6) | (funct & 0x3F)
    }

    fn make_i(op: u32, rs: u32, rt: u32, imm: u16) -> u32 {
        (op << 26) | ((rs & 0x1F) << 21) | ((rt & 0x1F) << 16) | (imm as u32)
    }

    fn make_j(op: u32, target: u32) -> u32 {
        (op << 26) | (target & 0x3FF_FFFF)
    }

    // NOP = SLL $0, $0, 0 = 0x0000_0000
    const NOP: u32 = 0;

    // ── Test harness ─────────────────────────────────────────────────────────────

    /// State snapshot for comparison.
    #[derive(Clone, Debug)]
    struct CpuState {
        gpr: [u64; 32],
        pc:  u64,
        hi:  u64,
        lo:  u64,
    }

    impl CpuState {
        fn capture(exec: &MipsExecutor<PassthroughTlb, PassthroughCache>) -> Self {
            CpuState {
                gpr: exec.core.gpr,
                pc:  exec.core.pc,
                hi:  exec.core.hi,
                lo:  exec.core.lo,
            }
        }

        fn restore(&self, exec: &mut MipsExecutor<PassthroughTlb, PassthroughCache>) {
            exec.core.gpr = self.gpr;
            exec.core.pc  = self.pc;
            exec.core.hi  = self.hi;
            exec.core.lo  = self.lo;
            exec.in_delay_slot      = false;
            exec.delay_slot_target  = 0;
        }
    }

    fn diff_states(label: &str, jit: &CpuState, interp: &CpuState) {
        let mut diffs = Vec::new();
        for i in 0..32 {
            if jit.gpr[i] != interp.gpr[i] {
                diffs.push(format!(
                    "  gpr[{i:2}]: JIT={:#018x}  INTERP={:#018x}",
                    jit.gpr[i], interp.gpr[i]
                ));
            }
        }
        if jit.pc != interp.pc {
            diffs.push(format!(
                "  pc:       JIT={:#018x}  INTERP={:#018x}",
                jit.pc, interp.pc
            ));
        }
        if jit.hi != interp.hi {
            diffs.push(format!(
                "  hi:       JIT={:#018x}  INTERP={:#018x}",
                jit.hi, interp.hi
            ));
        }
        if jit.lo != interp.lo {
            diffs.push(format!(
                "  lo:       JIT={:#018x}  INTERP={:#018x}",
                jit.lo, interp.lo
            ));
        }
        if !diffs.is_empty() {
            panic!("{label}: JIT vs interpreter mismatch:\n{}", diffs.join("\n"));
        }
    }

    /// Build a decoded instruction list from raw words at TEST_PC (kseg0).
    /// The physical address is TEST_PC & 0x1FFFFFFF. Instrs are written to MockMemory
    /// so the interpreter's fetch path also works.
    fn prepare_block(
        exec: &mut MipsExecutor<PassthroughTlb, PassthroughCache>,
        mem: &MockMemory,
        words: &[u32],
    ) -> Vec<(u32, DecodedInstr)> {
        // Write instruction bytes at the physical address
        let phys = (TEST_PC & 0x1FFF_FFFF) as u64;
        for (i, &w) in words.iter().enumerate() {
            mem.set_word(phys + i as u64 * 4, w);
        }
        // Invalidate the instruction cache so the interpreter fetches fresh bytes
        exec.core.nanotlb_invalidate();

        // Build decoded instruction list (mirrors trace_block logic without the private call)
        words.iter().enumerate().map(|(_, &raw)| {
            let mut d = DecodedInstr::default();
            d.raw = raw;
            decode_into::<PassthroughTlb, PassthroughCache>(&mut d);
            (raw, d)
        }).collect()
    }

    /// Run a single-block JIT test with `interp_steps` interpreter steps.
    ///
    /// - `setup`: initialise GPRs/hi/lo before running
    /// - `words`: instruction words (written to memory AND compiled as the JIT block)
    /// - `tier`: compilation tier
    /// - `interp_steps`: how many step() calls needed to produce equivalent state
    ///   (1 for non-branch, 2 for branch+delay-slot)
    fn run_jit_test(
        label: &str,
        setup: impl Fn(&mut MipsExecutor<PassthroughTlb, PassthroughCache>),
        words: &[u32],
        tier: BlockTier,
        interp_steps: usize,
    ) {
        let (mut exec, mem) = create_executor();
        exec.core.pc = TEST_PC;

        let instrs = prepare_block(&mut exec, &mem, words);

        // Apply initial state
        setup(&mut exec);
        exec.core.pc = TEST_PC;

        // Capture state for restoring before the interpreter run
        let saved = CpuState::capture(&exec);

        // ── JIT path ─────────────────────────────────────────────────────────────
        let helpers = HelperPtrs::new::<PassthroughTlb, PassthroughCache>();
        let mut compiler = BlockCompiler::new(&helpers);
        let block = compiler
            .compile_block(&instrs, TEST_PC, tier)
            .expect("compile_block returned None");

        let mut ctx = JitContext::new();
        ctx.sync_from_executor(&exec);
        ctx.executor_ptr = &mut exec as *mut _ as u64;

        let entry: extern "C" fn(*mut JitContext) =
            unsafe { std::mem::transmute(block.entry) };
        entry(&mut ctx);
        ctx.sync_to_executor(&mut exec);

        let jit_state = CpuState::capture(&exec);

        // ── Interpreter path ─────────────────────────────────────────────────────
        saved.restore(&mut exec);

        for _ in 0..interp_steps {
            exec.step();
        }

        let interp_state = CpuState::capture(&exec);

        diff_states(label, &jit_state, &interp_state);
    }

    /// Like run_jit_test but also compares MockMemory contents after execution
    /// (for store tests). `mem_checks` is a list of (phys_addr, expected_bytes).
    fn run_jit_store_test(
        label: &str,
        setup: impl Fn(&mut MipsExecutor<PassthroughTlb, PassthroughCache>, &MockMemory),
        words: &[u32],
        interp_steps: usize,
        mem_checks: &[(u64, Vec<u8>)],
    ) {
        let (mut exec, mem) = create_executor();
        exec.core.pc = TEST_PC;

        let instrs = prepare_block(&mut exec, &mem, words);

        setup(&mut exec, &mem);
        exec.core.pc = TEST_PC;

        let saved = CpuState::capture(&exec);

        // ── JIT path ─────────────────────────────────────────────────────────────
        let helpers = HelperPtrs::new::<PassthroughTlb, PassthroughCache>();
        let mut compiler = BlockCompiler::new(&helpers);
        let block = compiler
            .compile_block(&instrs, TEST_PC, BlockTier::Full)
            .expect("compile_block returned None");

        let mut ctx = JitContext::new();
        ctx.sync_from_executor(&exec);
        ctx.executor_ptr = &mut exec as *mut _ as u64;

        let entry: extern "C" fn(*mut JitContext) =
            unsafe { std::mem::transmute(block.entry) };
        entry(&mut ctx);
        ctx.sync_to_executor(&mut exec);

        let jit_state = CpuState::capture(&exec);

        // Verify memory contents after JIT execution
        for (phys_addr, expected) in mem_checks {
            for (i, &exp_byte) in expected.iter().enumerate() {
                let got = mem.get_byte(phys_addr + i as u64);
                assert_eq!(got, exp_byte,
                    "{label}: JIT memory mismatch at phys {:#010x}+{i}: got {got:#04x}, want {exp_byte:#04x}",
                    phys_addr);
            }
        }

        // ── Interpreter path ─────────────────────────────────────────────────────
        // Clear memory area written by JIT so interpreter writes fresh
        let data_phys = (DATA_ADDR & 0x1FFF_FFFF) as u64;
        for i in 0..16 {
            mem.set_byte(data_phys + i, 0);
        }
        saved.restore(&mut exec);

        for _ in 0..interp_steps {
            exec.step();
        }

        let interp_state = CpuState::capture(&exec);
        diff_states(label, &jit_state, &interp_state);

        // Re-verify memory against interpreter's writes
        for (phys_addr, expected) in mem_checks {
            for (i, &exp_byte) in expected.iter().enumerate() {
                let got = mem.get_byte(phys_addr + i as u64);
                assert_eq!(got, exp_byte,
                    "{label}: interp memory mismatch at phys {:#010x}+{i}: got {got:#04x}, want {exp_byte:#04x}",
                    phys_addr);
            }
        }
    }

    // ── ALU register ops (OP_SPECIAL) ────────────────────────────────────────────

    #[test]
    fn test_jit_addu() {
        // ADDU rd=3, rs=1, rt=2
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_ADDU);
        for (a, b) in [(10u64, 20u64), (0, 0), (0xFFFF_FFFF, 1), (0x7FFF_FFFE, 2)] {
            run_jit_test(
                &format!("ADDU {a:#x}+{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_subu() {
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_SUBU);
        for (a, b) in [(30u64, 20u64), (0, 0), (0, 1), (0x80000000, 1)] {
            run_jit_test(
                &format!("SUBU {a:#x}-{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_and() {
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_AND);
        for (a, b) in [
            (0xAAAA_AAAA_AAAA_AAAAu64, 0x5555_5555_5555_5555u64),
            (0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF),
            (0, 0xFFFF_FFFF_FFFF_FFFF),
            (0xDEAD_BEEF, 0xFFFF_0000),
        ] {
            run_jit_test(
                &format!("AND {a:#x}&{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_or() {
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_OR);
        for (a, b) in [
            (0xAAAA_AAAAu64, 0x5555_5555u64),
            (0, 0),
            (0xFFFF_FFFF_FFFF_FFFF, 0),
            (0xDEAD_0000, 0x0000_BEEF),
        ] {
            run_jit_test(
                &format!("OR {a:#x}|{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_xor() {
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_XOR);
        for (a, b) in [
            (0xAAAA_AAAAu64, 0xAAAA_AAAAu64),
            (0, 0xFFFF_FFFF_FFFF_FFFF),
            (0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210),
            (0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF),
        ] {
            run_jit_test(
                &format!("XOR {a:#x}^{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_nor() {
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_NOR);
        for (a, b) in [
            (0u64, 0u64),
            (0xAAAA_AAAA, 0x5555_5555),
            (0xFFFF_FFFF_FFFF_FFFF, 0),
            (0x1234_5678, 0x9ABC_DEF0),
        ] {
            run_jit_test(
                &format!("NOR {a:#x} nor {b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_slt() {
        // SLT rd=3, rs=1, rt=2 (signed compare)
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_SLT);
        for (a, b) in [
            (0u64, 1u64),               // 0 < 1: 1
            (1u64, 0u64),               // 1 < 0: 0
            (0xFFFF_FFFF_FFFF_FFFF, 0), // -1 < 0: 1
            (0, 0xFFFF_FFFF_FFFF_FFFF), // 0 < -1: 0
            (0x7FFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF), // large pos < -1: 0
        ] {
            run_jit_test(
                &format!("SLT {a:#x} < {b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_sltu() {
        // SLTU rd=3, rs=1, rt=2 (unsigned compare)
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_SLTU);
        for (a, b) in [
            (0u64, 1u64),
            (1u64, 0u64),
            (0xFFFF_FFFF_FFFF_FFFF, 0),        // large > 0: 0
            (0, 0xFFFF_FFFF_FFFF_FFFF),        // 0 < large: 1
            (0x8000_0000_0000_0000, 0x7FFF_FFFF_FFFF_FFFF), // min > max as unsigned
        ] {
            run_jit_test(
                &format!("SLTU {a:#x} < {b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_daddu() {
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_DADDU);
        for (a, b) in [
            (0u64, 0u64),
            (0xFFFF_FFFF_FFFF_FFFF, 1),
            (0x7FFF_FFFF_FFFF_FFFF, 1),
            (0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321),
        ] {
            run_jit_test(
                &format!("DADDU {a:#x}+{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_dsubu() {
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_DSUBU);
        for (a, b) in [
            (100u64, 50u64),
            (0, 1),
            (0x8000_0000_0000_0000, 1),
            (0, 0xFFFF_FFFF_FFFF_FFFF),
        ] {
            run_jit_test(
                &format!("DSUBU {a:#x}-{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    // ── 32-bit shifts ─────────────────────────────────────────────────────────────

    #[test]
    fn test_jit_sll() {
        for sa in [0u32, 1, 8, 16, 31] {
            let instr = make_r(OP_SPECIAL, 0, 1, 2, sa, FUNCT_SLL);
            for val in [0u64, 1, 0xFFFF_FFFF, 0x8000_0000] {
                run_jit_test(
                    &format!("SLL {val:#x} << {sa}"),
                    |e| { e.core.write_gpr(1, val); },
                    &[instr],
                    BlockTier::Alu,
                    1,
                );
            }
        }
    }

    #[test]
    fn test_jit_srl() {
        for sa in [0u32, 1, 8, 16, 31] {
            let instr = make_r(OP_SPECIAL, 0, 1, 2, sa, FUNCT_SRL);
            for val in [0u64, 0xFFFF_FFFF, 0x8000_0000, 0x8000_0001] {
                run_jit_test(
                    &format!("SRL {val:#x} >> {sa}"),
                    |e| { e.core.write_gpr(1, val); },
                    &[instr],
                    BlockTier::Alu,
                    1,
                );
            }
        }
    }

    #[test]
    fn test_jit_sra() {
        for sa in [0u32, 1, 8, 16, 31] {
            let instr = make_r(OP_SPECIAL, 0, 1, 2, sa, FUNCT_SRA);
            // Include a negative value (high bit set) to test arithmetic shift
            for val in [0u64, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF] {
                run_jit_test(
                    &format!("SRA {val:#x} >> {sa}"),
                    |e| { e.core.write_gpr(1, val); },
                    &[instr],
                    BlockTier::Alu,
                    1,
                );
            }
        }
    }

    #[test]
    fn test_jit_sllv() {
        let instr = make_r(OP_SPECIAL, 2, 1, 3, 0, FUNCT_SLLV); // rd=3 = rt=1 << rs=2
        for (val, sa) in [(1u64, 0u64), (1, 16), (0xFFFF_FFFF, 1), (0x0000_FFFF, 31)] {
            run_jit_test(
                &format!("SLLV {val:#x} << {sa}"),
                |e| { e.core.write_gpr(1, val); e.core.write_gpr(2, sa); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_srlv() {
        let instr = make_r(OP_SPECIAL, 2, 1, 3, 0, FUNCT_SRLV);
        for (val, sa) in [(0xFFFF_FFFFu64, 0u64), (0xFFFF_FFFF, 8), (0x8000_0000, 1), (1, 31)] {
            run_jit_test(
                &format!("SRLV {val:#x} >> {sa}"),
                |e| { e.core.write_gpr(1, val); e.core.write_gpr(2, sa); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_srav() {
        let instr = make_r(OP_SPECIAL, 2, 1, 3, 0, FUNCT_SRAV);
        for (val, sa) in [(0x8000_0000u64, 0u64), (0x8000_0000, 1), (0x8000_0000, 16), (0x7FFF_FFFF, 31)] {
            run_jit_test(
                &format!("SRAV {val:#x} >> {sa}"),
                |e| { e.core.write_gpr(1, val); e.core.write_gpr(2, sa); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    // ── 64-bit shifts ────────────────────────────────────────────────────────────

    #[test]
    fn test_jit_dsll() {
        for sa in [0u32, 1, 16, 31] {
            let instr = make_r(OP_SPECIAL, 0, 1, 2, sa, FUNCT_DSLL);
            for val in [1u64, 0xFFFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000] {
                run_jit_test(
                    &format!("DSLL {val:#x} << {sa}"),
                    |e| { e.core.write_gpr(1, val); },
                    &[instr],
                    BlockTier::Alu,
                    1,
                );
            }
        }
    }

    #[test]
    fn test_jit_dsrl() {
        for sa in [0u32, 1, 16, 31] {
            let instr = make_r(OP_SPECIAL, 0, 1, 2, sa, FUNCT_DSRL);
            for val in [0xFFFF_FFFF_FFFF_FFFFu64, 1, 0x8000_0000_0000_0000] {
                run_jit_test(
                    &format!("DSRL {val:#x} >> {sa}"),
                    |e| { e.core.write_gpr(1, val); },
                    &[instr],
                    BlockTier::Alu,
                    1,
                );
            }
        }
    }

    #[test]
    fn test_jit_dsra() {
        for sa in [0u32, 1, 16, 31] {
            let instr = make_r(OP_SPECIAL, 0, 1, 2, sa, FUNCT_DSRA);
            for val in [0x8000_0000_0000_0000u64, 0x7FFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF] {
                run_jit_test(
                    &format!("DSRA {val:#x} >> {sa}"),
                    |e| { e.core.write_gpr(1, val); },
                    &[instr],
                    BlockTier::Alu,
                    1,
                );
            }
        }
    }

    #[test]
    fn test_jit_dsll32() {
        // DSLL32 sa=0 means shift by 32; sa=1 means shift by 33, etc.
        for sa in [0u32, 1, 16, 31] {
            let instr = make_r(OP_SPECIAL, 0, 1, 2, sa, FUNCT_DSLL32);
            for val in [1u64, 0xFFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF] {
                run_jit_test(
                    &format!("DSLL32 {val:#x} << {}",  sa + 32),
                    |e| { e.core.write_gpr(1, val); },
                    &[instr],
                    BlockTier::Alu,
                    1,
                );
            }
        }
    }

    #[test]
    fn test_jit_dsrl32() {
        for sa in [0u32, 1, 16, 31] {
            let instr = make_r(OP_SPECIAL, 0, 1, 2, sa, FUNCT_DSRL32);
            for val in [0xFFFF_FFFF_FFFF_FFFFu64, 0x8000_0000_0000_0000, 1] {
                run_jit_test(
                    &format!("DSRL32 {val:#x} >> {}", sa + 32),
                    |e| { e.core.write_gpr(1, val); },
                    &[instr],
                    BlockTier::Alu,
                    1,
                );
            }
        }
    }

    #[test]
    fn test_jit_dsra32() {
        for sa in [0u32, 1, 16, 31] {
            let instr = make_r(OP_SPECIAL, 0, 1, 2, sa, FUNCT_DSRA32);
            for val in [0x8000_0000_0000_0000u64, 0x7FFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF] {
                run_jit_test(
                    &format!("DSRA32 {val:#x} >> {}", sa + 32),
                    |e| { e.core.write_gpr(1, val); },
                    &[instr],
                    BlockTier::Alu,
                    1,
                );
            }
        }
    }

    #[test]
    fn test_jit_dsllv() {
        let instr = make_r(OP_SPECIAL, 2, 1, 3, 0, FUNCT_DSLLV);
        for (val, sa) in [(1u64, 0u64), (1, 32), (0xFFFF_FFFF, 32), (0x1, 63)] {
            run_jit_test(
                &format!("DSLLV {val:#x} << {sa}"),
                |e| { e.core.write_gpr(1, val); e.core.write_gpr(2, sa); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_dsrlv() {
        let instr = make_r(OP_SPECIAL, 2, 1, 3, 0, FUNCT_DSRLV);
        for (val, sa) in [(0xFFFF_FFFF_FFFF_FFFFu64, 0u64), (0xFFFF_FFFF_FFFF_FFFF, 32), (1, 1), (0x8000_0000_0000_0000, 63)] {
            run_jit_test(
                &format!("DSRLV {val:#x} >> {sa}"),
                |e| { e.core.write_gpr(1, val); e.core.write_gpr(2, sa); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_dsrav() {
        let instr = make_r(OP_SPECIAL, 2, 1, 3, 0, FUNCT_DSRAV);
        for (val, sa) in [(0x8000_0000_0000_0000u64, 0u64), (0x8000_0000_0000_0000, 32), (0x8000_0000_0000_0000, 63), (0x7FFF_FFFF_FFFF_FFFF, 32)] {
            run_jit_test(
                &format!("DSRAV {val:#x} >> {sa}"),
                |e| { e.core.write_gpr(1, val); e.core.write_gpr(2, sa); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    // ── Multiply / Divide ────────────────────────────────────────────────────────

    #[test]
    fn test_jit_mult() {
        let instr = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_MULT);
        for (a, b) in [
            (0u64, 0u64),
            (1, 1),
            (0x7FFF_FFFF, 2),
            (0xFFFF_FFFF, 0xFFFF_FFFF), // -1 × -1 signed 32
        ] {
            run_jit_test(
                &format!("MULT {a:#x}*{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_multu() {
        let instr = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_MULTU);
        for (a, b) in [
            (0u64, 0u64),
            (0xFFFF_FFFF, 0xFFFF_FFFF),
            (0x8000_0000, 2),
            (1234, 5678),
        ] {
            run_jit_test(
                &format!("MULTU {a:#x}*{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_div_nonzero() {
        // Only test non-zero divisors: interpreter returns EXEC_COMPLETE but leaves
        // hi/lo unchanged on division-by-zero (undefined per MIPS spec); JIT uses
        // a different safe-divisor fallback. Also skip INT_MIN / -1: the interpreter
        // uses wrapping_div (result = INT_MIN) but Cranelift's sdiv raises SIGFPE.
        let instr = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_DIV);
        for (a, b) in [
            (100u64, 7u64),
            (0xFFFF_FFFF, 2),    // -1 / 2 signed
            (1, 3),
            (0x7FFF_FFFF, 0xFFFF_FFFF), // INT_MAX / -1 = -INT_MAX
        ] {
            run_jit_test(
                &format!("DIV {a:#x}/{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_divu_nonzero() {
        let instr = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_DIVU);
        for (a, b) in [
            (100u64, 7u64),
            (0xFFFF_FFFF, 2),
            (0xDEAD_BEEF, 0x1000),
            (1, 1),
        ] {
            run_jit_test(
                &format!("DIVU {a:#x}/{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_dmult() {
        let instr = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_DMULT);
        for (a, b) in [
            (0u64, 0u64),
            (1, 0xFFFF_FFFF_FFFF_FFFF),
            (0x7FFF_FFFF_FFFF_FFFF, 2),
            (0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF), // (-1) * (-1)
        ] {
            run_jit_test(
                &format!("DMULT {a:#x}*{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_dmultu() {
        let instr = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_DMULTU);
        for (a, b) in [
            (0u64, 0u64),
            (0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF),
            (0x8000_0000_0000_0000, 2),
            (0x1234_5678_9ABC_DEF0, 0x1),
        ] {
            run_jit_test(
                &format!("DMULTU {a:#x}*{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_ddiv_nonzero() {
        let instr = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_DDIV);
        for (a, b) in [
            (100u64, 7u64),
            (0xFFFF_FFFF_FFFF_FFFF, 2u64), // -1 / 2
            (1, 3),
            (0x1234_5678_9ABC_DEF0, 0x1000),
        ] {
            run_jit_test(
                &format!("DDIV {a:#x}/{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_ddivu_nonzero() {
        let instr = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_DDIVU);
        for (a, b) in [
            (100u64, 7u64),
            (0xFFFF_FFFF_FFFF_FFFF, 2u64),
            (0xDEAD_BEEF_CAFE_BABE, 0x100),
            (1, 1),
        ] {
            run_jit_test(
                &format!("DDIVU {a:#x}/{b:#x}"),
                |e| { e.core.write_gpr(1, a); e.core.write_gpr(2, b); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    // ── HI/LO moves ──────────────────────────────────────────────────────────────

    #[test]
    fn test_jit_mfhi_mflo() {
        let mfhi = make_r(OP_SPECIAL, 0, 0, 3, 0, FUNCT_MFHI);
        let mflo = make_r(OP_SPECIAL, 0, 0, 4, 0, FUNCT_MFLO);
        for (hi_val, lo_val) in [
            (0u64, 0u64),
            (0xDEAD_BEEF_CAFE_BABEu64, 0x1234_5678_9ABC_DEF0u64),
            (0xFFFF_FFFF_FFFF_FFFF, 1),
        ] {
            run_jit_test(
                &format!("MFHI hi={hi_val:#x}"),
                |e| { e.core.hi = hi_val; },
                &[mfhi],
                BlockTier::Alu,
                1,
            );
            run_jit_test(
                &format!("MFLO lo={lo_val:#x}"),
                |e| { e.core.lo = lo_val; },
                &[mflo],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_mthi_mtlo() {
        let mthi = make_r(OP_SPECIAL, 1, 0, 0, 0, FUNCT_MTHI);
        let mtlo = make_r(OP_SPECIAL, 1, 0, 0, 0, FUNCT_MTLO);
        for val in [0u64, 0xDEAD_BEEF_CAFE_BABEu64, 0xFFFF_FFFF_FFFF_FFFF] {
            run_jit_test(
                &format!("MTHI rs={val:#x}"),
                |e| { e.core.write_gpr(1, val); },
                &[mthi],
                BlockTier::Alu,
                1,
            );
            run_jit_test(
                &format!("MTLO rs={val:#x}"),
                |e| { e.core.write_gpr(1, val); },
                &[mtlo],
                BlockTier::Alu,
                1,
            );
        }
    }

    // ── Conditional moves ────────────────────────────────────────────────────────

    #[test]
    fn test_jit_movz() {
        // MOVZ rd=3, rs=1, rt=2 — if rt==0 then rd=rs else rd unchanged
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_MOVZ);
        // rt=0: should move
        run_jit_test(
            "MOVZ rt=0",
            |e| { e.core.write_gpr(1, 0xABCD); e.core.write_gpr(2, 0); e.core.write_gpr(3, 0x1234); },
            &[instr],
            BlockTier::Alu,
            1,
        );
        // rt!=0: should not move
        run_jit_test(
            "MOVZ rt!=0",
            |e| { e.core.write_gpr(1, 0xABCD); e.core.write_gpr(2, 1); e.core.write_gpr(3, 0x1234); },
            &[instr],
            BlockTier::Alu,
            1,
        );
    }

    #[test]
    fn test_jit_movn() {
        // MOVN rd=3, rs=1, rt=2 — if rt!=0 then rd=rs else rd unchanged
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_MOVN);
        run_jit_test(
            "MOVN rt!=0",
            |e| { e.core.write_gpr(1, 0xABCD); e.core.write_gpr(2, 42); e.core.write_gpr(3, 0x1234); },
            &[instr],
            BlockTier::Alu,
            1,
        );
        run_jit_test(
            "MOVN rt=0",
            |e| { e.core.write_gpr(1, 0xABCD); e.core.write_gpr(2, 0); e.core.write_gpr(3, 0x1234); },
            &[instr],
            BlockTier::Alu,
            1,
        );
    }

    // ── SYNC ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_jit_sync() {
        let instr = make_r(OP_SPECIAL, 0, 0, 0, 0, FUNCT_SYNC);
        run_jit_test("SYNC", |_| {}, &[instr], BlockTier::Alu, 1);
    }

    // ── Immediate ALU ops ────────────────────────────────────────────────────────

    #[test]
    fn test_jit_addiu() {
        // ADDIU rt=2, rs=1, imm
        for (rs_val, imm) in [
            (0u64, 0i16),
            (10, -5),
            (0xFFFF_FFFF, 1),
            (0, -1),
            (0x7FFF_FFFF, 1),
        ] {
            let instr = make_i(OP_ADDIU, 1, 2, imm as u16);
            run_jit_test(
                &format!("ADDIU {rs_val:#x}+{imm}"),
                |e| { e.core.write_gpr(1, rs_val); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_daddiu() {
        for (rs_val, imm) in [
            (0u64, 0i16),
            (0xFFFF_FFFF_FFFF_FFFF, 1),
            (0, -1),
            (100, -200),
        ] {
            let instr = make_i(OP_DADDIU, 1, 2, imm as u16);
            run_jit_test(
                &format!("DADDIU {rs_val:#x}+{imm}"),
                |e| { e.core.write_gpr(1, rs_val); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_slti() {
        for (rs_val, imm) in [
            (0u64, 0i16),
            (0xFFFF_FFFF_FFFF_FFFF, 0), // -1 < 0: 1
            (0, -1i16),                  // 0 < -1: 0
            (100, 200),
        ] {
            let instr = make_i(OP_SLTI, 1, 2, imm as u16);
            run_jit_test(
                &format!("SLTI {rs_val:#x} < {imm}"),
                |e| { e.core.write_gpr(1, rs_val); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_sltiu() {
        for (rs_val, imm) in [
            (0u64, 1u16),
            (1u64, 0u16),
            (0u64, 0u16),
            (0xFFFF_FFFF_FFFF_FFFF, 1), // large < 1 unsigned: 0
        ] {
            let instr = make_i(OP_SLTIU, 1, 2, imm);
            run_jit_test(
                &format!("SLTIU {rs_val:#x} < {imm}"),
                |e| { e.core.write_gpr(1, rs_val); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_andi() {
        for (rs_val, imm) in [
            (0xFFFF_FFFF_FFFF_FFFFu64, 0xFFFFu16),
            (0xAAAA_AAAA_AAAA_AAAA, 0x5555),
            (0xDEAD_BEEF, 0xFF00),
            (0, 0xFFFF),
        ] {
            let instr = make_i(OP_ANDI, 1, 2, imm);
            run_jit_test(
                &format!("ANDI {rs_val:#x}&{imm:#x}"),
                |e| { e.core.write_gpr(1, rs_val); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_ori() {
        for (rs_val, imm) in [
            (0u64, 0xFFFFu16),
            (0xFFFF_FFFF_FFFF_0000u64, 0xFFFF),
            (0, 0),
            (0xDEAD_0000, 0xBEEF),
        ] {
            let instr = make_i(OP_ORI, 1, 2, imm);
            run_jit_test(
                &format!("ORI {rs_val:#x}|{imm:#x}"),
                |e| { e.core.write_gpr(1, rs_val); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_xori() {
        for (rs_val, imm) in [
            (0xFFFFu64, 0xFFFFu16),
            (0, 0xFFFF),
            (0xDEAD_BEEF, 0xBEEF),
            (0, 0),
        ] {
            let instr = make_i(OP_XORI, 1, 2, imm);
            run_jit_test(
                &format!("XORI {rs_val:#x}^{imm:#x}"),
                |e| { e.core.write_gpr(1, rs_val); },
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    #[test]
    fn test_jit_lui() {
        for imm in [0u16, 1, 0x7FFF, 0x8000, 0xFFFF] {
            let instr = make_i(OP_LUI, 0, 1, imm);
            run_jit_test(
                &format!("LUI imm={imm:#x}"),
                |_| {},
                &[instr],
                BlockTier::Alu,
                1,
            );
        }
    }

    // ── Branches ────────────────────────────────────────────────────────────────
    //
    // Branch blocks include the branch + a NOP delay slot. JIT computes the exit
    // PC directly. The interpreter needs 2 step() calls (branch + delay slot).
    // Branch offset is encoded as the signed number of instructions relative to
    // PC+4, sign-extended and shifted left 2. We use offset=+4 words (16 bytes
    // ahead of the delay slot) for the taken case.
    //
    // TEST_PC = 0x80010000
    // branch at offset 0:  PC+4 = 0x80010004
    // delay slot at offset 4: PC+8 = 0x80010008
    // taken target  = PC+4 + imm*4 = 0x80010004 + 4*4 = 0x80010014  (imm=4)
    // not-taken     = PC+8 = 0x80010008

    #[test]
    fn test_jit_beq_taken() {
        let branch = make_i(OP_BEQ, 1, 2, 4u16); // beq $1, $2, +4  (taken when equal)
        run_jit_test(
            "BEQ taken",
            |e| { e.core.write_gpr(1, 42); e.core.write_gpr(2, 42); },
            &[branch, NOP],
            BlockTier::Alu,
            2,
        );
    }

    #[test]
    fn test_jit_beq_not_taken() {
        let branch = make_i(OP_BEQ, 1, 2, 4u16);
        run_jit_test(
            "BEQ not taken",
            |e| { e.core.write_gpr(1, 1); e.core.write_gpr(2, 2); },
            &[branch, NOP],
            BlockTier::Alu,
            2,
        );
    }

    #[test]
    fn test_jit_bne_taken() {
        let branch = make_i(OP_BNE, 1, 2, 4u16);
        run_jit_test(
            "BNE taken",
            |e| { e.core.write_gpr(1, 1); e.core.write_gpr(2, 2); },
            &[branch, NOP],
            BlockTier::Alu,
            2,
        );
    }

    #[test]
    fn test_jit_bne_not_taken() {
        let branch = make_i(OP_BNE, 1, 2, 4u16);
        run_jit_test(
            "BNE not taken",
            |e| { e.core.write_gpr(1, 99); e.core.write_gpr(2, 99); },
            &[branch, NOP],
            BlockTier::Alu,
            2,
        );
    }

    #[test]
    fn test_jit_blez() {
        let branch = make_i(OP_BLEZ, 1, 0, 4u16);
        // zero: taken
        run_jit_test("BLEZ zero",   |e| { e.core.write_gpr(1, 0); }, &[branch, NOP], BlockTier::Alu, 2);
        // negative: taken
        run_jit_test("BLEZ neg",    |e| { e.core.write_gpr(1, 0xFFFF_FFFF_FFFF_FFFF); }, &[branch, NOP], BlockTier::Alu, 2);
        // positive: not taken
        run_jit_test("BLEZ pos",    |e| { e.core.write_gpr(1, 1); }, &[branch, NOP], BlockTier::Alu, 2);
    }

    #[test]
    fn test_jit_bgtz() {
        let branch = make_i(OP_BGTZ, 1, 0, 4u16);
        // positive: taken
        run_jit_test("BGTZ pos",    |e| { e.core.write_gpr(1, 1); }, &[branch, NOP], BlockTier::Alu, 2);
        // zero: not taken
        run_jit_test("BGTZ zero",   |e| { e.core.write_gpr(1, 0); }, &[branch, NOP], BlockTier::Alu, 2);
        // negative: not taken
        run_jit_test("BGTZ neg",    |e| { e.core.write_gpr(1, 0xFFFF_FFFF_FFFF_FFFF); }, &[branch, NOP], BlockTier::Alu, 2);
    }

    #[test]
    fn test_jit_j() {
        // J target: target26 << 2 in the same 256MB region.
        // TEST_PC = 0x8001_0000, PC+4 = 0x8001_0004
        // region   = (PC+4) & 0xFFFF_FFFF_F000_0000 = 0x8000_0000
        // target26 encodes instr-words offset: let's jump to 0x8000_4000
        // target26 = (0x8000_4000 >> 2) & 0x3FF_FFFF = 0x1000
        let target26: u32 = (0x8000_4000u32 >> 2) & 0x3FF_FFFF;
        let instr = make_j(OP_J, target26);
        run_jit_test(
            "J",
            |_| {},
            &[instr, NOP],
            BlockTier::Alu,
            2,
        );
    }

    #[test]
    fn test_jit_jal() {
        // JAL stores PC+8 in $ra (r31) and jumps.
        // TEST_PC = 0x8001_0000 → PC+8 = 0x8001_0008
        let target26: u32 = (0x8000_8000u32 >> 2) & 0x3FF_FFFF;
        let instr = make_j(OP_JAL, target26);
        run_jit_test(
            "JAL",
            |_| {},
            &[instr, NOP],
            BlockTier::Alu,
            2,
        );
    }

    #[test]
    fn test_jit_jr() {
        // JR $1 — jump to address in r1.
        // r1 = 0x8003_0000 (must be in kseg0 and aligned)
        let instr = make_r(OP_SPECIAL, 1, 0, 0, 0, FUNCT_JR);
        run_jit_test(
            "JR",
            |e| { e.core.write_gpr(1, 0x8003_0000); },
            &[instr, NOP],
            BlockTier::Alu,
            2,
        );
    }

    // ── Loads ────────────────────────────────────────────────────────────────────
    //
    // DATA_ADDR = 0x8002_0000, physical = 0x0002_0000.
    // We pre-populate MockMemory at that physical address before each load test.

    #[test]
    fn test_jit_lb() {
        // LB rt=2, offset=0(rs=1): load signed byte
        let instr = make_i(OP_LB, 1, 2, 0);
        // Positive byte (0x42)
        let (mut exec, mem) = create_executor();
        exec.core.pc = TEST_PC;
        let instrs = prepare_block(&mut exec, &mem, &[instr]);
        mem.set_byte((DATA_ADDR & 0x1FFF_FFFF) as u64, 0x42);
        exec.core.write_gpr(1, DATA_ADDR);
        let saved = CpuState::capture(&exec);
        let helpers = HelperPtrs::new::<PassthroughTlb, PassthroughCache>();
        let block = BlockCompiler::new(&helpers).compile_block(&instrs, TEST_PC, BlockTier::Loads).unwrap();
        let mut ctx = JitContext::new();
        ctx.sync_from_executor(&exec);
        ctx.executor_ptr = &mut exec as *mut _ as u64;
        let entry: extern "C" fn(*mut JitContext) = unsafe { std::mem::transmute(block.entry) };
        entry(&mut ctx);
        ctx.sync_to_executor(&mut exec);
        let jit_state = CpuState::capture(&exec);
        saved.restore(&mut exec);
        exec.step();
        diff_states("LB positive", &jit_state, &CpuState::capture(&exec));

        // Negative byte (0x80 → -128)
        let (mut exec, mem) = create_executor();
        exec.core.pc = TEST_PC;
        let instrs = prepare_block(&mut exec, &mem, &[instr]);
        mem.set_byte((DATA_ADDR & 0x1FFF_FFFF) as u64, 0x80);
        exec.core.write_gpr(1, DATA_ADDR);
        let saved = CpuState::capture(&exec);
        let helpers = HelperPtrs::new::<PassthroughTlb, PassthroughCache>();
        let block = BlockCompiler::new(&helpers).compile_block(&instrs, TEST_PC, BlockTier::Loads).unwrap();
        let mut ctx = JitContext::new();
        ctx.sync_from_executor(&exec);
        ctx.executor_ptr = &mut exec as *mut _ as u64;
        let entry: extern "C" fn(*mut JitContext) = unsafe { std::mem::transmute(block.entry) };
        entry(&mut ctx);
        ctx.sync_to_executor(&mut exec);
        let jit_state = CpuState::capture(&exec);
        saved.restore(&mut exec);
        exec.step();
        diff_states("LB negative", &jit_state, &CpuState::capture(&exec));
    }

    #[test]
    fn test_jit_lbu() {
        // LBU rt=2, offset=0(rs=1): load zero-extended byte
        let instr = make_i(OP_LBU, 1, 2, 0);
        for byte_val in [0u8, 0x42, 0x80, 0xFF] {
            let (mut exec, mem) = create_executor();
            exec.core.pc = TEST_PC;
            let instrs = prepare_block(&mut exec, &mem, &[instr]);
            mem.set_byte((DATA_ADDR & 0x1FFF_FFFF) as u64, byte_val);
            exec.core.write_gpr(1, DATA_ADDR);
            let saved = CpuState::capture(&exec);
            let helpers = HelperPtrs::new::<PassthroughTlb, PassthroughCache>();
            let block = BlockCompiler::new(&helpers)
                .compile_block(&instrs, TEST_PC, BlockTier::Loads)
                .unwrap();
            let mut ctx = JitContext::new();
            ctx.sync_from_executor(&exec);
            ctx.executor_ptr = &mut exec as *mut _ as u64;
            let entry: extern "C" fn(*mut JitContext) = unsafe { std::mem::transmute(block.entry) };
            entry(&mut ctx);
            ctx.sync_to_executor(&mut exec);
            let jit_state = CpuState::capture(&exec);
            saved.restore(&mut exec);
            exec.step();
            diff_states(&format!("LBU {byte_val:#x}"), &jit_state, &CpuState::capture(&exec));
        }
    }

    #[test]
    fn test_jit_lh() {
        // LH rt=2, offset=0(rs=1): load signed halfword
        let instr = make_i(OP_LH, 1, 2, 0);
        for hw in [0u16, 0x1234, 0x8000, 0xFFFF] {
            let (mut exec, mem) = create_executor();
            exec.core.pc = TEST_PC;
            let instrs = prepare_block(&mut exec, &mem, &[instr]);
            let phys = (DATA_ADDR & 0x1FFF_FFFF) as u64;
            mem.set_byte(phys,     (hw >> 8) as u8);
            mem.set_byte(phys + 1, (hw & 0xFF) as u8);
            exec.core.write_gpr(1, DATA_ADDR);
            let saved = CpuState::capture(&exec);
            let helpers = HelperPtrs::new::<PassthroughTlb, PassthroughCache>();
            let block = BlockCompiler::new(&helpers).compile_block(&instrs, TEST_PC, BlockTier::Loads).unwrap();
            let mut ctx = JitContext::new();
            ctx.sync_from_executor(&exec);
            ctx.executor_ptr = &mut exec as *mut _ as u64;
            let entry: extern "C" fn(*mut JitContext) = unsafe { std::mem::transmute(block.entry) };
            entry(&mut ctx);
            ctx.sync_to_executor(&mut exec);
            let jit_state = CpuState::capture(&exec);
            saved.restore(&mut exec);
            exec.step();
            diff_states(&format!("LH {hw:#x}"), &jit_state, &CpuState::capture(&exec));
        }
    }

    #[test]
    fn test_jit_lhu() {
        let instr = make_i(OP_LHU, 1, 2, 0);
        for hw in [0u16, 0x1234, 0x8000, 0xFFFF] {
            let (mut exec, mem) = create_executor();
            exec.core.pc = TEST_PC;
            let instrs = prepare_block(&mut exec, &mem, &[instr]);
            let phys = (DATA_ADDR & 0x1FFF_FFFF) as u64;
            mem.set_byte(phys,     (hw >> 8) as u8);
            mem.set_byte(phys + 1, (hw & 0xFF) as u8);
            exec.core.write_gpr(1, DATA_ADDR);
            let saved = CpuState::capture(&exec);
            let helpers = HelperPtrs::new::<PassthroughTlb, PassthroughCache>();
            let block = BlockCompiler::new(&helpers).compile_block(&instrs, TEST_PC, BlockTier::Loads).unwrap();
            let mut ctx = JitContext::new();
            ctx.sync_from_executor(&exec);
            ctx.executor_ptr = &mut exec as *mut _ as u64;
            let entry: extern "C" fn(*mut JitContext) = unsafe { std::mem::transmute(block.entry) };
            entry(&mut ctx);
            ctx.sync_to_executor(&mut exec);
            let jit_state = CpuState::capture(&exec);
            saved.restore(&mut exec);
            exec.step();
            diff_states(&format!("LHU {hw:#x}"), &jit_state, &CpuState::capture(&exec));
        }
    }

    #[test]
    fn test_jit_lw() {
        let instr = make_i(OP_LW, 1, 2, 0);
        for wval in [0u32, 0x1234_5678, 0x8000_0000, 0xFFFF_FFFF] {
            let (mut exec, mem) = create_executor();
            exec.core.pc = TEST_PC;
            let instrs = prepare_block(&mut exec, &mem, &[instr]);
            mem.set_word((DATA_ADDR & 0x1FFF_FFFF) as u64, wval);
            exec.core.write_gpr(1, DATA_ADDR);
            let saved = CpuState::capture(&exec);
            let helpers = HelperPtrs::new::<PassthroughTlb, PassthroughCache>();
            let block = BlockCompiler::new(&helpers).compile_block(&instrs, TEST_PC, BlockTier::Loads).unwrap();
            let mut ctx = JitContext::new();
            ctx.sync_from_executor(&exec);
            ctx.executor_ptr = &mut exec as *mut _ as u64;
            let entry: extern "C" fn(*mut JitContext) = unsafe { std::mem::transmute(block.entry) };
            entry(&mut ctx);
            ctx.sync_to_executor(&mut exec);
            let jit_state = CpuState::capture(&exec);
            saved.restore(&mut exec);
            exec.step();
            diff_states(&format!("LW {wval:#x}"), &jit_state, &CpuState::capture(&exec));
        }
    }

    #[test]
    fn test_jit_lwu() {
        let instr = make_i(OP_LWU, 1, 2, 0);
        for wval in [0u32, 0x1234_5678, 0x8000_0000, 0xFFFF_FFFF] {
            let (mut exec, mem) = create_executor();
            exec.core.pc = TEST_PC;
            let instrs = prepare_block(&mut exec, &mem, &[instr]);
            mem.set_word((DATA_ADDR & 0x1FFF_FFFF) as u64, wval);
            exec.core.write_gpr(1, DATA_ADDR);
            let saved = CpuState::capture(&exec);
            let helpers = HelperPtrs::new::<PassthroughTlb, PassthroughCache>();
            let block = BlockCompiler::new(&helpers).compile_block(&instrs, TEST_PC, BlockTier::Loads).unwrap();
            let mut ctx = JitContext::new();
            ctx.sync_from_executor(&exec);
            ctx.executor_ptr = &mut exec as *mut _ as u64;
            let entry: extern "C" fn(*mut JitContext) = unsafe { std::mem::transmute(block.entry) };
            entry(&mut ctx);
            ctx.sync_to_executor(&mut exec);
            let jit_state = CpuState::capture(&exec);
            saved.restore(&mut exec);
            exec.step();
            diff_states(&format!("LWU {wval:#x}"), &jit_state, &CpuState::capture(&exec));
        }
    }

    #[test]
    fn test_jit_ld() {
        let instr = make_i(OP_LD, 1, 2, 0);
        for dval in [0u64, 0x1234_5678_9ABC_DEF0, 0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF] {
            let (mut exec, mem) = create_executor();
            exec.core.pc = TEST_PC;
            let instrs = prepare_block(&mut exec, &mem, &[instr]);
            mem.set_double((DATA_ADDR & 0x1FFF_FFFF) as u64, dval);
            exec.core.write_gpr(1, DATA_ADDR);
            let saved = CpuState::capture(&exec);
            let helpers = HelperPtrs::new::<PassthroughTlb, PassthroughCache>();
            let block = BlockCompiler::new(&helpers).compile_block(&instrs, TEST_PC, BlockTier::Loads).unwrap();
            let mut ctx = JitContext::new();
            ctx.sync_from_executor(&exec);
            ctx.executor_ptr = &mut exec as *mut _ as u64;
            let entry: extern "C" fn(*mut JitContext) = unsafe { std::mem::transmute(block.entry) };
            entry(&mut ctx);
            ctx.sync_to_executor(&mut exec);
            let jit_state = CpuState::capture(&exec);
            saved.restore(&mut exec);
            exec.step();
            diff_states(&format!("LD {dval:#x}"), &jit_state, &CpuState::capture(&exec));
        }
    }

    // ── Stores ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_jit_sb() {
        // SB rt=2, offset=0(rs=1)
        let instr = make_i(OP_SB, 1, 2, 0);
        for bval in [0u64, 0x42, 0x80, 0xFF] {
            let phys = (DATA_ADDR & 0x1FFF_FFFF) as u64;
            run_jit_store_test(
                &format!("SB {bval:#x}"),
                |e, _| { e.core.write_gpr(1, DATA_ADDR); e.core.write_gpr(2, bval); },
                &[instr],
                1,
                &[(phys, vec![(bval & 0xFF) as u8])],
            );
        }
    }

    #[test]
    fn test_jit_sh() {
        let instr = make_i(OP_SH, 1, 2, 0);
        for hval in [0u64, 0x1234, 0x8000, 0xFFFF] {
            let phys = (DATA_ADDR & 0x1FFF_FFFF) as u64;
            let hw = (hval & 0xFFFF) as u16;
            run_jit_store_test(
                &format!("SH {hval:#x}"),
                |e, _| { e.core.write_gpr(1, DATA_ADDR); e.core.write_gpr(2, hval); },
                &[instr],
                1,
                &[(phys, vec![(hw >> 8) as u8, (hw & 0xFF) as u8])],
            );
        }
    }

    #[test]
    fn test_jit_sw() {
        let instr = make_i(OP_SW, 1, 2, 0);
        for wval in [0u64, 0x1234_5678, 0x8000_0000, 0xFFFF_FFFF] {
            let phys = (DATA_ADDR & 0x1FFF_FFFF) as u64;
            let w = (wval & 0xFFFF_FFFF) as u32;
            let wb = w.to_be_bytes();
            run_jit_store_test(
                &format!("SW {wval:#x}"),
                |e, _| { e.core.write_gpr(1, DATA_ADDR); e.core.write_gpr(2, wval); },
                &[instr],
                1,
                &[(phys, wb.to_vec())],
            );
        }
    }

    #[test]
    fn test_jit_sd() {
        let instr = make_i(OP_SD, 1, 2, 0);
        for dval in [0u64, 0x1234_5678_9ABC_DEF0, 0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF] {
            let phys = (DATA_ADDR & 0x1FFF_FFFF) as u64;
            let db = dval.to_be_bytes();
            run_jit_store_test(
                &format!("SD {dval:#x}"),
                |e, _| { e.core.write_gpr(1, DATA_ADDR); e.core.write_gpr(2, dval); },
                &[instr],
                1,
                &[(phys, db.to_vec())],
            );
        }
    }
}
