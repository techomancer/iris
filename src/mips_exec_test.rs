#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::collections::HashMap;
    use crate::mips_core::{STATUS_KX, STATUS_CU1, STATUS_FR};
    use crate::mips_exec::{MipsExecutor, MipsCpuConfig, DecodedInstr, EXEC_COMPLETE, EXEC_BREAKPOINT, EXEC_IS_EXCEPTION, EXEC_IS_TLB_REFILL, exec_exception, EXC_SYS, EXC_BP, EXC_TR, EXC_OV, EXC_RI, EXC_ADEL};
    use crate::mips_isa::*;
    use crate::mips_tlb::PassthroughTlb;
    use crate::mips_cache_v2::{PassthroughCache, PassthroughCacheM4, MipsCache, R4400Cache, CpuModel};
    use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BusDevice};
    use std::sync::Arc;

    // Mock Memory Interface
    pub struct MockMemory {
        pub data: Mutex<HashMap<u64, u8>>,
    }

    impl MockMemory {
        pub fn new() -> Self {
            Self {
                data: Mutex::new(HashMap::new()),
            }
        }

        pub fn set_byte(&self, addr: u64, val: u8) {
            self.data.lock().unwrap().insert(addr, val);
        }

        pub fn get_byte(&self, addr: u64) -> u8 {
            *self.data.lock().unwrap().get(&addr).unwrap_or(&0)
        }

        pub fn set_word(&self, addr: u64, val: u32) {
            let bytes = val.to_be_bytes();
            for i in 0..4 {
                self.data.lock().unwrap().insert(addr + i as u64, bytes[i]);
            }
        }
        
        pub fn get_word(&self, addr: u64) -> u32 {
            let mut bytes = [0u8; 4];
            for i in 0..4 {
                bytes[i] = *self.data.lock().unwrap().get(&(addr + i as u64)).unwrap_or(&0);
            }
            u32::from_be_bytes(bytes)
        }

        pub fn set_double(&self, addr: u64, val: u64) {
            let bytes = val.to_be_bytes();
            for i in 0..8 {
                self.data.lock().unwrap().insert(addr + i as u64, bytes[i]);
            }
        }

        pub fn get_double(&self, addr: u64) -> u64 {
            let mut bytes = [0u8; 8];
            for i in 0..8 {
                bytes[i] = *self.data.lock().unwrap().get(&(addr + i as u64)).unwrap_or(&0);
            }
            u64::from_be_bytes(bytes)
        }
    }

    impl BusDevice for MockMemory {
        fn read8(&self, addr: u32) -> BusRead8 {
            BusRead8::ok(self.get_byte(addr as u64))
        }

        fn write8(&self, addr: u32, val: u8) -> u32 {
            self.set_byte(addr as u64, val);
            BUS_OK
        }

        fn read16(&self, addr: u32) -> BusRead16 {
            let aligned_addr = (addr & !1) as u64;
            let mut bytes = [0u8; 2];
            for i in 0..2 {
                bytes[i] = self.get_byte(aligned_addr + i as u64);
            }
            BusRead16::ok(u16::from_be_bytes(bytes))
        }

        fn write16(&self, addr: u32, val: u16) -> u32 {
            let aligned_addr = (addr & !1) as u64;
            let bytes = val.to_be_bytes();
            for i in 0..2 {
                self.set_byte(aligned_addr + i as u64, bytes[i]);
            }
            BUS_OK
        }

        fn read32(&self, addr: u32) -> BusRead32 {
            let aligned_addr = (addr & !3) as u64;
            BusRead32::ok(self.get_word(aligned_addr))
        }

        fn write32(&self, addr: u32, val: u32) -> u32 {
            let aligned_addr = (addr & !3) as u64;
            self.set_word(aligned_addr, val);
            BUS_OK
        }

        fn read64(&self, addr: u32) -> BusRead64 {
            let aligned_addr = (addr & !7) as u64;
            BusRead64::ok(self.get_double(aligned_addr))
        }

        fn write64(&self, addr: u32, val: u64) -> u32 {
            let aligned_addr = (addr & !7) as u64;
            self.set_double(aligned_addr, val);
            BUS_OK
        }

        // This whole file predates jitv2 and no test here calls
        // `install_jit_hooks` — every `create_executor*` helper below
        // explicitly disables both the real dispatch gate
        // (`jitv2_dispatch_enabled = false`) and every `jitv2_lockstep`
        // class (`lockstep_enabled.{alu,branch,load_store,fpu} = false`),
        // so no jitv2 gate variant ever touches these executors,
        // unconditionally, regardless of which jitv2 feature happens to be
        // compiled in. Only `test_r4000cache_step_sequence` wants real jitv2
        // dispatch — it builds its own executor with a real gen-tracking
        // memory and calls `install_jit_hooks` explicitly instead of using
        // this helper's `MockMemory`/`create_executor_with_r4000cache`.
        // Without disabling both gates, a jitv2 gate variant that runs
        // unconditionally (jitv2_lockstep's lockstep_check, in particular —
        // see rules/jitv2/codegen-gotchas.md) can compile and run a function
        // against an executor whose `core.handle_exception_fn` was never
        // installed, aborting the whole process the moment that function
        // raises an exception (`jit_hooks_not_installed_exception`,
        // `panic = "abort"`). This used to be enforced by returning a null
        // gen_ptr here (a page with no real gen tracking used to be flatly
        // "not compilable") — that inference no longer holds (a null
        // gen_ptr now just means "use the shared, never-bumped fallback
        // counter", not "never dispatchable" — see
        // `crate::jitv2::jitv2::NEVER_COMPILABLE_GEN`'s doc comment), so the
        // real enforcement moved to the executor-level switches below;
        // staying null here is still fine (and simplest — no per-page
        // counter storage needed in this basic MockMemory at all).
        #[cfg(feature = "jitv2")]
        fn gen_ptr(&self, _addr: u32) -> *const std::sync::atomic::AtomicU64 {
            std::ptr::null()
        }
    }

    /// Disable every jitv2 dispatch path on a freshly-constructed executor —
    /// see `MockMemory::gen_ptr`'s doc comment for why this, not a null
    /// `gen_ptr`, is what actually keeps jitv2 out of this file's tests now.
    #[cfg(feature = "jitv2")]
    fn disable_jitv2<T: crate::mips_tlb::Tlb, C: CpuModel>(exec: &mut MipsExecutor<T, C>) {
        // jitv2_dispatch_enabled=false turns the whole JIT dispatch gate off,
        // which is what drives lockstep now — so this alone keeps jitv2 (and its
        // lockstep instrumentation) out of these interpreter-only tests.
        exec.jitv2_dispatch_enabled = false;
    }
    #[cfg(not(feature = "jitv2"))]
    fn disable_jitv2<T: crate::mips_tlb::Tlb, C: CpuModel>(_exec: &mut MipsExecutor<T, C>) {}

    // Helper to create executor with mock memory
    fn create_executor() -> (MipsExecutor<PassthroughTlb, PassthroughCache>, Arc<MockMemory>) {
        let mem = Arc::new(MockMemory::new());
        let mem_bus: Arc<dyn BusDevice> = mem.clone();
        let cfg = MipsCpuConfig::indy();
        let mut exec = MipsExecutor::new(mem_bus, PassthroughTlb::default(), &cfg);
        disable_jitv2(&mut exec);
        (exec, mem)
    }

    fn create_executor_m4() -> (MipsExecutor<PassthroughTlb, PassthroughCacheM4>, Arc<MockMemory>) {
        let mem = Arc::new(MockMemory::new());
        let mem_bus: Arc<dyn BusDevice> = mem.clone();
        let cfg = MipsCpuConfig::indy();
        let mut exec = MipsExecutor::new(mem_bus, PassthroughTlb::default(), &cfg);
        disable_jitv2(&mut exec);
        (exec, mem)
    }

    // Helper to create executor with specific TLB
    fn create_executor_with_tlb<T: crate::mips_tlb::Tlb>(tlb: T) -> (MipsExecutor<T, PassthroughCache>, Arc<MockMemory>) {
        let mem = Arc::new(MockMemory::new());
        let mem_bus: Arc<dyn BusDevice> = mem.clone();
        let cfg = MipsCpuConfig::indy();
        let mut exec = MipsExecutor::new(mem_bus, tlb, &cfg);
        disable_jitv2(&mut exec);
        (exec, mem)
    }

    // Helper to create executor with R4400Cache for VCE testing
    fn create_executor_with_r4000cache() -> (MipsExecutor<PassthroughTlb, R4400Cache>, Arc<MockMemory>) {
        let mem = Arc::new(MockMemory::new());
        let mem_bus: Arc<dyn BusDevice> = mem.clone();
        let cfg = MipsCpuConfig::indy();
        let mut exec = MipsExecutor::new(mem_bus, PassthroughTlb::default(), &cfg);
        disable_jitv2(&mut exec);
        (exec, mem)
    }

    /// A model's identity must be live the moment the executor exists — MipsCore::new
    /// runs reset_registers before the model is known, so the constructor has to set
    /// both the reset value and the live register. Caught a real bug: the guest read
    /// R4400 out of an R5000 machine because only the reset value had been updated.
    #[test]
    fn cpu_model_identity_is_live_at_construction() {
        fn build<C: crate::mips_cache_v2::CpuModel + From<Arc<dyn BusDevice>>>()
            -> MipsExecutor<PassthroughTlb, C> {
            let mem: Arc<dyn BusDevice> = Arc::new(MockMemory::new());
            MipsExecutor::new(mem, PassthroughTlb::default(), &MipsCpuConfig::indy())
        }
        let e4 = build::<R4400Cache>();
        assert_eq!(e4.core.cp0_prid, 0x0000_0440, "R4400 PRId at construction");
        assert_eq!(e4.core.fpu_fir,  0x0000_0500, "R4400 FIR at construction");
        assert!(!R4400Cache::MIPS4, "R4400 is MIPS III");

        let e5 = build::<crate::mips_cache_v2::R5000Cache>();
        assert_eq!(e5.core.cp0_prid, 0x0000_2321, "R5000 PRId at construction");
        assert_eq!(e5.core.fpu_fir,  0x0000_2300, "R5000 FIR at construction");
        assert!(crate::mips_cache_v2::R5000Cache::MIPS4, "R5000 is MIPS IV");

        // and they must survive a power_on, which re-derives from the reset values
        let e5b = build::<crate::mips_cache_v2::R5000Cache>();
        let mut core = e5b.core;
        core.reset(false);
        assert_eq!(core.cp0_prid, 0x0000_2321, "R5000 PRId after reset");
        assert_eq!(core.fpu_fir,  0x0000_2300, "R5000 FIR after reset");
    }

    // Instruction builders
    fn make_r(op: u32, rs: u32, rt: u32, rd: u32, sa: u32, funct: u32) -> u32 {
        (op << 26) | ((rs & 0x1F) << 21) | ((rt & 0x1F) << 16) | ((rd & 0x1F) << 11) | ((sa & 0x1F) << 6) | (funct & 0x3F)
    }

    fn make_i(op: u32, rs: u32, rt: u32, imm: u16) -> u32 {
        (op << 26) | ((rs & 0x1F) << 21) | ((rt & 0x1F) << 16) | (imm as u32)
    }

    fn make_j(op: u32, target: u32) -> u32 {
        (op << 26) | (target & 0x3FFFFFF)
    }

    #[test]
    fn test_add_addu() {
        let (mut exec, _) = create_executor();

        // ADD r3, r1, r2
        // r1 = 10, r2 = 20 -> r3 = 30
        exec.core.write_gpr(1, 10);
        exec.core.write_gpr(2, 20);
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_ADD);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 30);

        // ADDU r3, r1, r2 (wrapping)
        // r1 = 0xFFFFFFFF, r2 = 1 -> r3 = 0 (32-bit wrap)
        exec.core.write_gpr(1, 0xFFFFFFFF);
        exec.core.write_gpr(2, 1);
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_ADDU);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0); // 32-bit wrapping add, result sign extended (0 -> 0)
    }

    #[test]
    fn test_add_overflow() {
        let (mut exec, _) = create_executor();

        // ADD r3, r1, r2
        // r1 = 0x7FFFFFFF (MAX_I32), r2 = 1 -> Overflow
        exec.core.write_gpr(1, 0x7FFFFFFF);
        exec.core.write_gpr(2, 1);
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_ADD);
        { let _s = exec.exec(instr); assert!(_s & EXEC_IS_EXCEPTION != 0, "Expected overflow exception"); assert_eq!((_s >> 2) & 0x1F, EXC_OV); }
    }

    #[test]
    fn test_sub_subu() {
        let (mut exec, _) = create_executor();

        // SUB r3, r1, r2
        // r1 = 30, r2 = 20 -> r3 = 10
        exec.core.write_gpr(1, 30);
        exec.core.write_gpr(2, 20);
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_SUB);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 10);

        // SUBU r3, r1, r2
        // r1 = 10, r2 = 20 -> r3 = -10 (0xFFFFFFF6)
        exec.core.write_gpr(1, 10);
        exec.core.write_gpr(2, 20);
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_SUBU);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        // Result is 32-bit wrapping sub: 10 - 20 = -10. Sign extended to 64-bit: -10 (0xFFFFFFFFFFFFFFF6)
        assert_eq!(exec.core.read_gpr(3) as i64, -10);
    }

    #[test]
    fn test_addi_addiu() {
        let (mut exec, _) = create_executor();

        // ADDI r2, r1, -5
        // r1 = 10 -> r2 = 5
        exec.core.write_gpr(1, 10);
        let instr = make_i(OP_ADDI, 1, 2, (-5i16) as u16);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 5);

        // ADDIU r2, r1, 0xFFFF (-1 signed)
        // r1 = 10 -> r2 = 9
        exec.core.write_gpr(1, 10);
        let instr = make_i(OP_ADDIU, 1, 2, 0xFFFF);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 9);
        
        // ADDIU sign extension check
        // r1 = 0, imm = -1 -> r2 = -1 (0xFFFFFFFFFFFFFFFF)
        exec.core.write_gpr(1, 0);
        let instr = make_i(OP_ADDIU, 1, 2, 0xFFFF);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0xFFFFFFFFFFFFFFFF);
    }

    #[test]
    fn test_logical() {
        let (mut exec, _) = create_executor();

        // AND r3, r1, r2
        exec.core.write_gpr(1, 0xF0F0F0F0);
        exec.core.write_gpr(2, 0x0F0F0F0F);
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_AND);
        exec.exec(instr);
        assert_eq!(exec.core.read_gpr(3), 0);

        // OR r3, r1, r2
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_OR);
        exec.exec(instr);
        assert_eq!(exec.core.read_gpr(3), 0xFFFFFFFF);

        // ANDI r2, r1, 0xFFFF
        exec.core.write_gpr(1, 0x12345678);
        let instr = make_i(OP_ANDI, 1, 2, 0xFFFF);
        exec.exec(instr);
        assert_eq!(exec.core.read_gpr(2), 0x5678);

        // LUI r1, 0x1234
        let instr = make_i(OP_LUI, 0, 1, 0x1234);
        exec.exec(instr);
        assert_eq!(exec.core.read_gpr(1), 0x12340000);
    }

    #[test]
    fn test_shifts() {
        let (mut exec, _) = create_executor();

        // SLL r2, r1, 4
        exec.core.write_gpr(1, 0x1);
        let instr = make_r(OP_SPECIAL, 0, 1, 2, 4, FUNCT_SLL);
        exec.exec(instr);
        assert_eq!(exec.core.read_gpr(2), 0x10);

        // SRA r2, r1, 4 (Arithmetic shift right, sign extend)
        exec.core.write_gpr(1, 0xFFFFFFF0); // -16 in 32-bit
        let instr = make_r(OP_SPECIAL, 0, 1, 2, 4, FUNCT_SRA);
        exec.exec(instr);
        // 0xFFFFFFF0 >> 4 = 0xFFFFFFFF (-1)
        assert_eq!(exec.core.read_gpr(2) as i32, -1);
    }

    #[test]
    fn test_branches() {
        let (mut exec, _) = create_executor();

        // BEQ r1, r2, offset
        // r1 == r2, taken
        exec.core.write_gpr(1, 10);
        exec.core.write_gpr(2, 10);
        exec.core.pc = 0x1000;
        let offset = 4; // Target = PC + 4 + 4*4 = 0x1000 + 4 + 16 = 0x1014
        let instr = make_i(OP_BEQ, 1, 2, offset as u16);
        
        { let _s = exec.exec(instr); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x1014); }

        // Reset executor for independent test
        let (mut exec, _) = create_executor();
        exec.core.write_gpr(1, 10);
        exec.core.write_gpr(2, 10);
        exec.core.pc = 0x1000;

        // BNE r1, r2, offset
        // r1 == r2, not taken — the delay slot still always executes per
        // spec (see test_branch_not_taken_delay_slot_sets_cause_bd for why
        // this matters beyond just where PC ends up): in_delay_slot must be
        // set with delay_slot_target = pc+8, exactly like the taken case
        // above but with the fixed not-taken target instead of a computed one.
        let instr = make_i(OP_BNE, 1, 2, offset as u16);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert!(exec.core.in_delay_slot, "not-taken branch must still arm the delay slot (spec: BD always executes)");
        assert_eq!(exec.core.delay_slot_target, 0x1008, "not-taken target is pc+8, not a computed branch target");
        assert_eq!(exec.core.pc, 0x1004); // PC advanced by 4 to the delay slot itself, same as the taken case
    }

    /// MIPS spec: the instruction after a branch always executes, taken or
    /// not — and if that delay-slot instruction itself faults, the exception
    /// must report Cause.BD=1 and EPC pointing at the *branch*, not the
    /// delay-slot instruction (so the OS handler can back up and re-execute
    /// the branch+slot pair together, per the ISA's exception-delivery rules
    /// for branch delay slots). Before this was fixed, a not-taken branch's
    /// delay slot ran with `in_delay_slot == false` (indistinguishable from
    /// an ordinary sequential instruction), so a fault there got Cause.BD=0
    /// and EPC pointing straight at the delay-slot instruction — wrong per
    /// spec even though the common case (no fault in the slot) executed with
    /// the correct final PC either way.
    #[test]
    fn test_branch_not_taken_delay_slot_sets_cause_bd() {
        let (mut exec, mem) = create_executor();
        // kseg0: PassthroughTlb masks to a physical addr (virt & 0x1FFFFFFF)
        // before MockMemory ever sees it, so the mock's writes below must
        // target that physical address, not pc_base directly (MockMemory's
        // flat map has no translation of its own — see
        // test_addr_calc_load_store_fusion_fault's identical pc_base/`0`
        // split for the same reason).
        let pc_base: u64 = 0xFFFFFFFF80001000;
        let phys_base: u64 = pc_base & 0x1FFFFFFF;

        // BEQ r1, r2 (unequal -> not taken), delay slot LW v0, 1(t0) (misaligned -> faults)
        exec.core.write_gpr(1, 1);
        exec.core.write_gpr(2, 2);
        exec.core.write_gpr(8, 0x3000); // t0, base for the misaligned load
        exec.core.pc = pc_base;
        let beq = make_i(OP_BEQ, 1, 2, 4);
        let lw = make_i(OP_LW as u32, 8, 2, 1); // odd offset misaligns any word base
        mem.set_word(phys_base, beq);
        mem.set_word(phys_base + 4, lw);

        let s1 = exec.step_int();
        assert_eq!(s1, EXEC_COMPLETE, "not-taken BEQ itself must not fault");
        assert!(exec.core.in_delay_slot, "not-taken branch must still arm the delay slot");
        assert_eq!(exec.core.pc, pc_base + 4, "pc must sit at the delay-slot instruction");

        let s2 = exec.step_int();
        assert!(s2 & EXEC_IS_EXCEPTION != 0, "misaligned load in the delay slot must fault");
        assert_eq!((s2 >> 2) & 0x1F, EXC_ADEL);
        assert_ne!(exec.core.cp0_cause & crate::mips_core::CAUSE_BD, 0,
            "Cause.BD must be set: the fault happened in a delay slot, taken or not");
        assert_eq!(exec.core.cp0_epc, pc_base,
            "EPC must point at the branch itself (pc_base), not the delay-slot instruction (pc_base+4)");
    }

    #[test]
    fn test_jumps() {
        let (mut exec, _) = create_executor();

        exec.core.pc = 0x1000;
        // J target
        let target = 0x100;
        let instr = make_j(OP_J, target);
        // Target address = (PC+4) & 0xF0000000 | (target << 2)
        // 0x1004 & ... | 0x400 = 0x400
        { let _s = exec.exec(instr); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x400); }

        // Reset executor for independent test
        let (mut exec, _) = create_executor();
        exec.core.pc = 0x1000;

        // JAL target
        let instr = make_j(OP_JAL, target);
        { let _s = exec.exec(instr); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x400); assert_eq!(exec.core.read_gpr(31), 0x1008); }
    }

    #[test]
    fn test_load_store() {
        let (mut exec, mem) = create_executor();
        
        // Initialize memory
        // 0x1000: 0x11223344
        mem.set_word(0x1000, 0x11223344);
        // 0x1004: 0x8899AABB (Negative signed values)
        mem.set_word(0x1004, 0x8899AABB);

        exec.core.write_gpr(2, 0x1000); // Base address

        // LB r1, 0(r2) -> 0x11
        let instr = make_i(OP_LB, 2, 1, 0);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 0x11);

        // LB r1, 1(r2) -> 0x22
        let instr = make_i(OP_LB, 2, 1, 1);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 0x22);

        // LB r1, 4(r2) -> 0x88 (signed -120 -> 0xFFFFFFFFFFFFFF88)
        let instr = make_i(OP_LB, 2, 1, 4);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1) as i64, -120);

        // LBU r1, 4(r2) -> 0x88 (unsigned 136 -> 0x0000000000000088)
        let instr = make_i(OP_LBU, 2, 1, 4);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 0x88);

        // LH r1, 0(r2) -> 0x1122
        let instr = make_i(OP_LH, 2, 1, 0);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 0x1122);

        // LH r1, 4(r2) -> 0x8899 (signed -> sign extended)
        let instr = make_i(OP_LH, 2, 1, 4);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        // 0x8899 = -30567. Sign extended: 0xFFFFFFFFFFFF8899
        assert_eq!(exec.core.read_gpr(1) as i64, -30567);

        // LHU r1, 4(r2) -> 0x8899 (unsigned)
        let instr = make_i(OP_LHU, 2, 1, 4);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 0x8899);

        // LW r1, 0(r2) -> 0x11223344
        let instr = make_i(OP_LW, 2, 1, 0);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 0x11223344);

        // SW r1, 4(r2)
        exec.core.write_gpr(1, 0xDEADBEEF);
        let instr = make_i(OP_SW, 2, 1, 4);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        
        // Check memory
        assert_eq!(mem.get_word(0x1004), 0xDEADBEEF);

        // SB r1, 8(r2) -> Store 0xEF (LSB of 0xDEADBEEF) to 0x1008
        let instr = make_i(OP_SB, 2, 1, 8);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(mem.get_byte(0x1008), 0xEF);

        // SH r1, 10(r2) -> Store 0xBEEF to 0x100A
        let instr = make_i(OP_SH, 2, 1, 10);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(mem.get_byte(0x100A), 0xBE);
        assert_eq!(mem.get_byte(0x100B), 0xEF);
    }

    #[test]
    fn test_load_store_pc_advances() {
        // Regression test: every load/store dispatch-target handler must
        // advance PC by 4 on a normal (non-faulting) completion. Handlers
        // that route their status through a shared finish/complete helper
        // (e.g. store instructions returning write_data's raw status) can
        // silently skip the PC-advancing step if that helper only special-
        // cases the exception path and forgets the plain-EXEC_COMPLETE path
        // — that exact bug (finish_status not calling handle_exec_complete)
        // let every SB/SH/SW/SD/SWL/SWR/SDL/SDR/SWC1/SDC1 instruction execute
        // successfully while leaving PC frozen, an infinite loop on real
        // firmware. Covers stores, loads, and the unaligned-access family,
        // since each uses a different internal plumbing path.
        let (mut exec, _mem) = create_executor();
        exec.core.cp0_status |= STATUS_KX; // enable 64-bit ops (SD/LD/SDL/SDR)
        exec.core.write_gpr(2, 0x2000); // base register for all accesses below
        exec.core.write_gpr(1, 0x1122334455667788); // value register

        let ops: &[(&str, u32)] = &[
            ("SB", OP_SB), ("SH", OP_SH), ("SW", OP_SW), ("SD", OP_SD),
            ("SWL", OP_SWL), ("SWR", OP_SWR), ("SDL", OP_SDL), ("SDR", OP_SDR),
            ("LB", OP_LB), ("LBU", OP_LBU), ("LH", OP_LH), ("LHU", OP_LHU),
            ("LW", OP_LW), ("LWU", OP_LWU), ("LD", OP_LD),
            ("LWL", OP_LWL), ("LWR", OP_LWR), ("LDL", OP_LDL), ("LDR", OP_LDR),
        ];
        for (name, op) in ops {
            let pc_before = exec.core.pc;
            let instr = make_i(*op, 2, 1, 0);
            let status = exec.exec(instr);
            assert_eq!(status, EXEC_COMPLETE, "{name} should report EXEC_COMPLETE");
            assert_eq!(exec.core.pc, pc_before.wrapping_add(4), "{name} must advance PC by 4");
        }
    }

    #[test]
    fn test_load_store_64bit() {
        let (mut exec, mem) = create_executor();

        // Enable 64-bit Kernel Mode
        exec.core.cp0_status |= STATUS_KX;

        // SD r1, 0(r2)
        exec.core.write_gpr(1, 0x1122334455667788);
        exec.core.write_gpr(2, 0x2000);
        let instr = make_i(OP_SD, 2, 1, 0);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(mem.get_double(0x2000), 0x1122334455667788);

        // LD r3, 0(r2)
        let instr = make_i(OP_LD, 2, 3, 0);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0x1122334455667788);

        // LWU r4, 0(r2) -> 0x11223344 (upper 32 bits zeroed)
        // Note: MIPS is Big Endian. 0x2000 contains 0x11, 0x2001 contains 0x22...
        // LWU loads word at 0x2000: 0x11223344.
        let instr = make_i(OP_LWU, 2, 4, 0);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(4), 0x11223344);
    }

    #[test]
    fn test_load_store_32bit_allowed() {
        let (mut exec, mem) = create_executor();

        // Default is 32-bit mode (KX=0)

        // LD r1, 0(r2) -> Should work
        mem.set_double(0x1000, 0x1122334455667788);
        exec.core.write_gpr(2, 0x1000);
        let instr = make_i(OP_LD, 2, 1, 0);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 0x1122334455667788);

        // SD r1, 0(r2) -> Should work
        exec.core.write_gpr(1, 0xAABBCCDDEEFF0011);
        exec.core.write_gpr(2, 0x2000);
        let instr = make_i(OP_SD, 2, 1, 0);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(mem.get_double(0x2000), 0xAABBCCDDEEFF0011);

        // LWU r1, 0(r2) -> Should work
        mem.set_word(0x3000, 0x87654321);
        exec.core.write_gpr(2, 0x3000);
        let instr = make_i(OP_LWU, 2, 1, 0);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 0x0000000087654321);
    }

    #[test]
    fn test_cache_attributes() {
        let (mut exec, mem) = create_executor();

        // Set PC to KSEG0 (cached kernel segment)
        // In 32-bit mode, addresses must be sign-extended to 64 bits
        exec.core.pc = 0x80000000u32 as i32 as i64 as u64; // Sign-extend from 32 bits
        mem.set_word(0x00000000, 0); // NOP at physical address 0

        exec.step_int();
        // KSEG0 is cacheable. NoOpCache returns 0.
        // If it went to memory (uncached), we would get NOP (0).
        // This test is hard to verify without inspecting internal routing or using a cache that returns different data.
        // But we can verify KSEG1 (uncached)

        // Set PC to KSEG1 (uncached kernel segment)
        exec.core.pc = 0xA0000000u32 as i32 as i64 as u64; // Sign-extend from 32 bits
        // Use COP2 instruction (0x12 << 26) which is Reserved
        mem.set_word(0x00000000, 0x48000000);

        // Fetch from KSEG1 should hit memory
        let status = exec.step_int();
        assert_eq!(status, exec_exception(EXC_RI)); // 0x12345678 is likely RI
    }

    #[test]
    fn test_xkphys_access_mode_switch() {
        let (mut exec, mem) = create_executor();

        // Physical address 0x10000000: NOP (0)
        mem.set_word(0x10000000, 0);

        // XKPHYS address: 0x90000000_10000000
        // - Top bits (63:62) = 10 (binary) = 2 -> XKPHYS
        // - Cache bits (61:59) = 010 (binary) = 2 -> Uncached
        // - Phys addr = 0x10000000
        let xkphys_addr = 0x90000000_10000000u64;
        exec.core.pc = xkphys_addr;

        // 1. Test in 32-bit mode (default, STATUS_KX=0)
        // Upper 32 bits of PC are ignored for translation; low 32 bits = 0x10000000.
        // NOP executes successfully, PC advances by 4 (full 64-bit PC increments).
        assert_eq!(exec.step_int(), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, xkphys_addr + 4);

        // 2. Switch to 64-bit kernel mode and restore PC.
        exec.core.pc = xkphys_addr;
        exec.core.cp0_status = STATUS_KX; // 64-bit kernel mode, no EXL/ERL/BEV

        // 3. Test in 64-bit mode: XKPHYS translates to phys 0x10000000, NOP executes.
        assert_eq!(exec.step_int(), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, xkphys_addr + 4);
    }

    #[test]
    fn test_lui_ori_sequence() {
        let (mut exec, _) = create_executor();

        // Load 0x12345678 into r1
        // LUI r1, 0x1234
        // ORI r1, r1, 0x5678

        let instr1 = make_i(OP_LUI, 0, 1, 0x1234);
        assert_eq!(exec.exec(instr1), EXEC_COMPLETE);
        // LUI sign-extends the 32-bit result. 0x12340000 is positive, so high bits are 0.
        assert_eq!(exec.core.read_gpr(1), 0x12340000);

        let instr2 = make_i(OP_ORI, 1, 1, 0x5678);
        assert_eq!(exec.exec(instr2), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 0x12345678);
    }

    #[test]
    fn test_tlb_instructions() {
        // MipsExecutor<MipsTlb> is ~512KB due to the inline vmap, and debug
        // builds don't elide the by-value moves through default()/new()/Self{},
        // so several copies can be live on the stack at once. 4MB proved
        // insufficient (overflowed in CI); use 16MB for headroom.
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| { test_tlb_instructions_inner(); })
            .unwrap().join().unwrap();
    }
    fn test_tlb_instructions_inner() {
        use crate::mips_tlb::MipsTlb;

        let (mut exec, _) = create_executor_with_tlb(MipsTlb::default());

        // Test TLBWI - Write Indexed TLB Entry
        // Setup CP0 registers for a TLB entry
        exec.core.cp0_index = 5;  // Write to index 5
        exec.core.cp0_pagemask = 0;  // 4KB pages
        exec.core.cp0_entryhi = (0x100 << 13) | 10;  // VPN2=0x100, ASID=10
        exec.core.cp0_entrylo0 = (0x50 << 6) | (3 << 3) | (1 << 2) | (1 << 1);  // PFN=0x50, Cacheable, Dirty, Valid
        exec.core.cp0_entrylo1 = (0x51 << 6) | (2 << 3) | (0 << 2) | (1 << 1);  // PFN=0x51, Uncached, Not Dirty, Valid

        // Execute TLBWI instruction
        // COP0 with RS=TLB (0x10) and FUNCT=TLBWI (0x02)
        let tlbwi_instr = (OP_COP0 << 26) | (0x10 << 21) | 0x02;
        assert_eq!(exec.exec(tlbwi_instr), EXEC_COMPLETE);

        // Test TLBR - Read Indexed TLB Entry
        // Clear the CP0 registers first
        exec.core.cp0_entryhi = 0;
        exec.core.cp0_entrylo0 = 0;
        exec.core.cp0_entrylo1 = 0;
        exec.core.cp0_pagemask = 0;

        // Execute TLBR instruction to read back entry at index 5
        // COP0 with RS=TLB (0x10) and FUNCT=TLBR (0x01)
        let tlbr_instr = (OP_COP0 << 26) | (0x10 << 21) | 0x01;
        assert_eq!(exec.exec(tlbr_instr), EXEC_COMPLETE);

        // Verify the values were read back correctly
        assert_eq!(exec.core.cp0_pagemask, 0);
        assert_eq!(exec.core.cp0_entryhi, (0x100 << 13) | 10);
        assert_eq!(exec.core.cp0_entrylo0, (0x50 << 6) | (3 << 3) | (1 << 2) | (1 << 1));
        assert_eq!(exec.core.cp0_entrylo1, (0x51 << 6) | (2 << 3) | (0 << 2) | (1 << 1));

        // Test TLBP - Probe TLB for Matching Entry
        // Set EntryHi to search for the entry we just wrote
        exec.core.cp0_entryhi = (0x100 << 13) | 10;  // VPN2=0x100, ASID=10
        exec.core.cp0_index = 0xFFFFFFFF;  // Clear index

        // Execute TLBP instruction
        // COP0 with RS=TLB (0x10) and FUNCT=TLBP (0x08)
        let tlbp_instr = (OP_COP0 << 26) | (0x10 << 21) | 0x08;
        assert_eq!(exec.exec(tlbp_instr), EXEC_COMPLETE);

        // Verify index was set to 5 (the matching entry)
        assert_eq!(exec.core.cp0_index, 5);

        // Test TLBP with no match
        exec.core.cp0_entryhi = (0x200 << 13) | 10;  // Different VPN2, same ASID
        exec.core.cp0_index = 0;

        assert_eq!(exec.exec(tlbp_instr), EXEC_COMPLETE);

        // Verify P bit is set (bit 31)
        assert_eq!(exec.core.cp0_index & 0x80000000, 0x80000000);
    }

    #[cfg(feature = "tlbcheck")]
    #[test]
    fn test_tlbwi_breaks_on_duplicate_entry() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| { test_tlbwi_breaks_on_duplicate_entry_inner(); })
            .unwrap().join().unwrap();
    }
    #[cfg(feature = "tlbcheck")]
    fn test_tlbwi_breaks_on_duplicate_entry_inner() {
        use crate::mips_tlb::MipsTlb;

        let (mut exec, _) = create_executor_with_tlb(MipsTlb::default());
        let tlbwi_instr = (OP_COP0 << 26) | (0x10 << 21) | 0x02;

        // Write a valid entry at index 5: VPN2=0x100, ASID=10.
        exec.core.cp0_index = 5;
        exec.core.cp0_pagemask = 0;
        exec.core.cp0_entryhi = (0x100 << 13) | 10;
        exec.core.cp0_entrylo0 = (0x50 << 6) | (3 << 3) | (1 << 2) | (1 << 1);
        exec.core.cp0_entrylo1 = (0x51 << 6) | (3 << 3) | (1 << 2) | (1 << 1);
        assert_eq!(exec.exec(tlbwi_instr), EXEC_COMPLETE);

        // Write the exact same VPN2/ASID at a different index (6): this is
        // precisely the "duplicate TLB entry" condition IRIX itself panics
        // on. tlbcheck should catch it right here and return EXEC_BREAKPOINT
        // instead of silently continuing, with PC already retired past the
        // TLBWI (matching how other breakpoints leave PC at the next instr).
        let pc_before = exec.core.pc;
        exec.core.cp0_index = 6;
        assert_eq!(exec.exec(tlbwi_instr), EXEC_BREAKPOINT);
        assert_eq!(exec.core.pc, pc_before.wrapping_add(4));

        // The checker re-scans the whole TLB on every write, so the 5/6
        // duplicate keeps tripping EXEC_BREAKPOINT on any further write until
        // it's actually resolved — confirming it isn't a one-shot fluke.
        //
        // Note there is no way to *resolve* this duplicate from here without
        // hitting the separate vmap-dangling bug covered by
        // test_tlbcheck_detects_vmap_left_dangling_by_overlap_repair: any
        // further write to index 6 (moving it to a new VPN2, or clearing it
        // to invalid) calls vmap_erase(6), which wipes vmap[0x100] back to
        // VMAP_MISS and orphans index 5's still-valid claim — so this test
        // stops at "detected and not a one-shot fluke" rather than also
        // trying to demonstrate a clean recovery.
        exec.core.cp0_index = 7;
        exec.core.cp0_entryhi = (0x200 << 13) | 10;
        assert_eq!(exec.exec(tlbwi_instr), EXEC_BREAKPOINT);
    }

    // Regression test for a real vmap bug tlbcheck surfaced (now fixed via
    // MipsTlb::vmap_replace, see rules/testing/tlbcheck-vmap-dangling-slot-bug.md):
    // when two entries transiently share a VPN2 (e.g. mid TLBWR churn picks a
    // colliding victim index) and the later one then moves to a new VPN2,
    // the vmap must NOT simply blank the shared slot — it needs to re-derive
    // whether some *other* entry still legitimately covers it and restore
    // that claim. This drives exactly that sequence and confirms the TLB
    // ends up fully consistent again (no lingering EXEC_BREAKPOINT) once the
    // real duplicate is resolved, with entry 5's original mapping intact.
    #[cfg(feature = "tlbcheck")]
    #[test]
    fn test_tlbcheck_vmap_survives_overlap_repair() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| { test_tlbcheck_vmap_survives_overlap_repair_inner(); })
            .unwrap().join().unwrap();
    }
    #[cfg(feature = "tlbcheck")]
    fn test_tlbcheck_vmap_survives_overlap_repair_inner() {
        use crate::mips_tlb::MipsTlb;

        let (mut exec, _) = create_executor_with_tlb(MipsTlb::default());
        let tlbwi_instr = (OP_COP0 << 26) | (0x10 << 21) | 0x02;

        // Index 5: VPN2=0x100, ASID=10 (valid).
        exec.core.cp0_index = 5;
        exec.core.cp0_pagemask = 0;
        exec.core.cp0_entryhi = (0x100 << 13) | 10;
        exec.core.cp0_entrylo0 = (0x50 << 6) | (3 << 3) | (1 << 2) | (1 << 1);
        exec.core.cp0_entrylo1 = (0x51 << 6) | (3 << 3) | (1 << 2) | (1 << 1);
        assert_eq!(exec.exec(tlbwi_instr), EXEC_COMPLETE);

        // Index 6: same VPN2/ASID — a transient overlap, as ordinary TLBWR
        // replacement could produce. tlbcheck flags it (see the duplicate-
        // detection test); here we drive past it to observe the vmap fallout.
        exec.core.cp0_index = 6;
        assert_eq!(exec.exec(tlbwi_instr), EXEC_BREAKPOINT);

        // Index 6 moves on to a fresh VPN2, as a real replacement algorithm
        // would do on its next victim pick — index 5's still-valid mapping at
        // VPN2=0x100 is never touched. This resolves the duplicate, and with
        // vmap_replace's restore step, entry 5 correctly reclaims its slot
        // instead of being left dangling at VMAP_MISS — no more violations.
        exec.core.cp0_index = 6;
        exec.core.cp0_entryhi = (0x300 << 13) | 10;
        assert_eq!(exec.exec(tlbwi_instr), EXEC_COMPLETE);

        // Confirm entry 5 is genuinely still reachable through the vmap fast
        // path, not just "no violation reported" — translate() must still
        // find it.
        use crate::mips_tlb::{AccessType, Tlb, TlbResult};
        let va = 0x100u64 << 13;
        match exec.tlb.translate::<0>(va, 10, AccessType::Read) {
            TlbResult::Hit { phys_addr, .. } => {
                assert_eq!(phys_addr, 0x50 << 12, "expected entry 5's PFN, got phys_addr={:#x}", phys_addr);
            }
            other => panic!("expected Hit against entry 5's still-valid mapping, got {:?}", other),
        }
    }

    #[test]
    fn test_mfc0_mtc0() {
        let (mut exec, _) = create_executor();

        // Test MTC0 - Move to CP0
        // Load value 0x12345678 into r1
        exec.core.write_gpr(1, 0x12345678);

        // MTC0 r1, Index (CP0 reg 0)
        // COP0 with RS=MTC0 (0x04), RT=1, RD=0
        let mtc0_instr = (OP_COP0 << 26) | (0x04 << 21) | (1 << 16) | (0 << 11);
        assert_eq!(exec.exec(mtc0_instr), EXEC_COMPLETE);

        // Verify CP0.Index was set
        assert_eq!(exec.core.cp0_index, 0x12345678);

        // Test MFC0 - Move from CP0
        // Clear r2
        exec.core.write_gpr(2, 0);

        // MFC0 r2, Index (CP0 reg 0)
        // COP0 with RS=MFC0 (0x00), RT=2, RD=0
        let mfc0_instr = (OP_COP0 << 26) | (0x00 << 21) | (2 << 16) | (0 << 11);
        assert_eq!(exec.exec(mfc0_instr), EXEC_COMPLETE);

        // Verify r2 was loaded with Index value (sign-extended to 64 bits)
        assert_eq!(exec.core.read_gpr(2), 0x12345678);
    }

    #[test]
    fn test_eret() {
        let (mut exec, _) = create_executor();

        // Test ERET from exception level (EXL=1)
        exec.core.cp0_epc = 0xBFC00100;
        exec.core.cp0_status = 0x02;  // EXL bit set
        exec.core.pc = 0xBFC00200;

        // Execute ERET instruction
        // COP0 with RS=TLB (0x10) and FUNCT=ERET (0x18)
        // ERET sets PC directly (no delay slot).
        let eret_instr = (OP_COP0 << 26) | (0x10 << 21) | 0x18;
        assert_eq!(exec.exec(eret_instr), EXEC_COMPLETE);

        // Verify PC was restored from EPC and EXL was cleared
        assert_eq!(exec.core.pc, 0xBFC00100);
        assert_eq!(exec.core.cp0_status & 0x02, 0);  // EXL should be cleared

        // Test ERET from error level (ERL=1)
        exec.core.cp0_errorepc = 0xBFC00300;
        exec.core.cp0_status = 0x04;  // ERL bit set
        exec.core.pc = 0xBFC00400;

        assert_eq!(exec.exec(eret_instr), EXEC_COMPLETE);

        // Verify PC was restored from ErrorEPC and ERL was cleared
        assert_eq!(exec.core.pc, 0xBFC00300);
        assert_eq!(exec.core.cp0_status & 0x04, 0);  // ERL should be cleared
    }

    #[test]
    fn test_branch_likely_taken() {
        let (mut exec, _) = create_executor();

        // Test BEQL (Branch on Equal Likely) - TAKEN
        exec.core.pc = 0x1000;
        exec.core.write_gpr(1, 42);
        exec.core.write_gpr(2, 42);

        // BEQL r1, r2, offset=4 (branch to 0x1000 + 4 + 4*4 = 0x1014)
        let beql_instr = make_i(OP_BEQL, 1, 2, 4);
        { let _s = exec.exec(beql_instr); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x1014 ); };

        // PC should be at 0x1004 (delay slot)
        assert_eq!(exec.core.pc, 0x1004);

        // Now execute a NOP in the delay slot
        let nop = make_r(0, 0, 0, 0, 0, 0);
        assert_eq!(exec.exec(nop), EXEC_COMPLETE);

        // After delay slot, PC should be at target
        assert_eq!(exec.core.pc, 0x1014);

        // Test BNEL (Branch on Not Equal Likely) - TAKEN
        exec.core.pc = 0x2000;
        exec.core.write_gpr(1, 42);
        exec.core.write_gpr(2, 100);

        // BNEL r1, r2, offset=8
        let bnel_instr = make_i(OP_BNEL, 1, 2, 8);
        { let _s = exec.exec(bnel_instr); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x2024 ); };

        // Execute delay slot NOP
        assert_eq!(exec.exec(nop), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x2024);

        // Test BLEZL (Branch on Less than or Equal to Zero Likely) - TAKEN
        exec.core.pc = 0x3000;
        exec.core.write_gpr(1, 0);

        let blezl_instr = make_i(OP_BLEZL, 1, 0, 2);
        { let _s = exec.exec(blezl_instr); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x300C ); };

        assert_eq!(exec.exec(nop), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x300C);

        // Test BGTZL (Branch on Greater Than Zero Likely) - TAKEN
        exec.core.pc = 0x4000;
        exec.core.write_gpr(1, 100);

        let bgtzl_instr = make_i(OP_BGTZL, 1, 0, 2);
        { let _s = exec.exec(bgtzl_instr); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x400C ); };

        assert_eq!(exec.exec(nop), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x400C);
    }

    #[test]
    fn test_branch_likely_not_taken() {
        let (mut exec, _) = create_executor();

        // Test BEQL (Branch on Equal Likely) - NOT TAKEN
        // The key test: delay slot should be NULLIFIED (not executed)
        exec.core.pc = 0x1000;
        exec.core.write_gpr(1, 42);
        exec.core.write_gpr(2, 100);
        exec.core.write_gpr(3, 0);  // r3 = 0

        // BEQL r1, r2, offset=4 (won't branch because r1 != r2)
        let beql_instr = make_i(OP_BEQL, 1, 2, 4);
        assert_eq!(exec.exec(beql_instr), EXEC_COMPLETE);

        // PC should skip both the branch instruction AND the delay slot (PC + 8)
        assert_eq!(exec.core.pc, 0x1008);

        // Verify that if we had an instruction in the delay slot that modifies r3,
        // it would NOT execute. Let's test this by setting up the scenario again:
        exec.core.pc = 0x2000;
        exec.core.write_gpr(1, 42);
        exec.core.write_gpr(2, 100);
        exec.core.write_gpr(3, 0);  // r3 = 0

        // BNEL r1, r2, offset=4 (won't branch because we'll make them equal)
        exec.core.write_gpr(1, 42);
        exec.core.write_gpr(2, 42);
        let bnel_instr = make_i(OP_BNEL, 1, 2, 4);
        assert_eq!(exec.exec(bnel_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x2008);

        // Test BLEZL - NOT TAKEN
        exec.core.pc = 0x3000;
        exec.core.write_gpr(1, 100);  // positive value

        let blezl_instr = make_i(OP_BLEZL, 1, 0, 2);
        assert_eq!(exec.exec(blezl_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x3008);

        // Test BGTZL - NOT TAKEN
        exec.core.pc = 0x4000;
        exec.core.write_gpr(1, 0);  // zero

        let bgtzl_instr = make_i(OP_BGTZL, 1, 0, 2);
        assert_eq!(exec.exec(bgtzl_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x4008);

        // Test BGTZL with negative - NOT TAKEN
        exec.core.pc = 0x5000;
        exec.core.write_gpr(1, -10i64 as u64);

        let bgtzl_instr = make_i(OP_BGTZL, 1, 0, 2);
        assert_eq!(exec.exec(bgtzl_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x5008);
    }

    #[test]
    fn test_branch_likely_regimm() {
        let (mut exec, _) = create_executor();

        // Test BLTZL (Branch on Less Than Zero Likely) - TAKEN
        exec.core.pc = 0x1000;
        exec.core.write_gpr(1, -10i64 as u64);

        let bltzl_instr = make_i(OP_REGIMM, 1, RT_BLTZL, 4);
        { let _s = exec.exec(bltzl_instr); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x1014 ); };

        let nop = make_r(0, 0, 0, 0, 0, 0);
        assert_eq!(exec.exec(nop), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x1014);

        // Test BLTZL - NOT TAKEN
        exec.core.pc = 0x2000;
        exec.core.write_gpr(1, 100);

        let bltzl_instr = make_i(OP_REGIMM, 1, RT_BLTZL, 4);
        assert_eq!(exec.exec(bltzl_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x2008);

        // Test BGEZL (Branch on Greater Than or Equal to Zero Likely) - TAKEN
        exec.core.pc = 0x3000;
        exec.core.write_gpr(1, 0);

        let bgezl_instr = make_i(OP_REGIMM, 1, RT_BGEZL, 4);
        { let _s = exec.exec(bgezl_instr); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x3014 ); };

        assert_eq!(exec.exec(nop), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x3014);

        // Test BGEZL - NOT TAKEN
        exec.core.pc = 0x4000;
        exec.core.write_gpr(1, -10i64 as u64);

        let bgezl_instr = make_i(OP_REGIMM, 1, RT_BGEZL, 4);
        assert_eq!(exec.exec(bgezl_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x4008);

        // Test BLTZALL (Branch on Less Than Zero And Link Likely) - TAKEN
        exec.core.pc = 0x5000;
        exec.core.write_gpr(1, -10i64 as u64);
        exec.core.write_gpr(31, 0);

        let bltzall_instr = make_i(OP_REGIMM, 1, RT_BLTZALL, 4);
        { let _s = exec.exec(bltzall_instr); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x5014 ); };
        assert_eq!(exec.core.read_gpr(31), 0x5008);  // Return address saved

        assert_eq!(exec.exec(nop), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x5014);

        // Test BGEZALL (Branch on Greater Than or Equal to Zero And Link Likely) - NOT TAKEN
        exec.core.pc = 0x6000;
        exec.core.write_gpr(1, -10i64 as u64);
        exec.core.write_gpr(31, 0);

        let bgezall_instr = make_i(OP_REGIMM, 1, RT_BGEZALL, 4);
        assert_eq!(exec.exec(bgezall_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(31), 0x6008);  // Return address still saved even though not taken
        assert_eq!(exec.core.pc, 0x6008);
    }

    #[test]
    fn test_unaligned_word_load_store() {
        let (mut exec, mem) = create_executor();

        // Set up 64-bit kernel mode
        exec.core.cp0_status |= STATUS_KX;
        exec.core.pc = 0xFFFFFFFF_A0001000;

        // Set up test data in memory at address 0x1000-0x1007
        // Memory layout (big-endian):
        // 0x1000: 0x11
        // 0x1001: 0x22
        // 0x1002: 0x33
        // 0x1003: 0x44
        // 0x1004: 0x55
        // 0x1005: 0x66
        // 0x1006: 0x77
        // 0x1007: 0x88
        mem.set_word(0x1000, 0x11223344);
        mem.set_word(0x1004, 0x55667788);

        // Test LWL and LWR to load unaligned word at 0x1001-0x1004
        // Expected result: 0x22334455

        // Initialize rt with pattern
        exec.core.write_gpr(2, 0xDEADBEEF);

        // LWL $2, 0x1001($0)  - loads bytes 0x22, 0x33, 0x44
        let lwl_instr = make_i(OP_LWL, 0, 2, 0x1001);
        assert_eq!(exec.exec(lwl_instr), EXEC_COMPLETE);
        // After LWL from offset 1: loads 3 bytes, result should be 0x223344EF
        assert_eq!(exec.core.read_gpr(2) as u32, 0x223344EF);

        // LWR $2, 0x1004($0)  - loads byte 0x55
        let lwr_instr = make_i(OP_LWR, 0, 2, 0x1004);
        assert_eq!(exec.exec(lwr_instr), EXEC_COMPLETE);
        // After LWR from offset 0: loads 1 byte, result should be 0x22334455
        assert_eq!(exec.core.read_gpr(2) as u32, 0x22334455);

        // Test SWL and SWR to store unaligned word
        // Store 0xAABBCCDD at 0x1009-0x100C

        exec.core.write_gpr(3, 0xAABBCCDD);

        // SWL $3, 0x1009($0)  - stores 0xAA, 0xBB, 0xCC
        let swl_instr = make_i(OP_SWL, 0, 3, 0x1009);
        assert_eq!(exec.exec(swl_instr), EXEC_COMPLETE);

        // SWR $3, 0x100C($0)  - stores 0xDD
        let swr_instr = make_i(OP_SWR, 0, 3, 0x100C);
        assert_eq!(exec.exec(swr_instr), EXEC_COMPLETE);

        // Verify the stored data
        assert_eq!(mem.get_byte(0x1009), 0xAA);
        assert_eq!(mem.get_byte(0x100A), 0xBB);
        assert_eq!(mem.get_byte(0x100B), 0xCC);
        assert_eq!(mem.get_byte(0x100C), 0xDD);
    }

    #[test]
    fn test_unaligned_word_all_offsets() {
        let (mut exec, mem) = create_executor();

        exec.core.cp0_status |= STATUS_KX;
        exec.core.pc = 0xFFFFFFFF_A0001000;

        // Test all 4 byte offsets for word unaligned access
        mem.set_word(0x2000, 0x11223344);
        mem.set_word(0x2004, 0x55667788);

        // Offset 0: LWL loads all 4 bytes
        exec.core.write_gpr(4, 0x00000000);
        let lwl = make_i(OP_LWL, 0, 4, 0x2000);
        exec.exec(lwl);
        assert_eq!(exec.core.read_gpr(4) as u32, 0x11223344);

        // Offset 1: LWL loads 3 bytes
        exec.core.write_gpr(4, 0x000000FF);
        let lwl = make_i(OP_LWL, 0, 4, 0x2001);
        exec.exec(lwl);
        assert_eq!(exec.core.read_gpr(4) as u32, 0x223344FF);

        // Offset 2: LWL loads 2 bytes
        exec.core.write_gpr(4, 0x0000FFFF);
        let lwl = make_i(OP_LWL, 0, 4, 0x2002);
        exec.exec(lwl);
        assert_eq!(exec.core.read_gpr(4) as u32, 0x3344FFFF);

        // Offset 3: LWL loads 1 byte
        exec.core.write_gpr(4, 0x00FFFFFF);
        let lwl = make_i(OP_LWL, 0, 4, 0x2003);
        exec.exec(lwl);
        assert_eq!(exec.core.read_gpr(4) as u32, 0x44FFFFFF);

        // Offset 0: LWR loads 1 byte
        exec.core.write_gpr(4, 0xFFFFFF00);
        let lwr = make_i(OP_LWR, 0, 4, 0x2000);
        exec.exec(lwr);
        assert_eq!(exec.core.read_gpr(4) as u32, 0xFFFFFF11);

        // Offset 1: LWR loads 2 bytes
        exec.core.write_gpr(4, 0xFFFF0000);
        let lwr = make_i(OP_LWR, 0, 4, 0x2001);
        exec.exec(lwr);
        assert_eq!(exec.core.read_gpr(4) as u32, 0xFFFF1122);

        // Offset 2: LWR loads 3 bytes
        exec.core.write_gpr(4, 0xFF000000);
        let lwr = make_i(OP_LWR, 0, 4, 0x2002);
        exec.exec(lwr);
        assert_eq!(exec.core.read_gpr(4) as u32, 0xFF112233);

        // Offset 3: LWR loads all 4 bytes
        exec.core.write_gpr(4, 0x00000000);
        let lwr = make_i(OP_LWR, 0, 4, 0x2003);
        exec.exec(lwr);
        assert_eq!(exec.core.read_gpr(4) as u32, 0x11223344);
    }

    #[test]
    fn test_unaligned_doubleword_load_store() {
        let (mut exec, mem) = create_executor();

        // Set up 64-bit kernel mode
        exec.core.cp0_status |= STATUS_KX;
        exec.core.pc = 0xFFFFFFFF_A0001000;

        // Set up test data in memory
        mem.set_double(0x3000, 0x1122334455667788);
        mem.set_double(0x3008, 0x99AABBCCDDEEFF00);

        // Test LDL and LDR to load unaligned doubleword at 0x3003-0x300A
        // LDL at 0x3003 (offset 3): loads from byte 3-7 (5 bytes: 0x44,0x55,0x66,0x77,0x88)
        // LDR at 0x300A (offset 2): loads from byte 0-2 (3 bytes: 0x99,0xAA,0xBB)
        // Combined result: 0x44556677_8899AABB

        exec.core.write_gpr(5, 0xFFFFFFFF_FFFFFFFF);

        // LDL $5, 0x3003($0)  - loads 5 bytes from offset 3
        let ldl_instr = make_i(OP_LDL, 0, 5, 0x3003);
        assert_eq!(exec.exec(ldl_instr), EXEC_COMPLETE);
        // After LDL: upper 5 bytes loaded, lower 3 preserved
        assert_eq!(exec.core.read_gpr(5), 0x44556677_88FFFFFF);

        // LDR $5, 0x300A($0)  - loads 3 bytes from offset 2
        let ldr_instr = make_i(OP_LDR, 0, 5, 0x300A);
        assert_eq!(exec.exec(ldr_instr), EXEC_COMPLETE);
        // After LDR: upper 5 bytes preserved, lower 3 loaded
        assert_eq!(exec.core.read_gpr(5), 0x44556677_8899AABB);

        // Test SDL and SDR to store unaligned doubleword
        // Value: 0xFEDCBA9876543210
        // SDL at 0x3013 (aligned 0x3010, offset 3): stores upper portion (5 bytes: FE,DC,BA,98,76)
        // SDR at 0x301A (aligned 0x3018, offset 2): stores lower portion (3 bytes: 54,32,10)
        exec.core.write_gpr(6, 0xFEDCBA98_76543210);

        // SDL $6, 0x3013($0)  - stores 5 bytes at offset 3 of doubleword 0x3010
        let sdl_instr = make_i(OP_SDL, 0, 6, 0x3013);
        assert_eq!(exec.exec(sdl_instr), EXEC_COMPLETE);

        // SDR $6, 0x301A($0)  - stores 3 bytes at offset 2 of doubleword 0x3018
        let sdr_instr = make_i(OP_SDR, 0, 6, 0x301A);
        assert_eq!(exec.exec(sdr_instr), EXEC_COMPLETE);

        // Verify the stored data
        // SDL at 0x3013-0x3017: FE, DC, BA, 98, 76
        assert_eq!(mem.get_byte(0x3013), 0xFE);
        assert_eq!(mem.get_byte(0x3014), 0xDC);
        assert_eq!(mem.get_byte(0x3015), 0xBA);
        assert_eq!(mem.get_byte(0x3016), 0x98);
        assert_eq!(mem.get_byte(0x3017), 0x76);
        // SDR at 0x3018-0x301A: 54, 32, 10
        assert_eq!(mem.get_byte(0x3018), 0x54);
        assert_eq!(mem.get_byte(0x3019), 0x32);
        assert_eq!(mem.get_byte(0x301A), 0x10);
    }

    #[test]
    fn test_unaligned_complete_word_sequence() {
        // Test a complete unaligned word load using LWL followed by LWR
        let (mut exec, mem) = create_executor();

        exec.core.cp0_status |= STATUS_KX;
        exec.core.pc = 0xFFFFFFFF_A0001000;

        // Place test value 0x12345678 at unaligned address 0x4001
        mem.set_word(0x4000, 0xAA123456);
        mem.set_word(0x4004, 0x78BBBBBB);

        // Load word at 0x4001 using LWL + LWR
        exec.core.write_gpr(7, 0);

        let lwl = make_i(OP_LWL, 0, 7, 0x4001);
        exec.exec(lwl);

        let lwr = make_i(OP_LWR, 0, 7, 0x4004);
        exec.exec(lwr);

        assert_eq!(exec.core.read_gpr(7) as u32, 0x12345678);

        // Store word 0xABCDEF01 at unaligned address 0x5002 using SWL + SWR
        exec.core.write_gpr(8, 0xABCDEF01);

        let swl = make_i(OP_SWL, 0, 8, 0x5002);
        exec.exec(swl);

        let swr = make_i(OP_SWR, 0, 8, 0x5005);
        exec.exec(swr);

        // Verify stored bytes
        assert_eq!(mem.get_byte(0x5002), 0xAB);
        assert_eq!(mem.get_byte(0x5003), 0xCD);
        assert_eq!(mem.get_byte(0x5004), 0xEF);
        assert_eq!(mem.get_byte(0x5005), 0x01);
    }

    #[test]
    fn test_variable_shifts() {
        let (mut exec, _) = create_executor();

        // Test SLLV (Shift Left Logical Variable)
        exec.core.write_gpr(1, 4);  // Shift amount in rs
        exec.core.write_gpr(2, 0x1); // Value to shift in rt
        let sllv_instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_SLLV);
        assert_eq!(exec.exec(sllv_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0x10); // 0x1 << 4 = 0x10

        // Test SLLV with shift amount > 31 (should use only low 5 bits)
        exec.core.write_gpr(1, 36);  // 36 & 0x1F = 4
        exec.core.write_gpr(2, 0x1);
        assert_eq!(exec.exec(sllv_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0x10); // Still shifts by 4

        // Test SRLV (Shift Right Logical Variable)
        exec.core.write_gpr(1, 4);  // Shift amount
        exec.core.write_gpr(2, 0x80000000); // Value with high bit set
        let srlv_instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_SRLV);
        assert_eq!(exec.exec(srlv_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3) as u32, 0x08000000); // Logical shift fills with 0

        // Test SRAV (Shift Right Arithmetic Variable)
        exec.core.write_gpr(1, 4);  // Shift amount
        exec.core.write_gpr(2, 0xFFFFFFF0); // Negative value in 32-bit
        let srav_instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_SRAV);
        assert_eq!(exec.exec(srav_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3) as i32, -1); // Arithmetic shift preserves sign
    }

    #[test]
    fn test_hi_lo_moves() {
        let (mut exec, _) = create_executor();

        // Test MTHI (Move To HI)
        exec.core.write_gpr(1, 0x12345678_9ABCDEF0);
        let mthi_instr = make_r(OP_SPECIAL, 1, 0, 0, 0, FUNCT_MTHI);
        assert_eq!(exec.exec(mthi_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.hi, 0x12345678_9ABCDEF0);

        // Test MFHI (Move From HI)
        exec.core.write_gpr(2, 0);
        let mfhi_instr = make_r(OP_SPECIAL, 0, 0, 2, 0, FUNCT_MFHI);
        assert_eq!(exec.exec(mfhi_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0x12345678_9ABCDEF0);

        // Test MTLO (Move To LO)
        exec.core.write_gpr(3, 0xFEDCBA98_76543210);
        let mtlo_instr = make_r(OP_SPECIAL, 3, 0, 0, 0, FUNCT_MTLO);
        assert_eq!(exec.exec(mtlo_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.lo, 0xFEDCBA98_76543210);

        // Test MFLO (Move From LO)
        exec.core.write_gpr(4, 0);
        let mflo_instr = make_r(OP_SPECIAL, 0, 0, 4, 0, FUNCT_MFLO);
        assert_eq!(exec.exec(mflo_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(4), 0xFEDCBA98_76543210);
    }

    #[test]
    fn test_multiplication() {
        let (mut exec, _) = create_executor();

        // Test MULT (Multiply Signed)
        // Positive * Positive
        exec.core.write_gpr(1, 1000);  // 1000
        exec.core.write_gpr(2, 2000);  // 2000
        let mult_instr = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_MULT);
        assert_eq!(exec.exec(mult_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.lo, 2000000); // Result = 2,000,000
        assert_eq!(exec.core.hi, 0);

        // Negative * Positive
        exec.core.write_gpr(1, 0xFFFFFFFF_FFFFFC18); // -1000 in 64-bit
        exec.core.write_gpr(2, 2000);
        assert_eq!(exec.exec(mult_instr), EXEC_COMPLETE);
        let result = (exec.core.hi as i64) << 32 | (exec.core.lo as u32 as u64) as i64;
        assert_eq!(result, -2000000);

        // Large positive multiplication that overflows 32-bit
        exec.core.write_gpr(1, 0x10000);  // 65536
        exec.core.write_gpr(2, 0x10000);  // 65536
        assert_eq!(exec.exec(mult_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.lo, 0);  // Low 32 bits
        assert_eq!(exec.core.hi as u32, 1);  // High 32 bits

        // Test MULTU (Multiply Unsigned)
        exec.core.write_gpr(1, 0xFFFFFFFF); // Max unsigned 32-bit value
        exec.core.write_gpr(2, 2);
        let multu_instr = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_MULTU);
        assert_eq!(exec.exec(multu_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.lo as u32, 0xFFFFFFFE); // Low 32 bits
        assert_eq!(exec.core.hi as u32, 1); // High 32 bits
    }

    #[test]
    fn test_division() {
        let (mut exec, _) = create_executor();

        // Test DIV (Divide Signed)
        exec.core.write_gpr(1, 100);  // Dividend
        exec.core.write_gpr(2, 7);    // Divisor
        let div_instr = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_DIV);
        assert_eq!(exec.exec(div_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.lo as i32, 14); // Quotient: 100 / 7 = 14
        assert_eq!(exec.core.hi as i32, 2);  // Remainder: 100 % 7 = 2

        // Test negative dividend
        exec.core.write_gpr(1, 0xFFFFFFFF_FFFFFF9C); // -100 in 64-bit
        exec.core.write_gpr(2, 7);
        assert_eq!(exec.exec(div_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.lo as i32, -14); // Quotient: -100 / 7 = -14
        assert_eq!(exec.core.hi as i32, -2);  // Remainder: -100 % 7 = -2

        // Test division by zero (should not crash)
        exec.core.write_gpr(1, 100);
        exec.core.write_gpr(2, 0);
        let old_hi = exec.core.hi;
        let old_lo = exec.core.lo;
        assert_eq!(exec.exec(div_instr), EXEC_COMPLETE);
        // HI and LO should be unchanged on division by zero
        assert_eq!(exec.core.hi, old_hi);
        assert_eq!(exec.core.lo, old_lo);

        // Test DIVU (Divide Unsigned)
        exec.core.write_gpr(1, 0xFFFFFFFF); // Max unsigned 32-bit value
        exec.core.write_gpr(2, 2);
        let divu_instr = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_DIVU);
        assert_eq!(exec.exec(divu_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.lo as u32, 0x7FFFFFFF); // Quotient
        assert_eq!(exec.core.hi as u32, 1);          // Remainder

        // Test DIVU division by zero
        exec.core.write_gpr(1, 100);
        exec.core.write_gpr(2, 0);
        let old_hi = exec.core.hi;
        let old_lo = exec.core.lo;
        assert_eq!(exec.exec(divu_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.hi, old_hi);
        assert_eq!(exec.core.lo, old_lo);
    }

    #[test]
    fn test_mult_div_integration() {
        // Test a complete multiply-divide sequence with HI/LO access
        let (mut exec, _) = create_executor();

        // Multiply two numbers
        exec.core.write_gpr(1, 123);
        exec.core.write_gpr(2, 456);
        let mult_instr = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_MULT);
        assert_eq!(exec.exec(mult_instr), EXEC_COMPLETE);

        // Read result from LO (low 32 bits of product)
        exec.core.write_gpr(5, 0);
        let mflo_instr = make_r(OP_SPECIAL, 0, 0, 5, 0, FUNCT_MFLO);
        assert_eq!(exec.exec(mflo_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(5), 123 * 456);

        // Read HI (high 32 bits, should be 0 for this small product)
        exec.core.write_gpr(6, 0);
        let mfhi_instr = make_r(OP_SPECIAL, 0, 0, 6, 0, FUNCT_MFHI);
        assert_eq!(exec.exec(mfhi_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(6), 0);

        // Now divide
        exec.core.write_gpr(3, 1000);
        exec.core.write_gpr(4, 13);
        let div_instr = make_r(OP_SPECIAL, 3, 4, 0, 0, FUNCT_DIV);
        assert_eq!(exec.exec(div_instr), EXEC_COMPLETE);

        // Read quotient from LO
        exec.core.write_gpr(7, 0);
        let mflo_instr2 = make_r(OP_SPECIAL, 0, 0, 7, 0, FUNCT_MFLO);
        assert_eq!(exec.exec(mflo_instr2), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(7) as i32, 76); // 1000 / 13 = 76

        // Read remainder from HI
        exec.core.write_gpr(8, 0);
        let mfhi_instr2 = make_r(OP_SPECIAL, 0, 0, 8, 0, FUNCT_MFHI);
        assert_eq!(exec.exec(mfhi_instr2), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(8) as i32, 12); // 1000 % 13 = 12
    }

    // ===== FPU (COP1) Tests =====

    // Helper to make FPU instructions
    fn make_cop1_move(rs: u32, rt: u32, fs: u32) -> u32 {
        // COP1 with RS field, RT (GPR), FS (FPR)
        (OP_COP1 << 26) | ((rs & 0x1F) << 21) | ((rt & 0x1F) << 16) | ((fs & 0x1F) << 11)
    }

    fn make_cop1_compute(fmt: u32, ft: u32, fs: u32, fd: u32, funct: u32) -> u32 {
        // COP1 with format (RS), FT, FS, FD, FUNCT
        (OP_COP1 << 26) | ((fmt & 0x1F) << 21) | ((ft & 0x1F) << 16) | ((fs & 0x1F) << 11) | ((fd & 0x1F) << 6) | (funct & 0x3F)
    }

    fn make_cop1_branch(cond: u32, offset: i16) -> u32 {
        // BC1F/BC1T/BC1FL/BC1TL
        (OP_COP1 << 26) | (RS_BC1 << 21) | ((cond & 0x1F) << 16) | ((offset as u16) as u32)
    }

    #[test]
    fn test_fpu_data_transfer() {
        let (mut exec, _) = create_executor();

        // Enable FPU (set CU1 bit in Status)
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Test MTC1 - Move Word To FPU
        exec.core.write_gpr(1, 0x3F800000); // 1.0 in IEEE 754 single precision
        let mtc1_instr = make_cop1_move(RS_MTC1, 1, 0);
        assert_eq!(exec.exec(mtc1_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_w(0), 0x3F800000);

        // Test MFC1 - Move Word From FPU
        exec.core.write_gpr(2, 0);
        let mfc1_instr = make_cop1_move(RS_MFC1, 2, 0);
        assert_eq!(exec.exec(mfc1_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0x3F800000);

        // Test DMTC1 - Move Doubleword To FPU (64-bit mode)
        exec.core.cp0_status |= STATUS_KX; // Enable 64-bit kernel mode
        exec.core.write_gpr(3, 0x3FF00000_00000000); // 1.0 in IEEE 754 double precision
        let dmtc1_instr = make_cop1_move(RS_DMTC1, 3, 1);
        assert_eq!(exec.exec(dmtc1_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_l(1), 0x3FF00000_00000000);

        // Test DMFC1 - Move Doubleword From FPU
        exec.core.write_gpr(4, 0);
        let dmfc1_instr = make_cop1_move(RS_DMFC1, 4, 1);
        assert_eq!(exec.exec(dmfc1_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(4), 0x3FF00000_00000000);
    }

    #[test]
    fn test_fpu_control_registers() {
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Test CTC1 - Move Control Word To FPU (FCSR)
        exec.core.write_gpr(1, 0x00000001); // Set rounding mode to RZ (Round toward Zero)
        let ctc1_instr = make_cop1_move(RS_CTC1, 1, 31); // Reg 31 = FCSR
        assert_eq!(exec.exec(ctc1_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.fpu_fcsr & 0x3, 0x1);

        // Test CFC1 - Move Control Word From FPU
        exec.core.write_gpr(2, 0);
        let cfc1_instr = make_cop1_move(RS_CFC1, 2, 31);
        assert_eq!(exec.exec(cfc1_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2) & 0x3, 0x1);
    }

    #[test]
    fn test_fpu_load_store() {
        let (mut exec, mem) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Test LWC1 - Load Word to FPU
        mem.set_word(0x2000, 0x40490FDB); // pi (3.14159...) in float
        exec.core.write_gpr(5, 0x2000);
        let lwc1_instr = make_i(OP_LWC1, 5, 0, 0); // ft=0 (f0)
        let pc_before = exec.core.pc;
        assert_eq!(exec.exec(lwc1_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_w(0), 0x40490FDB);
        assert_eq!(exec.core.pc, pc_before.wrapping_add(4), "LWC1 must advance PC by 4");
        // Verify it's actually pi
        let pi_val = exec.core.read_fpr_s(0);
        assert!((pi_val - 3.14159265).abs() < 0.0001);

        // Test SWC1 - Store Word from FPU
        mem.set_word(0x2004, 0);
        exec.core.write_gpr(6, 0x2004);
        let swc1_instr = make_i(OP_SWC1, 6, 0, 0);
        let pc_before = exec.core.pc;
        assert_eq!(exec.exec(swc1_instr), EXEC_COMPLETE);
        assert_eq!(mem.get_word(0x2004), 0x40490FDB);
        assert_eq!(exec.core.pc, pc_before.wrapping_add(4), "SWC1 must advance PC by 4");

        // Test LDC1 - Load Doubleword to FPU
        mem.set_double(0x3000, 0x400921FB_54442D18); // pi in double precision
        exec.core.write_gpr(7, 0x3000);
        let ldc1_instr = make_i(OP_LDC1, 7, 1, 0); // ft=1 (f1)
        let pc_before = exec.core.pc;
        assert_eq!(exec.exec(ldc1_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_l(1), 0x400921FB_54442D18);
        assert_eq!(exec.core.pc, pc_before.wrapping_add(4), "LDC1 must advance PC by 4");

        // Test SDC1 - Store Doubleword from FPU
        mem.set_double(0x3008, 0);
        exec.core.write_gpr(8, 0x3008);
        let sdc1_instr = make_i(OP_SDC1, 8, 1, 0);
        let pc_before = exec.core.pc;
        assert_eq!(exec.exec(sdc1_instr), EXEC_COMPLETE);
        assert_eq!(mem.get_double(0x3008), 0x400921FB_54442D18);
        assert_eq!(exec.core.pc, pc_before.wrapping_add(4), "SDC1 must advance PC by 4");
    }

    #[test]
    fn test_fpu_add_sub_single() {
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Test ADD.S - Single Precision Addition
        exec.core.write_fpr_s(1, 2.5);
        exec.core.write_fpr_s(2, 3.5);
        let add_s_instr = make_cop1_compute(RS_S, 2, 1, 0, FUNCT_FADD);
        assert_eq!(exec.exec(add_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(0), 6.0);

        // Test SUB.S - Single Precision Subtraction
        exec.core.write_fpr_s(3, 10.0);
        exec.core.write_fpr_s(4, 4.5);
        let sub_s_instr = make_cop1_compute(RS_S, 4, 3, 5, FUNCT_FSUB);
        assert_eq!(exec.exec(sub_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(5), 5.5);
    }

    /// R4000/VR5000 manuals: a denormalized operand to a computational
    /// instruction is Unimplemented Operation (Cause.E) — the hardware never
    /// computes an answer, so the destination is left untouched. See
    /// rules/testing/fpu-exception-model-vs-r4400.md.
    #[test]
    fn test_fpu_denorm_operand_is_unimplemented() {
        let (mut exec, _) = create_executor();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        exec.core.write_fpr_w(1, 0x0000_0001); // smallest denormal single
        exec.core.write_fpr_s(2, 1.0);
        exec.core.write_fpr_w(0, 0x5a5a5a5a); // sentinel, must survive a trap untouched

        let add_s = make_cop1_compute(RS_S, 2, 1, 0, FUNCT_FADD);
        let s = exec.exec(add_s);
        assert!(s & EXEC_IS_EXCEPTION != 0, "denormal operand must trap");
        assert_eq!((s >> 2) & 0x1F, crate::mips_exec::EXC_FPE);
        assert_eq!(exec.core.read_fpu_control(31) & 0x0002_0000, 0x0002_0000, "Cause.E must be set");
        assert_eq!(exec.core.read_fpr_w(0), 0x5a5a5a5a, "destination must be untouched on trap");
    }

    /// Same rule for a quiet NaN operand — the surprising half of the table:
    /// an R4400/R5000 does not propagate NaN through ADD in hardware.
    #[test]
    fn test_fpu_qnan_operand_is_unimplemented() {
        let (mut exec, _) = create_executor();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        exec.core.write_fpr_w(1, 0x7FC0_0000); // quiet NaN
        exec.core.write_fpr_s(2, 1.0);
        exec.core.write_fpr_w(0, 0x5a5a5a5a);

        let add_s = make_cop1_compute(RS_S, 2, 1, 0, FUNCT_FADD);
        let s = exec.exec(add_s);
        assert!(s & EXEC_IS_EXCEPTION != 0, "qNaN operand must trap");
        assert_eq!((s >> 2) & 0x1F, crate::mips_exec::EXC_FPE);
        assert_eq!(exec.core.read_fpr_w(0), 0x5a5a5a5a, "destination must be untouched on trap");
    }

    /// Denormalized result, FS clear: Unimplemented Operation, destination
    /// untouched. 2^-126 * 0.5 = 2^-127, denormal in single precision.
    #[test]
    fn test_fpu_denorm_result_without_fs_is_unimplemented() {
        let (mut exec, _) = create_executor();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        exec.core.write_fpr_s(1, f32::from_bits(0x0080_0000)); // 2^-126, smallest normal
        exec.core.write_fpr_s(2, 0.5);
        exec.core.write_fpr_w(0, 0x5a5a5a5a);
        exec.core.fpu_fcsr &= !0x0100_0000; // FS clear

        let mul_s = make_cop1_compute(RS_S, 2, 1, 0, FUNCT_FMUL);
        let s = exec.exec(mul_s);
        assert!(s & EXEC_IS_EXCEPTION != 0, "denormal result without FS must trap");
        assert_eq!((s >> 2) & 0x1F, crate::mips_exec::EXC_FPE);
        assert_eq!(exec.core.read_fpu_control(31) & 0x0002_0000, 0x0002_0000, "Cause.E must be set");
        assert_eq!(exec.core.read_fpr_w(0), 0x5a5a5a5a, "destination must be untouched on trap");
    }

    /// Denormalized result, FS set, neither U nor I trap enabled: flushed to
    /// a signed zero (sign of the intermediate result) and Cause.U+I set —
    /// no trap.
    #[test]
    fn test_fpu_denorm_result_flushed_with_fs() {
        let (mut exec, _) = create_executor();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        exec.core.write_fpr_s(1, f32::from_bits(0x0080_0000)); // 2^-126
        exec.core.write_fpr_s(2, 0.5);
        exec.core.fpu_fcsr = 0x0100_0000; // FS set, no enables

        let mul_s = make_cop1_compute(RS_S, 2, 1, 0, FUNCT_FMUL);
        assert_eq!(exec.exec(mul_s), EXEC_COMPLETE, "flush-to-zero must not trap");
        assert_eq!(exec.core.read_fpr_s(0), 0.0);
        assert!(exec.core.read_fpr_s(0).is_sign_positive());
        let fcsr = exec.core.read_fpu_control(31);
        assert_eq!(fcsr & 0x0000_3000, 0x0000_3000, "Cause.U and Cause.I must both be set");
        assert_eq!(fcsr & 0x0000_000c, 0x0000_000c, "Flag.U and Flag.I must both be set");

        // Sign of the intermediate result survives the flush.
        exec.core.write_fpr_s(1, f32::from_bits(0x8080_0000)); // -2^-126
        exec.core.fpu_fcsr = 0x0100_0000;
        assert_eq!(exec.exec(mul_s), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(0), 0.0);
        assert!(exec.core.read_fpr_s(0).is_sign_negative());
    }

    /// FS alone is not enough: with Underflow trapping enabled, the same
    /// multiply reverts to Unimplemented Operation instead of flushing.
    #[test]
    fn test_fpu_underflow_enable_forces_unimplemented() {
        let (mut exec, _) = create_executor();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        exec.core.write_fpr_s(1, f32::from_bits(0x0080_0000));
        exec.core.write_fpr_s(2, 0.5);
        exec.core.write_fpr_w(0, 0x5a5a5a5a);
        exec.core.fpu_fcsr = 0x0100_0000 | 0x0000_0100; // FS set, U enable set

        let mul_s = make_cop1_compute(RS_S, 2, 1, 0, FUNCT_FMUL);
        let s = exec.exec(mul_s);
        assert!(s & EXEC_IS_EXCEPTION != 0, "underflow-enable must force Unimplemented, not flush");
        assert_eq!((s >> 2) & 0x1F, crate::mips_exec::EXC_FPE);
        assert_eq!(exec.core.read_fpu_control(31) & 0x0002_0000, 0x0002_0000, "Cause.E must be set");
        assert_eq!(exec.core.read_fpu_control(31) & 0x0000_2000, 0, "Cause.U must NOT be set — this is E, not U");
        assert_eq!(exec.core.read_fpr_w(0), 0x5a5a5a5a, "destination must be untouched on trap");
    }

    /// Compare is explicitly exempt from the denormal-operand rule.
    #[test]
    fn test_fpu_denorm_compare_does_not_trap() {
        let (mut exec, _) = create_executor();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        exec.core.write_fpr_w(1, 0x0000_0001); // smallest denormal
        exec.core.write_fpr_s(2, 1.0);
        let c_lt_s = make_cop1_compute(RS_S, 2, 1, 0, FUNCT_FC_LT);
        assert_eq!(exec.exec(c_lt_s), EXEC_COMPLETE, "compare must not trap on a denormal operand");
        assert!(exec.core.get_fpu_cc(0), "denorm < 1.0 must be true");
    }

    /// MOV is exempt from the denormal-operand rule too — it's a bit copy,
    /// not an arithmetic operation.
    #[test]
    fn test_fpu_denorm_move_does_not_trap() {
        let (mut exec, _) = create_executor();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        exec.core.write_fpr_w(1, 0x0000_0001);
        let mov_s = make_cop1_compute(RS_S, 0, 1, 2, FUNCT_FMOV);
        assert_eq!(exec.exec(mov_s), EXEC_COMPLETE, "MOV must not trap on a denormal operand");
        assert_eq!(exec.core.read_fpr_w(2), 0x0000_0001);
    }

    #[test]
    fn test_fpu_mul_div_single() {
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Test MUL.S - Single Precision Multiplication
        exec.core.write_fpr_s(1, 2.0);
        exec.core.write_fpr_s(2, 3.0);
        let mul_s_instr = make_cop1_compute(RS_S, 2, 1, 0, FUNCT_FMUL);
        assert_eq!(exec.exec(mul_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(0), 6.0);

        // Test DIV.S - Single Precision Division
        exec.core.write_fpr_s(3, 10.0);
        exec.core.write_fpr_s(4, 4.0);
        let div_s_instr = make_cop1_compute(RS_S, 4, 3, 5, FUNCT_FDIV);
        assert_eq!(exec.exec(div_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(5), 2.5);
    }

    #[test]
    fn test_fpu_sqrt_abs_neg_single() {
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Test SQRT.S - Square Root
        exec.core.write_fpr_s(1, 9.0);
        let sqrt_s_instr = make_cop1_compute(RS_S, 0, 1, 0, FUNCT_FSQRT);
        assert_eq!(exec.exec(sqrt_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(0), 3.0);

        // Test ABS.S - Absolute Value
        exec.core.write_fpr_s(2, -5.5);
        let abs_s_instr = make_cop1_compute(RS_S, 0, 2, 3, FUNCT_FABS);
        assert_eq!(exec.exec(abs_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(3), 5.5);

        // Test NEG.S - Negate
        exec.core.write_fpr_s(4, 7.25);
        let neg_s_instr = make_cop1_compute(RS_S, 0, 4, 5, FUNCT_FNEG);
        assert_eq!(exec.exec(neg_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(5), -7.25);

        // Test MOV.S - Move
        exec.core.write_fpr_s(6, 42.0);
        let mov_s_instr = make_cop1_compute(RS_S, 0, 6, 7, FUNCT_FMOV);
        assert_eq!(exec.exec(mov_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(7), 42.0);
    }

    #[test]
    fn test_fpu_add_sub_double() {
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Test ADD.D - Double Precision Addition
        exec.core.write_fpr_d(1, 2.5);
        exec.core.write_fpr_d(2, 3.5);
        let add_d_instr = make_cop1_compute(RS_D, 2, 1, 0, FUNCT_FADD);
        assert_eq!(exec.exec(add_d_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(0), 6.0);

        // Test SUB.D - Double Precision Subtraction
        exec.core.write_fpr_d(3, 10.0);
        exec.core.write_fpr_d(4, 4.5);
        let sub_d_instr = make_cop1_compute(RS_D, 4, 3, 5, FUNCT_FSUB);
        assert_eq!(exec.exec(sub_d_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(5), 5.5);
    }

    #[test]
    fn test_fpu_mul_div_double() {
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Test MUL.D - Double Precision Multiplication
        exec.core.write_fpr_d(1, 2.0);
        exec.core.write_fpr_d(2, 3.0);
        let mul_d_instr = make_cop1_compute(RS_D, 2, 1, 0, FUNCT_FMUL);
        assert_eq!(exec.exec(mul_d_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(0), 6.0);

        // Test DIV.D - Double Precision Division
        exec.core.write_fpr_d(3, 10.0);
        exec.core.write_fpr_d(4, 4.0);
        let div_d_instr = make_cop1_compute(RS_D, 4, 3, 5, FUNCT_FDIV);
        assert_eq!(exec.exec(div_d_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(5), 2.5);
    }

    #[test]
    fn test_fpu_sqrt_abs_neg_double() {
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Test SQRT.D - Square Root
        exec.core.write_fpr_d(1, 16.0);
        let sqrt_d_instr = make_cop1_compute(RS_D, 0, 1, 0, FUNCT_FSQRT);
        assert_eq!(exec.exec(sqrt_d_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(0), 4.0);

        // Test ABS.D - Absolute Value
        exec.core.write_fpr_d(2, -12.75);
        let abs_d_instr = make_cop1_compute(RS_D, 0, 2, 3, FUNCT_FABS);
        assert_eq!(exec.exec(abs_d_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(3), 12.75);

        // Test NEG.D - Negate
        exec.core.write_fpr_d(4, 99.5);
        let neg_d_instr = make_cop1_compute(RS_D, 0, 4, 5, FUNCT_FNEG);
        assert_eq!(exec.exec(neg_d_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(5), -99.5);
    }

    #[test]
    fn test_fpu_conversions() {
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Test CVT.D.S - Single to Double
        exec.core.write_fpr_s(1, 3.14159);
        let cvt_d_s_instr = make_cop1_compute(RS_S, 0, 1, 0, FUNCT_FCVT_D);
        assert_eq!(exec.exec(cvt_d_s_instr), EXEC_COMPLETE);
        let result = exec.core.read_fpr_d(0);
        assert!((result - 3.14159).abs() < 0.00001);

        // Test CVT.S.D - Double to Single
        exec.core.write_fpr_d(2, 2.71828);
        let cvt_s_d_instr = make_cop1_compute(RS_D, 0, 2, 3, FUNCT_FCVT_S);
        assert_eq!(exec.exec(cvt_s_d_instr), EXEC_COMPLETE);
        let result = exec.core.read_fpr_s(3);
        assert!((result - 2.71828).abs() < 0.00001);

        // Test CVT.W.S - Single to Word
        exec.core.write_fpr_s(4, 42.7);
        let cvt_w_s_instr = make_cop1_compute(RS_S, 0, 4, 5, FUNCT_FCVT_W);
        assert_eq!(exec.exec(cvt_w_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_w(5) as i32, 43); // Rounds to nearest

        // Test CVT.S.W - Word to Single
        exec.core.write_fpr_w(6, 100u32);
        let cvt_s_w_instr = make_cop1_compute(RS_W, 0, 6, 7, FUNCT_FCVT_S);
        assert_eq!(exec.exec(cvt_s_w_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(7), 100.0);

        // Test CVT.D.W - Word to Double
        exec.core.write_fpr_w(8, 200u32);
        let cvt_d_w_instr = make_cop1_compute(RS_W, 0, 8, 9, FUNCT_FCVT_D);
        assert_eq!(exec.exec(cvt_d_w_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(9), 200.0);
    }

    #[test]
    fn test_fpu_rounding() {
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Test TRUNC.W.S - Truncate to Word
        exec.core.write_fpr_s(1, 3.9);
        let trunc_w_s_instr = make_cop1_compute(RS_S, 0, 1, 0, FUNCT_FTRUNC_W);
        assert_eq!(exec.exec(trunc_w_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_w(0) as i32, 3);

        // Test ROUND.W.S - Round to Word
        exec.core.write_fpr_s(2, 3.5);
        let round_w_s_instr = make_cop1_compute(RS_S, 0, 2, 3, FUNCT_FROUND_W);
        assert_eq!(exec.exec(round_w_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_w(3) as i32, 4);

        // Test CEIL.W.S - Ceiling to Word
        exec.core.write_fpr_s(4, 3.1);
        let ceil_w_s_instr = make_cop1_compute(RS_S, 0, 4, 5, FUNCT_FCEIL_W);
        assert_eq!(exec.exec(ceil_w_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_w(5) as i32, 4);

        // Test FLOOR.W.S - Floor to Word
        exec.core.write_fpr_s(6, 3.9);
        let floor_w_s_instr = make_cop1_compute(RS_S, 0, 6, 7, FUNCT_FFLOOR_W);
        assert_eq!(exec.exec(floor_w_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_w(7) as i32, 3);

        // Test negative values
        exec.core.write_fpr_s(8, -2.7);
        let trunc_neg_instr = make_cop1_compute(RS_S, 0, 8, 9, FUNCT_FTRUNC_W);
        assert_eq!(exec.exec(trunc_neg_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_w(9) as i32, -2); // Truncate toward zero

        exec.core.write_fpr_s(10, -2.7);
        let floor_neg_instr = make_cop1_compute(RS_S, 0, 10, 11, FUNCT_FFLOOR_W);
        assert_eq!(exec.exec(floor_neg_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_w(11) as i32, -3); // Floor toward -inf
    }

    #[test]
    fn test_fpu_comparisons() {
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Test C.EQ.S - Equal
        exec.core.write_fpr_s(1, 5.0);
        exec.core.write_fpr_s(2, 5.0);
        let c_eq_s_instr = make_cop1_compute(RS_S, 2, 1, 0, FUNCT_FC_EQ);
        assert_eq!(exec.exec(c_eq_s_instr), EXEC_COMPLETE);
        assert!(exec.core.get_fpu_cc(0)); // Condition code should be true

        // Test C.EQ.S - Not Equal
        exec.core.write_fpr_s(3, 5.0);
        exec.core.write_fpr_s(4, 6.0);
        let c_eq_s_instr2 = make_cop1_compute(RS_S, 4, 3, 0, FUNCT_FC_EQ);
        assert_eq!(exec.exec(c_eq_s_instr2), EXEC_COMPLETE);
        assert!(!exec.core.get_fpu_cc(0)); // Condition code should be false

        // Test C.LT.S - Less Than
        exec.core.write_fpr_s(5, 3.0);
        exec.core.write_fpr_s(6, 7.0);
        let c_lt_s_instr = make_cop1_compute(RS_S, 6, 5, 0, FUNCT_FC_LT);
        assert_eq!(exec.exec(c_lt_s_instr), EXEC_COMPLETE);
        assert!(exec.core.get_fpu_cc(0)); // 3.0 < 7.0

        // Test C.LE.S - Less Than or Equal
        exec.core.write_fpr_s(7, 4.0);
        exec.core.write_fpr_s(8, 4.0);
        let c_le_s_instr = make_cop1_compute(RS_S, 8, 7, 0, FUNCT_FC_LE);
        assert_eq!(exec.exec(c_le_s_instr), EXEC_COMPLETE);
        assert!(exec.core.get_fpu_cc(0)); // 4.0 <= 4.0

        // Test C.OLT.D - Ordered Less Than (Double)
        exec.core.write_fpr_d(9, 1.5);
        exec.core.write_fpr_d(10, 2.5);
        let c_olt_d_instr = make_cop1_compute(RS_D, 10, 9, 0, FUNCT_FC_OLT);
        assert_eq!(exec.exec(c_olt_d_instr), EXEC_COMPLETE);
        assert!(exec.core.get_fpu_cc(0)); // 1.5 < 2.5

        // Test C.OLE.D - Ordered Less Than or Equal (Double)
        exec.core.write_fpr_d(11, 3.5);
        exec.core.write_fpr_d(12, 3.5);
        let c_ole_d_instr = make_cop1_compute(RS_D, 12, 11, 0, FUNCT_FC_OLE);
        assert_eq!(exec.exec(c_ole_d_instr), EXEC_COMPLETE);
        assert!(exec.core.get_fpu_cc(0)); // 3.5 <= 3.5
    }

    #[test]
    fn test_fpu_branches() {
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();
        exec.core.pc = 0x1000;

        // Set condition code to true
        exec.core.set_fpu_cc(0, true);

        // Test BC1T - Branch on FPU True (should branch)
        let bc1t_instr = make_cop1_branch(1, 4); // offset=4 -> target=0x1000+4+4*4=0x1014
        { let _s = exec.exec(bc1t_instr); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x1014 ); };

        // Reset
        exec.core.pc = 0x2000;

        // Test BC1F - Branch on FPU False (should not branch, CC is true)
        let bc1f_instr = make_cop1_branch(0, 4);
        assert_eq!(exec.exec(bc1f_instr), EXEC_COMPLETE);

        // Set condition code to false
        exec.core.set_fpu_cc(0, false);
        exec.core.pc = 0x3000;

        // Test BC1F - Branch on FPU False (should branch, CC is false)
        let bc1f_instr2 = make_cop1_branch(0, 4);
        { let _s = exec.exec(bc1f_instr2); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x3014 ); };

        // Test BC1TL - Branch on FPU True Likely (should not branch and nullify)
        exec.core.set_fpu_cc(0, false);
        exec.core.pc = 0x4000;
        let bc1tl_instr = make_cop1_branch(3, 4); // bit 1 set = likely
        assert_eq!(exec.exec(bc1tl_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x4008); // PC+8 (skip delay slot)

        // Test BC1FL - Branch on FPU False Likely (should not branch and nullify)
        exec.core.set_fpu_cc(0, true);
        exec.core.pc = 0x5000;
        let bc1fl_instr = make_cop1_branch(2, 4); // bit 1 set = likely, bit 0 clear = false
        assert_eq!(exec.exec(bc1fl_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0x5008);
    }

    #[test]
    fn test_fpu_exception_disabled() {
        let (mut exec, _) = create_executor();

        // FPU is disabled (CU1 not set)
        exec.core.cp0_status &= !STATUS_CU1;

        // Attempt MTC1 - should trigger coprocessor unusable exception
        exec.core.write_gpr(1, 0x3F800000);
        let mtc1_instr = make_cop1_move(RS_MTC1, 1, 0);
        { let _s = exec.exec(mtc1_instr); assert!(_s & EXEC_IS_EXCEPTION != 0, "Expected coprocessor unusable exception"); assert_eq!((_s >> 2) & 0x1F, crate::mips_exec::EXC_CPU); }

        // Attempt LWC1 - should also trigger exception
        exec.core.write_gpr(2, 0x1000);
        let lwc1_instr = make_i(OP_LWC1, 2, 0, 0);
        { let _s = exec.exec(lwc1_instr); assert!(_s & EXEC_IS_EXCEPTION != 0, "Expected coprocessor unusable exception"); assert_eq!((_s >> 2) & 0x1F, crate::mips_exec::EXC_CPU); }
    }

    #[test]
    fn test_fpu_long_conversions() {
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR | STATUS_KX; // Enable FPU and 64-bit mode
        exec.update_fpr_mode();

        // Test CVT.L.S - Single to Long
        exec.core.write_fpr_s(1, 123456.75);
        let cvt_l_s_instr = make_cop1_compute(RS_S, 0, 1, 0, FUNCT_FCVT_L);
        assert_eq!(exec.exec(cvt_l_s_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_l(0) as i64, 123457); // Rounds to nearest

        // Test CVT.L.D - Double to Long
        exec.core.write_fpr_d(2, 987654.25);
        let cvt_l_d_instr = make_cop1_compute(RS_D, 0, 2, 3, FUNCT_FCVT_L);
        assert_eq!(exec.exec(cvt_l_d_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_l(3) as i64, 987654);

        // Test CVT.S.L - Long to Single
        exec.core.write_fpr_l(4, 555555i64 as u64);
        let cvt_s_l_instr = make_cop1_compute(RS_L, 0, 4, 5, FUNCT_FCVT_S);
        assert_eq!(exec.exec(cvt_s_l_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(5), 555555.0);

        // Test CVT.D.L - Long to Double
        exec.core.write_fpr_l(6, 999999i64 as u64);
        let cvt_d_l_instr = make_cop1_compute(RS_L, 0, 6, 7, FUNCT_FCVT_D);
        assert_eq!(exec.exec(cvt_d_l_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(7), 999999.0);

        // Test TRUNC.L.D - Truncate Double to Long
        exec.core.write_fpr_d(8, 12345.99);
        let trunc_l_d_instr = make_cop1_compute(RS_D, 0, 8, 9, FUNCT_FTRUNC_L);
        assert_eq!(exec.exec(trunc_l_d_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_l(9) as i64, 12345);

        // Test with negative values
        exec.core.write_fpr_d(10, -54321.75);
        let cvt_l_d_neg_instr = make_cop1_compute(RS_D, 0, 10, 11, FUNCT_FCVT_L);
        assert_eq!(exec.exec(cvt_l_d_neg_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_l(11) as i64, -54322); // Rounds to nearest
    }

    #[test]
    fn test_fpu_nan_comparisons() {
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Create NaN value
        let nan = f32::NAN;
        exec.core.write_fpr_s(1, nan);
        exec.core.write_fpr_s(2, 5.0);

        // Test C.EQ.S with NaN - should be false
        let c_eq_s_instr = make_cop1_compute(RS_S, 2, 1, 0, FUNCT_FC_EQ);
        assert_eq!(exec.exec(c_eq_s_instr), EXEC_COMPLETE);
        assert!(!exec.core.get_fpu_cc(0)); // NaN != anything

        // Test C.UN.S - Unordered (true if either operand is NaN)
        let c_un_s_instr = make_cop1_compute(RS_S, 2, 1, 0, FUNCT_FC_UN);
        assert_eq!(exec.exec(c_un_s_instr), EXEC_COMPLETE);
        assert!(exec.core.get_fpu_cc(0)); // Unordered because of NaN

        // Test C.UEQ.S - Unordered or Equal
        let c_ueq_s_instr = make_cop1_compute(RS_S, 2, 1, 0, FUNCT_FC_UEQ);
        assert_eq!(exec.exec(c_ueq_s_instr), EXEC_COMPLETE);
        assert!(exec.core.get_fpu_cc(0)); // Unordered (NaN)

        // Test with double precision NaN
        let nan_d = f64::NAN;
        exec.core.write_fpr_d(3, nan_d);
        exec.core.write_fpr_d(4, 10.0);

        let c_olt_d_instr = make_cop1_compute(RS_D, 4, 3, 0, FUNCT_FC_OLT);
        assert_eq!(exec.exec(c_olt_d_instr), EXEC_COMPLETE);
        assert!(!exec.core.get_fpu_cc(0)); // OLT (ordered) is false with NaN

        let c_ult_d_instr = make_cop1_compute(RS_D, 4, 3, 0, FUNCT_FC_ULT);
        assert_eq!(exec.exec(c_ult_d_instr), EXEC_COMPLETE);
        assert!(exec.core.get_fpu_cc(0)); // ULT (unordered or less) is true with NaN
    }

    #[test]
    fn test_fpu_complete_computation() {
        // Comprehensive test: (a + b) * (c - d) / e, then convert and compare
        let (mut exec, _) = create_executor();

        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Set up values: a=10, b=5, c=20, d=8, e=3
        exec.core.write_fpr_s(0, 10.0); // a
        exec.core.write_fpr_s(1, 5.0);  // b
        exec.core.write_fpr_s(2, 20.0); // c
        exec.core.write_fpr_s(3, 8.0);  // d
        exec.core.write_fpr_s(4, 3.0);  // e

        // Compute a + b -> f5
        let add_instr = make_cop1_compute(RS_S, 1, 0, 5, FUNCT_FADD);
        exec.exec(add_instr);
        assert_eq!(exec.core.read_fpr_s(5), 15.0);

        // Compute c - d -> f6
        let sub_instr = make_cop1_compute(RS_S, 3, 2, 6, FUNCT_FSUB);
        exec.exec(sub_instr);
        assert_eq!(exec.core.read_fpr_s(6), 12.0);

        // Compute f5 * f6 -> f7
        let mul_instr = make_cop1_compute(RS_S, 6, 5, 7, FUNCT_FMUL);
        exec.exec(mul_instr);
        assert_eq!(exec.core.read_fpr_s(7), 180.0);

        // Compute f7 / e -> f8
        let div_instr = make_cop1_compute(RS_S, 4, 7, 8, FUNCT_FDIV);
        exec.exec(div_instr);
        assert_eq!(exec.core.read_fpr_s(8), 60.0);

        // Convert to double precision
        let cvt_d_instr = make_cop1_compute(RS_S, 0, 8, 9, FUNCT_FCVT_D);
        exec.exec(cvt_d_instr);
        assert_eq!(exec.core.read_fpr_d(9), 60.0);

        // Convert to word
        let cvt_w_instr = make_cop1_compute(RS_S, 0, 8, 10, FUNCT_FCVT_W);
        exec.exec(cvt_w_instr);
        assert_eq!(exec.core.read_fpr_w(10) as i32, 60);

        // Compare result with 50.0 (check if 60.0 > 50.0)
        exec.core.write_fpr_s(11, 50.0);
        let cmp_instr = make_cop1_compute(RS_S, 8, 11, 0, FUNCT_FC_LT);
        exec.exec(cmp_instr);
        assert!(exec.core.get_fpu_cc(0)); // 50.0 < 60.0 is true
    }

    #[test]
    fn test_stack_pointer_operations() {
        let (mut exec, mem) = create_executor();

        // Initialize registers
        let sp = 29;
        let t0 = 8;
        let t1 = 9;
        let t2 = 10;
        let t3 = 11;
        
        // Use KSEG1 address (uncached, unmapped)
        // Must be sign-extended for 32-bit mode compatibility check
        let initial_sp = 0xFFFFFFFF_A0001000u64;
        let phys_sp = 0x1000u64;

        exec.core.write_gpr(sp, initial_sp);
        exec.core.write_gpr(t0, 0xDEADBEEF);
        exec.core.write_gpr(t1, 0xCAFEBABE);

        // 1. Allocate stack frame (8 bytes): ADDIU $sp, $sp, -8
        let alloc_instr = make_i(OP_ADDIU, sp, sp, (-8i16) as u16);
        assert_eq!(exec.exec(alloc_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(sp), initial_sp - 8);

        // 2. Store t0 at sp+4: SW $t0, 4($sp)
        let sw_t0 = make_i(OP_SW, sp, t0, 4);
        assert_eq!(exec.exec(sw_t0), EXEC_COMPLETE);

        // 3. Store t1 at sp+0: SW $t1, 0($sp)
        let sw_t1 = make_i(OP_SW, sp, t1, 0);
        assert_eq!(exec.exec(sw_t1), EXEC_COMPLETE);

        // Verify memory content (at physical address)
        assert_eq!(mem.get_word(phys_sp - 4), 0xDEADBEEF);
        assert_eq!(mem.get_word(phys_sp - 8), 0xCAFEBABE);

        // 4. Load from sp+4 into t2: LW $t2, 4($sp)
        let lw_t2 = make_i(OP_LW, sp, t2, 4);
        assert_eq!(exec.exec(lw_t2), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(t2), 0xFFFFFFFF_DEADBEEF);

        // 5. Load from sp+0 into t3: LW $t3, 0($sp)
        let lw_t3 = make_i(OP_LW, sp, t3, 0);
        assert_eq!(exec.exec(lw_t3), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(t3), 0xFFFFFFFF_CAFEBABE);

        // 6. Deallocate stack frame: ADDIU $sp, $sp, 8
        let dealloc_instr = make_i(OP_ADDIU, sp, sp, 8);
        assert_eq!(exec.exec(dealloc_instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(sp), initial_sp);
    }

    #[test]
    fn test_slt_operations() {
        let (mut exec, _) = create_executor();

        // SLT r3, r1, r2 (Signed)
        // 10 < 20 -> 1
        exec.core.write_gpr(1, 10);
        exec.core.write_gpr(2, 20);
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_SLT);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 1);

        // 20 < 10 -> 0
        exec.core.write_gpr(1, 20);
        exec.core.write_gpr(2, 10);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0);

        // -10 < 10 -> 1
        exec.core.write_gpr(1, -10i64 as u64);
        exec.core.write_gpr(2, 10);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 1);

        // SLTU r3, r1, r2 (Unsigned)
        // -1 (Max Int) < 10 -> 0
        let instr_u = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_SLTU);
        exec.core.write_gpr(1, -1i64 as u64); // 0xFF...FF
        exec.core.write_gpr(2, 10);
        assert_eq!(exec.exec(instr_u), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0);

        // SLTI r2, r1, imm (Signed)
        // 10 < 20 -> 1
        exec.core.write_gpr(1, 10);
        let instr_i = make_i(OP_SLTI, 1, 2, 20);
        assert_eq!(exec.exec(instr_i), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 1);

        // SLTIU r2, r1, imm (Unsigned)
        // 10 < 20 -> 1
        let instr_iu = make_i(OP_SLTIU, 1, 2, 20);
        assert_eq!(exec.exec(instr_iu), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 1);
        
        // SLTIU with sign-extended immediate treated as unsigned
        // r1 = 0, imm = -1 (0xFFFF -> 0xFF...FF)
        // 0 < Max -> 1
        exec.core.write_gpr(1, 0);
        let instr_iu_neg = make_i(OP_SLTIU, 1, 2, 0xFFFF);
        assert_eq!(exec.exec(instr_iu_neg), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 1);
    }

    #[test]
    fn test_extended_logic() {
        let (mut exec, _) = create_executor();

        // XOR r3, r1, r2
        exec.core.write_gpr(1, 0xF0F0F0F0);
        exec.core.write_gpr(2, 0x0F0F0F0F);
        let instr_xor = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_XOR);
        assert_eq!(exec.exec(instr_xor), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0xFFFFFFFF);

        // NOR r3, r1, r2
        // ~(0xF0... | 0x0F...) = ~0x...FFFFFFFF = 0xFFFFFFFF00000000
        let instr_nor = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_NOR);
        assert_eq!(exec.exec(instr_nor), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0xFFFFFFFF00000000);

        // XORI r2, r1, imm
        exec.core.write_gpr(1, 0xF0F0F0F0);
        let instr_xori = make_i(OP_XORI, 1, 2, 0xFFFF);
        assert_eq!(exec.exec(instr_xori), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0xF0F00F0F); // 0xF0F0F0F0 ^ 0x0000FFFF
    }

    #[test]
    fn test_srl() {
        let (mut exec, _) = create_executor();
        
        // SRL r2, r1, 4
        exec.core.write_gpr(1, 0xF0000000);
        let instr = make_r(OP_SPECIAL, 0, 1, 2, 4, FUNCT_SRL);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0x0F000000);
    }

    #[test]
    fn test_register_jumps() {
        let (mut exec, _) = create_executor();

        // JR r1
        exec.core.write_gpr(1, 0x1000);
        let instr_jr = make_r(OP_SPECIAL, 1, 0, 0, 0, FUNCT_JR);
        { let _s = exec.exec(instr_jr); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x1000); }

        // JALR r1, r31 (default)
        exec.core.pc = 0x2000;
        exec.core.write_gpr(1, 0x3000);
        let instr_jalr = make_r(OP_SPECIAL, 1, 0, 31, 0, FUNCT_JALR);
        { let _s = exec.exec(instr_jalr); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x3000); assert_eq!(exec.core.read_gpr(31), 0x2008); }
    }

    #[test]
    fn test_traps() {
        let (mut exec, _) = create_executor();

        // TGE r1, r2 (Trap if r1 >= r2)
        exec.core.write_gpr(1, 10);
        exec.core.write_gpr(2, 5);
        let instr_tge = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_TGE);
        { let _s = exec.exec(instr_tge); assert!(_s & EXEC_IS_EXCEPTION != 0, "Expected Trap exception"); assert_eq!((_s >> 2) & 0x1F, EXC_TR); }

        // TGE r1, r2 (No trap if r1 < r2)
        exec.core.write_gpr(1, 5);
        exec.core.write_gpr(2, 10);
        assert_eq!(exec.exec(instr_tge), EXEC_COMPLETE);

        // TEQI r1, imm (Trap if r1 == imm)
        exec.core.write_gpr(1, 10);
        let instr_teqi = make_i(OP_REGIMM, 1, RT_TEQI, 10);
        { let _s = exec.exec(instr_teqi); assert!(_s & EXEC_IS_EXCEPTION != 0, "Expected Trap exception"); assert_eq!((_s >> 2) & 0x1F, EXC_TR); }
    }

    #[test]
    fn test_conditional_moves() {
        let (mut exec, _) = create_executor_m4();

        // MOVZ r3, r1, r2 (Move r1 to r3 if r2 == 0)
        exec.core.write_gpr(1, 0xDEADBEEF);
        exec.core.write_gpr(2, 0);
        exec.core.write_gpr(3, 0);
        let instr_movz = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_MOVZ);
        assert_eq!(exec.exec(instr_movz), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0xDEADBEEF);

        // MOVZ (No move if r2 != 0)
        exec.core.write_gpr(2, 1);
        exec.core.write_gpr(3, 0);
        assert_eq!(exec.exec(instr_movz), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0);

        // MOVN r3, r1, r2 (Move r1 to r3 if r2 != 0)
        exec.core.write_gpr(2, 1);
        let instr_movn = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_MOVN);
        assert_eq!(exec.exec(instr_movn), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0xDEADBEEF);
    }

    #[test]
    fn test_atomic_ll_sc() {
        let (mut exec, mem) = create_executor();
        
        let addr = 0x1000;
        mem.set_word(addr, 0x12345678);
        exec.core.write_gpr(1, addr as u64);

        // LL r2, 0(r1)
        let instr_ll = make_i(OP_LL, 1, 2, 0);
        assert_eq!(exec.exec(instr_ll), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0x12345678);
        assert!(exec.cache.get_llbit());

        // SC r2, 0(r1) - Should succeed
        exec.core.write_gpr(2, 0xDEADBEEF);
        let instr_sc = make_i(OP_SC, 1, 2, 0);
        assert_eq!(exec.exec(instr_sc), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 1); // Success
        assert_eq!(mem.get_word(addr), 0xDEADBEEF);
        assert!(!exec.cache.get_llbit()); // Cleared

        // SC Fail case (LLBit clear)
        exec.core.write_gpr(2, 0xCAFEBABE);
        assert_eq!(exec.exec(instr_sc), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0); // Fail
        assert_eq!(mem.get_word(addr), 0xDEADBEEF); // Memory unchanged
    }

    #[test]
    fn test_syscall_break() {
        let (mut exec, _) = create_executor();

        let instr_syscall = make_r(OP_SPECIAL, 0, 0, 0, 0, FUNCT_SYSCALL);
        { let _s = exec.exec(instr_syscall); assert!(_s & EXEC_IS_EXCEPTION != 0, "Expected Syscall exception"); assert_eq!((_s >> 2) & 0x1F, EXC_SYS); }

        let instr_break = make_r(OP_SPECIAL, 0, 0, 0, 0, FUNCT_BREAK);
        { let _s = exec.exec(instr_break); assert!(_s & EXEC_IS_EXCEPTION != 0, "Expected Breakpoint exception"); assert_eq!((_s >> 2) & 0x1F, EXC_BP); }
    }

    #[test]
    fn test_remaining_branches() {
        let (mut exec, _) = create_executor();
        exec.core.pc = 0x1000;

        // BLTZ r1, offset
        exec.core.write_gpr(1, -1i64 as u64);
        let instr_bltz = make_i(OP_REGIMM, 1, RT_BLTZ, 4);
        { let _s = exec.exec(instr_bltz); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x1014); }

        // BGEZ r1, offset
        let (mut exec, _) = create_executor();
        exec.core.pc = 0x1000;
        exec.core.write_gpr(1, 0);
        let instr_bgez = make_i(OP_REGIMM, 1, RT_BGEZ, 4);
        { let _s = exec.exec(instr_bgez); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x1014); }

        // BLEZ r1, offset
        let (mut exec, _) = create_executor();
        exec.core.pc = 0x1000;
        exec.core.write_gpr(1, 0);
        let instr_blez = make_i(OP_BLEZ, 1, 0, 4);
        { let _s = exec.exec(instr_blez); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x1014); }

        // BGTZ r1, offset
        let (mut exec, _) = create_executor();
        exec.core.pc = 0x1000;
        exec.core.write_gpr(1, 1);
        let instr_bgtz = make_i(OP_BGTZ, 1, 0, 4);
        { let _s = exec.exec(instr_bgtz); assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x1014); }
    }

    #[test]
    fn test_cop0_64bit() {
        let (mut exec, _) = create_executor();
        exec.core.cp0_status |= STATUS_KX; // Enable 64-bit

        // DMTC0 r1, Index
        exec.core.write_gpr(1, 0x1234567890ABCDEF);
        let instr_dmtc0 = (OP_COP0 << 26) | (RS_DMTC0 << 21) | (1 << 16) | (0 << 11);
        assert_eq!(exec.exec(instr_dmtc0), EXEC_COMPLETE);
        assert_eq!(exec.core.cp0_index, 0x90ABCDEF); // Index is 32-bit, truncated
        
        // DMTC0 r1, Context (u64)
        let instr_dmtc0_ctx = (OP_COP0 << 26) | (RS_DMTC0 << 21) | (1 << 16) | (4 << 11);
        assert_eq!(exec.exec(instr_dmtc0_ctx), EXEC_COMPLETE);
        assert_eq!(exec.core.cp0_context, 0x1234567890ABCDEF);

        // DMFC0 r2, Context
        let instr_dmfc0 = (OP_COP0 << 26) | (RS_DMFC0 << 21) | (2 << 16) | (4 << 11);
        assert_eq!(exec.exec(instr_dmfc0), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0x1234567890ABCDEF);
    }

    #[test]
    fn test_tlb_random_write() {
        // See stack_size comment on test_tlb_instructions above.
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| { test_tlb_random_write_inner(); })
            .unwrap().join().unwrap();
    }
    fn test_tlb_random_write_inner() {
        use crate::mips_tlb::Tlb;
        let (mut exec, _) = create_executor_with_tlb(crate::mips_tlb::MipsTlb::default());

        // TLBWR
        exec.core.cp0_random = 10;
        exec.core.cp0_entryhi = 0xDEADBEEF;
        let instr_tlbwr = (OP_COP0 << 26) | (0x10 << 21) | 0x06;
        assert_eq!(exec.exec(instr_tlbwr), EXEC_COMPLETE);

        let entry = exec.tlb.read(10);
        // Per MIPS R4000 spec: G bit (bit 12) in EntryHi is computed from EntryLo0 and EntryLo1 G bits.
        // Since EntryLo0/1 are not set in this test, G bit will be cleared (bit 12 = 0).
        // EH_WM also zeroes reserved bits 11:8, so 0xDEADBEEF → 0xDEADA0EF.
        const EH_WM: u64 = 0xC000_00FF_FFFF_E0FF;
        assert_eq!(entry.entry_hi, 0xDEADBEEF & EH_WM & !0x1000); // 0xDEADA0EF
    }

    #[test]
    fn test_cache() {
        let (mut exec, _) = create_executor();
        // CACHE 0, 0(r1)
        exec.core.write_gpr(1, 0x1000);
        let instr_cache = make_i(OP_CACHE, 1, 0, 0);
        // Requires CP0 usable
        exec.core.cp0_status |= crate::mips_core::STATUS_CU0;
        assert_eq!(exec.exec(instr_cache), EXEC_COMPLETE);
    }

    #[test]
    fn test_pref() {
        let (mut exec, _) = create_executor_m4();
        exec.core.write_gpr(1, 0x1000);
        // PREF 0, 0(r1)
        let instr_pref = make_i(OP_PREF, 1, 0, 0);
        assert_eq!(exec.exec(instr_pref), EXEC_COMPLETE);
    }

    #[test]
    fn test_dadd_daddu() {
        let (mut exec, _) = create_executor();

        // DADD r3, r1, r2
        exec.core.write_gpr(1, 0x100000000);
        exec.core.write_gpr(2, 0x200000000);
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_DADD);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0x300000000);

        // DADD Overflow
        exec.core.write_gpr(1, i64::MAX as u64);
        exec.core.write_gpr(2, 1);
        { let _s = exec.exec(instr); assert!(_s & EXEC_IS_EXCEPTION != 0, "Expected overflow exception"); assert_eq!((_s >> 2) & 0x1F, EXC_OV); }

        // DADDU (No overflow)
        let instr_u = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_DADDU);
        assert_eq!(exec.exec(instr_u), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), i64::MIN as u64);
    }

    #[test]
    fn test_dsub_dsubu() {
        let (mut exec, _) = create_executor();

        // DSUB r3, r1, r2
        exec.core.write_gpr(1, 0x300000000);
        exec.core.write_gpr(2, 0x100000000);
        let instr = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_DSUB);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0x200000000);

        // DSUB Overflow
        exec.core.write_gpr(1, i64::MIN as u64);
        exec.core.write_gpr(2, 1);
        { let _s = exec.exec(instr); assert!(_s & EXEC_IS_EXCEPTION != 0, "Expected overflow exception"); assert_eq!((_s >> 2) & 0x1F, EXC_OV); }

        // DSUBU (No overflow)
        let instr_u = make_r(OP_SPECIAL, 1, 2, 3, 0, FUNCT_DSUBU);
        assert_eq!(exec.exec(instr_u), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), i64::MAX as u64);
    }

    #[test]
    fn test_daddi_daddiu() {
        let (mut exec, _) = create_executor();

        // DADDI r2, r1, imm
        exec.core.write_gpr(1, 0x100000000);
        let instr = make_i(OP_DADDI, 1, 2, 1);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0x100000001);

        // DADDI Overflow
        exec.core.write_gpr(1, i64::MAX as u64);
        let instr_ov = make_i(OP_DADDI, 1, 2, 1);
        { let _s = exec.exec(instr_ov); assert!(_s & EXEC_IS_EXCEPTION != 0, "Expected overflow exception"); assert_eq!((_s >> 2) & 0x1F, EXC_OV); }

        // DADDIU
        let instr_u = make_i(OP_DADDIU, 1, 2, 1);
        assert_eq!(exec.exec(instr_u), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), i64::MIN as u64);
    }

    #[test]
    fn test_dshifts() {
        let (mut exec, _) = create_executor();

        // DSLL r2, r1, sa
        exec.core.write_gpr(1, 1);
        let instr_dsll = make_r(OP_SPECIAL, 0, 1, 2, 1, FUNCT_DSLL);
        assert_eq!(exec.exec(instr_dsll), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 1 << 1);

        // DSRL
        exec.core.write_gpr(1, 1u64 << 63);
        let instr_dsrl = make_r(OP_SPECIAL, 0, 1, 2, 1, FUNCT_DSRL);
        assert_eq!(exec.exec(instr_dsrl), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 1u64 << 62);

        // DSRA (Arithmetic)
        exec.core.write_gpr(1, !0u64); // -1
        let instr_dsra = make_r(OP_SPECIAL, 0, 1, 2, 1, FUNCT_DSRA);
        assert_eq!(exec.exec(instr_dsra), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), !0u64);

        // DSLL32
        exec.core.write_gpr(1, 1);
        let instr_dsll32 = make_r(OP_SPECIAL, 0, 1, 2, 1, FUNCT_DSLL32); // Shift 1 + 32 = 33
        assert_eq!(exec.exec(instr_dsll32), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 1 << 33);

        // DSRL32
        exec.core.write_gpr(1, 1u64 << 33);
        let instr_dsrl32 = make_r(OP_SPECIAL, 0, 1, 2, 1, FUNCT_DSRL32); // Shift 1 + 32 = 33
        assert_eq!(exec.exec(instr_dsrl32), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 1);

        // DSRA32
        exec.core.write_gpr(1, !0u64);
        let instr_dsra32 = make_r(OP_SPECIAL, 0, 1, 2, 1, FUNCT_DSRA32);
        assert_eq!(exec.exec(instr_dsra32), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), !0u64);
    }

    #[test]
    fn test_dshifts_variable() {
        let (mut exec, _) = create_executor();

        // DSLLV r2, r1, rs (shift amount in rs)
        exec.core.write_gpr(1, 1); // Value
        exec.core.write_gpr(3, 33); // Shift amount
        let instr_dsllv = make_r(OP_SPECIAL, 3, 1, 2, 0, FUNCT_DSLLV);
        assert_eq!(exec.exec(instr_dsllv), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 1 << 33);

        // DSRLV
        exec.core.write_gpr(1, 1u64 << 33);
        exec.core.write_gpr(3, 33);
        let instr_dsrlv = make_r(OP_SPECIAL, 3, 1, 2, 0, FUNCT_DSRLV);
        assert_eq!(exec.exec(instr_dsrlv), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 1);

        // DSRAV
        exec.core.write_gpr(1, !0u64);
        exec.core.write_gpr(3, 33);
        let instr_dsrav = make_r(OP_SPECIAL, 3, 1, 2, 0, FUNCT_DSRAV);
        assert_eq!(exec.exec(instr_dsrav), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), !0u64);
    }

    #[test]
    fn test_dmult_ddiv() {
        let (mut exec, _) = create_executor();

        // DMULT
        exec.core.write_gpr(1, 0x100000000);
        exec.core.write_gpr(2, 0x100000000);
        let instr_dmult = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_DMULT);
        assert_eq!(exec.exec(instr_dmult), EXEC_COMPLETE);
        assert_eq!(exec.core.lo, 0);
        assert_eq!(exec.core.hi, 1);

        // DMULTU
        exec.core.write_gpr(1, !0u64);
        exec.core.write_gpr(2, !0u64);
        let instr_dmultu = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_DMULTU);
        assert_eq!(exec.exec(instr_dmultu), EXEC_COMPLETE);
        assert_eq!(exec.core.lo, 1);
        assert_eq!(exec.core.hi, !0u64 - 1);

        // DDIV
        exec.core.write_gpr(1, 100);
        exec.core.write_gpr(2, 3);
        let instr_ddiv = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_DDIV);
        assert_eq!(exec.exec(instr_ddiv), EXEC_COMPLETE);
        assert_eq!(exec.core.lo, 33);
        assert_eq!(exec.core.hi, 1);

        // DDIVU
        exec.core.write_gpr(1, 100);
        exec.core.write_gpr(2, 3);
        let instr_ddivu = make_r(OP_SPECIAL, 1, 2, 0, 0, FUNCT_DDIVU);
        assert_eq!(exec.exec(instr_ddivu), EXEC_COMPLETE);
        assert_eq!(exec.core.lo, 33);
        assert_eq!(exec.core.hi, 1);
    }

    // Helper for COP1X instructions (MIPS IV)
    fn make_cop1x(rs: u32, rt: u32, rd: u32, sa: u32, funct: u32) -> u32 {
        (OP_COP1X << 26) | ((rs & 0x1F) << 21) | ((rt & 0x1F) << 16) | ((rd & 0x1F) << 11) | ((sa & 0x1F) << 6) | (funct & 0x3F)
    }

    #[test]
    fn test_fpu_round_long() {
        let (mut exec, _) = create_executor();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // ROUND.L.S (Round to Long from Single)
        exec.core.write_fpr_s(1, 3.6);
        let round_l_s = make_cop1_compute(RS_S, 0, 1, 2, FUNCT_FROUND_L);
        assert_eq!(exec.exec(round_l_s), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_l(2) as i64, 4);

        // CEIL.L.D (Ceil to Long from Double)
        exec.core.write_fpr_d(3, 3.1);
        let ceil_l_d = make_cop1_compute(RS_D, 0, 3, 4, FUNCT_FCEIL_L);
        assert_eq!(exec.exec(ceil_l_d), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_l(4) as i64, 4);

        // FLOOR.L.S (Floor to Long from Single)
        exec.core.write_fpr_s(5, 3.9);
        let floor_l_s = make_cop1_compute(RS_S, 0, 5, 6, FUNCT_FFLOOR_L);
        assert_eq!(exec.exec(floor_l_s), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_l(6) as i64, 3);
    }

    #[test]
    fn test_fpu_mov_cond() {
        let (mut exec, _) = create_executor_m4();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // MOVZ.S fd, fs, rt (Move if GPR[rt] == 0)
        exec.core.write_fpr_s(1, 1.23);
        exec.core.write_fpr_s(2, 0.0);
        exec.core.write_gpr(3, 0); // Condition true
        
        // MOVZ.S f2, f1, r3
        // Format: COP1(17) | S(16) | rt(3) | fs(1) | fd(2) | MOVZ(18)
        // Note: make_cop1_compute takes (fmt, ft, fs, fd, funct).
        // For MOVZ, ft field is rt (GPR).
        let movz_s = make_cop1_compute(RS_S, 3, 1, 2, FUNCT_FMOVZ);
        assert_eq!(exec.exec(movz_s), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(2), 1.23);

        // MOVN.D fd, fs, rt (Move if GPR[rt] != 0)
        exec.core.write_fpr_d(4, 4.56);
        exec.core.write_fpr_d(5, 0.0);
        exec.core.write_gpr(6, 0); // Condition false
        
        let movn_d = make_cop1_compute(RS_D, 6, 4, 5, FUNCT_FMOVN);
        assert_eq!(exec.exec(movn_d), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(5), 0.0); // Should not move

        exec.core.write_gpr(6, 1); // Condition true
        assert_eq!(exec.exec(movn_d), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(5), 4.56);
    }

    #[test]
    fn test_fpu_recip_rsqrt() {
        let (mut exec, _) = create_executor_m4();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // RECIP.S
        exec.core.write_fpr_s(1, 2.0);
        let recip_s = make_cop1_compute(RS_S, 0, 1, 2, FUNCT_FRECIP);
        assert_eq!(exec.exec(recip_s), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(2), 0.5);

        // RSQRT.D
        exec.core.write_fpr_d(3, 4.0);
        let rsqrt_d = make_cop1_compute(RS_D, 0, 3, 4, FUNCT_FRSQRT);
        assert_eq!(exec.exec(rsqrt_d), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(4), 0.5); // 1 / sqrt(4) = 1/2 = 0.5
    }

    #[test]
    fn test_cop1x_load_store() {
        let (mut exec, mem) = create_executor_m4();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // LWXC1 fd, index(base)
        // Addr = base + index
        mem.set_word(0x1004, 0x3F800000); // 1.0
        exec.core.write_gpr(1, 0x1000); // Base
        exec.core.write_gpr(2, 4);      // Index
        
        // LWXC1 f3, r2(r1)
        // rs=base(1), rt=index(2), fd=3 (in sa field 10..6)
        let lwxc1 = make_cop1x(1, 2, 0, 3, FUNCT_LWXC1);
        let pc_before = exec.core.pc;
        assert_eq!(exec.exec(lwxc1), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(3), 1.0);
        assert_eq!(exec.core.pc, pc_before.wrapping_add(4), "LWXC1 must advance PC by 4");

        // SWXC1 fs, index(base)
        // fs is in rd field (15..11)
        exec.core.write_fpr_s(4, 2.0); // 0x40000000
        exec.core.write_gpr(5, 0x2000);
        exec.core.write_gpr(6, 8);

        // SWXC1 f4, r6(r5)
        // rs=5, rt=6, rd=4 (fs)
        let swxc1 = make_cop1x(5, 6, 4, 0, FUNCT_SWXC1);
        let pc_before = exec.core.pc;
        assert_eq!(exec.exec(swxc1), EXEC_COMPLETE);
        assert_eq!(mem.get_word(0x2008), 0x40000000);
        assert_eq!(exec.core.pc, pc_before.wrapping_add(4), "SWXC1 must advance PC by 4");

        // LDXC1 / SDXC1 (Double)
        mem.set_double(0x3008, 0x3FF0000000000000); // 1.0 double
        exec.core.write_gpr(7, 0x3000);
        exec.core.write_gpr(8, 8);

        // LDXC1 f0, r8(r7)
        let ldxc1 = make_cop1x(7, 8, 0, 0, FUNCT_LDXC1);
        let pc_before = exec.core.pc;
        assert_eq!(exec.exec(ldxc1), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(0), 1.0);
        assert_eq!(exec.core.pc, pc_before.wrapping_add(4), "LDXC1 must advance PC by 4");

        // SDXC1 f0, r8(r7) -> store 1.0 double to 0x3008
        mem.set_double(0x3008, 0);
        let sdxc1 = make_cop1x(7, 8, 0, 0, FUNCT_SDXC1);
        let pc_before = exec.core.pc;
        assert_eq!(exec.exec(sdxc1), EXEC_COMPLETE);
        assert_eq!(mem.get_double(0x3008), 0x3FF0000000000000);
        assert_eq!(exec.core.pc, pc_before.wrapping_add(4), "SDXC1 must advance PC by 4");
    }

    #[test]
    fn test_cop1x_madd() {
        let (mut exec, _) = create_executor_m4();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // MADD.S fd, fr, fs, ft
        // fd = fs * ft + fr
        exec.core.write_fpr_s(1, 2.0); // fr
        exec.core.write_fpr_s(2, 3.0); // fs
        exec.core.write_fpr_s(3, 4.0); // ft
        
        // MADD.S f4, f1, f2, f3
        // rs=fr(1), rt=ft(3), rd=fs(2), sa=fd(4)
        let madd_s = make_cop1x(1, 3, 2, 4, FUNCT_MADD_S);
        assert_eq!(exec.exec(madd_s), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(4), 14.0); // 3*4 + 2 = 14

        // MSUB.D fd, fr, fs, ft
        // fd = fs * ft - fr
        exec.core.write_fpr_d(5, 2.0); // fr
        exec.core.write_fpr_d(6, 3.0); // fs
        exec.core.write_fpr_d(7, 4.0); // ft
        
        // MSUB.D f8, f5, f6, f7
        let msub_d = make_cop1x(5, 7, 6, 8, FUNCT_MSUB_D);
        assert_eq!(exec.exec(msub_d), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(8), 10.0); // 3*4 - 2 = 10
    }

    // Covers the remaining MIPS IV COP1X fused ops not exercised by test_cop1x_madd
    // (MADD.D, MSUB.S, NMADD.S/D, NMSUB.S/D) — fd = fs*ft +/- fr, negated for NMADD/NMSUB.
    #[test]
    fn test_cop1x_madd_remaining_variants() {
        let (mut exec, _) = create_executor_m4();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // MADD.D fd, fr, fs, ft: fd = fs*ft + fr
        exec.core.write_fpr_d(1, 2.0); // fr
        exec.core.write_fpr_d(2, 3.0); // fs
        exec.core.write_fpr_d(3, 4.0); // ft
        let madd_d = make_cop1x(1, 3, 2, 4, FUNCT_MADD_D);
        assert_eq!(exec.exec(madd_d), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(4), 14.0); // 3*4 + 2 = 14

        // MSUB.S fd, fr, fs, ft: fd = fs*ft - fr
        exec.core.write_fpr_s(5, 2.0); // fr
        exec.core.write_fpr_s(6, 3.0); // fs
        exec.core.write_fpr_s(7, 4.0); // ft
        let msub_s = make_cop1x(5, 7, 6, 8, FUNCT_MSUB_S);
        assert_eq!(exec.exec(msub_s), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(8), 10.0); // 3*4 - 2 = 10

        // NMADD.S fd, fr, fs, ft: fd = -(fs*ft + fr)
        exec.core.write_fpr_s(9, 2.0);  // fr
        exec.core.write_fpr_s(10, 3.0); // fs
        exec.core.write_fpr_s(11, 4.0); // ft
        let nmadd_s = make_cop1x(9, 11, 10, 12, FUNCT_NMADD_S);
        assert_eq!(exec.exec(nmadd_s), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(12), -14.0); // -(3*4 + 2) = -14

        // NMADD.D fd, fr, fs, ft: fd = -(fs*ft + fr)
        exec.core.write_fpr_d(13, 2.0); // fr
        exec.core.write_fpr_d(14, 3.0); // fs
        exec.core.write_fpr_d(15, 4.0); // ft
        let nmadd_d = make_cop1x(13, 15, 14, 16, FUNCT_NMADD_D);
        assert_eq!(exec.exec(nmadd_d), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(16), -14.0); // -(3*4 + 2) = -14

        // NMSUB.S fd, fr, fs, ft: fd = -(fs*ft - fr)
        exec.core.write_fpr_s(17, 2.0); // fr
        exec.core.write_fpr_s(18, 3.0); // fs
        exec.core.write_fpr_s(19, 4.0); // ft
        let nmsub_s = make_cop1x(17, 19, 18, 20, FUNCT_NMSUB_S);
        assert_eq!(exec.exec(nmsub_s), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(20), -10.0); // -(3*4 - 2) = -10

        // NMSUB.D fd, fr, fs, ft: fd = -(fs*ft - fr)
        exec.core.write_fpr_d(21, 2.0); // fr
        exec.core.write_fpr_d(22, 3.0); // fs
        exec.core.write_fpr_d(23, 4.0); // ft
        let nmsub_d = make_cop1x(21, 23, 22, 24, FUNCT_NMSUB_D);
        assert_eq!(exec.exec(nmsub_d), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(24), -10.0); // -(3*4 - 2) = -10
    }

    // Covers SDXC1 (missing from test_cop1x_load_store, which only checks LWXC1/SWXC1/LDXC1)
    // and PREFX (COP1X prefetch — architecturally a hint/no-op).
    #[test]
    fn test_cop1x_sdxc1_and_prefx() {
        let (mut exec, mem) = create_executor_m4();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // SDXC1 fs, index(base): store doubleword FPR to base+index
        exec.core.write_fpr_d(4, 1.0); // 0x3FF0000000000000
        exec.core.write_gpr(5, 0x4000); // base
        exec.core.write_gpr(6, 8);      // index

        // SDXC1 f4, r6(r5) — rs=base(5), rt=index(6), rd=fs(4)
        let sdxc1 = make_cop1x(5, 6, 4, 0, FUNCT_SDXC1);
        assert_eq!(exec.exec(sdxc1), EXEC_COMPLETE);
        assert_eq!(mem.get_double(0x4008), 0x3FF0000000000000);

        // PREFX is a hint; must complete without side effects.
        let prefx = make_cop1x(5, 6, 0, 0, FUNCT_PREFX);
        assert_eq!(exec.exec(prefx), EXEC_COMPLETE);
    }

    // Covers LLD/SCD (64-bit load-linked/store-conditional), the MIPS III siblings
    // of LL/SC that had no dedicated test.
    #[test]
    fn test_lld_scd() {
        let (mut exec, mem) = create_executor();

        let addr = 0x2000u64;
        mem.set_double(addr, 0x0123456789ABCDEF);
        exec.core.write_gpr(1, addr);

        // LLD r2, 0(r1)
        let instr_lld = make_i(OP_LLD, 1, 2, 0);
        assert_eq!(exec.exec(instr_lld), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0x0123456789ABCDEF);
        assert!(exec.cache.get_llbit());

        // SCD r2, 0(r1) — should succeed
        exec.core.write_gpr(2, 0xFEDCBA9876543210);
        let instr_scd = make_i(OP_SCD, 1, 2, 0);
        assert_eq!(exec.exec(instr_scd), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 1); // Success
        assert_eq!(mem.get_double(addr), 0xFEDCBA9876543210);
        assert!(!exec.cache.get_llbit()); // Cleared

        // SCD fail case (LLBit clear from prior successful store)
        exec.core.write_gpr(2, 0xAAAAAAAAAAAAAAAA);
        assert_eq!(exec.exec(instr_scd), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0); // Fail
        assert_eq!(mem.get_double(addr), 0xFEDCBA9876543210); // Memory unchanged
    }

    #[test]
    fn test_movci() {
        let (mut exec, _) = create_executor_m4();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Helper to create MOVCI instruction
        // MOVCI: op=SPECIAL, rs=rs, cc|nd|tf in bits [20:16], rd=rd, funct=MOVCI
        fn make_movci(rd: u32, rs: u32, cc: u32, tf: bool) -> u32 {
            let tf_bit = if tf { 1 } else { 0 };
            (0 << 26) | (rs << 21) | (cc << 18) | (tf_bit << 16) | (rd << 11) | FUNCT_MOVCI
        }

        // Test MOVT (move if CC true) with CC0
        exec.core.set_fpu_cc(0, true);
        exec.core.write_gpr(2, 0x1234);
        exec.core.write_gpr(3, 0x5678);

        // MOVT r3, r2, $fcc0 - should move because CC0 is true
        let movt_cc0 = make_movci(3, 2, 0, true);
        assert_eq!(exec.exec(movt_cc0), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0x1234);

        // Test MOVF (move if CC false) with CC0
        exec.core.set_fpu_cc(0, true);
        exec.core.write_gpr(4, 0xAAAA);
        exec.core.write_gpr(5, 0xBBBB);

        // MOVF r5, r4, $fcc0 - should NOT move because CC0 is true
        let movf_cc0 = make_movci(5, 4, 0, false);
        assert_eq!(exec.exec(movf_cc0), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(5), 0xBBBB); // Unchanged

        // Test with CC3
        exec.core.set_fpu_cc(3, false);
        exec.core.write_gpr(6, 0xDEAD);
        exec.core.write_gpr(7, 0xBEEF);

        // MOVF r7, r6, $fcc3 - should move because CC3 is false
        let movf_cc3 = make_movci(7, 6, 3, false);
        assert_eq!(exec.exec(movf_cc3), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(7), 0xDEAD);

        // Test MOVT with false condition - no move
        exec.core.set_fpu_cc(5, false);
        exec.core.write_gpr(8, 0x1111);
        exec.core.write_gpr(9, 0x2222);

        // MOVT r9, r8, $fcc5 - should NOT move because CC5 is false
        let movt_cc5 = make_movci(9, 8, 5, true);
        assert_eq!(exec.exec(movt_cc5), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(9), 0x2222); // Unchanged
    }

    #[test]
    fn test_fpu_movcf() {
        let (mut exec, _) = create_executor_m4();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Helper to create MOVCF.S/D instruction
        // Format: op=COP1, rs=fmt, cc|nd|tf in bits [20:16], fs=fs, fd=fd, funct=FMOVCF
        fn make_fmovcf(fmt: u32, fd: u32, fs: u32, cc: u32, tf: bool) -> u32 {
            let tf_bit = if tf { 1 } else { 0 };
            (OP_COP1 << 26) | (fmt << 21) | (cc << 18) | (tf_bit << 16) | (fs << 11) | (fd << 6) | FUNCT_FMOVCF
        }

        // Test MOVT.S with CC0 true
        exec.core.set_fpu_cc(0, true);
        exec.core.write_fpr_s(1, 3.14);
        exec.core.write_fpr_s(2, 2.71);

        // MOVT.S f2, f1, $fcc0 - should move
        let movt_s_cc0 = make_fmovcf(RS_S, 2, 1, 0, true);
        assert_eq!(exec.exec(movt_s_cc0), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(2), 3.14);

        // Test MOVF.D with CC2 false
        exec.core.set_fpu_cc(2, false);
        exec.core.write_fpr_d(3, 1.414);
        exec.core.write_fpr_d(4, 2.718);

        // MOVF.D f4, f3, $fcc2 - should move
        let movf_d_cc2 = make_fmovcf(RS_D, 4, 3, 2, false);
        assert_eq!(exec.exec(movf_d_cc2), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(4), 1.414);

        // Test MOVT.S with false condition - no move
        exec.core.set_fpu_cc(4, false);
        exec.core.write_fpr_s(5, 9.99);
        exec.core.write_fpr_s(6, 8.88);

        // MOVT.S f6, f5, $fcc4 - should NOT move
        let movt_s_cc4 = make_fmovcf(RS_S, 6, 5, 4, true);
        assert_eq!(exec.exec(movt_s_cc4), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_s(6), 8.88); // Unchanged

        // Test MOVF.D with true condition - no move
        exec.core.set_fpu_cc(7, true);
        assert_eq!(exec.core.get_fpu_cc(7), true, "CC7 should be set to true");
        exec.core.write_fpr_d(10, 5.55);
        assert_eq!(exec.core.get_fpu_cc(7), true, "CC7 should still be true after write_fpr_d");
        exec.core.write_fpr_d(12, 6.66);
        assert_eq!(exec.core.get_fpu_cc(7), true, "CC7 should still be true");

        // MOVF.D f12, f10, $fcc7 - should NOT move
        let movf_d_cc7 = make_fmovcf(RS_D, 12, 10, 7, false);
        assert_eq!(exec.exec(movf_d_cc7), EXEC_COMPLETE);
        assert_eq!(exec.core.read_fpr_d(12), 6.66); // Unchanged
    }

    #[test]
    fn test_fpu_multi_cc_compare_and_branch() {
        let (mut exec, _) = create_executor();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        // Helper to create FPU compare with CC field
        // c.eq.s $cc, fs, ft
        // Format: op=COP1, rs=S, ft=ft, fs=fs, cc in bits [10:8], funct=FC_EQ
        fn make_compare_s(cc: u32, fs: u32, ft: u32, cond: u32) -> u32 {
            (OP_COP1 << 26) | (RS_S << 21) | (ft << 16) | (fs << 11) | ((cc & 0x7) << 8) | cond
        }

        // Helper to create BC1T/BC1F with CC field
        // bc1t $cc, offset
        // Format: op=COP1, rs=BC1, cc in bits [20:18], tf in bit [16], likely in bit [17], offset
        fn make_bc1(cc: u32, tf: bool, likely: bool, offset: i16) -> u32 {
            let tf_bit = if tf { 1 } else { 0 };
            let likely_bit = if likely { 2 } else { 0 };
            let rt = tf_bit | likely_bit | (cc << 2); // cc goes in upper bits of rt field
            (OP_COP1 << 26) | (RS_BC1 << 21) | (rt << 16) | (offset as u16 as u32)
        }

        // Set up FPU registers
        exec.core.write_fpr_s(1, 5.0);
        exec.core.write_fpr_s(2, 5.0);
        exec.core.write_fpr_s(3, 3.0);
        exec.core.write_fpr_s(4, 7.0);

        // Compare equal on CC0: f1 == f2 (5.0 == 5.0) -> true
        let cmp_eq_cc0 = make_compare_s(0, 1, 2, FUNCT_FC_EQ);
        assert_eq!(exec.exec(cmp_eq_cc0), EXEC_COMPLETE);
        assert_eq!(exec.core.get_fpu_cc(0), true);

        // Compare less than on CC3: f3 < f4 (3.0 < 7.0) -> true
        let cmp_lt_cc3 = make_compare_s(3, 3, 4, FUNCT_FC_LT);
        assert_eq!(exec.exec(cmp_lt_cc3), EXEC_COMPLETE);
        assert_eq!(exec.core.get_fpu_cc(3), true);

        // Compare equal on CC5: f3 == f4 (3.0 == 7.0) -> false
        let cmp_eq_cc5 = make_compare_s(5, 3, 4, FUNCT_FC_EQ);
        assert_eq!(exec.exec(cmp_eq_cc5), EXEC_COMPLETE);
        assert_eq!(exec.core.get_fpu_cc(5), false);

        // Test BC1T on CC0 (should branch - CC0 is true)
        exec.core.pc = 0x1000;
        let bc1t_cc0 = make_bc1(0, true, false, 4); // offset 4 -> PC+4+(4<<2) = 0x1000+4+16 = 0x1014
        let result = exec.exec(bc1t_cc0);
        { assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x1014); }

        // Test BC1F on CC0 (should not branch - CC0 is true)
        exec.core.pc = 0x2000;
        let bc1f_cc0 = make_bc1(0, false, false, 4);
        let result = exec.exec(bc1f_cc0);
        assert_eq!(result, EXEC_COMPLETE);

        // Test BC1T on CC3 (should branch - CC3 is true)
        exec.core.pc = 0x3000;
        let bc1t_cc3 = make_bc1(3, true, false, 8); // offset 8 -> PC+4+(8<<2) = 0x3000+4+32 = 0x3024
        let result = exec.exec(bc1t_cc3);
        { assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x3024); }

        // Test BC1F on CC5 (should branch - CC5 is false)
        exec.core.pc = 0x4000;
        let bc1f_cc5 = make_bc1(5, false, false, -2); // offset -2 -> PC+4+(-2<<2) = 0x4000+4-8 = 0x3FFC
        let result = exec.exec(bc1f_cc5);
        { assert!(exec.core.in_delay_slot, "Expected BranchDelay"); assert_eq!(exec.core.delay_slot_target, 0x3FFC); }

        // Test BC1TL (likely) on CC5 (should not take, nullify delay slot - CC5 is false)
        exec.core.pc = 0x5000;
        let bc1tl_cc5 = make_bc1(5, true, true, 4);
        let result = exec.exec(bc1tl_cc5);
        assert_eq!(result, EXEC_COMPLETE);
    }

    #[test]
    fn test_fpu_cc1_7_visible_via_fcsr_readback() {
        // Regression test: cc1..cc7 (FCSR bits 25..31) must be readable via a plain
        // `cfc1 $rt, $31` (FCSR), not just via get_fpu_cc()/BC1. Before this was fixed,
        // c.cond.fmt with a non-zero cc field only updated a side `fpu_fccr` cache and
        // synced cc0 into FCSR bit 23, silently dropping cc1..cc7 from FCSR itself --
        // so guest code doing `c.lt.d $fcc2, ...; cfc1 $t0, $31; andi $t0, 1<<27` would
        // always see 0 for that condition, regardless of the actual compare result.
        let (mut exec, _) = create_executor();
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();

        for cc in 0u32..8 {
            exec.core.set_fpu_cc(cc, true);
            let expected_bit = if cc == 0 { 23 } else { 24 + cc };

            let cfc1_instr = make_cop1_move(RS_CFC1, 10, 31);
            assert_eq!(exec.exec(cfc1_instr), EXEC_COMPLETE);
            let fcsr = exec.core.read_gpr(10) as u32;
            assert_eq!((fcsr >> expected_bit) & 1, 1, "cc{cc} not visible in FCSR readback");

            exec.core.set_fpu_cc(cc, false);
            assert_eq!(exec.core.read_fpu_control(31) >> expected_bit & 1, 0);
        }

        // Multiple cc bits set simultaneously must all coexist in one FCSR read.
        exec.core.set_fpu_cc(0, true);
        exec.core.set_fpu_cc(2, true);
        exec.core.set_fpu_cc(7, true);
        let fcsr = exec.core.read_fpu_control(31);
        assert_eq!((fcsr >> 23) & 1, 1, "cc0");
        assert_eq!((fcsr >> 26) & 1, 1, "cc2");
        assert_eq!((fcsr >> 31) & 1, 1, "cc7");
        assert_eq!(exec.core.get_fpu_cc(1), false);
        assert_eq!(exec.core.get_fpu_cc(3), false);

        // Writing FCSR directly (ctc1 $31) must also make cc1..cc7 visible to
        // get_fpu_cc()/BC1, not just cc0.
        exec.core.write_gpr(11, (1u64 << 23) | (1u64 << 30)); // cc0 and cc6
        let ctc1_instr = make_cop1_move(RS_CTC1, 11, 31);
        assert_eq!(exec.exec(ctc1_instr), EXEC_COMPLETE);
        assert!(exec.core.get_fpu_cc(0));
        assert!(exec.core.get_fpu_cc(6));
        assert!(!exec.core.get_fpu_cc(2));

        // FCCR (control reg 25) must round-trip the same 8 condition bits as FCSR.
        let fccr = exec.core.read_fpu_control(25);
        assert_eq!(fccr & 0x1, 1, "FCCR cc0");
        assert_eq!((fccr >> 6) & 0x1, 1, "FCCR cc6");
        exec.core.write_fpu_control(25, 0xFF);
        for cc in 0u32..8 {
            assert!(exec.core.get_fpu_cc(cc), "cc{cc} should be set after FCCR write");
        }
    }

    /// Same shape as `MockMemory`, but with a real per-page gen counter —
    /// used only by `test_r4000cache_step_sequence`, the one test in this
    /// file that actually wants jitv2 dispatch to fire. Every other test
    /// uses the shared `MockMemory` via `create_executor`/friends, which
    /// disable every jitv2 gate at the executor level (see
    /// `disable_jitv2`'s doc comment) so jitv2 never intercepts them.
    #[cfg(feature = "jitv2")]
    struct JitCapableMockMemory {
        data: Mutex<HashMap<u64, u8>>,
        gens: Mutex<HashMap<u32, Box<std::sync::atomic::AtomicU64>>>,
    }

    #[cfg(feature = "jitv2")]
    impl JitCapableMockMemory {
        fn new() -> Self {
            Self { data: Mutex::new(HashMap::new()), gens: Mutex::new(HashMap::new()) }
        }
        fn set_double(&self, addr: u64, val: u64) {
            for (i, b) in val.to_be_bytes().iter().enumerate() {
                self.data.lock().unwrap().insert(addr + i as u64, *b);
            }
        }
        fn get_word(&self, addr: u64) -> u32 {
            let mut bytes = [0u8; 4];
            for i in 0..4 { bytes[i] = *self.data.lock().unwrap().get(&(addr + i as u64)).unwrap_or(&0); }
            u32::from_be_bytes(bytes)
        }
        fn get_double(&self, addr: u64) -> u64 {
            let mut bytes = [0u8; 8];
            for i in 0..8 { bytes[i] = *self.data.lock().unwrap().get(&(addr + i as u64)).unwrap_or(&0); }
            u64::from_be_bytes(bytes)
        }
    }

    #[cfg(feature = "jitv2")]
    impl BusDevice for JitCapableMockMemory {
        fn read8(&self, addr: u32) -> BusRead8 {
            BusRead8::ok(*self.data.lock().unwrap().get(&(addr as u64)).unwrap_or(&0))
        }
        fn write8(&self, addr: u32, val: u8) -> u32 { self.data.lock().unwrap().insert(addr as u64, val); BUS_OK }
        fn read16(&self, addr: u32) -> BusRead16 {
            let a = (addr & !1) as u64;
            let mut b = [0u8; 2];
            for i in 0..2 { b[i] = *self.data.lock().unwrap().get(&(a + i as u64)).unwrap_or(&0); }
            BusRead16::ok(u16::from_be_bytes(b))
        }
        fn write16(&self, addr: u32, val: u16) -> u32 {
            let a = (addr & !1) as u64;
            for (i, b) in val.to_be_bytes().iter().enumerate() { self.data.lock().unwrap().insert(a + i as u64, *b); }
            BUS_OK
        }
        fn read32(&self, addr: u32) -> BusRead32 { BusRead32::ok(self.get_word((addr & !3) as u64)) }
        fn write32(&self, addr: u32, val: u32) -> u32 {
            let a = (addr & !3) as u64;
            for (i, b) in val.to_be_bytes().iter().enumerate() { self.data.lock().unwrap().insert(a + i as u64, *b); }
            BUS_OK
        }
        fn read64(&self, addr: u32) -> BusRead64 { BusRead64::ok(self.get_double((addr & !7) as u64)) }
        fn write64(&self, addr: u32, val: u64) -> u32 { self.set_double((addr & !7) as u64, val); BUS_OK }
        fn gen_ptr(&self, addr: u32) -> *const std::sync::atomic::AtomicU64 {
            let page = addr / crate::jitv2::PAGE_SIZE;
            let mut gens = self.gens.lock().unwrap();
            let counter = gens.entry(page).or_insert_with(|| Box::new(std::sync::atomic::AtomicU64::new(0)));
            counter.as_ref() as *const std::sync::atomic::AtomicU64
        }
    }

    #[test]
    #[cfg(feature = "jitv2")]
    fn test_r4000cache_step_sequence() {
        // Exercise the full R4400Cache path: kseg0 fetch → L2 fill → L1I fill → exec_decoded.
        // PC 0x80000000 → phys 0x00000000 (kseg0, cacheable). Uses
        // JitCapableMockMemory (not the shared MockMemory) plus
        // install_jit_hooks specifically so jitv2 dispatch is exercisable
        // here, unlike every other test in this file.
        let mem = Arc::new(JitCapableMockMemory::new());
        let mem_bus: Arc<dyn BusDevice> = mem.clone();
        let cfg = MipsCpuConfig::indy();
        let mut exec: MipsExecutor<PassthroughTlb, R4400Cache> = MipsExecutor::new(mem_bus, PassthroughTlb::default(), &cfg);
        exec.install_jit_hooks();

        // Write instructions into memory as big-endian u32s packed into u64 chunks.
        // Offset 0:  ADDIU r1, r0, 42
        // Offset 4:  ADDIU r2, r0, 7
        // Offset 8:  ADDU  r3, r1, r2
        // Offset 12: NOP (SLL r0,r0,0)
        let addiu_r1 = make_i(OP_ADDIU as u32, 0, 1, 42);
        let addiu_r2 = make_i(OP_ADDIU as u32, 0, 2, 7);
        let addu_r3  = make_r(OP_SPECIAL as u32, 1, 2, 3, 0, FUNCT_ADDU as u32);
        let nop      = 0u32;

        // L2 fills via read64; pack two instructions per 8-byte chunk (big-endian MIPS)
        mem.set_double(0,  ((addiu_r1 as u64) << 32) | addiu_r2 as u64);
        mem.set_double(8,  ((addu_r3  as u64) << 32) | nop as u64);

        // kseg0 PC must be sign-extended to 64-bit (0x80000000 has sign bit set)
        let pc_base: u64 = 0xFFFFFFFF80000000;
        exec.core.pc = pc_base;

        assert_eq!(exec.step_int(), EXEC_COMPLETE, "ADDIU r1,r0,42");
        assert_eq!(exec.core.read_gpr(1), 42,  "r1 should be 42");

        assert_eq!(exec.step_int(), EXEC_COMPLETE, "ADDIU r2,r0,7");
        assert_eq!(exec.core.read_gpr(2), 7,   "r2 should be 7");

        assert_eq!(exec.step_int(), EXEC_COMPLETE, "ADDU r3,r1,r2");
        assert_eq!(exec.core.read_gpr(3), 49,  "r3 should be 49");

        // Re-execute the same sequence — this time all three instructions hit L1I
        // (same virtual address, same cache line still valid).
        exec.core.pc = pc_base;
        exec.core.write_gpr(1, 0);
        exec.core.write_gpr(2, 0);
        exec.core.write_gpr(3, 0);
        assert_eq!(exec.step_int(), EXEC_COMPLETE, "cache-hit ADDIU r1,r0,42");
        assert_eq!(exec.core.read_gpr(1), 42);
        assert_eq!(exec.step_int(), EXEC_COMPLETE, "cache-hit ADDIU r2,r0,7");
        assert_eq!(exec.core.read_gpr(2), 7);
        assert_eq!(exec.step_int(), EXEC_COMPLETE, "cache-hit ADDU r3,r1,r2");
        assert_eq!(exec.core.read_gpr(3), 49);
    }

    #[test]
    fn test_decoded_instr_size() {
        // DecodedInstr lives in the flat L2 decoded-instruction array (SIZE/4
        // entries — hundreds of thousands at the default 1MB L2), so any growth
        // here multiplies across the whole array and degrades cache-line packing
        // for every fetch, not just branches. A prior attempt to add delay-slot
        // fusion lookahead as `next: Option<u32>` grew this from 24 to 32 bytes
        // (Option<u32> can't be niche-packed) and measurably regressed
        // whetstone/dhrystone. The flags-byte + transient-imm design must not
        // reintroduce that growth.
        assert_eq!(std::mem::size_of::<DecodedInstr>(), 24, "DecodedInstr grew past its 24-byte budget");
    }

    #[test]
    #[cfg(feature = "opcodefusion")]
    fn test_branch_nop_fusion() {
        // Exercise the fused branch+NOP-delay-slot path (see FLAG_IMM_IS_NEXT):
        // BEQ/BNE/JR taken through the real I-cache with a literal NOP in the
        // delay slot must jump straight to the target in one step(), never
        // setting in_delay_slot, and must still execute a normal instruction
        // placed after the NOP when not taken.
        let (mut exec, mem) = create_executor_with_r4000cache();
        let pc_base: u64 = 0xFFFFFFFF80000000;

        // Offset 0:  BEQ r0, r0, +3 (always taken -> offset 0 + 4 + 3*4 = 16)
        // Offset 4:  NOP (delay slot, fused away)
        // Offset 8:  ADDIU r1, r0, 111  (must NOT execute — skipped by the branch)
        // Offset 12: ADDIU r2, r0, 222  (must NOT execute)
        // Offset 16: ADDIU r3, r0, 42   (branch target)
        let beq_taken = make_i(OP_BEQ as u32, 0, 0, 3);
        let nop = 0u32;
        let addiu_r1 = make_i(OP_ADDIU as u32, 0, 1, 111);
        let addiu_r2 = make_i(OP_ADDIU as u32, 0, 2, 222);
        let addiu_r3 = make_i(OP_ADDIU as u32, 0, 3, 42);
        mem.set_double(0,  ((beq_taken as u64) << 32) | nop as u64);
        mem.set_double(8,  ((addiu_r1 as u64) << 32) | addiu_r2 as u64);
        mem.set_double(16, ((addiu_r3 as u64) << 32) | nop as u64);

        // Not-taken case lives in a separate 128-byte L2 line (offset 128+) so
        // writing it doesn't mutate a line that's already been fetched/cached
        // above — memory must be written before it's ever fetched, same as real
        // hardware requires a cache flush for genuinely self-modifying code.
        let bne_never = make_i(OP_BNE as u32, 0, 0, 3);
        mem.set_double(128, ((bne_never as u64) << 32) | nop as u64);
        mem.set_double(136, ((addiu_r1 as u64) << 32) | nop as u64);

        // JR ra fused case in its own line too.
        let jr_ra = make_r(OP_SPECIAL as u32, 31, 0, 0, 0, FUNCT_JR as u32);
        mem.set_double(256, ((jr_ra as u64) << 32) | nop as u64);

        exec.core.pc = pc_base;
        let s = exec.step_int();
        assert_eq!(s, EXEC_COMPLETE, "fused taken BEQ sets PC directly, no delay slot");
        assert_eq!(exec.core.pc, pc_base + 16, "PC should jump straight to branch target");
        assert!(!exec.core.in_delay_slot, "fused path must never set in_delay_slot");

        let s = exec.step_int();
        assert_eq!(s, EXEC_COMPLETE, "ADDIU r3,r0,42");
        assert_eq!(exec.core.read_gpr(3), 42);
        assert_eq!(exec.core.read_gpr(1), 0, "skipped ADDIU r1 must not have executed");
        assert_eq!(exec.core.read_gpr(2), 0, "skipped ADDIU r2 must not have executed");

        // Not-taken case: BNE r0,r0 (never taken) + NOP delay slot, then a real
        // instruction right after — must execute (PC+=8 skip of branch+NOP only).
        exec.core.pc = pc_base + 128;
        exec.core.write_gpr(1, 0);
        exec.step_int();
        assert_eq!(exec.core.pc, pc_base + 136, "PC should skip branch+NOP (PC+=8)");
        assert!(!exec.core.in_delay_slot);
        assert_eq!(exec.step_int(), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 111, "instruction after skipped NOP must execute");

        // JR ra fused with NOP delay slot: unconditional, PC=target directly.
        exec.core.pc = pc_base + 256;
        exec.core.write_gpr(31, pc_base + 300);
        let s = exec.step_int();
        assert_eq!(s, EXEC_COMPLETE, "fused JR sets PC directly");
        assert_eq!(exec.core.pc, pc_base + 300, "PC should jump straight to r31");
        assert!(!exec.core.in_delay_slot);
    }

    #[test]
    #[cfg(feature = "opcodefusion")]
    fn test_branch_nop_fusion_self_modifying_delay_slot() {
        // If the delay slot word is rewritten AFTER the branch has already been
        // fetched/decoded once (picking the fused handler), the writeback path
        // must mark the branch's slot FLAG_NOT_DECODED again — otherwise the
        // cached fused handler pointer goes stale and the newly-written
        // delay-slot instruction would be silently skipped forever.
        let (mut exec, mem) = create_executor_with_r4000cache();
        let pc_base: u64 = 0xFFFFFFFF80000000;

        let beq_taken = make_i(OP_BEQ as u32, 0, 0, 3);
        let nop = 0u32;
        let addiu_r3 = make_i(OP_ADDIU as u32, 0, 3, 42);
        mem.set_double(0, ((beq_taken as u64) << 32) | nop as u64);
        mem.set_double(16, ((addiu_r3 as u64) << 32) | nop as u64);

        // First execution: decode-time fusion picks exec_beq_nop.
        exec.core.pc = pc_base;
        exec.step_int();
        assert_eq!(exec.core.pc, pc_base + 16);
        exec.step_int();
        assert_eq!(exec.core.read_gpr(3), 42);

        // Now overwrite the delay slot (offset 4) with a real instruction via the
        // standard self-modifying-code idiom real guest code must use: SW, then
        // CACHE Hit_Writeback_Invalidate (D) to push the store to L2, then CACHE
        // Hit_Invalidate (I) so the next fetch re-reads L2 instead of a stale
        // I-cache line. A raw backing-memory poke wouldn't be visible to the
        // cache at all — this exercises the real invalidation path in
        // writeback_l1d_line that syncs l2.instrs (and, via our fetch() change,
        // recomputes `next` and forces a re-decode).
        let addiu_r1 = make_i(OP_ADDIU as u32, 0, 1, 77);
        exec.core.write_gpr(4, addiu_r1 as u64);
        exec.core.write_gpr(5, pc_base);
        let sw = make_i(OP_SW, 5, 4, 4); // SW r4, 4(r5)
        assert_eq!(exec.exec(sw), EXEC_COMPLETE);
        let cache_hwbinv_d = make_i(OP_CACHE, 5, C_HWBINV | CACH_PD, 4);
        assert_eq!(exec.exec(cache_hwbinv_d), EXEC_COMPLETE);
        let cache_hinv_i = make_i(OP_CACHE, 5, C_HINV | CACH_PI, 4);
        assert_eq!(exec.exec(cache_hinv_i), EXEC_COMPLETE);

        exec.core.pc = pc_base;
        exec.core.write_gpr(1, 0);
        exec.core.write_gpr(3, 0);
        let s = exec.step_int();
        assert!(exec.core.in_delay_slot, "must re-decode as unfused branch once delay slot is no longer a NOP");
        assert_eq!(exec.core.delay_slot_target, pc_base + 16);
        let s2 = exec.step_int();
        assert_eq!(s2, EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 77, "rewritten delay-slot instruction must execute, not be skipped");
    }

    #[test]
    #[cfg(feature = "opcodefusion")]
    fn test_lui_ori_addiu_fusion() {
        // LUI+ORI and LUI+ADDIU are the standard 32-bit-immediate materialization
        // idiom (`lui rX,hi; {ori,addiu} rX,rX,lo`) and should fuse into a single
        // handler call (exec_lui_imm32 / exec_lui_simm32) when the second
        // instruction is same-register. Neither ORI nor ADDIU can fault, so
        // there's no exception-exclusion concern like with branch delay slots.
        let (mut exec, mem) = create_executor_with_r4000cache();
        let pc_base: u64 = 0xFFFFFFFF80000000;

        // All instruction words are written before the first fetch touches this
        // L2 line — writing to `mem` after a line is cached doesn't invalidate
        // it (that's what real self-modifying code needs an explicit CACHE flush
        // for; see test_branch_nop_fusion_self_modifying_delay_slot), so every
        // word this test depends on must be in place up front.
        //
        // Offset 0:  LUI v1, 0xa874 ; ORI v1, v1, 0x74f0 (no carry: bit15 of lo
        //            set, but ORI never carries — just verifies OR-combine).
        // Offset 8:  LUI v0, 0x1234 ; ADDIU v0, v0, 0x5678 (lo16 bit15 clear: no carry).
        // Offset 16: LUI v0, 0x1234 ; ADDIU v0, v0, 0xFFFF (lo16 = -1: carries,
        //            borrowing 1 from the hi16 half — must match exec_addiu's
        //            wrapping-add semantics exactly, not a naive OR).
        // Offset 24: LUI v0, 0x1234 ; ADDIU v1, v0, 0x5678 — different DEST
        //            register (v1 vs v0) — must NOT fuse.
        // Offset 32: NOP NOP (landing pad for the unfused case above).
        let lui = make_i(OP_LUI as u32, 0, 3, 0xa874);
        let ori = make_i(OP_ORI as u32, 3, 3, 0x74f0);
        let lui2 = make_i(OP_LUI as u32, 0, 2, 0x1234);
        let addiu2 = make_i(OP_ADDIU as u32, 2, 2, 0x5678);
        let lui3 = make_i(OP_LUI as u32, 0, 2, 0x1234);
        let addiu3 = make_i(OP_ADDIU as u32, 2, 2, 0xFFFF);
        let lui4 = make_i(OP_LUI as u32, 0, 2, 0x1234);
        let addiu4 = make_i(OP_ADDIU as u32, 2, 3, 0x5678);
        let nop = 0u32;
        mem.set_double(0, ((lui as u64) << 32) | ori as u64);
        mem.set_double(8, ((lui2 as u64) << 32) | addiu2 as u64);
        mem.set_double(16, ((lui3 as u64) << 32) | addiu3 as u64);
        mem.set_double(24, ((lui4 as u64) << 32) | addiu4 as u64);
        mem.set_double(32, ((nop as u64) << 32) | nop as u64);

        exec.core.pc = pc_base;
        let s = exec.step_int();
        assert_eq!(s, EXEC_COMPLETE, "fused LUI+ORI should complete in one step");
        assert_eq!(exec.core.pc, pc_base + 8, "PC should skip both fused instructions");
        assert_eq!(exec.core.read_gpr(3) as u32, 0xa87474f0);
        // Upper 32 bits: LUI's result is sign-extended (0xa874<<16 has bit31 set),
        // and ORI never touches or clears the upper bits — must still be all 1s.
        assert_eq!(exec.core.read_gpr(3), 0xFFFFFFFFa87474f0);

        let s = exec.step_int();
        assert_eq!(s, EXEC_COMPLETE, "fused LUI+ADDIU should complete in one step");
        assert_eq!(exec.core.pc, pc_base + 16);
        assert_eq!(exec.core.read_gpr(2), 0x12345678);

        let s = exec.step_int();
        assert_eq!(s, EXEC_COMPLETE);
        assert_eq!(exec.core.pc, pc_base + 24);
        // lui=0x12340000, addiu adds sign_extend(0xFFFF)=-1 -> 0x1233FFFF
        assert_eq!(exec.core.read_gpr(2), 0x1233FFFF);

        exec.core.pc = pc_base + 24;
        exec.core.write_gpr(2, 0);
        exec.core.write_gpr(3, 0);
        let s = exec.step_int();
        assert_eq!(s, EXEC_COMPLETE, "unfused LUI should report plain EXEC_COMPLETE");
        assert_eq!(exec.core.pc, pc_base + 28, "unfused: PC advances by 4, not 8");
        // 0x1234's bit 15 is clear, so LUI's sign-extension is a no-op here.
        assert_eq!(exec.core.read_gpr(2), 0x0000000012340000, "LUI alone must have run");
        let s2 = exec.step_int();
        assert_eq!(s2, EXEC_COMPLETE);
        assert_eq!(exec.core.pc, pc_base + 32);
        assert_eq!(exec.core.read_gpr(3), 0x12345678, "ADDIU v1,v0,0x5678 using v0 set by LUI");
    }

    #[test]
    #[cfg(feature = "opcodefusion")]
    fn test_addr_calc_load_store_fusion() {
        // ADDU/SUBU/ADDIU immediately followed by a load/store using the
        // arithmetic op's destination as base register should fuse into one
        // handler call (exec_addu_ls/exec_subu_ls/exec_addiu_ls), skipping the
        // separate fetch/decode of the load/store — any offset is fine, the
        // fused handler reads the load/store's own offset field.
        let (mut exec, mem) = create_executor_with_r4000cache();
        let pc_base: u64 = 0xFFFFFFFF80000000;
        let nop = 0u32;

        // Offset 0: ADDIU t0, a0, 4 ; LW v0, 8(t0)  (nonzero base-calc imm AND
        // nonzero load offset, to prove neither is dropped).
        let addiu = make_i(OP_ADDIU as u32, 4 /*a0*/, 8 /*t0*/, 4);
        let lw = make_i(OP_LW as u32, 8 /*t0*/, 2 /*v0*/, 8);
        mem.set_double(0, ((addiu as u64) << 32) | lw as u64);

        // Offset 8: ADDU t1, a0, a1 ; SW v0, 0(t1)
        let addu = make_r(OP_SPECIAL as u32, 4, 5, 9 /*t1*/, 0, FUNCT_ADDU as u32);
        let sw = make_i(OP_SW, 9 /*t1*/, 2 /*v0*/, 0);
        mem.set_double(8, ((addu as u64) << 32) | sw as u64);

        // Backing memory for the LW at effective address a0(4)+4+8.
        let base_addr = 0x2000u64;
        mem.set_word(base_addr + 4 + 8, 0xCAFEF00D);
        mem.set_double(16, ((nop as u64) << 32) | nop as u64); // landing pad

        exec.core.write_gpr(4 /*a0*/, base_addr);
        exec.core.pc = pc_base;
        let s = exec.step_int();
        assert_eq!(s, EXEC_COMPLETE, "fused ADDIU+LW should report plain EXEC_COMPLETE");
        assert_eq!(exec.core.pc, pc_base + 8, "PC should skip both fused instructions");
        assert_eq!(exec.core.read_gpr(8 /*t0*/), base_addr + 4, "ADDIU result must still be written");
        // 0xCAFEF00D has bit 31 set, so LW sign-extends to 64 bits.
        assert_eq!(exec.core.read_gpr(2 /*v0*/), 0xFFFFFFFFCAFEF00D, "LW must load through the fused address");

        exec.core.write_gpr(5 /*a1*/, 0x10);
        let s = exec.step_int();
        assert_eq!(s, EXEC_COMPLETE, "fused ADDU+SW should report plain EXEC_COMPLETE");
        assert_eq!(exec.core.pc, pc_base + 16);
        assert_eq!(exec.core.read_gpr(9 /*t1*/), base_addr + 0x10);
        assert_eq!(mem.get_word(base_addr + 0x10), 0xCAFEF00D, "SW must store v0 through the fused address");
    }

    #[test]
    #[cfg(feature = "opcodefusion")]
    fn test_addr_calc_load_store_fusion_fault() {
        // A fused pair whose load/store faults (misaligned access here, for
        // simplicity) must report the exception, and cp0_epc must point at
        // the LOAD/STORE's own address — not the address-calc instruction's —
        // exactly as if the two had executed unfused. This is why the fused
        // handler advances PC by 4 for the address-calc BEFORE running the
        // load/store: handle_exception reads self.core.pc directly.
        let (mut exec, mem) = create_executor_with_r4000cache();
        let pc_base: u64 = 0xFFFFFFFF80000000;

        // ADDIU t0, a0, 0 ; LW v0, 1(t0)  -- offset 1 misaligns any word base.
        let addiu = make_i(OP_ADDIU as u32, 4, 8, 0);
        let lw = make_i(OP_LW as u32, 8, 2, 1);
        mem.set_double(0, ((addiu as u64) << 32) | lw as u64);

        exec.core.write_gpr(4, 0x3000);
        exec.core.pc = pc_base;
        let s = exec.step_int();
        assert!(s & EXEC_IS_EXCEPTION != 0, "misaligned fused load must fault");
        assert_eq!((s >> 2) & 0x1F, EXC_ADEL);
        assert_eq!(exec.core.cp0_epc, pc_base + 4,
            "cp0_epc must point at the LOAD's address (pc_base+4), not the ADDIU's (pc_base)");
        assert_eq!(exec.core.read_gpr(8), 0x3000, "ADDIU result must still have been written before the fault");
    }

    #[test]
    #[cfg(feature = "opcodefusion")]
    fn test_addr_calc_load_store_fusion_in_delay_slot() {
        // If the fused ADDIU+LW is itself sitting in another branch's delay
        // slot, it can't also skip to the load/store — a delay slot is
        // exactly one instruction. Must fall back to plain ADDIU completion
        // (which lets exec_decoded's normal EXEC_COMPLETE-in-delay-slot path
        // redirect PC to delay_slot_target), leaving the LW to be
        // fetched/decoded/executed normally as the real next instruction.
        let (mut exec, mem) = create_executor_with_r4000cache();
        let pc_base: u64 = 0xFFFFFFFF80000000;
        let nop = 0u32;

        // Offset 0: BEQ r0,r0,+3 (always taken -> target = 0+4+3*4 = 16)
        // Offset 4: ADDIU t0,a0,4 ; LW v0,8(t0)   <- delay slot (NOT a NOP,
        //           so branch fusion itself does not kick in here; this is
        //           purely testing addr-calc+load fusion's own delay-slot check)
        // Offset 16: branch target landing pad
        let beq_taken = make_i(OP_BEQ as u32, 0, 0, 3);
        let addiu = make_i(OP_ADDIU as u32, 4, 8, 4);
        let lw = make_i(OP_LW as u32, 8, 2, 8);
        mem.set_double(0, ((beq_taken as u64) << 32) | addiu as u64);
        mem.set_double(8, ((lw as u64) << 32) | nop as u64);
        mem.set_double(16, ((nop as u64) << 32) | nop as u64);

        let base_addr = 0x4000u64;
        mem.set_word(base_addr + 4 + 8, 0xABCD1234);
        exec.core.write_gpr(4, base_addr);

        exec.core.pc = pc_base;
        let s = exec.step_int();
        assert!(exec.core.in_delay_slot, "BEQ taken, real delay slot begins");
        assert_eq!(exec.core.pc, pc_base + 4);

        // Execute the ADDIU (in the delay slot): must NOT fuse away the LW —
        // it should behave like plain ADDIU and let PC redirect to the branch
        // target via the delay-slot mechanism.
        let s2 = exec.step_int();
        assert_eq!(s2, EXEC_COMPLETE, "ADDIU in delay slot falls back to plain completion");
        assert_eq!(exec.core.pc, pc_base + 16, "PC must redirect to branch target, not pc+8");
        assert!(!exec.core.in_delay_slot);
        assert_eq!(exec.core.read_gpr(8), base_addr + 4, "ADDIU result must still be written");
        assert_eq!(exec.core.read_gpr(2), 0, "LW must NOT have executed as part of the fused pair");

        // The LW at offset 8 was never consumed — confirm it's still there
        // and executes normally on its own next time it's reached. (Not
        // reached by falling through here since branch redirected past it;
        // this just documents that the fused handler didn't corrupt/consume
        // that memory word.)
        assert_eq!(mem.get_word(8), lw);
    }

    #[test]
    #[cfg(not(feature = "r5k"))]
    fn test_virtual_coherency_exception() {
        #[allow(unused_imports)]
        use crate::mips_exec::{MipsExecutor, MipsCpuConfig, EXC_VCED, EXC_VCEI};
        use crate::mips_tlb::PassthroughTlb;
        #[allow(unused_imports)]
        use crate::mips_cache_v2::R4400Cache;
        #[allow(unused_imports)]
        use crate::mips_core::STATUS_KX;
        use crate::traits::{BUS_OK, BUS_VCE};

        let (exec, mem) = create_executor_with_r4000cache();

        // Set up two virtual addresses with different bits [14:12] that map to same physical address
        // Virtual address bits [14:12] become PIdx in L2 cache tag
        //
        // virt1 = 0x0000_0000 (bits [14:12] = 0b000)
        // virt2 = 0x0000_1000 (bits [14:12] = 0b001)
        // Both map to same physical address through PassthroughTlb
        let phys_addr = 0x100000u64;  // Some physical address
        let virt1 = phys_addr;         // PIdx = (phys_addr >> 12) & 0x7 = 0
        let virt2 = phys_addr ^ 0x1000; // PIdx = ((phys_addr ^ 0x1000) >> 12) & 0x7 = 1

        // Store some data at the physical address
        mem.set_word(phys_addr, 0xDEADBEEF);

        // First access with virt1 - this fills L2 with PIdx=0
        let result = exec.cache.read::<4>(virt1, phys_addr);
        assert_eq!(result.status, BUS_OK, "First access should succeed");
        assert_eq!(result.data, 0xDEADBEEF, "First access should return correct data");

        // Second access with virt2 mapping to same phys_addr but different virt PIdx
        // This should trigger VCE because:
        // - L1 miss (different virtual index)
        // - L2 hit (same physical address)
        // - virt PIdx (1) != stored PIdx in L2 tag (0)
        let result = exec.cache.read::<4>(virt2, phys_addr);
        assert_eq!(result.status, BUS_VCE,
                   "Second access with different virtual index should trigger VCE");

        // Test VCEI (instruction fetch)
        mem.set_word(phys_addr + 0x1000, 0x00000000);

        let result = exec.cache.fetch(virt1, phys_addr + 0x1000);
        assert_eq!(result.status, crate::mips_exec::EXEC_COMPLETE, "First fetch should succeed");

        let result = exec.cache.fetch(virt2, phys_addr + 0x1000);
        assert_eq!(result.status, crate::mips_exec::exec_exception_const(crate::mips_exec::EXC_VCEI),
                   "Second fetch with different virtual index should trigger VCEI");
    }

    // Tests verifying correctness of pre-processed imm field in DecodedInstr.
    // Each test exercises a specific encoding path set at decode time.

    #[test]
    fn test_imm_sign_extension() {
        let (mut exec, _) = create_executor();

        // ADDIU with negative immediate: imm = 0xFFFF = -1 sign-extended
        // r1=0 + (-1) = -1 = 0xFFFFFFFFFFFFFFFF
        exec.core.write_gpr(1, 0);
        let instr = make_i(OP_ADDIU, 1, 2, 0xFFFF);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0xFFFFFFFFFFFFFFFF);

        // ADDIU with negative immediate sign-extends to 64-bit result
        // r1=1 + (-2) = -1 = 0xFFFFFFFFFFFFFFFF
        exec.core.write_gpr(1, 1);
        let instr = make_i(OP_ADDIU, 1, 2, 0xFFFE); // -2
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0xFFFFFFFFFFFFFFFF);

        // DADDIU with negative immediate
        exec.core.write_gpr(1, 0x100000000u64);
        let instr = make_i(OP_DADDIU, 1, 2, 0xFFFF); // r1 + (-1)
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0xFFFFFFFF);
    }

    #[test]
    fn test_lui_sign_extension() {
        let (mut exec, _) = create_executor();

        // LUI 0x1234 -> 0x0000000012340000 (positive, no sign extension needed)
        let instr = make_i(OP_LUI, 0, 1, 0x1234);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 0x0000000012340000);

        // LUI 0x8000 -> (0x8000 << 16) as i32 = 0x80000000 as i32 = -2147483648
        // sign-extended to 64-bit: 0xFFFFFFFF80000000
        let instr = make_i(OP_LUI, 0, 1, 0x8000);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 0xFFFFFFFF80000000);

        // LUI 0xFFFF -> (0xFFFF << 16) as i32 = 0xFFFF0000 as i32 = -65536
        // sign-extended to 64-bit: 0xFFFFFFFFFFFF0000
        let instr = make_i(OP_LUI, 0, 1, 0xFFFF);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 0xFFFFFFFFFFFF0000);
    }

    #[test]
    fn test_imm_zero_extension() {
        let (mut exec, _) = create_executor();

        // ANDI with 0xFFFF: zero-extended, not sign-extended.
        // r1 = 0xFFFFFFFFFFFFFFFF; r1 & 0xFFFF = 0x000000000000FFFF
        exec.core.write_gpr(1, 0xFFFFFFFFFFFFFFFF);
        let instr = make_i(OP_ANDI, 1, 2, 0xFFFF);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0x000000000000FFFF);

        // ORI with 0x8000: zero-extended (not sign-extended to 0xFFFFFFFFFFFF8000)
        exec.core.write_gpr(1, 0);
        let instr = make_i(OP_ORI, 1, 2, 0x8000);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0x0000000000008000);

        // XORI with 0xFFFF: zero-extended
        exec.core.write_gpr(1, 0xFFFFFFFFFFFF0000);
        let instr = make_i(OP_XORI, 1, 2, 0xFFFF);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0xFFFFFFFFFFFFFFFF);
    }

    #[test]
    fn test_branch_negative_offset() {
        let (mut exec, _) = create_executor();

        // BEQ with negative offset: branch backwards
        // PC=0x2000, offset=-4 (0xFFFC), target = 0x2004 + (-4)*4 = 0x2004 - 16 = 0x1FF4
        exec.core.pc = 0x2000;
        exec.core.write_gpr(1, 5);
        exec.core.write_gpr(2, 5);
        let instr = make_i(OP_BEQ, 1, 2, 0xFFFC); // -4
        let s = exec.exec(instr);
        assert!(exec.core.in_delay_slot);
        assert_eq!(exec.core.delay_slot_target, 0x2004u64.wrapping_add((-4i64 * 4) as u64));

        // BGTZ with negative offset (not taken because rs=0)
        exec.core.pc = 0x2000;
        exec.core.write_gpr(1, 0);
        let instr = make_i(OP_BGTZ, 1, 0, 0xFFFC);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);

        // BLTZ with negative offset (taken because rs < 0)
        exec.core.pc = 0x2000;
        exec.core.write_gpr(1, (-1i64) as u64);
        let instr = make_i(OP_REGIMM, 1, RT_BLTZ as u32, 0xFFFC);
        let s = exec.exec(instr);
        assert!(exec.core.in_delay_slot);
        assert_eq!(exec.core.delay_slot_target, 0x2004u64.wrapping_add((-4i64 * 4) as u64));
    }

    #[test]
    fn test_j_target_preshifted() {
        let (mut exec, _) = create_executor();

        // J with target=1: address = (PC+4 & ~0x0FFFFFFF) | (1 << 2) = 4
        exec.core.pc = 0x1000;
        let instr = make_j(OP_J, 1);
        let s = exec.exec(instr);
        assert!(exec.core.in_delay_slot);
        assert_eq!(exec.core.delay_slot_target, 4);

        // J with max target=0x3FFFFFF: low 28 bits = 0xFFFFFFFC
        exec.core.pc = 0x1000;
        let instr = make_j(OP_J, 0x3FFFFFF);
        let s = exec.exec(instr);
        assert!(exec.core.in_delay_slot);
        assert_eq!(exec.core.delay_slot_target, (0x1004u64 & 0xFFFFFFFF_F0000000) | 0x0FFFFFFC);

        // JAL saves return address
        exec.core.pc = 0x1000;
        let instr = make_j(OP_JAL, 0x100);
        let s = exec.exec(instr);
        assert!(exec.core.in_delay_slot);
        assert_eq!(exec.core.delay_slot_target, (0x1004u64 & 0xFFFFFFFF_F0000000) | 0x400);
        assert_eq!(exec.core.read_gpr(31), 0x1008);
    }

    #[test]
    fn test_sltiu_sign_extended_unsigned_compare() {
        let (mut exec, _) = create_executor();

        // SLTIU: imm=0xFFFF sign-extends to 0xFFFFFFFFFFFFFFFF (max u64).
        // Any rs < 0xFFFFFFFFFFFFFFFF should set rt=1.
        exec.core.write_gpr(1, 0);
        let instr = make_i(OP_SLTIU, 1, 2, 0xFFFF);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 1); // 0 < 0xFFFFFFFFFFFFFFFF

        // rs = 0xFFFFFFFFFFFFFFFF should NOT be less than 0xFFFFFFFFFFFFFFFF
        exec.core.write_gpr(1, 0xFFFFFFFFFFFFFFFF);
        let instr = make_i(OP_SLTIU, 1, 2, 0xFFFF);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0);

        // SLTI: imm=0xFFFF = -1 signed. rs=0 is NOT less than -1 (signed).
        exec.core.write_gpr(1, 0);
        let instr = make_i(OP_SLTI, 1, 2, 0xFFFF);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0); // 0 >= -1

        // rs = -2 IS less than -1.
        exec.core.write_gpr(1, (-2i64) as u64);
        let instr = make_i(OP_SLTI, 1, 2, 0xFFFF);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 1);
    }

    #[test]
    fn test_load_store_negative_offset() {
        let (mut exec, mem) = create_executor();

        // LW with negative offset: base=0x1010, offset=-16 (0xFFF0), addr=0x1000
        mem.set_word(0x1000, 0xDEADBEEF);
        exec.core.write_gpr(1, 0x1010);
        let instr = make_i(OP_LW, 1, 2, 0xFFF0); // -16
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0xFFFFFFFFDEADBEEF); // LW sign-extends

        // SW with negative offset
        exec.core.write_gpr(2, 0x12345678);
        let instr = make_i(OP_SW, 1, 2, 0xFFF0);
        assert_eq!(exec.exec(instr), EXEC_COMPLETE);
        assert_eq!(mem.get_word(0x1000), 0x12345678);
    }

    // =========================================================================
    // FR=0 mode tests
    // =========================================================================
    //
    // In FR=0 the 32 architectural FPRs map onto 16 physical 64-bit slots:
    //   physical fpr[n]  bits 31:0  = FPR(2n)   (even register)
    //   physical fpr[n]  bits 63:32 = FPR(2n+1) (odd register)
    //
    // A double/long in FPR(2n) occupies the entire 64-bit even slot.
    // Only even registers may be used as double/long operands.

    // Helper: switch executor to FR=0 (CU1 set, FR clear)
    fn setup_fr0(exec: &mut MipsExecutor<PassthroughTlb, PassthroughCache>) {
        exec.core.cp0_status = (exec.core.cp0_status | STATUS_CU1) & !STATUS_FR;
        exec.update_fpr_mode();
    }

    // Helper: switch executor to FR=1 (CU1 set, FR set)
    fn setup_fr1(exec: &mut MipsExecutor<PassthroughTlb, PassthroughCache>) {
        exec.core.cp0_status |= STATUS_CU1 | STATUS_FR;
        exec.update_fpr_mode();
    }

    /// FR=0: odd register (FPR1) is the upper 32 bits of physical slot 0.
    /// Writing FPR1 via MTC1 must land in fpr[0] bits 63:32.
    /// Writing FPR0 via MTC1 must land in fpr[0] bits 31:0.
    #[test]
    fn test_fr0_register_aliasing_mtc1_mfc1() {
        let (mut exec, _) = create_executor();
        setup_fr0(&mut exec);

        // MTC1 r1 -> FPR0 (even): should write lower half of slot 0
        exec.core.write_gpr(1, 0x3F800000u64); // 1.0f
        assert_eq!(exec.exec(make_cop1_move(RS_MTC1, 1, 0)), EXEC_COMPLETE);

        // MTC1 r2 -> FPR1 (odd): should write upper half of slot 0
        exec.core.write_gpr(2, 0x40000000u64); // 2.0f
        assert_eq!(exec.exec(make_cop1_move(RS_MTC1, 2, 1)), EXEC_COMPLETE);

        // Verify slot 0 contains both words: hi=2.0f lo=1.0f
        assert_eq!(exec.core.fpr[0], 0x40000000_3F800000u64,
            "fpr[0] should hold FPR1 in upper half and FPR0 in lower half");

        // MFC1 FPR0 -> r3: should get lower half (1.0f)
        assert_eq!(exec.exec(make_cop1_move(RS_MFC1, 3, 0)), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(3), 0x3F800000u64);

        // MFC1 FPR1 -> r4: should get upper half (2.0f)
        assert_eq!(exec.exec(make_cop1_move(RS_MFC1, 4, 1)), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(4), 0x40000000u64);
    }

    /// FR=0: LWC1/SWC1 with odd register accesses upper half of even slot.
    #[test]
    fn test_fr0_lwc1_swc1_odd_register() {
        let (mut exec, mem) = create_executor();
        setup_fr0(&mut exec);

        // LWC1 to FPR3 (odd): loads into upper half of slot 2 (fpr[2])
        mem.set_word(0x1000, 0x40400000); // 3.0f
        exec.core.write_gpr(5, 0x1000);
        assert_eq!(exec.exec(make_i(OP_LWC1, 5, 3, 0)), EXEC_COMPLETE);
        assert_eq!((exec.core.fpr[2] >> 32) as u32, 0x40400000,
            "LWC1 to odd FPR3 should write upper half of slot 2");

        // SWC1 from FPR3 (odd): stores upper half of slot 2
        mem.set_word(0x1004, 0);
        exec.core.write_gpr(6, 0x1004);
        assert_eq!(exec.exec(make_i(OP_SWC1, 6, 3, 0)), EXEC_COMPLETE);
        assert_eq!(mem.get_word(0x1004), 0x40400000);
    }

    /// FR=0: ADD.S with odd source/dest registers uses upper halves correctly.
    #[test]
    fn test_fr0_single_arithmetic_odd_regs() {
        let (mut exec, _) = create_executor();
        setup_fr0(&mut exec);

        // Load 2.5 into FPR1 (odd = upper half of slot 0)
        // Load 3.5 into FPR3 (odd = upper half of slot 2)
        exec.core.write_gpr(1, 2.5f32.to_bits() as u64);
        exec.core.write_gpr(2, 3.5f32.to_bits() as u64);
        assert_eq!(exec.exec(make_cop1_move(RS_MTC1, 1, 1)), EXEC_COMPLETE); // FPR1
        assert_eq!(exec.exec(make_cop1_move(RS_MTC1, 2, 3)), EXEC_COMPLETE); // FPR3

        // ADD.S FPR1 + FPR3 -> FPR5 (odd = upper half of slot 4)
        let add_s = make_cop1_compute(RS_S, 3, 1, 5, FUNCT_FADD);
        assert_eq!(exec.exec(add_s), EXEC_COMPLETE);

        // Read back FPR5 via MFC1
        assert_eq!(exec.exec(make_cop1_move(RS_MFC1, 10, 5)), EXEC_COMPLETE);
        let result = f32::from_bits(exec.core.read_gpr(10) as u32);
        assert_eq!(result, 6.0f32);
    }

    /// FR=0: ADD.D uses the full 64-bit even slot; reg must be even.
    #[test]
    fn test_fr0_double_arithmetic() {
        let (mut exec, _) = create_executor();
        setup_fr0(&mut exec);

        // Write doubles directly into even physical slots
        exec.core.fpr[0] = 2.5f64.to_bits(); // FPR0
        exec.core.fpr[2] = 3.5f64.to_bits(); // FPR2

        // ADD.D FPR0 + FPR2 -> FPR4
        let add_d = make_cop1_compute(RS_D, 2, 0, 4, FUNCT_FADD);
        assert_eq!(exec.exec(add_d), EXEC_COMPLETE);
        assert_eq!(f64::from_bits(exec.core.fpr[4]), 6.0f64);

        // MUL.D FPR0 * FPR2 -> FPR4
        exec.core.fpr[0] = 4.0f64.to_bits();
        exec.core.fpr[2] = 3.0f64.to_bits();
        let mul_d = make_cop1_compute(RS_D, 2, 0, 4, FUNCT_FMUL);
        assert_eq!(exec.exec(mul_d), EXEC_COMPLETE);
        assert_eq!(f64::from_bits(exec.core.fpr[4]), 12.0f64);
    }

    /// FR=0: LDC1/SDC1 load/store the full 64-bit even slot.
    #[test]
    fn test_fr0_ldc1_sdc1() {
        let (mut exec, mem) = create_executor();
        setup_fr0(&mut exec);

        let pi_bits = std::f64::consts::PI.to_bits();
        mem.set_double(0x2000, pi_bits);
        exec.core.write_gpr(1, 0x2000);

        // LDC1 to FPR0 (even): loads into slot 0
        assert_eq!(exec.exec(make_i(OP_LDC1, 1, 0, 0)), EXEC_COMPLETE);
        assert_eq!(exec.core.fpr[0], pi_bits);
        assert_eq!(f64::from_bits(exec.core.fpr[0]), std::f64::consts::PI);

        // SDC1 from FPR0 (even): stores slot 0
        mem.set_double(0x2008, 0);
        exec.core.write_gpr(2, 0x2008);
        assert_eq!(exec.exec(make_i(OP_SDC1, 2, 0, 0)), EXEC_COMPLETE);
        assert_eq!(mem.get_double(0x2008), pi_bits);
    }

    /// FR=0: Writing a double into an even slot makes both halves visible
    /// as the corresponding even/odd single-precision registers.
    #[test]
    fn test_fr0_double_single_aliasing() {
        let (mut exec, _) = create_executor();
        setup_fr0(&mut exec);

        // Write a known bit pattern as a double into FPR0 (even slot 0)
        // lo=0x11111111, hi=0x22222222
        exec.core.fpr[0] = 0x22222222_11111111u64;

        // MFC1 FPR0 -> should get lo word (0x11111111)
        assert_eq!(exec.exec(make_cop1_move(RS_MFC1, 1, 0)), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(1), 0x11111111u64);

        // MFC1 FPR1 -> should get hi word (0x22222222)
        assert_eq!(exec.exec(make_cop1_move(RS_MFC1, 2, 1)), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(2), 0x22222222u64);

        // Write 0xAAAAAAAA into FPR1 (odd) via MTC1 — only hi changes
        exec.core.write_gpr(3, 0xAAAAAAAAu64);
        assert_eq!(exec.exec(make_cop1_move(RS_MTC1, 3, 1)), EXEC_COMPLETE);
        assert_eq!(exec.core.fpr[0], 0xAAAAAAAA_11111111u64,
            "writing odd FPR1 must update upper half only, leave lower half intact");
    }

    /// Switching FR=1 -> FR=0: data already in even slots is preserved and
    /// accessible as doubles; odd slots are now aliased to upper halves.
    /// Switching FR=0 -> FR=1: even slot data unchanged; upper-half values
    /// written under FR=0 are visible in odd slots under FR=1.
    #[test]
    fn test_fr_mode_switch_aliasing() {
        let (mut exec, _) = create_executor();

        // --- Start in FR=1 ---
        setup_fr1(&mut exec);

        // Write independent values into all four slots 0-3
        exec.core.fpr[0] = 0x00000000_3F800000u64; // fpr[0] lo=1.0f, hi=0
        exec.core.fpr[1] = 0x00000000_40000000u64; // fpr[1] lo=2.0f, hi=0
        exec.core.fpr[2] = 0x00000000_40400000u64; // fpr[2] lo=3.0f, hi=0
        exec.core.fpr[3] = 0x00000000_40800000u64; // fpr[3] lo=4.0f, hi=0

        // Under FR=1: MFC1 FPR1 reads fpr[1] bits 31:0 = 2.0f
        assert_eq!(exec.exec(make_cop1_move(RS_MFC1, 10, 1)), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(10), 2.0f32.to_bits() as u64);

        // --- Switch to FR=0 ---
        setup_fr0(&mut exec);

        // slot 0 is unchanged: lo=1.0f, hi=0
        // Under FR=0: FPR0 = lo of slot 0 = 1.0f
        assert_eq!(exec.exec(make_cop1_move(RS_MFC1, 10, 0)), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(10), 1.0f32.to_bits() as u64);

        // Under FR=0: FPR1 = hi of slot 0 = 0 (was 0 when we set it in FR=1)
        assert_eq!(exec.exec(make_cop1_move(RS_MFC1, 10, 1)), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(10), 0u64);

        // Write 5.0f into FPR1 (odd) via MTC1 under FR=0 -> goes into hi of slot 0
        exec.core.write_gpr(11, 5.0f32.to_bits() as u64);
        assert_eq!(exec.exec(make_cop1_move(RS_MTC1, 11, 1)), EXEC_COMPLETE);
        assert_eq!(exec.core.fpr[0], 0x40A00000_3F800000u64,
            "FR=0 MTC1->FPR1 should write upper half of slot 0");

        // --- Switch back to FR=1 ---
        setup_fr1(&mut exec);

        // slot 0 is now 0x40A00000_3F800000; under FR=1 fpr[0] is untouched
        assert_eq!(exec.core.fpr[0], 0x40A00000_3F800000u64);

        // Under FR=1: MFC1 FPR1 reads fpr[1] bits 31:0 (the original 2.0f)
        assert_eq!(exec.exec(make_cop1_move(RS_MFC1, 10, 1)), EXEC_COMPLETE);
        assert_eq!(exec.core.read_gpr(10), 2.0f32.to_bits() as u64);
    }
}
