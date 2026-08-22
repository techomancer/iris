//! JIT v2 vs. interpreter equivalence test harness (§4.6 "verification,
//! day-one, not retrofit"). Building blocks for per-instruction lockstep
//! tests: run the interpreter's `MipsExecutor::exec` and a `Codegen`-compiled
//! single-instruction region against two independently-seeded `MipsCore`s
//! from the same starting state, then compare the architectural fields each
//! instruction class can touch.
//!
//! Deliberately bounded in scope for now — `CoreSnapshot` only covers GPRs/
//! pc/hi/lo, the fields the currently-wired semantics emitters
//! (`jitv2/codegen.rs`'s `lookup_semantics`) can touch. Grows alongside the
//! emitter table (CP0 fields for MTC0-adjacent instructions, FPU registers
//! for CP1 ops, etc.) rather than trying to cover the whole (non-`Clone`,
//! non-`PartialEq`) `MipsCore` struct up front — see `CoreSnapshot`'s doc
//! comment for why a whole-struct comparison isn't the right shape anyway.

#[cfg(all(test, feature = "jitv2"))]
mod tests {
    use crate::jitv2::analyzer::Analyzer;
    use crate::jitv2::codegen::Codegen;
    use crate::jitv2::{JitFn, ENTRIES_PER_PAGE, PAGE_SIZE};
    use crate::mips_core::MipsCore;
    use crate::mips_exec::{MipsCpuConfig, MipsExecutor};
    use crate::mips_tlb::PassthroughTlb;
    use crate::mips_cache_v2::PassthroughCache;
    use crate::traits::{BusDevice, BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BUS_BUSY};
    use std::sync::atomic::AtomicU64;
    use std::sync::{Arc, Mutex};

    /// Snapshot of the architectural fields a compiled single-instruction
    /// region can observe or mutate. Not a whole-`MipsCore` comparison:
    /// `MipsCore` holds non-comparable bookkeeping (`Arc<AtomicU64>` cycle
    /// counters shared with other threads, `std::time::Instant` calibration
    /// timestamps, a raw callback pointer, a `HashMap` debug stats table)
    /// that has no "equal" notion meaningful to an instruction-semantics
    /// test and were never going to be touched by `lookup_semantics`'
    /// emitters anyway. Extend this struct's fields (and `capture`/`assert_eq_to`)
    /// as new instruction classes get emitters that touch new state (CP0
    /// regs, FPRs, HI/LO for mult/div, ...).
    #[derive(Debug, Clone, PartialEq)]
    struct CoreSnapshot {
        gpr: [u64; 32],
        pc: u64,
        hi: u64,
        lo: u64,
        cp0_epc: u64,
        cp0_cause: u32,
        cp0_status: u32,
        cp0_badvaddr: u64,
        fpr: [u64; 32],
        fpu_fcsr: u32,
        fpu_fccr: u32,
        fpu_fexr: u32,
        fpu_fenr: u32,
    }

    impl CoreSnapshot {
        fn capture(core: &MipsCore) -> Self {
            Self {
                gpr: core.gpr, pc: core.pc, hi: core.hi, lo: core.lo,
                cp0_epc: core.cp0_epc, cp0_cause: core.cp0_cause,
                cp0_status: core.cp0_status, cp0_badvaddr: core.cp0_badvaddr,
                fpr: core.fpr, fpu_fcsr: core.fpu_fcsr,
                fpu_fccr: core.fpu_fccr, fpu_fexr: core.fpu_fexr, fpu_fenr: core.fpu_fenr,
            }
        }
    }

    /// Same minimal `BusDevice` `MockMemory` shape as `mips_exec_test.rs`
    /// uses, duplicated here rather than shared: that one lives inside a
    /// `#[cfg(test)] mod tests` block (not reachable from another module),
    /// and this harness needs its own independent instances anyway (one per
    /// engine under comparison, never shared state between them).
    struct MockMemory {
        data: Mutex<std::collections::HashMap<u64, u8>>,
        /// One well-known word address (word-aligned; see
        /// `with_magic_responses`) whose `read32` is intercepted entirely,
        /// popping the next `(status, data)` pair off a caller-supplied
        /// queue instead of touching `data` at all — every access beyond
        /// the queue's length repeats its last entry. A single fixed
        /// address (rather than an address *set*, like an earlier version
        /// of this mock) sidesteps needing to reason about virtual/physical
        /// address translation (kseg0/1 masking, `PassthroughTlb`'s own
        /// identity-map cutoff) at all in a test — the queue is keyed by
        /// *access order*, not by address, which is also exactly what a
        /// real transiently-busy or always-erroring device's behavior
        /// actually depends on (call sequence), not the address per se.
        magic_addr: u64,
        magic_responses: Mutex<std::collections::VecDeque<(u32, u32)>>, // (status, data)
        #[cfg(feature = "jitv2")]
        gens: Mutex<std::collections::HashMap<u32, Box<AtomicU64>>>,
        /// When true, `seeded_executor_over` pre-claims and permanently
        /// denylists (`PhysicalCodePage::denylist_all`) pc's own physical
        /// page, and disables every `jitv2_lockstep` class
        /// (`lockstep_enabled.{alu,branch,load_store,fpu} = false`), right
        /// after constructing the executor — so every jitv2 dispatch gate
        /// variant (real/inline/lockstep) falls straight through to the
        /// interpreter unconditionally, regardless of which jitv2 feature is
        /// compiled in. Both are needed: `denylist_all` blocks the real
        /// (async/inline) dispatch gate, but `jitv2_lockstep`'s
        /// `lockstep_check` deliberately bypasses page state entirely (never
        /// touches publish/entries/ENTRY_DENYLISTED — see its own doc comment)
        /// so it can intercept a word before it's ever compiled; only the
        /// executor-level switch stops it. Needed by harnesses like
        /// `run_interpreter_page` whose dispatch-count arithmetic assumes
        /// exactly one instruction retires per `exec()` call — a real jitv2
        /// gate is free to redispatch a compiled multi-instruction region on
        /// a later call, which would silently break that count once a real
        /// compile actually landed in time (previously only ever true by
        /// luck, since the async compile thread rarely won the race within
        /// these short loops — see rules/jitv2/codegen-gotchas.md).
        #[cfg(feature = "jitv2")]
        no_jitv2: bool,
    }

    impl MockMemory {
        fn new() -> Self {
            Self {
                data: Mutex::new(std::collections::HashMap::new()),
                magic_addr: u64::MAX, // never matches a real word-aligned address
                magic_responses: Mutex::new(std::collections::VecDeque::new()),
                #[cfg(feature = "jitv2")]
                gens: Mutex::new(std::collections::HashMap::new()),
                #[cfg(feature = "jitv2")]
                no_jitv2: false,
            }
        }
        /// See the `no_jitv2` field doc comment.
        #[cfg(feature = "jitv2")]
        fn new_not_compilable() -> Self {
            Self { no_jitv2: true, ..Self::new() }
        }
        /// Arm the magic-address response queue (see `magic_addr`'s doc
        /// comment) — `addr` (word-aligned) becomes the one intercepted
        /// word; `read32` against it pops `responses` in order (repeating
        /// the last entry once exhausted) instead of touching `data`.
        fn with_magic_responses(mut self, addr: u64, responses: Vec<(u32, u32)>) -> Self {
            // read32's own addr parameter is the *physical* address
            // (already translated by the time BusDevice sees it) — a
            // kseg0/kseg1 virtual address (`translate_32bit_impl`'s
            // segment 4/5 cases) translates to `virt_addr32 & 0x1FFFFFFF`,
            // so a caller's natural 64-bit virtual address (e.g.
            // 0xFFFFFFFF9F0F0200) must go through the exact same masking
            // here, not just a plain `as u32` truncation, to match what
            // read32 actually receives (this was the mismatch an earlier
            // version of this fix still had: comparing the unmasked 32-bit
            // truncation against the real, kseg-masked physical address —
            // still silently never intercepting anything).
            self.magic_addr = ((addr as u32) & 0x1FFF_FFFF & !3) as u64;
            *self.magic_responses.get_mut().unwrap() = responses.into();
            self
        }
        fn get_byte(&self, addr: u64) -> u8 {
            *self.data.lock().unwrap().get(&addr).unwrap_or(&0)
        }
        fn set_byte(&self, addr: u64, val: u8) {
            self.data.lock().unwrap().insert(addr, val);
        }
        fn get_word(&self, addr: u64) -> u32 {
            let mut bytes = [0u8; 4];
            for i in 0..4 { bytes[i] = self.get_byte(addr + i as u64); }
            u32::from_be_bytes(bytes)
        }
        fn set_word(&self, addr: u64, val: u32) {
            for (i, b) in val.to_be_bytes().iter().enumerate() { self.set_byte(addr + i as u64, *b); }
        }
    }

    impl BusDevice for MockMemory {
        fn read8(&self, addr: u32) -> BusRead8 {
            BusRead8::ok(self.get_byte(addr as u64))
        }
        fn write8(&self, addr: u32, val: u8) -> u32 { self.set_byte(addr as u64, val); BUS_OK }
        fn read16(&self, addr: u32) -> BusRead16 {
            let a = (addr & !1) as u64;
            let mut b = [0u8; 2];
            for i in 0..2 { b[i] = self.get_byte(a + i as u64); }
            BusRead16::ok(u16::from_be_bytes(b))
        }
        fn write16(&self, addr: u32, val: u16) -> u32 {
            let a = (addr & !1) as u64;
            for (i, b) in val.to_be_bytes().iter().enumerate() { self.set_byte(a + i as u64, *b); }
            BUS_OK
        }
        fn read32(&self, addr: u32) -> BusRead32 {
            let a = (addr & !3) as u64;
            if a == self.magic_addr {
                let mut responses = self.magic_responses.lock().unwrap();
                let (status, data) = if responses.len() > 1 {
                    responses.pop_front().unwrap()
                } else {
                    *responses.front().expect("with_magic_responses must supply at least one response")
                };
                return BusRead32 { status, data };
            }
            BusRead32::ok(self.get_word(a))
        }
        fn write32(&self, addr: u32, val: u32) -> u32 { self.set_word((addr & !3) as u64, val); BUS_OK }
        fn read64(&self, addr: u32) -> BusRead64 {
            let a = (addr & !7) as u64;
            let hi = self.get_word(a) as u64;
            let lo = self.get_word(a + 4) as u64;
            BusRead64::ok((hi << 32) | lo)
        }
        fn write64(&self, addr: u32, val: u64) -> u32 {
            let a = (addr & !7) as u64;
            self.set_word(a, (val >> 32) as u32);
            self.set_word(a + 4, val as u32);
            BUS_OK
        }
        #[cfg(feature = "jitv2")]
        fn gen_ptr(&self, addr: u32) -> *const AtomicU64 {
            // Real enforcement of no_jitv2 is seeded_executor_over's
            // denylist_all() call, not this null return (a null gen_ptr no
            // longer means "not dispatchable" — see the no_jitv2 field's own
            // doc comment) — kept null here anyway since it's still
            // factually true that this device has no real gen tracking.
            if self.no_jitv2 {
                return std::ptr::null();
            }
            let page = addr / PAGE_SIZE;
            let mut gens = self.gens.lock().unwrap();
            let counter = gens.entry(page).or_insert_with(|| Box::new(AtomicU64::new(0)));
            counter.as_ref() as *const AtomicU64
        }
    }

    fn make_r(op: u32, rs: u32, rt: u32, rd: u32, sa: u32, funct: u32) -> u32 {
        (op << 26) | ((rs & 0x1F) << 21) | ((rt & 0x1F) << 16) | ((rd & 0x1F) << 11) | ((sa & 0x1F) << 6) | (funct & 0x3F)
    }

    /// Seed a fresh `MipsCore` with the given GPR contents (r0 forced to 0,
    /// matching architecture) and pc, with nothing pending so the
    /// pending-interrupt preamble never bails during the test — an
    /// equivalence test wants the instruction's own semantics exercised,
    /// not a preamble bail path. (No timer fields to silence: the compare
    /// timer is an hptimer-thread interrupt now and no manager is wired in
    /// tests.)
    fn seeded_core(gpr: [u64; 32], pc: u64) -> MipsCore {
        let mut core = MipsCore::new();
        core.gpr = gpr;
        core.gpr[0] = 0;
        core.pc = pc;
        core.hot.interrupts.store(0, std::sync::atomic::Ordering::Relaxed);
        core
    }

    /// Build a fresh `MipsExecutor` over its own `MockMemory`, seeded
    /// identically for both engines: gpr/pc as given, nothing pending so
    /// the pending-interrupt preamble never bails during the test (that's a
    /// separate concern, already covered by `codegen.rs`'s preamble tests —
    /// an equivalence test wants the instruction's own semantics exercised,
    /// not a preamble bail path).
    fn seeded_executor(gpr: [u64; 32], pc: u64) -> (MipsExecutor<PassthroughTlb, PassthroughCache>, Arc<MockMemory>) {
        seeded_executor_over(MockMemory::new(), gpr, pc)
    }

    /// Same seeding as `seeded_executor`, but over a caller-supplied
    /// `MockMemory` — used by `run_interpreter_page` to pass
    /// `MockMemory::new_not_compilable()` (see its doc comment).
    fn seeded_executor_over(mem: MockMemory, gpr: [u64; 32], pc: u64) -> (MipsExecutor<PassthroughTlb, PassthroughCache>, Arc<MockMemory>) {
        let mem = Arc::new(mem);
        let mem_bus: Arc<dyn BusDevice> = mem.clone();
        let cfg = MipsCpuConfig::indy();
        let mut exec: MipsExecutor<PassthroughTlb, PassthroughCache> =
            MipsExecutor::new(mem_bus, PassthroughTlb::default(), &cfg);
        exec.core.gpr = gpr;
        exec.core.gpr[0] = 0;
        exec.core.pc = pc;
        exec.core.hot.interrupts.store(0, std::sync::atomic::Ordering::Relaxed);
        // no_jitv2 (MockMemory::new_not_compilable's doc comment): pre-claim
        // pc's own physical page and denylist every offset on it right here,
        // in the test setup, rather than relying on gen_ptr returning null
        // to imply "never compilable" — that inference doesn't hold anymore
        // (PhysicalCodePage::gen is never null; a null gen_ptr just means
        // "use the shared, never-bumped fallback counter", not "never
        // dispatchable" — see NEVER_COMPILABLE_GEN's doc comment). Denylisting
        // every offset is the real, still-current mechanism for "this page
        // must never compile/dispatch, everything runs on the interpreter".
        #[cfg(feature = "jitv2")]
        if mem.no_jitv2 {
            let phys_base = (pc as u32) & 0x1FFF_FFFF & !(PAGE_SIZE as u32 - 1);
            let pfn = phys_base / PAGE_SIZE;
            let mut jit = exec.jitv2.lock();
            let slot = jit.page_for(pfn, phys_base, exec.sysad.as_ref())
                .expect("fresh Jitv2 pool must have room for one page");
            unsafe { (*jit.page_ptr(slot)).denylist_all(); }
            drop(jit);
            // Under jitv2_lockstep the real dispatch gate is ON and forces
            // inline compile, so denylist_all alone isn't enough (a fresh page
            // arrival could still compile). Turn the gate off outright for this
            // pure-interpreter reference executor — `run_interpreter` needs an
            // uninstrumented run, and the redesigned lockstep verifies via the
            // gate itself, so "no gate" == "no lockstep instrumentation" here.
            #[cfg(feature = "jitv2_lockstep")]
            {
                exec.jitv2_dispatch_enabled = false;
            }
        }
        (exec, mem)
    }

    /// Run `instr` through the real interpreter (`MipsExecutor::exec`,
    /// exactly `mips_exec_test.rs`'s entry point) starting from `gpr`/`pc`
    /// with `mem` pre-populated for memory-referencing instructions.
    ///
    /// no_jitv2 (see `MockMemory::new_not_compilable`'s doc comment): this
    /// executor never calls `install_jit_hooks`, so if jitv2 (in particular
    /// `jitv2_lockstep`, which — unlike the real gate — runs unconditionally
    /// on every ALU dispatch regardless of call count, see
    /// `MipsExecutor::lockstep_check`) ever compiled and ran a function
    /// against `exec.core` here, any exception it raises would call through
    /// `core.handle_exception_fn`'s never-installed sentinel and abort the
    /// whole process (`panic = "abort"`) — this must stay a pure interpreter
    /// run.
    fn run_interpreter(instr: u32, gpr: [u64; 32], pc: u64, mem_init: &[(u64, u32)]) -> CoreSnapshot {
        let (mut exec, mem) = seeded_executor_over(MockMemory::new_not_compilable(), gpr, pc);
        for &(addr, val) in mem_init { mem.set_word(addr, val); }
        exec.exec(instr);
        CoreSnapshot::capture(&exec.core)
    }

    /// Compile a single-instruction region (`instr`, wrapped by an entry
    /// jump and the shared exit block, no fusion/branch handling) and run it
    /// against a fresh `MipsExecutor` seeded identically to
    /// `run_interpreter`'s, with `install_jit_hooks` wired so
    /// memory-referencing/exception-raising instructions work exactly as
    /// they would through the real dispatcher. `word_offset` is where
    /// `instr` lives within its (single-page) region — callers that don't
    /// care can pass any in-page offset; word 0 is the simplest choice
    /// unless the instruction under test needs to be away from page
    /// boundaries for some other reason (0xFFC hazard tests, etc, not
    /// exercised by this straight-line-only harness).
    ///
    /// Returns `None` if `Codegen::compile_region` declined (no emitter for
    /// `instr` yet, or the analyzer classified it as non-`Sequential`) —
    /// callers should treat that as a test setup bug (the whole point of
    /// calling this is that the instruction IS supposed to be compilable),
    /// not a silently-skipped comparison.
    ///
    /// **Safety note on executor placement**: `install_jit_hooks` captures
    /// `&mut exec` as `jit_ctx` and that pointer must never go stale (see
    /// `MipsExecutor::install_jit_hooks`'s doc comment — the same discipline
    /// as `interrupts_ptr`). `exec` is boxed here specifically so its address
    /// is stable from the hook-install point onward: a bare stack local
    /// would still be at a fixed address for this function's body, but
    /// boxing makes that invariant explicit and immune to any future
    /// refactor that might move `exec` around before calling `jit_fn`.
    fn run_jit(instr: u32, gpr: [u64; 32], pc: u64, word_offset: u16, mem_init: &[(u64, u32)]) -> Option<CoreSnapshot> {
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[word_offset as usize] = instr;

        // page_base must be pc's real page (upper bits included), not 0:
        // codegen derives absolute virtual addresses from it (off-page
        // branch/jump targets, delay-slot exception EPC) — a fake all-zero
        // page silently produces wrong addresses for any test that exercises
        // those paths (see rules/jitv2/codegen-gotchas.md).
        let page_base = (pc & !(PAGE_SIZE as u64 - 1)) as u32;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page, word_offset, page_base, 1);
        assert!(non_empty, "entry instruction must not be excluded — check the test's instruction encoding");
        let mut instrs_owned = *walked;

        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word_offset, true, false)?;

        let (exec, mem) = seeded_executor(gpr, pc);
        let mut exec = Box::new(exec);
        for &(addr, val) in mem_init { mem.set_word(addr, val); }
        exec.install_jit_hooks();

        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        // Leak the module: the JIT-compiled code must stay valid for the
        // `jit_fn` call above and this function's return; `codegen` going
        // out of scope would otherwise free the code page it points into.
        // Fine for a test — mirrors codegen.rs's own test harness.
        std::mem::forget(codegen);

        Some(CoreSnapshot::capture(&exec.core))
    }

    /// Multi-instruction counterpart of `run_interpreter`: `page` is a
    /// sparse (word offset, raw instruction) list laid out into a full
    /// page, executed by repeatedly calling `exec.exec(instr)` at the
    /// current `core.pc` until `steps` instructions have retired — needed
    /// for branch/jump equivalence tests, where a single compiled unit
    /// (branch + inlined slot) corresponds to **two** interpreter dispatches
    /// (the branch itself, which arms `core.in_delay_slot` and doesn't
    /// retire, then the delay slot instruction, which is what actually
    /// advances PC to the target) — see `MipsExecutor::branch_delay`. Fixed
    /// `steps` rather than "run until PC leaves the region" keeps this
    /// harness from needing any exit-condition guessing.
    fn run_interpreter_page(page: &[(u16, u32)], gpr: [u64; 32], pc: u64, steps: usize) -> CoreSnapshot {
        // no_jitv2: see MockMemory::new_not_compilable's doc comment — this
        // harness's dispatch-count arithmetic (the `steps` contract above)
        // assumes exactly one instruction retires per exec() call, which a
        // real jitv2 dispatch is free to violate the moment it has a
        // compiled multi-instruction region ready.
        let (mut exec, mem) = seeded_executor_over(MockMemory::new_not_compilable(), gpr, pc);
        let page_base = pc & !(PAGE_SIZE as u64 - 1);
        for &(word, raw) in page {
            mem.set_word(page_base + (word as u64) * 4, raw);
        }
        for _ in 0..steps {
            let fetch_pc = exec.core.pc;
            let instr = mem.get_word(fetch_pc & !3);
            exec.exec(instr);
        }
        CoreSnapshot::capture(&exec.core)
    }

    /// Multi-instruction counterpart of `run_jit`: compiles the region
    /// starting at `entry_word` from a sparse (word offset, raw instruction)
    /// page layout, with an instruction budget of `max_instrs` — *head*
    /// instructions only (`Analyzer::walk_bounded`'s budget; a branch/jump's
    /// mandatory delay slot, however deep a nested slot-chain goes, is
    /// never charged against it — `analyzer::visit_slot`). A single branch
    /// or jump plus its slot wants `max_instrs=1`, not 2 — callers size this
    /// to how many *additional* head instructions past the entry their test
    /// layout needs to reach (not a runtime step count like
    /// `run_interpreter_page`'s `steps`). One JIT call executes the whole
    /// compiled region in a single native call — unlike the interpreter
    /// side, there's no per-instruction step loop here, since the compiled
    /// function runs until it exits the region (returns to the interpreter)
    /// on its own.
    fn run_jit_page(page: &[(u16, u32)], gpr: [u64; 32], pc: u64, entry_word: u16, max_instrs: usize, mem_init: &[(u64, u32)]) -> Option<CoreSnapshot> {
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, raw) in page {
            page_words[word as usize] = raw;
        }

        // page_base must be pc's real page, not 0: analyzer::jump_target
        // (J/JAL's on/off-page classification — unlike branch_target, which
        // is pure word-offset PC-relative math, J/JAL's target is an
        // absolute address within a 256MB region per the ISA, so on-page-
        // ness genuinely depends on where the page sits, not just the entry
        // word) reconstructs an absolute address from page_base to compare
        // against. A page_base that doesn't match the executor's real pc
        // below (previously hardcoded 0 here) makes the analyzer classify
        // against a fictional page while codegen's own address emitters
        // correctly derive everything from the live core.pc at runtime — the
        // two silently disagree whenever a test's real pc isn't page 0,
        // exactly the kind of bug this harness exists to catch, not cause.
        let page_base = (pc & !(PAGE_SIZE as u64 - 1)) as u32;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, entry_word, page_base, max_instrs);
        assert!(non_empty, "entry instruction must not be excluded — check the test's instruction encoding");
        let mut instrs_owned = *walked;

        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, entry_word, true, false)?;

        let (exec, mem) = seeded_executor(gpr, pc);
        let mut exec = Box::new(exec);
        // Store code at both the (unmasked) virtual page base AND the physical
        // (kseg-masked) base. A native-emitter region never re-fetches, so the
        // virtual copy alone sufficed historically — but an interpreter-fallback
        // head (emit_interp_fallback_head) re-fetches through the bus, which
        // sees the translated physical address; without the physical copy it
        // would read 0 (a NOP) and silently mis-execute the fallback word.
        let phys_base = (page_base & 0x1FFF_FFFF) as u64;
        for &(word, raw) in page {
            mem.set_word(page_base as u64 + (word as u64) * 4, raw);
            mem.set_word(phys_base + (word as u64) * 4, raw);
        }
        for &(addr, val) in mem_init { mem.set_word(addr, val); }
        exec.install_jit_hooks();

        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        std::mem::forget(codegen);

        Some(CoreSnapshot::capture(&exec.core))
    }

    /// Same as `run_jit_page`, but also returns `core.hot.cycles` (not part
    /// of `CoreSnapshot` — see its doc comment) for tests that specifically
    /// need to check the cycle count, e.g. `try_emit_fused_nop_slot`'s
    /// branch+NOP fusion (`codegen.rs`), which must still advance `cycles`
    /// by exactly 2 per fused pair even though it skips the slot's normal
    /// per-instruction dispatch.
    fn run_jit_page_with_cycles(page: &[(u16, u32)], gpr: [u64; 32], pc: u64, entry_word: u16, max_instrs: usize, mem_init: &[(u64, u32)]) -> Option<(CoreSnapshot, u64)> {
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, raw) in page {
            page_words[word as usize] = raw;
        }

        let page_base = (pc & !(PAGE_SIZE as u64 - 1)) as u32;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, entry_word, page_base, max_instrs);
        assert!(non_empty, "entry instruction must not be excluded — check the test's instruction encoding");
        let mut instrs_owned = *walked;

        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, entry_word, true, false)?;

        let (exec, mem) = seeded_executor(gpr, pc);
        let mut exec = Box::new(exec);
        let phys_base = (page_base & 0x1FFF_FFFF) as u64;
        for &(word, raw) in page {
            mem.set_word(page_base as u64 + (word as u64) * 4, raw);
            mem.set_word(phys_base + (word as u64) * 4, raw);
        }
        for &(addr, val) in mem_init { mem.set_word(addr, val); }
        exec.install_jit_hooks();

        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        std::mem::forget(codegen);

        Some((CoreSnapshot::capture(&exec.core), exec.core.hot.cycles))
    }

    /// Multi-page, `step()`-driven equivalence harness. Unlike `run_jit_page`
    /// (one pre-compiled single-page region), this loads instructions at
    /// arbitrary virtual addresses spanning several physical pages and drives
    /// the *real* dispatch gate via `step()`, so cross-page `jal`/`jr`
    /// transfers flow through the interpreter's dispatch exactly as they do
    /// live — every page compiles into its own JIT region, control handed back
    /// and forth through the gate. `code` is `(vaddr, raw)` pairs (stored at
    /// both the virtual address and its kseg-masked physical, so fallback
    /// re-fetches through the bus see the same bytes). Runs `steps` dispatches;
    /// caller sizes it to reach a known quiescent PC on both engines.
    ///
    /// `jit`: `true` builds a JIT executor with inline (synchronous, no async
    /// thread — deterministic) compilation and interpreter-fallback ON; `false`
    /// builds a pure interpreter (`new_not_compilable`, no hooks). Same seeded
    /// gpr/pc/mem otherwise.
    #[cfg(not(feature = "lightning"))]
    fn run_multipage(code: &[(u64, u32)], data: &[(u64, u32)], gpr: [u64; 32], pc: u64, steps: usize, jit: bool) -> CoreSnapshot {
        let mem = if jit { MockMemory::new() } else { MockMemory::new_not_compilable() };
        let (exec, mem) = seeded_executor_over(mem, gpr, pc);
        // Box so the executor's address is stable from install_jit_hooks onward
        // (jit_ctx captures &mut exec — same discipline as run_jit).
        let mut exec = Box::new(exec);
        let store = |vaddr: u64, val: u32| {
            mem.set_word(vaddr, val);              // virtual
            mem.set_word(vaddr & 0x1FFF_FFFF, val); // kseg-masked physical (bus/fallback re-fetch)
        };
        for &(vaddr, raw) in code { store(vaddr, raw); }
        for &(vaddr, val) in data { store(vaddr, val); }
        if jit {
            exec.jitv2_inline_compile = true;
            exec.install_jit_hooks();
        }
        for _ in 0..steps {
            exec.step();
        }
        CoreSnapshot::capture(&exec.core)
    }

    /// The developer per-instruction hook (`emit_dev_trace_bp` ->
    /// `core.dev_trace_bp_fn`) must make JIT-executed instructions show up in
    /// the traceback tagged `jit=true`, and a PC breakpoint set inside a
    /// compiled region must fire (stopping before the instruction, pc left at
    /// it for resume). Drives a small straight-line region through the real
    /// `step()` gate with inline compile + hooks so the compiled path runs.
    #[cfg(all(feature = "developer", not(feature = "lightning")))]
    #[test]
    fn dev_hook_traces_and_breakpoints_jit_instructions() {
        let pc = 0xFFFF_FFFF_8000_1000u64;
        // Three ADDIUs then a JR to exit the region cleanly.
        let code = [
            (pc + 0x00, make_i(crate::mips_isa::OP_ADDIU, 0, 1, 1)),
            (pc + 0x04, make_i(crate::mips_isa::OP_ADDIU, 0, 2, 2)),
            (pc + 0x08, make_i(crate::mips_isa::OP_ADDIU, 0, 3, 3)),
            (pc + 0x0c, make_r(crate::mips_isa::OP_SPECIAL, 31, 0, 0, 0, crate::mips_isa::FUNCT_JR)),
            (pc + 0x10, 0), // jr delay slot
        ];
        let mut gpr = [0u64; 32];
        gpr[31] = pc + 0x200; // jr target (off region)

        // --- Part 1: traceback captures JIT-executed instructions, tagged jit.
        {
            let mem = MockMemory::new();
            let (exec, mem) = seeded_executor_over(mem, gpr, pc);
            let mut exec = Box::new(exec);
            for &(vaddr, raw) in &code {
                mem.set_word(vaddr, raw);
                mem.set_word(vaddr & 0x1FFF_FFFF, raw);
            }
            exec.jitv2_inline_compile = true;
            exec.install_jit_hooks();
            // First step compiles + runs the region (the region's own
            // interior instructions record via the dev hook, tagged jit;
            // the entry word is recorded by exec_decoded's own JIT-hit push,
            // right before it jumps into the compiled function — also
            // tagged jit, NOT interp: this is a real external JIT dispatch,
            // it just isn't recorded via the dev hook (that path stays
            // internal-edge-only, see emit_dev_trace_bp's own doc comment).
            exec.step(); // one step runs the whole first region under inline compile

            let entries = exec.test_traceback_last(64);
            let find = |want: u64| entries.iter().find(|&&(epc, _, _)| epc == want)
                .unwrap_or_else(|| panic!("pc {:#x} missing from traceback", want));
            assert!(find(pc).2, "entry word's external JIT dispatch must be tagged jit (recorded by exec_decoded's own JIT-hit push)");
            // The interior instructions ran inside the compiled region, recorded
            // by the dev hook — must be jit-tagged.
            for off in [4u64, 8] {
                assert!(find(pc + off).2,
                    "instruction at {:#x} ran under JIT but wasn't tagged jit in the traceback", pc + off);
            }
        }

        // --- Part 2: a PC breakpoint inside the region fires from compiled code.
        {
            let mem = MockMemory::new();
            let (exec, mem) = seeded_executor_over(mem, gpr, pc);
            let mut exec = Box::new(exec);
            for &(vaddr, raw) in &code {
                mem.set_word(vaddr, raw);
                mem.set_word(vaddr & 0x1FFF_FFFF, raw);
            }
            exec.jitv2_inline_compile = true;
            exec.install_jit_hooks();
            // Breakpoint on the 3rd ADDIU (pc+8), which lives inside the
            // compiled region (not the entry word).
            exec.add_breakpoint(1, pc + 8, crate::mips_exec::BpType::Pc);

            let mut hit = false;
            for _ in 0..6 {
                if exec.step() == crate::mips_exec::EXEC_BREAKPOINT { hit = true; break; }
            }
            assert!(hit, "PC breakpoint inside a compiled region must fire from the dev hook");
            assert_eq!(exec.core.pc, pc + 8,
                "breakpoint must stop with pc at the instruction (before it executes), for resume");
            assert_eq!(exec.core.gpr[3], 0,
                "the breakpointed instruction (addiu r3) must not have executed yet");
        }
    }

    /// Every `InstrOrigin` arrival class actually gets recorded with its own
    /// distinct tag: plain JIT body, an inlined delay slot, a fallback word,
    /// and a fallback successor — all in one straight-line region, all
    /// reached via their real external/first-time arrival (not a back-edge;
    /// see the two back-edge tests below for those). Region:
    ///   word0 (entry): ADDIU r2 = r1+1        -> Jit (recorded by
    ///                                             exec_decoded's own JIT-hit
    ///                                             push, right before it
    ///                                             jumps in — a real JIT
    ///                                             dispatch, just not routed
    ///                                             through the dev hook)
    ///   word1: BEQ r0,r0,+1 (always taken)    -> Jit (a plain head; itself
    ///                                             not delay-slot-tagged)
    ///   word2: delay slot: ADDIU r3 = r1+2    -> JitDelaySlot
    ///   word3: BC1F cc0, not taken (fallback head) -> FallbackWord
    ///   word4: fallback successor (BC1's delay slot): ADDIU r5=r1+3 -> FallbackSuccessor
    ///
    /// The fallback must be a BC1 specifically, not a plain excluded
    /// instruction like MTC0: only a fallback that is itself a branch marks
    /// its successor `is_branch_fallback_successor` (`analyzer::visit`'s
    /// `is_fallback_branch` check) — an MTC0's successor is just an ordinary
    /// fallthrough word, tagged `Jit` like any other plain head (found via
    /// this exact test failing with an MTC0 fallback: successor came back
    /// `Jit`, not `FallbackSuccessor`, correctly per that rule).
    #[test]
    #[cfg(feature = "developer")]
    fn every_instr_origin_class_is_tagged_in_traceback() {
        let _fb = fallback_on_guard();
        let pc = 0xFFFF_FFFF_8000_2000u64;
        let entry0 = ((pc & 0xFFF) / 4) as u16;
        let mut gpr = [0u64; 32];
        gpr[1] = 0x10;

        let addiu0 = make_i(crate::mips_isa::OP_ADDIU, 1, 2, 1);
        let beq = make_i(crate::mips_isa::OP_BEQ, 0, 0, 1); // always taken, target = word1+1+1 = word3
        let slot = make_i(crate::mips_isa::OP_ADDIU, 1, 3, 2);
        // BC1F cc0, +10: branch-if-false, condition = !cc. cc0 set true
        // below -> condition = false -> NOT taken -> falls through to its
        // own delay slot (word4), same as bc1f in the back-edge test.
        let bc1f = (crate::mips_isa::OP_COP1 << 26) | (crate::mips_isa::RS_BC1 << 21) | 10u32;
        let succ = make_i(crate::mips_isa::OP_ADDIU, 1, 5, 3);
        let page = [
            (entry0, addiu0),
            (entry0 + 1, beq),
            (entry0 + 2, slot),
            (entry0 + 3, bc1f),
            (entry0 + 4, succ),
            // Boundary sentinel: without this, the rest of MockMemory's
            // zero-filled page decodes as valid NOPs (SLL r0,r0,0) and the
            // analyzer just keeps walking the region all the way to the
            // page's end — filling the 64-entry traceback window with ~250
            // more Jit-tagged NOPs and pushing this test's 5 real entries
            // out of it entirely (found via this exact test failing empty).
            (entry0 + 5, crate::mips_isa::JIT_REGION_BOUNDARY_SENTINEL),
        ];

        // Drive through the REAL dispatch gate (step()), not a direct jit_fn
        // call: only step() records the entry word's external Interp tag —
        // a direct compile_region+call harness (like run_jit_page) never
        // goes through step() at all, so it can't exercise that distinction.
        let mem = MockMemory::new();
        let (exec, mem) = seeded_executor_over(mem, gpr, pc);
        let mut exec = Box::new(exec);
        for &(w, raw) in &page {
            let vaddr = ((pc & !0xFFFu64) as u64) + (w as u64) * 4;
            mem.set_word(vaddr, raw);
            mem.set_word(vaddr & 0x1FFF_FFFF, raw);
        }
        exec.core.cp0_status |= crate::mips_core::STATUS_CU1;
        exec.core.set_fpu_cc(0, true); // cc0=true -> BC1F not taken
        exec.update_fpr_mode();
        exec.jitv2_inline_compile = true;
        exec.install_jit_hooks();
        exec.step(); // one step compiles + runs the whole region under inline compile

        // Sanity: all three ADDIUs actually ran (proves the region really
        // executed entry+slot+fallback+successor, not just compiled).
        assert_eq!(exec.core.gpr[2], 0x11, "entry word ADDIU did not run");
        assert_eq!(exec.core.gpr[3], 0x12, "delay slot ADDIU did not run");
        assert_eq!(exec.core.gpr[5], 0x13, "fallback successor ADDIU did not run");

        let entries = exec.test_traceback_last_origin(64);
        let find = |want: u64| entries.iter().find(|&&(epc, _, _)| epc == want)
            .unwrap_or_else(|| panic!("pc {:#x} missing from traceback: {:x?}", want, entries));

        assert_eq!(find(pc).2, crate::mips_exec::InstrOrigin::JitEntry,
            "entry word's external JIT dispatch must be tagged JitEntry (recorded by exec_decoded's own JIT-hit push, not the dev hook)");
        assert_eq!(find(pc + 4).2, crate::mips_exec::InstrOrigin::Jit,
            "the BEQ head must be tagged Jit");
        assert_eq!(find(pc + 8).2, crate::mips_exec::InstrOrigin::JitDelaySlot,
            "the BEQ's inlined delay slot must be tagged JitDelaySlot");
        assert_eq!(find(pc + 12).2, crate::mips_exec::InstrOrigin::FallbackWord,
            "the MTC0 fallback head must be tagged FallbackWord");
        assert_eq!(find(pc + 16).2, crate::mips_exec::InstrOrigin::FallbackSuccessor,
            "the word after the fallback must be tagged FallbackSuccessor");
    }

    /// The region's entry word, reached a SECOND time via an internal
    /// back-edge (a backward branch looping to word0), is tagged
    /// `JitEntryBackEdge` — distinct from its first, external arrival
    /// (`Jit`, recorded by `exec_decoded`'s own JIT-hit push). Loop: word0
    /// (entry, decrement counter) -> word1 (BNE back to word0 while counter
    /// != 0) -> word2 (delay slot) -> falls through to word3 (boundary
    /// sentinel) once done.
    #[test]
    #[cfg(feature = "developer")]
    fn entry_word_back_edge_is_tagged_distinctly() {
        let pc = 0xFFFF_FFFF_8000_3000u64;
        let entry0 = ((pc & 0xFFF) / 4) as u16;
        let mut gpr = [0u64; 32];
        gpr[1] = 2; // loop counter: 2 iterations -> 1 back-edge

        let dec = make_i(crate::mips_isa::OP_ADDIU, 1, 1, 0xFFFF); // -1
        let bne = make_i(crate::mips_isa::OP_BNE, 1, 0, (-2i16) as u16); // target = word0
        let page = [
            (entry0, dec),
            (entry0 + 1, bne),
            (entry0 + 2, 0), // bne's delay slot (nop)
            (entry0 + 3, crate::mips_isa::JIT_REGION_BOUNDARY_SENTINEL),
        ];

        let mem = MockMemory::new();
        let (exec, mem) = seeded_executor_over(mem, gpr, pc);
        let mut exec = Box::new(exec);
        for &(w, raw) in &page {
            let vaddr = ((pc & !0xFFFu64) as u64) + (w as u64) * 4;
            mem.set_word(vaddr, raw);
            mem.set_word(vaddr & 0x1FFF_FFFF, raw);
        }
        exec.jitv2_inline_compile = true;
        exec.install_jit_hooks();
        exec.step(); // one step compiles + runs the whole looping region

        assert_eq!(exec.core.gpr[1], 0, "loop must run to completion (counter 2 -> 0)");

        let entries = exec.test_traceback_last_origin(64);
        let entry_hits: Vec<_> = entries.iter().filter(|&&(epc, _, _)| epc == pc).collect();
        assert_eq!(entry_hits.len(), 2, "entry word must be dispatched twice (external + one back-edge): {:x?}", entries);
        assert_eq!(entry_hits[0].2, crate::mips_exec::InstrOrigin::JitEntry,
            "first (external) entry arrival must be tagged JitEntry (exec_decoded's own JIT-hit push)");
        assert_eq!(entry_hits[1].2, crate::mips_exec::InstrOrigin::JitEntryBackEdge,
            "second (looped) entry arrival must be tagged JitEntryBackEdge");
    }

    /// A fallback successor word, reached a SECOND time via an internal
    /// back-edge (a backward branch looping to it, not to the fallback
    /// itself), still gets recorded and doesn't silently vanish from `dt`.
    /// Only a fallback that is ITSELF a branch (BC1 — see
    /// `analyzer::is_fallback_branch`) marks its successor
    /// `is_branch_fallback_successor`; a plain fallback (e.g. MTC0) does not
    /// — its successor is just an ordinary fallthrough word, so this test
    /// must use BC1, not `benign_excluded_mtc0`. Loop: word0 (fallback head,
    /// BC1 not-taken) -> word1 (successor/BC1 delay slot, decrement counter)
    /// -> word2 (BNE back to word1 while counter != 0) -> word3 (delay slot)
    /// -> falls through to word4 (boundary sentinel).
    ///
    /// `is_branch_fallback_successor` is walk-local, not persistent: a
    /// fallback's own compiled region ends right after its successor (found
    /// live: one step() landed exactly at the BNE's own address without
    /// dispatching it), so the BNE and its loop body compile as a SEPARATE
    /// region on the next dispatch, starting a fresh analyzer walk from the
    /// BNE itself — a walk that never visits word0's fallback at all, so it
    /// never learns word1 is architecturally a fallback successor. The
    /// second region sees word1 purely as an ordinary backward-branch
    /// target, tagged `Jit` like any other in-region head — correctly, since
    /// on THIS arrival path (a plain taken branch, not a delay-slot-armed
    /// transfer) `in_delay_slot` really is false and no foreign-slot
    /// treatment is needed. So the two arrivals get two different, both
    /// individually-correct tags: `FallbackSuccessor` (first, from the
    /// fallback's own region) then `Jit` (second, from the loop body's own
    /// separately-compiled region) — not the same tag twice.
    #[test]
    #[cfg(feature = "developer")]
    fn fallback_successor_back_edge_is_still_recorded() {
        let _fb = fallback_on_guard();
        let pc = 0xFFFF_FFFF_8000_4000u64;
        let entry0 = ((pc & 0xFFF) / 4) as u16;
        let mut gpr = [0u64; 32];
        gpr[1] = 2; // loop counter: 2 iterations through the successor -> 1 back-edge

        // BC1F cc0, +10: branch-if-false, condition = !cc. With cc0=true
        // below, condition = !true = false -> NOT taken -> falls through to
        // its own delay slot (word1), same as any non-taken branch. (Offset
        // value is irrelevant on the not-taken path — only used if taken.)
        let bc1f = (crate::mips_isa::OP_COP1 << 26) | (crate::mips_isa::RS_BC1 << 21) | 10u32;
        let dec = make_i(crate::mips_isa::OP_ADDIU, 1, 1, 0xFFFF); // -1, BC1's delay slot / successor word
        let bne = make_i(crate::mips_isa::OP_BNE, 1, 0, (-2i16) as u16); // target = word1 (the successor)
        let page = [
            (entry0, bc1f),
            (entry0 + 1, dec),
            (entry0 + 2, bne),
            (entry0 + 3, 0), // bne's delay slot (nop)
            (entry0 + 4, crate::mips_isa::JIT_REGION_BOUNDARY_SENTINEL),
        ];

        let mem = MockMemory::new();
        let (exec, mem) = seeded_executor_over(mem, gpr, pc);
        let mut exec = Box::new(exec);
        for &(w, raw) in &page {
            let vaddr = ((pc & !0xFFFu64) as u64) + (w as u64) * 4;
            mem.set_word(vaddr, raw);
            mem.set_word(vaddr & 0x1FFF_FFFF, raw);
        }
        exec.core.cp0_status |= crate::mips_core::STATUS_CU1;
        exec.core.set_fpu_cc(0, true); // cc0=true -> BC1F (branch-if-false) not taken -> falls through
        exec.update_fpr_mode();
        exec.jitv2_inline_compile = true;
        exec.install_jit_hooks();
        // A fallback word ends its own compiled region right after its
        // successor (found live: one step() call landed exactly at the BNE's
        // own address, EXEC_COMPLETE, without dispatching it — the BNE and
        // its delay slot compile as a fresh, separate region on the next
        // dispatch) — unlike a pure-JIT loop, which can stay in one step()
        // call for its whole run. Loop step() with a safety cap instead of
        // assuming one call suffices.
        for _ in 0..16 {
            if exec.core.gpr[1] == 0 { break; }
            exec.step();
        }

        assert_eq!(exec.core.gpr[1], 0, "loop must run to completion (counter 2 -> 0)");

        let successor_pc = pc + 4;
        let entries = exec.test_traceback_last_origin(64);
        let hits: Vec<_> = entries.iter().filter(|&&(epc, _, _)| epc == successor_pc).collect();
        assert_eq!(hits.len(), 2, "fallback successor must be dispatched twice (first arrival + one back-edge): {:x?}", entries);
        assert_eq!(hits[0].2, crate::mips_exec::InstrOrigin::FallbackSuccessor,
            "first arrival (from the fallback's own region) must be tagged FallbackSuccessor");
        assert_eq!(hits[1].2, crate::mips_exec::InstrOrigin::Jit,
            "second arrival (the loop back-edge) lands inside an already-compiled region as a plain interior word (not a fresh external entry, per emit_dev_trace_bp's normal path) — must be tagged Jit, not silently dropped from the trace");
    }

    /// Diagnostic (not a strict regression gate): loads the REAL page
    /// contents captured live from `wd93_init`'s physical page (pfn=0x800b,
    /// `ignore/mem.txt`, a `m 0xffffffff8800b000 1024` dump) and walks the
    /// analyzer from word 0x150 (`0xffffffff8800b540`) — the exact entry
    /// that live-boot's `handle_request` logged as `instr_count=2
    /// visited_words=[150, 151]`. Confirms whether that's a real analyzer
    /// bug or architecturally correct: word 0x150 is `jal 0x805de0` (a real
    /// call, mandatory delay slot at 0x151, then control leaves this
    /// page/walk) — a 2-instruction region is what a JAL entry SHOULD
    /// produce, not evidence of a truncation bug.
    #[test]
    fn wd93_init_real_page_walk_from_jal_entry() {
        let raw = include_str!("../../ignore/mem.txt");
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        let mut count = 0;
        for line in raw.lines() {
            let line = line.trim();
            let Some((addr_str, word_str)) = line.split_once(':') else { continue };
            let addr_str = addr_str.trim();
            let word_str = word_str.trim();
            let Ok(addr) = u64::from_str_radix(addr_str.trim_start_matches("0x").trim_start_matches("ffffffff"), 16) else { continue };
            let Ok(word) = u32::from_str_radix(word_str, 16) else { continue };
            let off = ((addr & 0xFFF) / 4) as usize;
            if off < ENTRIES_PER_PAGE {
                page_words[off] = word;
                count += 1;
            }
        }
        assert_eq!(count, 1024, "must have parsed all 1024 words from ignore/mem.txt — got {}", count);

        // Sanity: word 0x150 (0xffffffff8800b540) must be the JAL we expect,
        // confirming the file parsed at the right offsets.
        assert_eq!(page_words[0x150], 0x0e0177b8, "word 0x150 must be the jal seen live");
        assert_eq!(page_words[0x151], 0x02002025, "word 0x151 must be jal's delay slot");

        let mut analyzer = Analyzer::new();
        let entry_word: u16 = 0x150;
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, entry_word, 0x8800b000, usize::MAX);
        assert!(non_empty, "region must not be empty");
        let visited: Vec<u16> = crate::jitv2::analyzer::instrs_linear(walked).map(|i| i.word).collect();
        eprintln!("DEBUG real-page walk from word {:#x}: visited = {:x?}", entry_word, visited);
        // A JAL is a control transfer with a mandatory delay slot — the walk
        // SHOULD stop right after the slot (nothing else is reachable by
        // straight-line fallthrough from a jal), so instr_count=2 here is
        // architecturally correct, not a truncation bug. This assertion
        // documents that expectation explicitly.
        assert_eq!(visited, vec![0x150u16, 0x151],
            "jal entry should walk exactly 2 words (itself + mandatory delay slot)");

        // The other case from the live log: walking from word 0x154 DOES
        // include 0x150/0x151 — initially looked like a bug (a "forward"
        // walk reaching backward-earlier words), but it isn't: word 0x15a
        // (0xffffffff8800b568) is `bne t9,zero,-11`, a backward branch whose
        // target is exactly word 0x150 ((0x15a+1)-11 = 0x150) — a real loop
        // back-edge inside this same region. `visit` correctly follows it
        // and promotes 0x150/0x151 to reachable heads. Not a bug; documents
        // the real (non-forward-only) reachability semantics.
        let mut analyzer2 = Analyzer::new();
        let entry2: u16 = 0x154;
        let (walked2, non_empty2) = analyzer2.walk_bounded(&page_words, entry2, 0x8800b000, usize::MAX);
        assert!(non_empty2, "region must not be empty");
        let visited2: Vec<u16> = crate::jitv2::analyzer::instrs_linear(walked2).map(|i| i.word).collect();
        eprintln!("DEBUG real-page walk from word {:#x}: visited = {:x?}", entry2, visited2);
        assert!(visited2.contains(&0x150) && visited2.contains(&0x151),
            "0x150/0x151 SHOULD be included here — they're the target of a real backward branch (bne at 0x15a) inside this region: {:x?}", visited2);
    }

    #[test]
    #[cfg(feature = "developer")]
    fn code_size_scales_sublinearly_with_repeated_exception_exit_sites() {
        // Diagnostic, not a strict regression gate: confirms the shared
        // exception_exit_block (codegen.rs) actually shrinks output for a
        // region hitting emit_exception_exit many times, rather than just
        // trusting the refactor's intent. Before the shared block existed,
        // every ADD here would inline its own full copy of the delay-slot
        // check + handle_exception_fn call — size would scale roughly
        // linearly with instruction count; after, only the constant-size
        // jump-with-args at each site should scale, with the shared block's
        // body paid once regardless of how many ADDs are chained.
        let add = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_ADD);
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let page_base = (pc & !(PAGE_SIZE as u64 - 1)) as u32;

        let size_for_n_instrs = |n: u16| -> u32 {
            let mut page_words = [0u32; ENTRIES_PER_PAGE];
            for w in 0..n { page_words[w as usize] = add; }
            let mut analyzer = Analyzer::new();
            let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, n as usize);
            assert!(non_empty);
            let mut instrs_owned = *walked;
            let mut codegen = Codegen::new();
            let _jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
                .expect("ADD chain must be compilable for this test to be meaningful");
            let size = codegen.last_code_size();
            std::mem::forget(codegen);
            size
        };

        let size_1 = size_for_n_instrs(1);
        let size_16 = size_for_n_instrs(16);
        let per_instr_growth = (size_16 - size_1) as f64 / 15.0;

        // Baseline: ADDU has no exception-exit call at all (never overflow-
        // traps) — its marginal per-instruction cost is purely preamble +
        // add + write-back, with zero contribution from emit_exception_exit
        // either way. Comparing ADD's marginal growth against this baseline
        // (rather than an arbitrary hardcoded byte ceiling) isolates
        // specifically what the overflow-check + exception-exit call site
        // costs on top of a plain instruction, which is what the shared
        // block is actually supposed to shrink.
        let addu = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_ADDU);
        let size_for_n_addu = |n: u16| -> u32 {
            let mut page_words = [0u32; ENTRIES_PER_PAGE];
            for w in 0..n { page_words[w as usize] = addu; }
            let mut analyzer = Analyzer::new();
            let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, n as usize);
            assert!(non_empty);
            let mut instrs_owned = *walked;
            let mut codegen = Codegen::new();
            let _jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
                .expect("ADDU chain must be compilable for this test to be meaningful");
            let size = codegen.last_code_size();
            std::mem::forget(codegen);
            size
        };
        let addu_size_1 = size_for_n_addu(1);
        let addu_size_16 = size_for_n_addu(16);
        let addu_per_instr_growth = (addu_size_16 - addu_size_1) as f64 / 15.0;

        println!("ADD  (traps): 1 = {size_1} bytes, 16 = {size_16} bytes, ~{per_instr_growth:.1} bytes/instr marginal");
        println!("ADDU (never traps): 1 = {addu_size_1} bytes, 16 = {addu_size_16} bytes, ~{addu_per_instr_growth:.1} bytes/instr marginal");
        println!("overflow-check + exception-exit call site overhead: ~{:.1} bytes/instr", per_instr_growth - addu_per_instr_growth);

        // The shared block's whole point is that this delta stays small and
        // roughly constant (a compare + conditional jump-with-args to the
        // one shared block) instead of scaling with a full inlined copy of
        // the delay-slot-check + handle_exception_fn call (~15-20
        // instructions) at every site. Sanity ceiling, not a tight bound —
        // exact bytes are host-ISA/Cranelift-version-dependent.
        let exception_exit_overhead = per_instr_growth - addu_per_instr_growth;
        assert!(exception_exit_overhead < 40.0,
            "overflow-check + exception-exit overhead per site ({exception_exit_overhead:.1} bytes) looks like a full inlined copy, not a shared-block jump");
    }

    /// Run a multi-instruction page through both engines and assert their
    /// resulting snapshots match exactly. `steps` is the interpreter's
    /// dispatch count (see `run_interpreter_page`); `max_instrs` is the
    /// JIT's analyzer walk budget (see `run_jit_page`) — for a single
    /// branch + slot with both arms one instruction each landing back in
    /// the same tiny region, `max_instrs` typically wants to be generous
    /// (e.g. the whole page's live instruction count) since the walker
    /// explores both arms.
    fn assert_jit_matches_interpreter_page(page: &[(u16, u32)], gpr: [u64; 32], pc: u64, entry_word: u16, steps: usize, max_instrs: usize) {
        let interp = run_interpreter_page(page, gpr, pc, steps);
        let jit = run_jit_page(page, gpr, pc, entry_word, max_instrs, &[])
            .expect("region must be compilable for this test to be meaningful");
        assert_eq!(jit, interp, "JIT and interpreter diverged for page={:x?} entry_word={} pc=0x{:x}", page, entry_word, pc);
    }

    // ---- FPU (CP1) test harness ------
    //
    // Separate from the integer harness above: FPU tests need FPR contents
    // seeded and CU1 (+ FR for double/long-format tests) set in cp0_status,
    // which the shared `seeded_core`/`seeded_executor` deliberately don't
    // set by default (every non-FPU test relies on CU1 being irrelevant).

    /// no_jitv2 (see `MockMemory::new_not_compilable`'s doc comment): every
    /// caller that actually wants jitv2 involved (`run_jit_fpu` and the
    /// individual `jit_exec` sites) calls `install_jit_hooks` explicitly
    /// before touching a compiled function or `exec_decoded`, so whether
    /// this executor's page is denylisted makes no difference to them — but
    /// the `interp_exec`/`run_interpreter_fpu` callers that never install
    /// hooks would abort if a jitv2 gate variant ever intercepted their
    /// plain `exec.exec()` call (as `jitv2_lockstep`'s `lockstep_check`
    /// already does unconditionally for ALU ops, with FPU ops a natural next
    /// target — see `add_overflow_traps_and_matches_interpreter`'s
    /// pre-fix history in rules/jitv2/codegen-gotchas.md). Always
    /// denylisted, unconditionally, is simplest and safe for every caller
    /// here.
    fn fpu_seeded_executor(gpr: [u64; 32], fpr: [u64; 32], pc: u64, fr1: bool) -> (MipsExecutor<PassthroughTlb, PassthroughCache>, Arc<MockMemory>) {
        let (mut exec, mem) = seeded_executor_over(MockMemory::new_not_compilable(), gpr, pc);
        exec.core.fpr = fpr;
        exec.core.cp0_status = crate::mips_core::STATUS_CU1 | if fr1 { crate::mips_core::STATUS_FR } else { 0 };
        // Setting cp0_status directly (bypassing write_cp0's
        // on_cp0_status_changed callback) leaves the executor's
        // fpr_read_w/fpr_write_w fn pointers stale at whatever FR mode
        // MipsExecutor::new() initialized them to — update_fpr_mode()
        // re-derives them from the live STATUS_FR bit, same as the real
        // write_cp0 path would trigger automatically.
        exec.update_fpr_mode();
        (exec, mem)
    }

    fn run_interpreter_fpu(instr: u32, gpr: [u64; 32], fpr: [u64; 32], pc: u64, fr1: bool) -> CoreSnapshot {
        let (mut exec, _mem) = fpu_seeded_executor(gpr, fpr, pc, fr1);
        exec.exec(instr);
        CoreSnapshot::capture(&exec.core)
    }

    fn run_jit_fpu(instr: u32, gpr: [u64; 32], fpr: [u64; 32], pc: u64, word_offset: u16, fr1: bool) -> Option<CoreSnapshot> {
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[word_offset as usize] = instr;

        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page, word_offset, 0, 1);
        assert!(non_empty, "entry instruction must not be excluded — check the test's instruction encoding");
        let mut instrs_owned = *walked;

        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word_offset, fr1, false)?;

        let (exec, _mem) = fpu_seeded_executor(gpr, fpr, pc, fr1);
        let mut exec = Box::new(exec);
        exec.install_jit_hooks();

        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        std::mem::forget(codegen);

        Some(CoreSnapshot::capture(&exec.core))
    }

    fn assert_fpu_matches_interpreter(instr: u32, gpr: [u64; 32], fpr: [u64; 32], fr1: bool) {
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let word_offset = (pc as u16 / 4) & 0x3FF;
        let interp = run_interpreter_fpu(instr, gpr, fpr, pc, fr1);
        let jit = run_jit_fpu(instr, gpr, fpr, pc, word_offset, fr1)
            .expect("FPU instruction must be compilable for this test to be meaningful");
        assert_eq!(jit, interp, "JIT and interpreter diverged for instr=0x{:08x} fr1={}", instr, fr1);
    }

    /// `run_interpreter_fpu`/`run_jit_fpu`'s counterpart for LWC1/LDC1 —
    /// same CU1/FR-mode seeding, plus `mem_init` for the load to actually
    /// read something (both harnesses declined to unify since only these
    /// two instructions in the whole FPU family need memory pre-population
    /// at all).
    fn run_interpreter_fpu_mem(instr: u32, gpr: [u64; 32], fpr: [u64; 32], pc: u64, fr1: bool, mem_init: &[(u64, u32)]) -> CoreSnapshot {
        let (mut exec, mem) = fpu_seeded_executor(gpr, fpr, pc, fr1);
        for &(addr, val) in mem_init { mem.set_word(addr, val); }
        exec.exec(instr);
        CoreSnapshot::capture(&exec.core)
    }

    fn run_jit_fpu_mem(instr: u32, gpr: [u64; 32], fpr: [u64; 32], pc: u64, word_offset: u16, fr1: bool, mem_init: &[(u64, u32)]) -> Option<CoreSnapshot> {
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[word_offset as usize] = instr;

        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page, word_offset, 0, 1);
        assert!(non_empty, "entry instruction must not be excluded — check the test's instruction encoding");
        let mut instrs_owned = *walked;

        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word_offset, fr1, false)?;

        let (exec, mem) = fpu_seeded_executor(gpr, fpr, pc, fr1);
        let mut exec = Box::new(exec);
        for &(addr, val) in mem_init { mem.set_word(addr, val); }
        exec.install_jit_hooks();

        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        std::mem::forget(codegen);

        Some(CoreSnapshot::capture(&exec.core))
    }

    fn assert_fpu_matches_interpreter_mem(instr: u32, gpr: [u64; 32], fpr: [u64; 32], fr1: bool, mem_init: &[(u64, u32)]) {
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let word_offset = (pc as u16 / 4) & 0x3FF;
        let interp = run_interpreter_fpu_mem(instr, gpr, fpr, pc, fr1, mem_init);
        let jit = run_jit_fpu_mem(instr, gpr, fpr, pc, word_offset, fr1, mem_init)
            .expect("FPU load/store must be compilable for this test to be meaningful");
        assert_eq!(jit, interp, "JIT and interpreter diverged for instr=0x{:08x} fr1={}", instr, fr1);
    }

    /// Run `instr` through both engines from the same seeded state (and, for
    /// memory-referencing instructions, the same pre-populated memory) and
    /// assert their resulting `CoreSnapshot`s match exactly (§4.6's per-unit
    /// lockstep contract, applied at test time instead of at runtime).
    fn assert_jit_matches_interpreter(instr: u32, gpr: [u64; 32], pc: u64) {
        assert_jit_matches_interpreter_mem(instr, gpr, pc, &[]);
    }

    fn assert_jit_matches_interpreter_mem(instr: u32, gpr: [u64; 32], pc: u64, mem_init: &[(u64, u32)]) {
        let interp = run_interpreter(instr, gpr, pc, mem_init);
        let word_offset = (pc as u16 / 4) & 0x3FF;
        let jit = run_jit(instr, gpr, pc, word_offset, mem_init)
            .expect("instruction must be compilable for this test to be meaningful");
        assert_eq!(jit, interp, "JIT and interpreter diverged for instr=0x{:08x} pc=0x{:x}", instr, pc);
    }

    /// Guards `Codegen::set_opt_level_speed`'s process-wide global static for
    /// the one test below that needs to flip it — every other test in this
    /// (parallel, by default) suite constructs its own `Codegen` via
    /// `Codegen::new()`, which reads that same global at construction time;
    /// mutating it unguarded mid-suite could make a concurrently-running
    /// test's `Codegen::new()` observe `speed` when it expected `none` (or
    /// vice versa) purely by scheduling luck. This lock only protects
    /// against two copies of *this* test's own logic overlapping — it can't
    /// stop some other, unrelated test's `Codegen::new()` from reading
    /// whatever the global happens to be while this test has it flipped to
    /// `speed`, which is a real (if narrow) source of flakiness this file
    /// otherwise has no mechanism to prevent. Acceptable here specifically
    /// because opt_level only affects codegen *quality*, never architectural
    /// correctness — a test elsewhere that happens to run under `speed`
    /// during this window still produces a correct compile, just not
    /// necessarily the exact one its own comments assume, which no test in
    /// this file actually depends on.
    static OPT_LEVEL_SPEED_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Forces interpreter-fallback ON for the returned guard's lifetime (the
    /// analyzer's `FALLBACK_ENABLED` defaults OFF now that it's a runtime `j2
    /// fallback` toggle). Delegates to the crate-wide
    /// `analyzer::test_fallback_guard` so this module's fallback tests share the
    /// ONE lock with the analyzer module's — two separate locks would not be
    /// mutually exclusive and would race on the shared flag (a real flake this
    /// consolidation fixes). Fallback affects *correctness* (whether an
    /// Excluded word is admitted to a region), so the guard is held for the
    /// whole test body; tests using only native instructions or the
    /// `RegionBoundary` sentinel are flag-independent and don't need it.
    fn fallback_on_guard() -> crate::jitv2::analyzer::TestFallbackGuard {
        crate::jitv2::analyzer::test_fallback_guard()
    }

    /// Not a diff against `assert_jit_matches_interpreter`'s `opt_level=none`
    /// codepath — that's covered by every other test in this file. This
    /// confirms `opt_level=speed` itself (real Cranelift optimization
    /// passes, exercised for the first time anywhere in this suite) doesn't
    /// miscompile the specific paths most likely to be sensitive to
    /// reordering/optimization: the atomic pending-interrupt load
    /// (`emit_pending_interrupt_preamble`'s `atomic_load` fix — this is
    /// exactly the class of bug `opt_level=none` could never have caught,
    /// since no reordering passes run under it) and every block this session
    /// marked `set_cold_block` (overflow traps, TEQ/TNE-family traps, FPU
    /// exception paths) — a cold-block hint is a scheduling nudge, not
    /// something with its own distinct execution semantics, but confirming
    /// equivalence under `speed` for the exact blocks that hint touches is
    /// the cheapest way to be sure the hint didn't accidentally interact
    /// with a real optimization pass in some code-motion-unsafe way.
    #[test]
    fn opt_level_speed_matches_interpreter_for_traps_and_overflow_and_fpu() {
        let _guard = OPT_LEVEL_SPEED_TEST_LOCK.lock().unwrap();
        Codegen::set_opt_level_speed(true);
        let restore = std::panic::AssertUnwindSafe(|| Codegen::set_opt_level_speed(false));
        let result = std::panic::catch_unwind(|| {
            // ADDU: plain, no cold blocks — baseline "does speed mode even work at all."
            let mut gpr = [0u64; 32];
            gpr[1] = 10;
            gpr[2] = 20;
            let addu = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_ADDU);
            assert_jit_matches_interpreter(addu, gpr, 0xFFFF_FFFF_8000_1000);

            // ADD: overflow-trap cold block, both taken and not-taken.
            let mut gpr_ok = [0u64; 32];
            gpr_ok[1] = 5;
            gpr_ok[2] = 10;
            let add = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_ADD);
            assert_jit_matches_interpreter(add, gpr_ok, 0xFFFF_FFFF_8000_1000);

            let mut gpr_overflow = [0u64; 32];
            gpr_overflow[1] = 0xFFFF_FFFF_7FFF_FFFF; // sign-extended i32::MAX
            gpr_overflow[2] = 1;
            assert_jit_matches_interpreter(add, gpr_overflow, 0xFFFF_FFFF_8000_1000);

            // TEQ: trap-family cold block, both taken and not-taken.
            let mut gpr_trap_taken = [0u64; 32];
            gpr_trap_taken[1] = 7;
            gpr_trap_taken[2] = 7;
            let teq = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 0, 0, crate::mips_isa::FUNCT_TEQ);
            assert_jit_matches_interpreter(teq, gpr_trap_taken, 0xFFFF_FFFF_8000_1000);

            let mut gpr_trap_not_taken = [0u64; 32];
            gpr_trap_not_taken[1] = 7;
            gpr_trap_not_taken[2] = 8;
            assert_jit_matches_interpreter(teq, gpr_trap_not_taken, 0xFFFF_FFFF_8000_1000);
        });
        restore.0();
        result.unwrap();
    }

    #[test]
    fn addu_matches_interpreter_basic() {
        let mut gpr = [0u64; 32];
        gpr[1] = 10;
        gpr[2] = 20;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_ADDU);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn addu_matches_interpreter_32bit_wraparound() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF;
        gpr[2] = 1;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_ADDU);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn addu_matches_interpreter_sign_extension() {
        // rs+rt's 32-bit result has its high bit set -> must sign-extend to
        // 64 bits, same as MipsExecutor::exec_addu.
        let mut gpr = [0u64; 32];
        gpr[1] = 0x7FFF_FFFF;
        gpr[2] = 1; // sum = 0x8000_0000 (32-bit), sign-extends to 0xFFFFFFFF_80000000
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_ADDU);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn addu_writing_r0_is_a_noop() {
        let mut gpr = [0u64; 32];
        gpr[1] = 5;
        gpr[2] = 5;
        // rd = 0: architecturally a no-op write.
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 0, 0, crate::mips_isa::FUNCT_ADDU);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn addu_same_register_as_both_operands() {
        let mut gpr = [0u64; 32];
        gpr[4] = 7;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 4, 4, 4, 0, crate::mips_isa::FUNCT_ADDU);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn analyzer_admits_unimplemented_instruction_as_a_fallback_entry() {
        let _fb = fallback_on_guard();
        // PREFX has no emitter (opcode_support::has_emitter) and, unlike most
        // other gaps closed in this file's history, never will: exec_prefx
        // checks STATUS_CU1 and raises cpu_unusable when it's clear, a genuine
        // COP0-adjacent side effect this codebase's hard no on privilege/
        // COP0-touching instructions excludes from *native* jitv2 codegen
        // permanently.
        //
        // With interpreter-fallback, "no native emitter" no longer means "not
        // in a region": the analyzer admits an Excluded instruction as a
        // fallback head (is_fallback), and codegen runs it through the real
        // interpreter (emit_interp_fallback_head), which delivers the
        // cpu_unusable exception correctly. So a lone PREFX entry is now a
        // one-instruction fallback region, not an empty one. (Pre-fallback this
        // asserted non_empty == false — see git history / the analyzer's own
        // walk_excluded_entry_* tests, updated in the same change.)
        let instr = make_r(crate::mips_isa::OP_COP1X, 1, 2, 3, 4, crate::mips_isa::FUNCT_PREFX);
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = instr;
        let mut analyzer = Analyzer::new();
        let (result, non_empty) = analyzer.walk(&page, 0, 0);
        assert!(non_empty, "PREFX has no native emitter but is now a compilable fallback region");
        assert!(result[0].visited && result[0].is_fallback, "PREFX entry must be a fallback head");
    }

    fn make_i(op: u32, rs: u32, rt: u32, imm: u16) -> u32 {
        (op << 26) | ((rs & 0x1F) << 21) | ((rt & 0x1F) << 16) | (imm as u32)
    }

    fn alu_imm_case(op: u32, rs_val: u64, imm: u16) {
        let mut gpr = [0u64; 32];
        gpr[1] = rs_val;
        let instr = make_i(op, 1, 2, imm);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn addiu_matches_interpreter() {
        alu_imm_case(crate::mips_isa::OP_ADDIU, 10, 20);
        alu_imm_case(crate::mips_isa::OP_ADDIU, 0xFFFF_FFFF, 1); // 32-bit wraparound
        alu_imm_case(crate::mips_isa::OP_ADDIU, 5, 0x8000); // negative immediate
    }

    #[test]
    fn addi_no_overflow_matches_interpreter() {
        alu_imm_case(crate::mips_isa::OP_ADDI, 10, 20);
    }

    #[test]
    fn addi_overflow_traps_and_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0x7FFF_FFFF;
        let instr = make_i(crate::mips_isa::OP_ADDI, 1, 2, 1);
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let interp = run_interpreter(instr, gpr, pc, &[]);
        let jit = run_jit(instr, gpr, pc, (pc as u16 / 4) & 0x3FF, &[])
            .expect("ADDI must be compilable for this test to be meaningful");
        assert_eq!(jit, interp);
        assert_ne!(jit.pc, pc, "overflow must vector via handle_exception");
    }

    #[test]
    fn daddi_no_overflow_matches_interpreter() {
        alu_imm_case(crate::mips_isa::OP_DADDI, 10, 20);
        // 32-bit boundary is NOT an overflow for DADDI (unlike ADDI) —
        // confirms the emitter is genuinely 64-bit-wide, not just ADDI with
        // a different opcode byte.
        alu_imm_case(crate::mips_isa::OP_DADDI, 0x7FFF_FFFF, 1);
    }

    #[test]
    fn daddi_overflow_traps_and_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0x7FFF_FFFF_FFFF_FFFF; // i64::MAX
        let instr = make_i(crate::mips_isa::OP_DADDI, 1, 2, 1);
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let interp = run_interpreter(instr, gpr, pc, &[]);
        let jit = run_jit(instr, gpr, pc, (pc as u16 / 4) & 0x3FF, &[])
            .expect("DADDI must be compilable for this test to be meaningful");
        assert_eq!(jit, interp);
        assert_ne!(jit.pc, pc, "64-bit overflow must vector via handle_exception");
    }

    #[test]
    fn daddiu_matches_interpreter() {
        alu_imm_case(crate::mips_isa::OP_DADDIU, 10, 20);
        // 64-bit wraparound is NOT a trap for DADDIU (unlike DADDI) — confirms
        // the emitter is genuinely wrapping, not DADDI with a different opcode byte.
        alu_imm_case(crate::mips_isa::OP_DADDIU, 0xFFFF_FFFF_FFFF_FFFF, 1);
        alu_imm_case(crate::mips_isa::OP_DADDIU, 5, 0x8000); // negative immediate
    }

    #[test]
    fn andi_ori_xori_match_interpreter() {
        for op in [crate::mips_isa::OP_ANDI, crate::mips_isa::OP_ORI, crate::mips_isa::OP_XORI] {
            alu_imm_case(op, 0xFFFF_FFFF_FFFF_0000, 0xFFFF); // imm is zero-extended, not sign-extended
            alu_imm_case(op, 0, 0);
        }
    }

    #[test]
    fn slti_sltiu_match_interpreter() {
        alu_imm_case(crate::mips_isa::OP_SLTI, u64::MAX, 1); // -1 <s 1 -> true
        alu_imm_case(crate::mips_isa::OP_SLTIU, u64::MAX, 1); // huge <u 1 -> false (imm sign-extends to huge too, but check boundary)
        alu_imm_case(crate::mips_isa::OP_SLTI, 1, 1);
        alu_imm_case(crate::mips_isa::OP_SLTIU, 0, 1);
    }

    #[test]
    fn lui_matches_interpreter() {
        let mut gpr = [0u64; 32];
        let instr = make_i(crate::mips_isa::OP_LUI, 0, 2, 0x8000); // high bit set -> must sign-extend
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
        let instr2 = make_i(crate::mips_isa::OP_LUI, 0, 2, 0x1234);
        gpr[2] = 0;
        assert_jit_matches_interpreter(instr2, gpr, 0xFFFF_FFFF_8000_1000);
    }

    /// `try_emit_fused_lui` (codegen.rs): `lui r2,0x1234; ori r2,r2,0x5678`
    /// must produce the same combined r2 as running the two instructions
    /// unfused, and must still fall through correctly to whatever follows
    /// the ORI (word 2 here, a marker ADDIU) — proving the fused LUI's edge
    /// targets word+2, not word+1 (which was folded in, not independently
    /// re-executed).
    #[test]
    fn lui_ori_fuses_and_matches_interpreter() {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let page = vec![
            (0u16, make_i(crate::mips_isa::OP_LUI, 0, 2, 0x1234)),
            (1, make_i(crate::mips_isa::OP_ORI, 2, 2, 0x5678)),
            (2, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)), // marker: proves word 2 still ran
        ];
        assert_jit_matches_interpreter_page(&page, [0u64; 32], pc, 0, 3, 3);
        let jit = run_jit_page(&page, [0u64; 32], pc, 0, 3, &[]).expect("region must compile");
        assert_eq!(jit.gpr[2], 0x1234_5678, "fused LUI+ORI must combine hi/lo exactly");
        assert_eq!(jit.gpr[5], 1, "word 2 (past the fused pair) must still run");
    }

    /// Same as `lui_ori_fuses_and_matches_interpreter`, but ADDIU — whose
    /// sign-extending add can carry into bit 16 when the low half's sign bit
    /// is set (unlike ORI's pure OR), so this specifically exercises that
    /// carry path (`lo16 = 0xFFFF` = -1) matching `exec_addiu`'s
    /// wrapping-add semantics exactly.
    #[test]
    fn lui_addiu_fuses_with_carry_and_matches_interpreter() {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let page = vec![
            (0u16, make_i(crate::mips_isa::OP_LUI, 0, 2, 0x1234)),
            (1, make_i(crate::mips_isa::OP_ADDIU, 2, 2, 0xFFFF)), // lo16 = -1, carries out of bit 16
            (2, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)),
        ];
        assert_jit_matches_interpreter_page(&page, [0u64; 32], pc, 0, 3, 3);
        let jit = run_jit_page(&page, [0u64; 32], pc, 0, 3, &[]).expect("region must compile");
        // 0x1234_0000 + (-1) = 0x1233_FFFF (borrows out of bit 16, matching a
        // real wrapping add rather than a plain OR of the two halves).
        assert_eq!(jit.gpr[2], 0x1233_FFFF, "fused LUI+ADDIU must carry, not just OR, the halves");
    }

    /// Correctness guard for the `is_branch_target` exclusion
    /// (`try_emit_fused_lui`'s doc comment): when something in the region
    /// jumps directly to the word right after an LUI, that word must NOT be
    /// silently folded away — the branch's own arrival needs the real,
    /// independently-addressable ORI to run.
    ///
    /// Modeled on `backward_branch_loop_body_matches_interpreter`'s
    /// loop-with-sentinel shape. Layout: word 0 `lui r2,0x1234` (fusion
    /// candidate — entry, runs once), word 1 `ori r2,r2,0x5678` (its
    /// would-be fusion partner, ALSO the loop body / branch target: word 2
    /// increments r4 on every pass so the test can tell how many times word
    /// 1 actually ran), word 2 `addiu r4,r4,1`, words 3-4 nop padding
    /// (mirroring the reference test's own spacing), word 5 BNE r1,r0,imm
    /// looping back to word 1 while r1 (seeded directly via `gpr`,
    /// independent of r2/the LUI) hasn't hit zero, word 6 delay slot
    /// decrementing r1, word 7 boundary sentinel.
    ///
    /// If fusion incorrectly triggered on word 0 (ignoring
    /// `is_branch_target`), word 1 would never compile as its own
    /// independently-reachable block the way the loop back-edge needs —
    /// except `block_for_word` always allocates it regardless (pass 1 is
    /// unconditional); what fusion getting this wrong would actually corrupt
    /// is r2 (case where the *entry* arrival's LUI folds word 1's ORI into
    /// itself AND word 1 still runs again on every backward-branch pass,
    /// double-applying the ORI on the first iteration) — caught by comparing
    /// against the interpreter's fully-unfused r2 exactly.
    #[test]
    fn lui_not_fused_when_next_word_is_a_branch_target() {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let target_word: i32 = 1; // loop back to word 1 (the ORI), not word 0 (the LUI)
        let imm16 = (target_word - (5 + 1)) as i16 as u16;
        let mut gpr = [0u64; 32];
        gpr[1] = 2; // loop counter, independent of r2
        let page = vec![
            (0u16, make_i(crate::mips_isa::OP_LUI, 0, 2, 0x1234)),
            (1, make_i(crate::mips_isa::OP_ORI, 2, 2, 0x5678)),
            (2, make_i(crate::mips_isa::OP_ADDIU, 4, 4, 1)), // counts word-1 passes
            (3, 0),
            (4, 0),
            (5, make_i(crate::mips_isa::OP_BNE, 1, 0, imm16)), // while r1 != 0
            (6, make_i(crate::mips_isa::OP_ADDIU, 1, 1, 0xFFFF)), // delay slot: r1 -= 1
            (7, crate::mips_isa::JIT_REGION_BOUNDARY_SENTINEL),
        ];
        // word0 (1) + 3 passes of word1..word6 (6 each: r1 starts at 2, BNE
        // tests-then-slot-decrements each pass — taken at r1=2, taken at
        // r1=1, not-taken at r1=0 but the slot still dispatches regardless,
        // matching branch_delay_slot_always_executes_regardless_of_taken) =
        // 1 + 6*3 = 19.
        let steps = 1 + 6 * 3;
        assert_jit_matches_interpreter_page(&page, gpr, pc, 0, steps, /*max_instrs=*/16);
        let jit = run_jit_page(&page, gpr, pc, 0, 16, &[]).expect("region must compile");
        assert_eq!(jit.gpr[2], 0x1234_5678, "r2 must match the fully-unfused LUI+ORI result even though word 0 is also a branch-target predecessor's LUI");
        assert_eq!(jit.gpr[4], 3, "word 1 (the ORI) must run exactly once per loop pass, not be silently skipped or double-applied");
    }

    /// Correctness guard for `try_emit_fused_lui`'s foreign-delay-slot
    /// exclusion — mirrors the interpreter's own `exec_lui_imm32`/
    /// `exec_lui_simm32` guard (`if self.core.in_delay_slot { ...don't
    /// fuse... }`, mips_exec.rs) and `standalone_compile_of_a_foreign_delay_slot_honors_pending_transfer`'s
    /// model above, but with the standalone entry word itself being a LUI
    /// that would otherwise fuse with its own word+1.
    ///
    /// `exec_decoded`'s dispatch gate can probe *any* word for standalone
    /// compilation — including one that, at this particular arrival, is
    /// actually running as some other, outside-the-region branch's delay
    /// slot (`core.in_delay_slot`/`core.delay_slot_target` pre-armed by that
    /// branch, not visible to this region's own analysis). If the entry
    /// word happens to look like `lui r2,hi` and word+1 happens to look like
    /// a same-register `ori r2,r2,lo` — coincidentally, since they belong to
    /// unrelated control-flow paths — fusion must not fire: word+1 has
    /// nothing to do with this LUI, and the real next PC is
    /// `delay_slot_target`, decided only at runtime, not word+2.
    ///
    /// Before the entry/branch-fallback-successor guard was added to
    /// `try_emit_fused_lui`, this would have unconditionally run the
    /// unrelated word+1 as if it were the ORI half of the pair (corrupting
    /// r2) and jumped to word+2 instead of honoring the pending transfer.
    #[test]
    fn lui_not_fused_when_entry_word_is_a_foreign_delay_slot() {
        let pc_word0 = 0xFFFF_FFFF_BFC0_0000u64;
        let target = 0xFFFF_FFFF_BFC0_03C0u64; // arbitrary foreign delay-slot target
        let page = vec![
            (0u16, make_i(crate::mips_isa::OP_LUI, 0, 2, 0x1234)),
            (1, make_i(crate::mips_isa::OP_ORI, 2, 2, 0x5678)), // unrelated at runtime; must NOT be folded in
        ];

        let page_base = 0x1FC0_0000u32;
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, r) in &page { page_words[word as usize] = r; }
        let mut analyzer = Analyzer::new();
        // budget=2 so word 1 (the would-be ORI fusion partner) is actually
        // visited/compiled as part of this region — with budget=1 it stays
        // `!visited` and `try_emit_fused_lui`'s existing "next not visited"
        // check alone would already suppress fusion, defeating the point of
        // this test.
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, 2);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
            .expect("standalone word-0 region must compile");

        let (mut jit_exec, jit_mem) = seeded_executor([0u64; 32], pc_word0);
        for &(word, r) in &page { jit_mem.set_word(pc_word0 + (word as u64) * 4, r); }
        jit_exec.core.in_delay_slot = true;
        jit_exec.core.delay_slot_target = target;

        let status = unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        assert_eq!(status, crate::mips_exec::EXEC_COMPLETE);

        assert_eq!(jit_exec.core.pc, target,
            "a standalone-compiled LUI that's actually running as a foreign delay slot must exit via delay_slot_target (0x{:x}), not a fused fallthrough to word+2 (got 0x{:x})",
            target, jit_exec.core.pc);
        assert!(!jit_exec.core.in_delay_slot, "in_delay_slot must be cleared once the pending transfer is honored");
        assert_eq!(jit_exec.core.gpr[2], 0x1234_0000u64 as i32 as i64 as u64,
            "unfused-LUI semantics only (rt = hi16<<16, sign-extended) — the unrelated word+1 ORI must never have been folded in");
    }

    #[test]
    fn lw_matches_interpreter_basic() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0000; // base
        let instr = make_i(crate::mips_isa::OP_LW, 1, 2, 8); // LW r2, 8(r1)
        assert_jit_matches_interpreter_mem(instr, gpr, 0xFFFF_FFFF_8000_1000,
            &[(0xFFFF_FFFF_8010_0008, 0x1234_5678)]);
    }

    #[test]
    fn lw_matches_interpreter_sign_extends_negative_word() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0000;
        let instr = make_i(crate::mips_isa::OP_LW, 1, 2, 0);
        assert_jit_matches_interpreter_mem(instr, gpr, 0xFFFF_FFFF_8000_1000,
            &[(0xFFFF_FFFF_8010_0000, 0x8000_0001)]); // high bit set -> sign-extends to 0xFFFFFFFF_80000001
    }

    #[test]
    fn sw_matches_interpreter_basic() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0000;
        gpr[2] = 0xDEAD_BEEF;
        let instr = make_i(crate::mips_isa::OP_SW, 1, 2, 4); // SW r2, 4(r1)
        assert_jit_matches_interpreter_mem(instr, gpr, 0xFFFF_FFFF_8000_1000, &[]);
    }

    #[test]
    fn lw_unaligned_address_raises_adel_and_matches_interpreter() {
        // Address error (ADEL): unaligned LW must deliver the exception via
        // handle_exception (§4.2 single-implementation delivery) — EPC/Cause/
        // Status/pc-at-vector must all match the interpreter exactly.
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0001; // misaligned by 1 byte
        let instr = make_i(crate::mips_isa::OP_LW, 1, 2, 0);
        let pc = 0xFFFF_FFFF_8000_1000u64;

        let interp = run_interpreter(instr, gpr, pc, &[]);
        let jit = run_jit(instr, gpr, pc, (pc as u16 / 4) & 0x3FF, &[])
            .expect("LW must be compilable for this test to be meaningful");
        assert_eq!(jit, interp, "JIT and interpreter diverged on ADEL delivery");
        // Sanity: pc must have actually moved to the exception vector, not
        // stayed at the faulting instruction — confirms handle_exception_fn
        // really ran rather than the test accidentally passing on a no-op.
        assert_ne!(jit.pc, pc, "pc must be vectored by handle_exception, not left at the faulting instruction");
    }

    // ---- Exception delivery: EPC/Cause/BD correctness with runway ------
    //
    // `lw_unaligned_address_raises_adel_and_matches_interpreter` above only
    // ever compiles a *single*-instruction region, so its own entry_word is
    // trivially the faulting word — that harness cannot tell "EPC landed on
    // the right instruction" from "EPC landed on whatever core.pc happened
    // to hold", since there's nothing else in the compiled unit for pc to
    // have drifted from. A real compiled unit's non-faulting head
    // instructions never write core.pc per-instruction (only exit points
    // do, to keep a straight-line run cheap) — before this was fixed
    // (emit_exception_exit synthesizing core.pc from ctx.word), a fault on
    // any instruction *after* the first in a compiled region reported EPC
    // from wherever core.pc was last left, not the real faulting
    // instruction (found live: an ADEL inside a compiled unit reported EPC
    // pointing at an unrelated, later PROM routine). These tests give every
    // case at least one instruction of runway ahead of the fault so a wrong,
    // stale core.pc has something real to diverge from.

    /// word 0: `ADDIU r5, r0, 1` (runway — retires cleanly, marks that it
    /// ran via gpr[5], and is exactly the kind of instruction whose body
    /// never used to touch core.pc). word 1: the faulting instruction.
    /// `entry_word=0, max_instrs=2` compiles both into one region;
    /// `steps=2` matches the interpreter dispatching both in turn.
    fn assert_head_fault_matches_interpreter(fault_instr: u32, gpr: [u64; 32]) {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let page = vec![
            (0, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)),
            (1, fault_instr),
        ];
        let interp = run_interpreter_page(&page, gpr, pc, 2);
        let jit = run_jit_page(&page, gpr, pc, 0, 2, &[])
            .expect("region must be compilable for this test to be meaningful");
        assert_eq!(jit, interp, "JIT and interpreter diverged on exception delivery after runway");
        assert_ne!(jit.pc, pc, "pc must be vectored by handle_exception, not left at the faulting instruction");
        // The real bug this guards: EPC must be the faulting instruction's
        // own address (word 1), not word 0's (stale core.pc left over from
        // a compiled unit that never updates it per-instruction) or the
        // region's page base.
        assert_eq!(jit.cp0_epc, pc + 4, "EPC must point at the faulting instruction (word 1), not stale/wrong core.pc");
        assert_eq!(jit.cp0_cause & crate::mips_core::CAUSE_BD, 0, "a head-instruction fault must not report BD set");
    }

    #[test]
    fn adel_after_runway_epc_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0001; // misaligned by 1 byte
        let instr = make_i(crate::mips_isa::OP_LW, 1, 2, 0);
        assert_head_fault_matches_interpreter(instr, gpr);
    }

    #[test]
    fn ades_after_runway_epc_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0001; // misaligned by 1 byte
        gpr[2] = 0xDEAD_BEEF;
        let instr = make_i(crate::mips_isa::OP_SW, 1, 2, 0);
        assert_head_fault_matches_interpreter(instr, gpr);
    }

    #[test]
    fn overflow_trap_after_runway_epc_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0x7FFF_FFFF; // ADD r3, r1, r2 with r2=1 overflows i32
        gpr[2] = 1;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_ADD);
        assert_head_fault_matches_interpreter(instr, gpr);
    }

    /// word 0: an always-taken branch (BEQ r0,r0 — no runway needed ahead of
    /// it, the branch itself never faults). word 1: its mandatory delay
    /// slot, which is the faulting instruction — `emit_slot_semantics`
    /// already wrote core.pc to the slot's own address before this session's
    /// fix (that's why delay-slot faults were never the observed bug), but
    /// covering it here locks that in as a real equivalence test rather
    /// than an inference from reading the code. EPC must be slot_addr - 4
    /// (`deliver_exception`'s in_delay_slot branch) with CAUSE_BD set,
    /// matching the interpreter exactly.
    fn assert_slot_fault_matches_interpreter(fault_instr: u32, gpr: [u64; 32]) {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let page = vec![
            (0, make_i(crate::mips_isa::OP_BEQ, 0, 0, 0)), // BEQ r0, r0, +0 -> always taken, target = slot's own addr + 0
            (1, fault_instr),
        ];
        let interp = run_interpreter_page(&page, gpr, pc, 2);
        let jit = run_jit_page(&page, gpr, pc, 0, 1, &[])
            .expect("region must be compilable for this test to be meaningful");
        assert_eq!(jit, interp, "JIT and interpreter diverged on delay-slot exception delivery");
        assert_ne!(jit.pc, pc, "pc must be vectored by handle_exception, not left at the branch");
        let slot_addr = pc + 4;
        assert_eq!(jit.cp0_epc, slot_addr - 4, "EPC must be slot_addr - 4 per deliver_exception's in_delay_slot branch");
        assert_ne!(jit.cp0_cause & crate::mips_core::CAUSE_BD, 0, "a delay-slot fault must report BD set");
    }

    #[test]
    fn adel_in_delay_slot_epc_and_bd_match_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0001; // misaligned by 1 byte
        let instr = make_i(crate::mips_isa::OP_LW, 1, 2, 0);
        assert_slot_fault_matches_interpreter(instr, gpr);
    }

    #[test]
    fn ades_in_delay_slot_epc_and_bd_match_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0001; // misaligned by 1 byte
        gpr[2] = 0xDEAD_BEEF;
        let instr = make_i(crate::mips_isa::OP_SW, 1, 2, 0);
        assert_slot_fault_matches_interpreter(instr, gpr);
    }

    // overflow's own delay-slot EPC/BD case is already covered by
    // `overflow_in_delay_slot_traps_with_epc_and_bd_matching_interpreter`
    // below (predates this section) — not duplicated here.

    /// Regression test for a distinct entry-word ambiguity, found by
    /// inspection while chasing the runway-fault bug above (not yet observed
    /// live): a physical word can be compiled once as an ordinary entry
    /// (`entry_word == this word`, reached e.g. as a plain sequential
    /// arrival or a jump target — never through `emit_slot_semantics`, which
    /// only ever runs for a word inlined as *another* instruction's slot).
    /// The exact same compiled function can *also*, on a separate later
    /// dispatch, be entered because the interpreter's own dispatch loop
    /// landed on this word as some *other* branch's delay slot
    /// (`core.in_delay_slot` and `core.pc` already correct, set by the
    /// interpreter's `branch_delay`/`handle_exec_complete` before
    /// `exec_decoded`'s JIT gate is even consulted — see `MipsCore`'s own
    /// dispatch, unrelated to `emit_slot_semantics`'s explicit slot
    /// handling). If this compiled entry then faults, `emit_exception_exit`
    /// must notice the *already-armed* `core.in_delay_slot` and leave
    /// `core.pc` alone (EPC = branch_addr, BD set) rather than
    /// unconditionally overwriting it with its own word's address (EPC =
    /// this word's own address, BD clear) — the runway-fault fix by itself
    /// would get this specific case wrong were it not for the additional
    /// `ctx.word == ctx.entry_word` check in `emit_exception_exit`.
    ///
    /// This compiles the faulting instruction as a **standalone one-word
    /// region** (`run_jit`, exactly as if first reached as a plain
    /// sequential/jump-target arrival — the ordinary, non-slot compile
    /// path), then invokes it with `core.in_delay_slot`/`core.pc`
    /// pre-armed exactly as `branch_delay` would leave them, simulating
    /// arrival via a real branch located elsewhere (never compiled itself,
    /// deliberately — this isolates the entry-word ambiguity from
    /// `emit_slot_semantics`'s own, already-covered, in-region case).
    fn assert_entry_word_reached_as_foreign_slot_matches_interpreter(fault_instr: u32, gpr: [u64; 32]) {
        let branch_pc = 0xFFFF_FFFF_8000_0000u64; // BEQ r0,r0,+0 (not compiled — real interpreter dispatch)
        let slot_pc = branch_pc + 4; // fault_instr's own address; also entry_word's address below
        let page = vec![
            (0u16, make_i(crate::mips_isa::OP_BEQ, 0, 0, 0)),
            (1u16, fault_instr),
        ];

        // Ground truth: real interpreter dispatch, branch then its slot —
        // in_delay_slot/EPC/BD end up exactly where the MIPS spec puts them.
        let interp = run_interpreter_page(&page, gpr, branch_pc, 2);

        // JIT: compile ONLY word 1, standalone, as an ordinary entry
        // (entry_word == 1, never touched by emit_slot_semantics) — then
        // invoke it with in_delay_slot/pc pre-armed by hand, exactly as
        // branch_delay would leave them after dispatching the BEQ for real.
        let word_offset = (slot_pc as u16 / 4) & 0x3FF;
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        page_words[word_offset as usize] = fault_instr;
        let page_base = (slot_pc & !(PAGE_SIZE as u64 - 1)) as u32;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, word_offset, page_base, 1);
        assert!(non_empty, "entry instruction must not be excluded");
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word_offset, true, false)
            .expect("region must be compilable for this test to be meaningful");

        let (exec, _mem) = seeded_executor(gpr, slot_pc);
        let mut exec = Box::new(exec);
        exec.core.in_delay_slot = true; // pre-armed, as if branch_delay just ran for the BEQ above
        exec.install_jit_hooks();
        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        std::mem::forget(codegen);
        let jit = CoreSnapshot::capture(&exec.core);

        assert_eq!(jit, interp, "JIT and interpreter diverged when a standalone-compiled entry word was reached as a foreign delay slot");
        assert_ne!(jit.pc, slot_pc, "pc must be vectored by handle_exception");
        assert_eq!(jit.cp0_epc, branch_pc, "EPC must be the real branch's address (core.pc - 4), not this word's own address");
        assert_ne!(jit.cp0_cause & crate::mips_core::CAUSE_BD, 0, "BD must be set — this word was reached as a delay slot, even though it was compiled as a plain entry");
    }

    #[test]
    fn adel_in_entry_word_reached_as_foreign_slot_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0001; // misaligned by 1 byte
        let instr = make_i(crate::mips_isa::OP_LW, 1, 2, 0);
        assert_entry_word_reached_as_foreign_slot_matches_interpreter(instr, gpr);
    }

    #[test]
    fn ades_in_entry_word_reached_as_foreign_slot_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0001; // misaligned by 1 byte
        gpr[2] = 0xDEAD_BEEF;
        let instr = make_i(crate::mips_isa::OP_SW, 1, 2, 0);
        assert_entry_word_reached_as_foreign_slot_matches_interpreter(instr, gpr);
    }

    #[test]
    fn overflow_trap_in_entry_word_reached_as_foreign_slot_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0x7FFF_FFFF;
        gpr[2] = 1;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_ADD);
        assert_entry_word_reached_as_foreign_slot_matches_interpreter(instr, gpr);
    }

    /// Sanity counterpart: the SAME standalone-compiled entry word, invoked
    /// the ordinary way (`in_delay_slot` false, ordinary arrival) must still
    /// get the runway-fault fix's plain synthesis — confirms the new
    /// `ctx.word == ctx.entry_word` branch in `emit_exception_exit` didn't
    /// regress the already-fixed, more common case by always taking the
    /// "leave core.pc alone" arm for entry words.
    #[test]
    fn adel_in_entry_word_ordinary_arrival_still_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0001;
        let instr = make_i(crate::mips_isa::OP_LW, 1, 2, 0);
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let interp = run_interpreter(instr, gpr, pc, &[]);
        let jit = run_jit(instr, gpr, pc, (pc as u16 / 4) & 0x3FF, &[])
            .expect("LW must be compilable for this test to be meaningful");
        assert_eq!(jit, interp);
        assert_eq!(jit.cp0_epc, pc, "ordinary (non-delay-slot) entry-word fault must still report EPC = this instruction's own address");
        assert_eq!(jit.cp0_cause & crate::mips_core::CAUSE_BD, 0, "ordinary entry-word fault must not report BD set");
    }

    /// Regression test for a live `jitcheck` divergence: a compiled unit's
    /// `lw` reading from a genuinely-erroring bus address (real hardware
    /// equivalent: `physical.rs`'s `ErrorBus`, an always-BUS_ERR device for
    /// unmapped GIO space — PROM's own hardware-presence probes deliberately
    /// touch it) must fault identically whether reached via the interpreter
    /// or real JIT dispatch. Live symptom: JIT vectored to the exception
    /// handler (`cp0_status.EXL` set, `cp0_epc` = the faulting `lw`'s own
    /// address) while the interpreter, given the exact same `gpr` (including
    /// the base register — confirmed unchanged, ruling out an upstream
    /// register-value divergence) at the exact same instruction count,
    /// did NOT fault at all. Since address translation for this range needs
    /// no TLB (kseg1-equivalent, fixed mapping) and both engines share the
    /// one `read_data_impl`/`BusDevice::read32` path, a real, always-on
    /// bus error should be completely deterministic — reached via either
    /// engine, it must fault every time, not just sometimes.
    ///
    /// Runway (word 0: harmless ADDIU) is required for the same reason as
    /// `assert_head_fault_matches_interpreter` — a single-instruction
    /// compiled region can't distinguish "faulted via the real path" from
    /// "some other divergence already happened before this ran".
    #[test]
    fn lw_from_always_erroring_address_matches_interpreter() {
        let error_addr: u64 = 0xFFFF_FFFF_9F0F_0100; // kseg1-equivalent, arbitrary unmapped-GIO-style address
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let mut gpr = [0u64; 32];
        gpr[16] = error_addr - 256; // s0; LW t4, 256(s0) below reaches error_addr
        let page: Vec<(u16, u32)> = vec![
            (0, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)), // runway: r5 = 1
            (1, make_i(crate::mips_isa::OP_LW, 16, 12, 256)), // LW t4(r12), 256(s0/r16) -> error_addr
        ];
        let page_base = pc & !(PAGE_SIZE as u64 - 1);
        let responses = vec![(BUS_ERR, 0)]; // always errors — single entry repeats forever

        // Interpreter pass.
        let interp = {
            let (mut exec, mem) = seeded_executor_over(
                MockMemory::new_not_compilable().with_magic_responses(error_addr, responses.clone()),
                gpr, pc,
            );
            for &(word, raw) in &page {
                mem.set_word(page_base + (word as u64) * 4, raw);
            }
            for _ in 0..2 {
                let fetch_pc = exec.core.pc;
                let instr = mem.get_word(fetch_pc & !3);
                exec.exec(instr);
            }
            CoreSnapshot::capture(&exec.core)
        };

        // JIT pass: compile both words as one region, run it once.
        let jit = {
            let mut page_words = [0u32; ENTRIES_PER_PAGE];
            for &(word, raw) in &page {
                page_words[word as usize] = raw;
            }
            let mut analyzer = Analyzer::new();
            let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base as u32, 2);
            assert!(non_empty, "entry instruction must not be excluded");
            let mut instrs_owned = *walked;
            let mut codegen = Codegen::new();
            let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
                .expect("region must be compilable for this test to be meaningful");

            let (exec, mem) = seeded_executor_over(
                MockMemory::new().with_magic_responses(error_addr, responses),
                gpr, pc,
            );
            let mut exec = Box::new(exec);
            for &(word, raw) in &page {
                mem.set_word(page_base + (word as u64) * 4, raw);
            }
            exec.install_jit_hooks();
            unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
            std::mem::forget(codegen);
            CoreSnapshot::capture(&exec.core)
        };

        assert_eq!(jit.gpr[16], interp.gpr[16], "s0 (the LW's own base register) must be unchanged and identical going into the LW — if this fails, the divergence is upstream of the LW, not in exception delivery");
        assert_eq!(jit, interp, "JIT and interpreter diverged on a genuinely-erroring uncached read: {:x?} vs {:x?}", jit, interp);
        // EXL, not a bare pc != pc check: this region has a runway
        // instruction ahead of the LW, so pc legitimately advances even
        // without any fault — only EXL can only be set by a real exception
        // actually being delivered.
        assert_ne!(jit.cp0_status & crate::mips_core::STATUS_EXL, 0, "a real bus error on this address must vector via handle_exception in both engines");
    }

    /// Regression test for a second live `jitcheck` divergence, found via the
    /// new `step_status` digest field: a transiently-busy bus read
    /// (`BUS_BUSY`/`EXEC_RETRY` — not `EXEC_IS_EXCEPTION`, see that
    /// constant's doc comment) was being routed by `emit_check_mem_exc` into
    /// `emit_exception_exit` exactly like a real exception, vectoring the
    /// JIT pass to the exception handler for something the interpreter just
    /// silently retried and moved past. `step_status` showed it directly:
    /// `interp=EXEC_RETRY(0x100) jit=EXEC_COMPLETE(0)` at the same
    /// instruction count, with the JIT side's `pc`/`cp0_status.EXL` already
    /// at the vector — an exception the interpreter never took.
    ///
    /// Both engines see the identical bus device, busy for the same fixed
    /// number of attempts (`with_busy_addr`), so both should retry the exact
    /// same number of times and land on bit-identical final state once the
    /// device stops being busy — retried "for free" here since neither
    /// engine's own status/retry loop is itself timing-dependent (a fixed
    /// countdown, not the wall-clock-driven busy devices `HW_READ_FIXUP_ADDRS`
    /// exists for — see that const's doc comment for the distinction: this
    /// tests that any given retry status is *handled* identically, not that
    /// retry *counts* stay in sync against real hardware, which is a
    /// separate, deliberately out-of-scope concern for `jitcheck` itself).
    #[test]
    fn lw_from_transiently_busy_address_matches_interpreter() {
        let addr: u64 = 0xFFFF_FFFF_9F0F_0200;
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let mut gpr = [0u64; 32];
        gpr[16] = addr - 64; // s0; LW t4, 64(s0) below reaches addr
        let instr = make_i(crate::mips_isa::OP_LW, 16, 12, 64); // LW t4(r12), 64(s0/r16)
        const MAX_ATTEMPTS: usize = 10; // generous bound; a real bail-and-redispatch loop, not expected to ever approach this
        // Busy for the first two attempts, then succeeds with a real value —
        // the last entry repeats forever once popped, so both engines can
        // loop past it exactly as many times as they individually need to.
        let responses = vec![(BUS_BUSY, 0), (BUS_BUSY, 0), (BUS_OK, 0x1234_5678)];

        // Interpreter pass: call exec() repeatedly, exactly like the real
        // outer dispatch loop would, until the status is no longer a retry.
        let interp = {
            let (mut exec, _mem) = seeded_executor_over(
                MockMemory::new_not_compilable().with_magic_responses(addr, responses.clone()),
                gpr, pc,
            );
            let mut attempts = 0;
            loop {
                let status = exec.exec(instr);
                attempts += 1;
                if status != crate::mips_exec::EXEC_RETRY { break; }
                assert!(attempts < MAX_ATTEMPTS, "interpreter retried more than expected — magic-response queue is broken");
            }
            CoreSnapshot::capture(&exec.core)
        };

        // JIT pass: compile the single-instruction region once, then call
        // the SAME jit_fn repeatedly — exactly what exec_decoded's real gate
        // does, since a retry-bail leaves core.pc unchanged (or, per the
        // entry-word fix, correctly reflecting an already-armed delay slot)
        // and the interpreter's own dispatch loop would simply re-enter and
        // re-dispatch this exact word again next step(), landing right back
        // in this same compiled entry.
        let jit = {
            let word_offset = (pc as u16 / 4) & 0x3FF;
            let mut page_words = [0u32; ENTRIES_PER_PAGE];
            page_words[word_offset as usize] = instr;
            let page_base = (pc & !(PAGE_SIZE as u64 - 1)) as u32;
            let mut analyzer = Analyzer::new();
            let (walked, non_empty) = analyzer.walk_bounded(&page_words, word_offset, page_base, 1);
            assert!(non_empty, "entry instruction must not be excluded");
            let mut instrs_owned = *walked;
            let mut codegen = Codegen::new();
            let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word_offset, true, false)
                .expect("region must be compilable for this test to be meaningful");

            let (exec, _mem) = seeded_executor_over(
                MockMemory::new().with_magic_responses(addr, responses),
                gpr, pc,
            );
            let mut exec = Box::new(exec);
            exec.install_jit_hooks();
            // Unlike the interpreter's exec()/step(), a retry-bail's own
            // return value is always EXEC_COMPLETE (emit_exit_block_body's
            // shared exit stub hardcodes it — see emit_check_mem_exc's doc
            // comment: this matches the real exec_decoded dispatch loop,
            // which never inspects the specific status either, just
            // re-enters and gets a fresh result next call). So the retry
            // condition here has to be "did this word actually retire" —
            // core.pc still at the entry word's own address — not the
            // return value.
            let mut attempts = 0;
            loop {
                unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
                attempts += 1;
                if exec.core.pc != pc { break; }
                assert!(attempts < MAX_ATTEMPTS, "JIT retried more than expected — either the magic-response queue or the retry-bail fix is broken");
            }
            std::mem::forget(codegen);
            CoreSnapshot::capture(&exec.core)
        };

        assert_eq!(jit, interp, "JIT and interpreter diverged on a transiently-busy uncached read: {:x?} vs {:x?}", jit, interp);
        assert_eq!(jit.pc, pc + 4, "the read must eventually succeed and retire normally, not vector as an exception");
        assert_eq!(jit.cp0_status & crate::mips_core::STATUS_EXL, 0, "a retry must never leave EXL set — no exception was ever real");
        assert_eq!(jit.gpr[12], 0x1234_5678, "t4 must hold the value from the read that finally succeeded");
    }

    // ---- Branches and jumps: batch A (plain conditional) ------
    //
    // Layout convention for these tests: word 0 = the branch, word 1 = its
    // delay slot (an ADDIU whose effect is independently observable in
    // gpr[5], so a test can tell whether the slot actually ran — it always
    // should, taken or not). The branch's imm16 is chosen so the taken
    // target lands outside the tiny compiled region (word 0's max_instrs is
    // small), forcing both arms through an exit bail — this isolates the
    // branch mechanism itself (condition + slot inlining + link, where
    // applicable) from chaining into further compiled code, which the
    // Sequential-chain tests already cover independently.

    /// word 0: `op rs, rt, imm=BRANCH_IMM` (target = 0 + 1 + BRANCH_IMM = 7,
    /// i.e. word 7 — outside the 2-instruction compiled region so the taken
    /// arm always exits; target is relative to the delay slot's own address,
    /// word 1, not word 2). word 1: `ADDIU r5, r0, 1` (delay slot, marks
    /// that it ran). Not-taken falls through to word 2, also outside the region.
    const BRANCH_IMM: u16 = 6;

    fn cond_branch_layout(op: u32, rs: u32, rt: u32) -> Vec<(u16, u32)> {
        vec![
            (0, make_i(op, rs, rt, BRANCH_IMM)),
            (1, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)), // r5 = 1, marks the slot ran
        ]
    }

    fn regimm_branch_layout(rt_field: u32, rs: u32) -> Vec<(u16, u32)> {
        vec![
            (0, make_i(crate::mips_isa::OP_REGIMM, rs, rt_field, BRANCH_IMM)),
            (1, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)),
        ]
    }

    /// `steps=2`: the interpreter dispatches the branch itself (arms
    /// `core.in_delay_slot`, doesn't retire) then the delay slot (which is
    /// what actually lands PC at the target, per `branch_delay`/
    /// `handle_exec_complete`). `max_instrs=1`: one *head* instruction (the
    /// branch) — its mandatory slot is never charged against this budget
    /// (`analyzer::visit_slot`), so both the taken and not-taken arms still
    /// exit the compiled region exactly as before slots stopped counting
    /// (see module-level doc comment above).
    fn assert_branch_matches_interpreter(page: Vec<(u16, u32)>, gpr: [u64; 32]) {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        assert_jit_matches_interpreter_page(&page, gpr, pc, 0, 2, 1);
    }

    #[test]
    fn beq_taken_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 42;
        gpr[2] = 42; // equal -> taken
        assert_branch_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BEQ, 1, 2), gpr);
    }

    #[test]
    fn beq_not_taken_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 42;
        gpr[2] = 43; // not equal -> not taken
        assert_branch_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BEQ, 1, 2), gpr);
    }

    #[test]
    fn bne_taken_and_not_taken_match_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 1;
        gpr[2] = 2;
        assert_branch_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BNE, 1, 2), gpr);
        gpr[2] = 1;
        assert_branch_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BNE, 1, 2), gpr);
    }

    #[test]
    fn blez_bgtz_match_interpreter() {
        let mut gpr = [0u64; 32];
        for rs_val in [0u64, 1, 0xFFFF_FFFF_FFFF_FFFF /* -1 */] {
            gpr[1] = rs_val;
            assert_branch_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BLEZ, 1, 0), gpr);
            assert_branch_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BGTZ, 1, 0), gpr);
        }
    }

    #[test]
    fn bltz_bgez_match_interpreter() {
        let mut gpr = [0u64; 32];
        for rs_val in [0u64, 1, 0xFFFF_FFFF_FFFF_FFFF /* -1 */] {
            gpr[1] = rs_val;
            assert_branch_matches_interpreter(regimm_branch_layout(crate::mips_isa::RT_BLTZ, 1), gpr);
            assert_branch_matches_interpreter(regimm_branch_layout(crate::mips_isa::RT_BGEZ, 1), gpr);
        }
    }

    #[test]
    fn branch_delay_slot_always_executes_regardless_of_taken() {
        // The slot's effect (r5 = 1) must be visible whether or not the
        // branch itself is taken — this is what distinguishes "delay slot"
        // from a normal conditional skip.
        let mut gpr = [0u64; 32];
        gpr[1] = 1; gpr[2] = 1; // taken
        let jit = run_jit_page(&cond_branch_layout(crate::mips_isa::OP_BEQ, 1, 2), gpr, 0xFFFF_FFFF_8000_0000, 0, 1, &[])
            .expect("BEQ region must compile");
        assert_eq!(jit.gpr[5], 1, "slot must execute on the taken path");

        gpr[2] = 2; // not taken
        let jit = run_jit_page(&cond_branch_layout(crate::mips_isa::OP_BEQ, 1, 2), gpr, 0xFFFF_FFFF_8000_0000, 0, 1, &[])
            .expect("BEQ region must compile");
        assert_eq!(jit.gpr[5], 1, "slot must execute on the not-taken path too");
    }

    #[test]
    fn backward_branch_loop_body_matches_interpreter() {
        // A loop-shaped region: word 4 = BNE looping back to word 0 while a
        // counter (r1) hasn't hit zero, word 5 = delay slot decrementing r1.
        // Entry at word 0 (a plain ADDIU so there's a real head instruction
        // there distinct from the branch target computation), falls through
        // to the branch at word 4. This exercises the two-pass block wiring
        // needed for a backward edge (word 4's taken target, word 0, is
        // *before* word 4 in offset order) — `compile_region`'s pass 1 must
        // have already created word 0's block before pass 2 wires the
        // back-edge into it.
        // target_word = word + 1 + imm16 (relative to the delay slot's own
        // address, word 5, not word 6); want the branch (at word 4) to land
        // back at word 1 (the loop body), not word 0 (which would re-run
        // the counter's own initialization every iteration).
        let target_word: i32 = 1;
        let imm16 = (target_word - (4 + 1)) as i16 as u16;
        // word 6: JIT_REGION_BOUNDARY_SENTINEL — a hard region end (the
        // analyzer stops the walk at the predecessor's edge, never visiting or
        // running it), so the not-taken fallthrough (word 4+2=6) has a
        // deterministic boundary. An MTC0 was used here historically, but with
        // interpreter-fallback an Excluded instruction no longer ends the
        // region (it's kept as a fallback head), so a dedicated boundary
        // sentinel — zero side effects, no delay slot, never executed — is the
        // right marker now. The interpreter's PC just lands here after the loop
        // exits; it never dispatches this word (step count unchanged from the
        // original 16).
        let page = vec![
            (0, make_i(crate::mips_isa::OP_ADDIU, 0, 1, 3)), // r1 = 3
            (1, make_i(crate::mips_isa::OP_ADDIU, 1, 1, 0xFFFF)), // loop body / branch target: r1 -= 1
            (2, 0),
            (3, 0),
            (4, make_i(crate::mips_isa::OP_BNE, 1, 0, imm16)), // while r1 != 0
            (5, 0), // delay slot: nop
            (6, crate::mips_isa::JIT_REGION_BOUNDARY_SENTINEL), // hard region end
        ];
        let pc = 0xFFFF_FFFF_8000_0000u64;
        // Interpreter dispatch count: word0 ADDIU (1), then per loop
        // iteration: word1 (decrement, the branch's landing target) + word2
        // (nop) + word3 (nop) + word4 (BNE) + word5 (delay slot) = 5
        // dispatches each, taken or not (the slot always runs). r1 starts
        // at 3 and the loop runs while r1 != 0 at the BNE test, so it
        // executes 3 full iterations (r1: 3->2->1->0, the third BNE finds
        // r1==0 and doesn't re-take, but its own dispatch + slot still
        // count): 1 + 5*3 = 16 dispatches, ending at word 6 (past word 5) —
        // the interpreter never actually executes word 6's sentinel, it's just
        // where PC lands after the loop exits.
        let steps = 1 + 5 * 3;
        assert_jit_matches_interpreter_page(&page, [0u64; 32], pc, 0, /*max_instrs=*/16, steps);
    }

    // ---- Branches and jumps: batch B (unconditional J/JAL) ------

    fn make_j(op: u32, target26: u32) -> u32 {
        (op << 26) | (target26 & 0x3FFFFFF)
    }

    /// word 0: J/JAL target=7 (word 7 — outside the 2-instruction compiled
    /// region, same exit-isolation rationale as `cond_branch_layout`).
    /// word 1: delay slot, marks that it ran.
    fn jump_layout(op: u32) -> Vec<(u16, u32)> {
        vec![
            (0, make_j(op, 7)),
            (1, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)),
        ]
    }

    #[test]
    fn j_matches_interpreter() {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        assert_jit_matches_interpreter_page(&jump_layout(crate::mips_isa::OP_J), [0u64; 32], pc, 0, 2, 1);
    }

    #[test]
    fn jal_matches_interpreter() {
        // JAL writes r31 = this instruction's address + 8 (past the delay
        // slot) — verified as part of the full snapshot comparison.
        let pc = 0xFFFF_FFFF_8000_0000u64;
        assert_jit_matches_interpreter_page(&jump_layout(crate::mips_isa::OP_JAL), [0u64; 32], pc, 0, 2, 1);
    }

    /// Regression test: JAL's link-register write (`emit_write_link_register`)
    /// used to compute `vbase | (link_word * 4)` — an OR, not an add. When
    /// the JAL sits at a page's second-to-last word (`entry_word =
    /// ENTRIES_PER_PAGE - 2`), `link_word = entry_word + 2 ==
    /// ENTRIES_PER_PAGE`, i.e. the return address (this instruction + 8)
    /// falls on the *next* page — `link_word * 4 == PAGE_SIZE`, which needs
    /// to carry into vbase's page-number bits. OR can't carry: whenever
    /// vbase's own bit 12 already happened to be 1 (any page whose low 13
    /// bits, as a page number, are odd), ORing in PAGE_SIZE (bit 12 alone)
    /// silently changed nothing, linking back into the *same* page instead
    /// of the next one (observed live on an IRIX 6.5 boot: JAL at
    /// ...173ff8 linked to ...173000 instead of ...174000). Picking a pc
    /// whose page number is odd (`0x8000_1000`, not `0x8000_0000`)
    /// reproduces this deterministically rather than relying on which
    /// physical page a real boot happens to place the JAL on.
    #[test]
    fn jal_at_page_penultimate_word_links_into_next_page() {
        let pc_page = 0xFFFF_FFFF_8000_1000u64; // page number 1 (odd) -> bit 12 of vbase is set
        let entry_word: u16 = (PAGE_SIZE / 4 - 2) as u16; // second-to-last word
        let pc = pc_page + (entry_word as u64) * 4;
        // target=0: off-page (start of the 256MB region) — exercises the
        // link-register bug in isolation from the taken-target address,
        // which run_jit_page/analyzer::jump_target now compute correctly
        // against pc's real page (see run_jit_page's doc comment).
        let page = vec![
            (entry_word, make_j(crate::mips_isa::OP_JAL, 0)),
            (entry_word + 1, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)), // delay slot, marks that it ran
        ];
        assert_jit_matches_interpreter_page(&page, [0u64; 32], pc, entry_word, 2, 1);
    }

    #[test]
    fn jump_delay_slot_always_executes() {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let jit = run_jit_page(&jump_layout(crate::mips_isa::OP_J), [0u64; 32], pc, 0, 1, &[])
            .expect("J region must compile");
        assert_eq!(jit.gpr[5], 1, "slot must execute");
    }

    /// `try_emit_fused_nop_slot` (codegen.rs): a real NOP (`raw == 0`) in a
    /// J/JAL's delay slot must skip the full `emit_slot_semantics`
    /// bracketing entirely (only outside `jitv2_lockstep`/`developer`, where
    /// the fast path is disabled — see that function's doc comment) while
    /// still landing on the right target and advancing `core.hot.cycles` by
    /// exactly 2 (branch + slot), matching the interpreter's own
    /// `exec_j_nop`-style fusion cycle count for the same pair.
    #[test]
    fn j_with_nop_slot_fuses_and_still_advances_cycles_by_two() {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let page = vec![(0u16, make_j(crate::mips_isa::OP_J, 7)), (1, 0)]; // word 1: real NOP
        let (jit, cycles) = run_jit_page_with_cycles(&page, [0u64; 32], pc, 0, 1, &[])
            .expect("J+NOP region must compile");
        let expected_target = (pc & 0xFFFF_FFFF_F000_0000) | (7 * 4);
        assert_eq!(jit.pc, expected_target, "J's target must still be correct with a fused NOP slot");
        #[cfg(not(any(feature = "jitv2_lockstep", feature = "developer")))]
        assert_eq!(cycles, 2, "fused branch+NOP pair must still advance cycles by exactly 2");
        #[cfg(any(feature = "jitv2_lockstep", feature = "developer"))]
        assert_eq!(cycles, 2, "unfused branch+NOP pair (lockstep/developer) must also advance cycles by exactly 2");
    }

    /// Regression test: `exec_decoded`'s JIT dispatch gate probes a PC for
    /// compilation when `core.jit_trigger` is set (or it's word-offset 4, or
    /// already published) — set by the interpreter's own
    /// `handle_exec_complete`/`exec_complete_pc_set` on every taken
    /// branch/jump, but previously *never* set when the taken transfer was
    /// executed by JIT-compiled code itself (`emit_absolute_pc_exit`/
    /// `emit_runtime_pc_exit` wrote `core.pc` directly and returned with no
    /// trigger at all). That meant a jump reached exclusively via
    /// JIT-to-JIT control transfer, landing on a fresh word that happened to
    /// sit at neither offset 4 nor an already-published entry, would never
    /// get probed for compilation — silently stuck interpreting that address
    /// forever even under a hot loop. Fixed by having the jump/branch exit
    /// stubs set `core.jit_trigger` themselves before returning.
    ///
    /// This test compiles a JIT function for a bare `J 7` (target word 7,
    /// deliberately off the compiled region so it must exit via
    /// `emit_absolute_pc_exit`), runs it directly (bypassing
    /// `exec_decoded`'s gate entirely — this test is about what the JIT
    /// *itself* leaves behind, not the gate's own logic), and asserts
    /// `core.jit_trigger` is `true` afterward.
    #[test]
    fn jit_jump_exit_sets_jit_trigger_for_the_interpreter_gate() {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let page = jump_layout(crate::mips_isa::OP_J);
        let mut page_words = [0u32; crate::jitv2::ENTRIES_PER_PAGE];
        for &(word, raw) in &page {
            page_words[word as usize] = raw;
        }
        let page_base = (pc & !(crate::jitv2::PAGE_SIZE as u64 - 1)) as u32;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, 1);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
            .expect("J region must compile");

        let (mut exec, mem) = seeded_executor([0u64; 32], pc);
        for &(word, raw) in &page { mem.set_word(page_base as u64 + (word as u64) * 4, raw); }
        exec.install_jit_hooks();
        exec.core.jit_trigger = false; // explicit: must be set BY the exit, not already true

        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        std::mem::forget(codegen);

        assert!(exec.core.jit_trigger, "JIT jump exit must set core.jit_trigger so the interpreter's dispatch gate probes the target for compilation");
    }

    #[test]
    fn j_off_page_exits_directly_to_the_real_target() {
        // Regression test: a J/JAL whose target is off-page (analyzer
        // classifies it Classify::Jump{target: None}, taken_exit =
        // Some(PageLeaving)) used to bail to jump_target_word's page-masked
        // (and for an off-page target, meaningless) word offset — observed
        // live as a JIT'd J landing thousands of words away from the real
        // target on an IRIX 5.3 boot trace.
        //
        // The fix is NOT "bail to the J's own pc instead" — the delay slot
        // has already executed inline by the time this exit fires (§6.1.4:
        // unconditional, before the target logic), so re-dispatching the J
        // from its own pc would have the interpreter run that slot a SECOND
        // time (exec_j's branch_delay sets up in_delay_slot, and the slot
        // gets dispatched as an ordinary instruction on the next step,
        // exactly like it does when the interpreter runs the J for the first
        // time). Instead, the compiled unit must write the real, fully
        // computed target address directly into core.pc and exit — no
        // re-dispatch of anything — exactly matching a from-scratch
        // interpreter run in one JIT call.
        let pc = 0xFFFF_FFFF_8000_0000u64;
        // J target26 chosen so (pc+4 & 0xF0000000) | (target26<<2) lands on
        // a different 4KB page than pc (pc's page is 0x8000_0000..0x8000_1000;
        // target26=0x0400 -> byte target 0x1000, well outside it).
        let page = vec![
            (0u16, make_j(crate::mips_isa::OP_J, 0x0400)),
            (1u16, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)), // delay slot, marks that it ran
        ];

        let page_base = pc as u32; // pc is already page-aligned in this test
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, raw) in &page { page_words[word as usize] = raw; }
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, 2);
        assert!(non_empty);
        assert_eq!(walked[0].taken_exit, Some(crate::jitv2::analyzer::StopReason::PageLeaving));
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
            .expect("J region must compile even though its target is off-page");

        let (mut jit_exec, jit_mem) = seeded_executor([0u64; 32], pc);
        for &(word, raw) in &page { jit_mem.set_word(pc + (word as u64) * 4, raw); }
        let status = unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        assert_eq!(status, crate::mips_exec::EXEC_COMPLETE);

        // A from-scratch 2-step interpreter run (J itself, then its delay
        // slot, which is where the real branch_delay-driven transfer lands)
        // is the ground truth this single JIT call must match exactly —
        // both the real off-page pc and the slot's one-time-only effect.
        let interp_final = run_interpreter_page(&page, [0u64; 32], pc, 2);
        let jit_final = CoreSnapshot::capture(&jit_exec.core);
        assert_eq!(jit_final, interp_final, "JIT's direct-to-target exit must match a from-scratch interpreter run in one call");
        assert_eq!(jit_final.gpr[5], 1, "delay slot must have executed exactly once");
    }

    #[test]
    fn j_at_reset_vector_reaches_realstart() {
        // Regression repro: the real IRIX PROM reset vector at
        // 0xFFFFFFFF_BFC00000 is `j 0xbfc003c0` (<realstart>) with a `nop`
        // delay slot — raw word 0x0bf000f0. Observed live: JIT dispatch on,
        // single-stepping this exact instruction leaves core.pc at
        // 0xbfc00008 (the next sequential word after the delay slot)
        // instead of 0xbfc003c0 — the jump silently fails to redirect
        // control flow, as if it had been a no-op. Uses the exact real PC/
        // instruction word from the live boot trace rather than a
        // synthetic address, since j_off_page_exits_directly_to_the_real_target
        // (word 0 @ 0x8000_0000, target off-page by construction) already
        // passes — something about *this* real page/target combination
        // specifically must differ for the bug to reproduce here and not
        // there.
        let pc = 0xFFFF_FFFF_BFC0_0000u64;
        let raw = 0x0bf000f0u32; // j 0xfc003c0 <realstart>
        assert_eq!((raw >> 26) & 0x3F, crate::mips_isa::OP_J, "sanity: raw must decode as J");
        let expected_target = 0xFFFF_FFFF_BFC0_03C0u64;

        let page = vec![
            (0u16, raw),
            (1u16, 0u32), // nop delay slot
        ];

        // Real dispatch (comp::handle_request) walks against the page's
        // PHYSICAL base (page.pfn * PAGE_SIZE), not the virtual pc — kseg1's
        // 0xFFFFFFFF_BFC00000 physically aliases 0x1FC00000. Also uses the
        // real MAX_INSTRS_PER_COMPILE budget (1), not an arbitrarily larger
        // one — both diverge from the earlier (passing) version of this test.
        let page_base = 0x1FC0_0000u32; // physical alias of pc's kseg1 page
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, r) in &page { page_words[word as usize] = r; }
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, 1);
        assert!(non_empty);
        assert_eq!(walked[0].taken_exit, Some(crate::jitv2::analyzer::StopReason::PageLeaving),
            "J is always classified PageLeaving regardless of on-page-ness (analyzer::jump_target's doc comment)");
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
            .expect("J region must compile");

        let (mut jit_exec, jit_mem) = seeded_executor([0u64; 32], pc);
        for &(word, r) in &page { jit_mem.set_word(pc + (word as u64) * 4, r); }
        let status = unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        assert_eq!(status, crate::mips_exec::EXEC_COMPLETE);

        assert_eq!(jit_exec.core.pc, expected_target,
            "JIT'd reset-vector J must land on realstart (0x{:x}), not fall through to the next word (got 0x{:x})",
            expected_target, jit_exec.core.pc);

        let interp_final = run_interpreter_page(&page, [0u64; 32], pc, 2);
        assert_eq!(interp_final.pc, expected_target, "sanity: interpreter itself must also reach realstart");
    }

    #[test]
    fn standalone_compile_of_a_foreign_delay_slot_honors_pending_transfer() {
        // Precise repro of the real bug mechanism (the test above only
        // covers a J's own delay slot inlined via emit_slot_semantics, which
        // was never broken): exec_decoded's dispatch gate (mips_exec.rs) can
        // probe any word for standalone compilation (via jit_trigger,
        // page.is_published, or — since entry_offset==0 became the
        // always-probe offset — the always-probe itself), regardless of
        // whether that word is "supposed" to be a delay slot right now. On
        // the original real boot trace that found this bug (back when
        // entry_offset==1 was the always-probe offset),
        // the interpreter ran word 0 (`j realstart`), which armed
        // core.in_delay_slot=true / core.delay_slot_target=<realstart> and
        // advanced core.pc to word 1 (the delay slot) via branch_delay. The
        // *next* dispatch, for word 1 alone, satisfied entry_offset==1 and
        // got compiled+run as an ordinary standalone Sequential entry (its
        // own region, with no idea it's running inside someone else's delay
        // slot) — which used to blindly fall through to word 2, discarding
        // the pending transfer to realstart entirely.
        //
        // This test compiles word 1 (a plain ADDIU, standing in for the
        // nop) completely on its own, with core.in_delay_slot/
        // delay_slot_target pre-armed exactly as the interpreter would
        // leave them, and asserts the compiled function redirects to
        // delay_slot_target instead of its own compile-time fallthrough.
        let pc_word1 = 0xFFFF_FFFF_BFC0_0004u64; // word 1 of the same real page
        let target = 0xFFFF_FFFF_BFC0_03C0u64; // realstart
        let page = vec![
            (1u16, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)), // stands in for the delay-slot nop; marks that it ran
        ];

        let page_base = 0x1FC0_0000u32; // physical alias, matching real handle_request
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, r) in &page { page_words[word as usize] = r; }
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 1, page_base, 1);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 1, true, false)
            .expect("standalone word-1 region must compile");

        let (mut jit_exec, jit_mem) = seeded_executor([0u64; 32], pc_word1);
        for &(word, r) in &page { jit_mem.set_word(pc_word1 - 4 + (word as u64) * 4, r); }
        jit_exec.core.in_delay_slot = true;
        jit_exec.core.delay_slot_target = target;

        let status = unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        assert_eq!(status, crate::mips_exec::EXEC_COMPLETE);

        assert_eq!(jit_exec.core.pc, target,
            "a standalone-compiled word that's actually running as a foreign delay slot must exit via delay_slot_target (0x{:x}), not its own compile-time fallthrough (got 0x{:x})",
            target, jit_exec.core.pc);
        assert!(!jit_exec.core.in_delay_slot, "in_delay_slot must be cleared once the pending transfer is honored");
        assert_eq!(jit_exec.core.gpr[5], 1, "the slot instruction's own semantics must still have executed");
    }

    /// entry_offset==0 became the dispatch gate's always-probe offset
    /// (mips_exec.rs) specifically because the same `in_delay_slot`/
    /// `delay_slot_target` runtime check proven above for a same-page
    /// inherited slot (word 1, `standalone_compile_of_a_foreign_delay_slot_honors_pending_transfer`)
    /// is page-agnostic — it reads plain `MipsCore` fields `branch_delay`
    /// sets identically no matter which page the branch itself lived on.
    /// This is the cross-page case that used to be refused statically
    /// (§6.1.4's original "offset 0 is never an entry" — the branch at the
    /// *previous* page's 0xFFC is never compiled, per the 0xFFC rule, so
    /// this page's word 0 has no way to know statically whether it's an
    /// ordinary instruction or someone else's inherited delay slot).
    /// Covers both the taken and not-taken arms of the previous page's
    /// conditional branch — `handle_branch_not_taken` arms
    /// `in_delay_slot`/`delay_slot_target` exactly the same way
    /// `branch_delay` does for the taken arm (target = pc+8, the natural
    /// fallthrough, instead of the branch's real target), so both must
    /// resolve correctly through the same standalone-compiled word-0 unit.
    #[test]
    fn standalone_compile_of_page0_honors_delay_slot_inherited_from_previous_pages_taken_branch() {
        // Previous page's last word (0xFFC) is a conditional branch — never
        // compiled itself (0xFFC rule), so only the interpreter ever runs
        // it; this test starts from the moment its delay slot (this page's
        // word 0) is about to dispatch, exactly as the interpreter would
        // have left `core.in_delay_slot`/`core.delay_slot_target` after
        // running that branch for real (taken arm).
        let pc_word0 = 0xFFFF_FFFF_9FC1_0000u64; // page B, word 0 (the inherited slot)
        let target = 0xFFFF_FFFF_8000_0400u64; // where page A's taken branch actually went
        let page = vec![
            (0u16, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)), // stands in for the delay-slot instruction; marks that it ran
        ];

        let page_base = 0x9FC1_0000u32;
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, r) in &page { page_words[word as usize] = r; }
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, 1);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
            .expect("standalone word-0 region must compile");

        let (mut jit_exec, jit_mem) = seeded_executor([0u64; 32], pc_word0);
        for &(word, r) in &page { jit_mem.set_word(pc_word0 + (word as u64) * 4, r); }
        jit_exec.core.in_delay_slot = true;
        jit_exec.core.delay_slot_target = target;

        let status = unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        assert_eq!(status, crate::mips_exec::EXEC_COMPLETE);

        assert_eq!(jit_exec.core.pc, target,
            "word 0, standalone-compiled, running as a delay slot inherited from the *previous page's* taken branch must exit via delay_slot_target (0x{:x}), not its own compile-time fallthrough (got 0x{:x})",
            target, jit_exec.core.pc);
        assert!(!jit_exec.core.in_delay_slot, "in_delay_slot must be cleared once the pending transfer is honored");
        assert_eq!(jit_exec.core.gpr[5], 1, "the slot instruction's own semantics must still have executed");
    }

    #[test]
    fn standalone_compile_of_page0_honors_delay_slot_inherited_from_previous_pages_not_taken_branch() {
        // Mirror of the taken-arm test above, for a conditional branch that
        // wasn't taken: handle_branch_not_taken still arms
        // in_delay_slot/delay_slot_target (target = the branch's own pc+8,
        // i.e. natural fallthrough past the slot — real MIPS semantics: the
        // delay slot always executes, taken or not), so the standalone word-0
        // compile must still honor it, landing exactly where the interpreter
        // would have: this same word, +4 (word 1, still on page B).
        let pc_word0 = 0xFFFF_FFFF_9FC1_0000u64; // page B, word 0 (the inherited slot)
        let target = pc_word0.wrapping_add(4); // not-taken: natural fallthrough past the slot
        let page = vec![
            (0u16, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)), // stands in for the delay-slot instruction; marks that it ran
        ];

        let page_base = 0x9FC1_0000u32;
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, r) in &page { page_words[word as usize] = r; }
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, 1);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
            .expect("standalone word-0 region must compile");

        let (mut jit_exec, jit_mem) = seeded_executor([0u64; 32], pc_word0);
        for &(word, r) in &page { jit_mem.set_word(pc_word0 + (word as u64) * 4, r); }
        jit_exec.core.in_delay_slot = true;
        jit_exec.core.delay_slot_target = target;

        let status = unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        assert_eq!(status, crate::mips_exec::EXEC_COMPLETE);

        assert_eq!(jit_exec.core.pc, target,
            "word 0, standalone-compiled, running as a delay slot inherited from the *previous page's* not-taken branch must exit via delay_slot_target (0x{:x}), not its own compile-time fallthrough (got 0x{:x})",
            target, jit_exec.core.pc);
        assert!(!jit_exec.core.in_delay_slot, "in_delay_slot must be cleared once the pending transfer is honored");
        assert_eq!(jit_exec.core.gpr[5], 1, "the slot instruction's own semantics must still have executed");
    }

    #[test]
    fn beq_off_page_taken_exits_directly_to_the_real_target() {
        // branch_target_addr's counterpart to j_off_page_exits_directly_to_the_real_target
        // — a conditional branch's taken arm can leave the page too
        // (Classify::Branch{target: None}), and went through the exact same
        // emit_target_edge path, so needs its own coverage: branch_target_addr
        // is a separate address computation from jump_target_addr (relative
        // to the delay slot's own address, not J/JAL's absolute-region math).
        let pc = 0xFFFF_FFFF_8000_0010u64; // word 4 within its page
        // BEQ r0,r0 (always taken), imm16 = -0x100 -> target word =
        // 4 + 1 + (-0x100) = -251, well before word 0 -> off-page backward.
        let page = vec![
            (4u16, make_i(crate::mips_isa::OP_BEQ, 0, 0, 0xFF00u16 /* -0x100 as i16 */)),
            (5u16, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)), // delay slot, marks that it ran
        ];

        let page_base = (pc & !0xFFFu64) as u32;
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, raw) in &page { page_words[word as usize] = raw; }
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 4, page_base, 2);
        assert!(non_empty);
        assert_eq!(walked[4].taken_exit, Some(crate::jitv2::analyzer::StopReason::PageLeaving));
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 4, true, false)
            .expect("BEQ region must compile even though its taken target is off-page");

        let (mut jit_exec, jit_mem) = seeded_executor([0u64; 32], pc);
        for &(word, raw) in &page { jit_mem.set_word((page_base as u64) + (word as u64) * 4, raw); }
        let status = unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        assert_eq!(status, crate::mips_exec::EXEC_COMPLETE);

        let interp_final = run_interpreter_page(&page, [0u64; 32], pc, 2);
        let jit_final = CoreSnapshot::capture(&jit_exec.core);
        assert_eq!(jit_final, interp_final, "JIT's direct-to-target exit must match a from-scratch interpreter run in one call");
        assert_eq!(jit_final.gpr[5], 1, "delay slot must have executed exactly once");
    }

    #[test]
    fn jal_link_register_matches_interpreter() {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let interp = run_interpreter_page(&jump_layout(crate::mips_isa::OP_JAL), [0u64; 32], pc, 2);
        let jit = run_jit_page(&jump_layout(crate::mips_isa::OP_JAL), [0u64; 32], pc, 0, 1, &[])
            .expect("JAL region must compile");
        assert_eq!(jit.gpr[31], interp.gpr[31]);
        assert_eq!(jit.gpr[31], pc + 8, "link register must be this instruction's address + 8");
    }

    // ---- Branches and jumps: batch C (link-writing conditional) ------

    #[test]
    fn bltzal_bgezal_taken_match_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_FFFF_FFFF; // -1: BLTZAL taken, BGEZAL not taken
        assert_branch_matches_interpreter(regimm_branch_layout(crate::mips_isa::RT_BLTZAL, 1), gpr);
        gpr[1] = 0; // BGEZAL taken, BLTZAL not taken
        assert_branch_matches_interpreter(regimm_branch_layout(crate::mips_isa::RT_BGEZAL, 1), gpr);
    }

    #[test]
    fn bltzal_bgezal_not_taken_match_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0; // BLTZAL not taken
        assert_branch_matches_interpreter(regimm_branch_layout(crate::mips_isa::RT_BLTZAL, 1), gpr);
        gpr[1] = 0xFFFF_FFFF_FFFF_FFFF; // BGEZAL not taken
        assert_branch_matches_interpreter(regimm_branch_layout(crate::mips_isa::RT_BGEZAL, 1), gpr);
    }

    #[test]
    fn bltzal_writes_link_register_even_when_not_taken() {
        // Link write is unconditional on BLTZAL/BGEZAL (happens before the
        // condition test) — must match the interpreter whether or not the
        // branch itself is taken.
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let mut gpr = [0u64; 32];
        gpr[1] = 0; // BLTZAL: not taken
        let page = regimm_branch_layout(crate::mips_isa::RT_BLTZAL, 1);
        let interp = run_interpreter_page(&page, gpr, pc, 2);
        let jit = run_jit_page(&page, gpr, pc, 0, 1, &[]).expect("BLTZAL must compile");
        assert_eq!(jit.gpr[31], interp.gpr[31]);
        assert_eq!(jit.gpr[31], pc + 8);
    }

    // ---- Branches and jumps: batch D (annulling "Likely" branches) ------

    fn regimm_likely_layout(rt_field: u32, rs: u32) -> Vec<(u16, u32)> {
        vec![
            (0, make_i(crate::mips_isa::OP_REGIMM, rs, rt_field, BRANCH_IMM)),
            (1, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)),
        ]
    }

    /// Likely branches need their own step-count model: not-taken never
    /// dispatches the slot at all (`handle_branch_likely_skip` sets pc
    /// directly, one dispatch total), unlike a plain branch's not-taken arm
    /// (which still dispatches the slot as an ordinary next instruction,
    /// two total). `taken_steps`/`not_taken_steps` let each call site state
    /// its own expectation explicitly rather than hard-coding one constant
    /// that would be wrong for one of the two arms.
    fn assert_likely_taken_matches_interpreter(page: Vec<(u16, u32)>, gpr: [u64; 32]) {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        assert_jit_matches_interpreter_page(&page, gpr, pc, 0, 2, 1);
    }
    fn assert_likely_not_taken_matches_interpreter(page: Vec<(u16, u32)>, gpr: [u64; 32]) {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        assert_jit_matches_interpreter_page(&page, gpr, pc, 0, 1, 1);
    }

    #[test]
    fn beql_bnel_taken_match_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 1; gpr[2] = 1;
        assert_likely_taken_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BEQL, 1, 2), gpr);
        gpr[2] = 2;
        assert_likely_taken_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BNEL, 1, 2), gpr);
    }

    #[test]
    fn beql_bnel_not_taken_annuls_slot_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 1; gpr[2] = 2; // not equal -> BEQL not taken
        assert_likely_not_taken_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BEQL, 1, 2), gpr);
        gpr[2] = 1; // equal -> BNEL not taken
        assert_likely_not_taken_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BNEL, 1, 2), gpr);
    }

    #[test]
    fn likely_branch_not_taken_does_not_execute_slot() {
        // The defining difference from a plain branch: the slot's effect
        // (r5=1) must be ABSENT on the not-taken path, since it's annulled,
        // never dispatched at all.
        let mut gpr = [0u64; 32];
        gpr[1] = 1; gpr[2] = 2; // not equal -> BEQL not taken
        let jit = run_jit_page(&cond_branch_layout(crate::mips_isa::OP_BEQL, 1, 2), gpr, 0xFFFF_FFFF_8000_0000, 0, 1, &[])
            .expect("BEQL region must compile");
        assert_eq!(jit.gpr[5], 0, "annulled slot must not execute");

        gpr[2] = 1; // equal -> taken
        let jit = run_jit_page(&cond_branch_layout(crate::mips_isa::OP_BEQL, 1, 2), gpr, 0xFFFF_FFFF_8000_0000, 0, 1, &[])
            .expect("BEQL region must compile");
        assert_eq!(jit.gpr[5], 1, "slot must execute on the taken path");
    }

    #[test]
    fn blezl_bgtzl_match_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0;
        assert_likely_taken_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BLEZL, 1, 0), gpr);
        assert_likely_not_taken_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BGTZL, 1, 0), gpr);
        gpr[1] = 1;
        assert_likely_not_taken_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BLEZL, 1, 0), gpr);
        assert_likely_taken_matches_interpreter(cond_branch_layout(crate::mips_isa::OP_BGTZL, 1, 0), gpr);
    }

    #[test]
    fn bltzl_bgezl_match_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_FFFF_FFFF; // -1
        assert_likely_taken_matches_interpreter(regimm_likely_layout(crate::mips_isa::RT_BLTZL, 1), gpr);
        assert_likely_not_taken_matches_interpreter(regimm_likely_layout(crate::mips_isa::RT_BGEZL, 1), gpr);
        gpr[1] = 0;
        assert_likely_not_taken_matches_interpreter(regimm_likely_layout(crate::mips_isa::RT_BLTZL, 1), gpr);
        assert_likely_taken_matches_interpreter(regimm_likely_layout(crate::mips_isa::RT_BGEZL, 1), gpr);
    }

    #[test]
    fn bltzall_bgezall_match_interpreter_including_link() {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_FFFF_FFFF; // -1: BLTZALL taken
        let page = regimm_likely_layout(crate::mips_isa::RT_BLTZALL, 1);
        let interp = run_interpreter_page(&page, gpr, pc, 2);
        let jit = run_jit_page(&page, gpr, pc, 0, 1, &[]).expect("BLTZALL must compile");
        assert_eq!(jit, interp);
        assert_eq!(jit.gpr[31], pc + 8, "link register written even though this is a Likely branch");

        // Not-taken: link is still written (link write is unconditional,
        // like BLTZAL), but the slot is annulled. BGEZALL is not-taken when
        // rs < 0.
        gpr[1] = 0xFFFF_FFFF_FFFF_FFFF; // -1
        let page = regimm_likely_layout(crate::mips_isa::RT_BGEZALL, 1);
        let interp = run_interpreter_page(&page, gpr, pc, 1);
        let jit = run_jit_page(&page, gpr, pc, 0, 1, &[]).expect("BGEZALL must compile");
        assert_eq!(jit, interp);
    }

    // ---- Branches and jumps: batch E (register-indirect JR/JALR) ------

    /// word 0: JR/JALR rs. word 1: delay slot, marks that it ran. The
    /// target comes from a register, so unlike the other layouts there's no
    /// static "outside the region" trick needed — RegJump always exits
    /// (`taken_exit` is unconditionally `Some(RegJump)`), so a 2-instruction
    /// region is inherently isolated already.
    fn regjump_layout(op_funct: u32, rs: u32, rd: u32) -> Vec<(u16, u32)> {
        vec![
            (0, make_r(crate::mips_isa::OP_SPECIAL, rs, 0, rd, 0, op_funct)),
            (1, make_i(crate::mips_isa::OP_ADDIU, 0, 5, 1)),
        ]
    }

    #[test]
    fn jr_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8000_5000; // target register value (an arbitrary valid-looking address)
        assert_branch_matches_interpreter(regjump_layout(crate::mips_isa::FUNCT_JR, 1, 0), gpr);
    }

    #[test]
    fn jr_delay_slot_always_executes() {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8000_5000;
        let jit = run_jit_page(&regjump_layout(crate::mips_isa::FUNCT_JR, 1, 0), gpr, pc, 0, 1, &[])
            .expect("JR region must compile");
        assert_eq!(jit.gpr[5], 1, "slot must execute");
        assert_eq!(jit.pc, gpr[1], "pc must be the register's own value, unmodified");
    }

    /// Same fusion coverage as `j_with_nop_slot_fuses_and_still_advances_cycles_by_two`,
    /// for the `emit_regjump` call site (JR's own top-level slot, not the
    /// nested-branch-in-slot path).
    #[test]
    fn jr_with_nop_slot_fuses_and_still_advances_cycles_by_two() {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8000_5000;
        let page = vec![(0u16, make_r(crate::mips_isa::OP_SPECIAL, 1, 0, 0, 0, crate::mips_isa::FUNCT_JR)), (1, 0)]; // word 1: real NOP
        let (jit, cycles) = run_jit_page_with_cycles(&page, gpr, pc, 0, 1, &[])
            .expect("JR+NOP region must compile");
        assert_eq!(jit.pc, gpr[1], "pc must be the register's own value, unmodified, with a fused NOP slot");
        assert_eq!(cycles, 2, "fused branch+NOP pair must still advance cycles by exactly 2");
    }

    // ---- Branches and jumps at the 0xFFC page boundary ------
    //
    // A branch/jump/regjump sitting at the page's last word (offset 0xFFC,
    // word 1023) has its mandatory delay slot on the *next* page, which a
    // single-page `Analyzer` walk can't see. Earlier behavior rejected these
    // as an entry entirely (denylisted, always interpreted) — codegen's only
    // option for a *compiled* 0xFFC word used to be a self-referential bail
    // (set core.pc back to its own address, return EXEC_COMPLETE), which
    // published and dispatched again would infinite-loop without ever
    // reaching the interpreter fallback it was meant to defer to (observed
    // live on an IRIX boot, rules/jitv2/codegen-gotchas.md).
    //
    // Now that every exit already materializes core.pc/in_delay_slot/
    // delay_slot_target explicitly (the entry_offset==0 foreign-slot work,
    // `standalone_compile_of_page0_honors_delay_slot_inherited_from_previous_pages_*`
    // above), a 0xFFC branch/jump/regjump compiles normally: instead of
    // inlining a slot that isn't on this page, it arms
    // core.in_delay_slot/core.delay_slot_target exactly like the
    // interpreter's own `branch_delay`/`handle_branch_not_taken`, and
    // advances core.pc onto the next page's word 0 — where dispatch actually
    // executes the (real, next-page) delay slot instruction and then honors
    // the pending transfer, per the foreign-slot entry check.
    //
    // These tests compile *only* the 0xFFC word (a 1-instruction region,
    // max_instrs=1 — its slot was never on this page to charge against the
    // budget anyway) and run it standalone, checking the resulting
    // core.pc/in_delay_slot/delay_slot_target against what the real
    // interpreter leaves behind for the same instruction — the delay slot
    // instruction's own execution is out of scope here (that's the
    // `standalone_compile_of_page0_...` tests' job, from the consuming side).

    const LAST_WORD: u16 = (ENTRIES_PER_PAGE - 1) as u16;
    const PAGE_A_BASE: u64 = 0xFFFF_FFFF_9FC1_0000u64;
    const PC_0XFFC: u64 = PAGE_A_BASE + (LAST_WORD as u64) * 4;
    const PAGE_B_WORD0: u64 = PAGE_A_BASE + PAGE_SIZE as u64; // next page, word 0
    const PAGE_B_WORD4: u64 = PAGE_B_WORD0 + 4; // next page, word 1 (natural not-taken fallthrough)

    /// Compile `instr` standalone at `LAST_WORD` and run it, returning the
    /// resulting core (pc/in_delay_slot/delay_slot_target/gpr) — no slot
    /// instruction is ever placed on this page (there's nowhere on-page to
    /// put it), matching every 0xFFC test below.
    fn run_0xffc_jit(instr: u32, gpr: [u64; 32]) -> MipsCore {
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        page_words[LAST_WORD as usize] = instr;
        let page_base = (PAGE_A_BASE & !(PAGE_SIZE as u64 - 1)) as u32;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, LAST_WORD, page_base, 1);
        assert!(non_empty, "0xFFC word must not be excluded — check the test's instruction encoding");
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, LAST_WORD, true, false)
            .expect("0xFFC region must compile (ForeignPageSlot, not a decline)");

        let (exec, _mem) = seeded_executor(gpr, PC_0XFFC);
        let mut exec = Box::new(exec);
        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        std::mem::forget(codegen);
        exec.core
    }

    /// Run `instr` through the real interpreter for exactly one dispatch —
    /// the branch/jump/regjump itself, which arms `in_delay_slot`/
    /// `delay_slot_target` and advances pc by 4, but does not itself run the
    /// (off-page) delay slot. Ground truth for `run_0xffc_jit`'s comparison.
    fn run_0xffc_interp(instr: u32, gpr: [u64; 32]) -> MipsCore {
        let (mut exec, _mem) = seeded_executor_over(MockMemory::new_not_compilable(), gpr, PC_0XFFC);
        exec.exec(instr);
        exec.core
    }

    fn assert_0xffc_matches_interpreter(instr: u32, gpr: [u64; 32]) {
        let interp = run_0xffc_interp(instr, gpr);
        let jit = run_0xffc_jit(instr, gpr);
        assert_eq!(jit.pc, interp.pc, "pc mismatch for 0xFFC instr=0x{:08x}", instr);
        assert_eq!(jit.in_delay_slot, interp.in_delay_slot, "in_delay_slot mismatch for 0xFFC instr=0x{:08x}", instr);
        assert_eq!(jit.delay_slot_target, interp.delay_slot_target, "delay_slot_target mismatch for 0xFFC instr=0x{:08x}", instr);
        assert_eq!(jit.gpr, interp.gpr, "gpr mismatch for 0xFFC instr=0x{:08x}", instr);
    }

    #[test]
    fn beq_taken_at_0xffc_arms_foreign_page_slot() {
        let mut gpr = [0u64; 32];
        gpr[1] = 42;
        gpr[2] = 42; // equal -> taken
        let instr = make_i(crate::mips_isa::OP_BEQ, 1, 2, 0); // target irrelevant: ForeignPageSlot never resolves it on-page
        assert_0xffc_matches_interpreter(instr, gpr);
        let jit = run_0xffc_jit(instr, gpr);
        assert!(jit.in_delay_slot, "taken branch must arm in_delay_slot");
        assert_eq!(jit.pc, PAGE_B_WORD0, "pc must advance into the (next-page) delay slot, not to the branch's own target yet");
    }

    #[test]
    fn beq_not_taken_at_0xffc_arms_foreign_page_slot_with_fallthrough_target() {
        let mut gpr = [0u64; 32];
        gpr[1] = 1;
        gpr[2] = 2; // not equal -> not taken
        let instr = make_i(crate::mips_isa::OP_BEQ, 1, 2, 0);
        assert_0xffc_matches_interpreter(instr, gpr);
        let jit = run_0xffc_jit(instr, gpr);
        assert!(jit.in_delay_slot, "delay slot always executes, taken or not (§6.1.4) — must still be armed");
        assert_eq!(jit.delay_slot_target, PC_0XFFC + 8, "not-taken target is the branch's own pc+8, matching handle_branch_not_taken");
        assert_eq!(jit.pc, PAGE_B_WORD0, "pc must advance into the (next-page) delay slot");
    }

    #[test]
    fn beql_taken_at_0xffc_arms_foreign_page_slot() {
        let mut gpr = [0u64; 32];
        gpr[1] = 42;
        gpr[2] = 42; // equal -> taken
        let instr = make_i(crate::mips_isa::OP_BEQL, 1, 2, 0);
        assert_0xffc_matches_interpreter(instr, gpr);
        let jit = run_0xffc_jit(instr, gpr);
        assert!(jit.in_delay_slot, "annulling Likely's taken arm still executes its slot");
        assert_eq!(jit.pc, PAGE_B_WORD0);
    }

    #[test]
    fn beql_not_taken_at_0xffc_skips_slot_and_lands_on_fallthrough_directly() {
        // Annulling Likely, not taken: the slot never executes at all
        // (handle_branch_likely_skip's direct pc+=8), so there's nothing to
        // arm — pc must land directly on the next page's word 4 (past where
        // the never-executed slot would have been), with in_delay_slot left
        // false.
        let mut gpr = [0u64; 32];
        gpr[1] = 1;
        gpr[2] = 2; // not equal -> not taken -> annulled
        let instr = make_i(crate::mips_isa::OP_BEQL, 1, 2, 0);
        assert_0xffc_matches_interpreter(instr, gpr);
        let jit = run_0xffc_jit(instr, gpr);
        assert!(!jit.in_delay_slot, "annulled slot never executes — nothing to arm");
        assert_eq!(jit.pc, PAGE_B_WORD4, "pc must land directly on the next page's word 4, matching handle_branch_likely_skip's pc+=8");
    }

    #[test]
    fn j_at_0xffc_arms_foreign_page_slot() {
        let gpr = [0u64; 32];
        let instr = make_j(crate::mips_isa::OP_J, 0x0400);
        assert_0xffc_matches_interpreter(instr, gpr);
        let jit = run_0xffc_jit(instr, gpr);
        assert!(jit.in_delay_slot, "J is unconditional — its slot always executes");
        assert_eq!(jit.pc, PAGE_B_WORD0);
    }

    #[test]
    fn jal_at_0xffc_arms_foreign_page_slot_and_writes_link_register() {
        let gpr = [0u64; 32];
        let instr = make_j(crate::mips_isa::OP_JAL, 0x0400);
        assert_0xffc_matches_interpreter(instr, gpr);
        let jit = run_0xffc_jit(instr, gpr);
        assert!(jit.in_delay_slot);
        assert_eq!(jit.gpr[31], PAGE_B_WORD4, "JAL's link register is this instr + 8, one word past the (next-page) delay slot it's skipping over");
    }

    #[test]
    fn jr_at_0xffc_arms_foreign_page_slot() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8000_5000;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 0, 0, 0, crate::mips_isa::FUNCT_JR);
        assert_0xffc_matches_interpreter(instr, gpr);
        let jit = run_0xffc_jit(instr, gpr);
        assert!(jit.in_delay_slot);
        assert_eq!(jit.delay_slot_target, gpr[1], "JR's target is the register's own value, unmodified");
        assert_eq!(jit.pc, PAGE_B_WORD0);
    }

    #[test]
    fn jalr_at_0xffc_arms_foreign_page_slot_and_writes_link_register() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8000_5000;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 0, 2, 0, crate::mips_isa::FUNCT_JALR); // rd=2
        assert_0xffc_matches_interpreter(instr, gpr);
        let jit = run_0xffc_jit(instr, gpr);
        assert!(jit.in_delay_slot);
        assert_eq!(jit.delay_slot_target, gpr[1]);
        assert_eq!(jit.gpr[2], PAGE_B_WORD4, "JALR writes its own rd (this instr + 8), not always r31");
    }

    #[test]
    fn sequential_pair_ending_at_0xffc_falls_through_to_next_page() {
        // Plain regression coverage for the ordinary (non-branch) case at
        // the same boundary: two straight-line instructions, the second
        // landing exactly on 0xFFC, must fall through to the next page's
        // word 0 — already covered in isolation by
        // `sequential_at_last_word_falls_through_to_next_page_not_same_page`
        // below; this is the same shape but as a genuine 2-instruction
        // compiled region (entry at word 1022) rather than a synthetic
        // single-word walk, exercising pass 2's ordinary fallthrough wiring
        // between two head instructions immediately adjacent to the
        // boundary.
        let entry_word = LAST_WORD - 1;
        let pc = PAGE_A_BASE + (entry_word as u64) * 4;
        let page = vec![
            (entry_word, make_i(crate::mips_isa::OP_ADDIU, 0, 1, 5)), // ADDIU r1,r0,5
            (LAST_WORD, make_i(crate::mips_isa::OP_ADDIU, 1, 2, 7)),  // ADDIU r2,r1,7
        ];
        let gpr = [0u64; 32];
        let interp = run_interpreter_page(&page, gpr, pc, 2);
        let jit = run_jit_page(&page, gpr, pc, entry_word, 2, &[])
            .expect("two-instruction region ending at 0xFFC must compile");
        assert_eq!(jit, interp);
        assert_eq!(jit.gpr[1], 5);
        assert_eq!(jit.gpr[2], 12);
        assert_eq!(jit.pc, PAGE_B_WORD0, "must fall through onto the next page's word 0, not wrap back onto this page");
    }

    #[test]
    fn skip_entry_preamble_true_bypasses_the_entry_words_own_checks() {
        // skip_entry_preamble=true (comp.rs's production path): the
        // interpreter's step() dispatch loop already ran the equivalent
        // IP7/pending-interrupt checks for this exact PC immediately before
        // calling into compiled code, so the compiled function's own entry
        // path must NOT re-check and bail — it should run the entry
        // instruction's real semantics directly. Proven here by seeding a
        // pending interrupt (core.hot.interrupts != 0) that would normally make
        // emit_pending_interrupt_preamble bail immediately (status
        // EXEC_COMPLETE, pc unchanged) and confirming the entry
        // instruction's semantics ran anyway (register written, pc
        // advanced past it) instead of bailing.
        let word: u16 = 10;
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        page_words[word as usize] = make_i(crate::mips_isa::OP_ADDIU, 1, 1, 5); // ADDIU r1,r1,5

        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, word, 0, 1);
        assert!(non_empty);
        let mut instrs_owned = *walked;

        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word, false, true)
            .expect("region must compile");

        let pc = 0xFFFF_FFFF_8000_0000u64 + (word as u64) * 4;
        let (mut exec, mem) = seeded_executor([0u64; 32], pc);
        exec.core.hot.interrupts.store(1, std::sync::atomic::Ordering::Relaxed); // pending -- would normally bail
        let status = unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        // status is EXEC_COMPLETE either way here (a 1-instruction
        // region's own normal end-of-region bail uses the same status code
        // as a preamble bail would) -- gpr[1]/pc are what actually
        // distinguish "ran for real, then hit the region boundary" from "the
        // preamble caught the pending interrupt and bailed before running
        // anything": gpr[1] must show the ADDIU actually executed, and pc
        // must have advanced past it (word+1), not stayed at word (which is
        // what a preamble bail — Some(Hazard)/interrupt-caught — would do).
        assert_eq!(status, crate::mips_exec::EXEC_COMPLETE);
        assert_eq!(exec.core.gpr[1], 5, "entry instruction's real semantics must run despite the pending interrupt -- the preamble that would have bailed on it must be skipped");
        assert_eq!(exec.core.pc, pc + 4, "must have bailed at the normal end-of-region boundary (word+1, past the ADDIU), not at word itself (which is what a preamble bail on the pending interrupt would produce)");
    }

    /// Regression test for the entry-word exception-exit materialization fix
    /// (`compile_region_uncommitted`'s `word == entry_word` arm, just above
    /// its jump into `entry_word_body_block`), isolated to ONLY the internal
    /// back-edge path into the region's real `entry_word`.
    ///
    /// `emit_target_edge`'s internal jump (`None` arm — a bare, zero-arg
    /// `jump`) never touches `core.pc`/`core.in_delay_slot`. A stale
    /// `core.pc` (left over from wherever this call's external entry
    /// happened to be — here, entry_word's own address, since the first
    /// pass runs via the ordinary `entry_block` -> `entry_word_body_block`
    /// bypass) and a stale `core.in_delay_slot` would still be sitting in
    /// `core` when the back-edge lands on `entry_word_block` a second time,
    /// if `entry_word_block`'s own preamble didn't materialize them itself
    /// first. Both values happen to already be individually correct-looking
    /// after just one pass here (pc already equals entry_word's own
    /// address, in_delay_slot already false) — which is exactly why this
    /// needs its own isolated test: with only one pass, the materialization
    /// fix and "do nothing" produce the same result, so the fix must be
    /// exercised via a genuine second, internal-only visit to actually
    /// distinguish them. The external-entry path itself is deliberately
    /// excluded from consideration here (a pre-armed foreign delay slot at
    /// the external entry is separately covered by
    /// `assert_entry_word_reached_as_foreign_slot_matches_interpreter` and
    /// `adel_in_entry_word_ordinary_arrival_still_matches_interpreter`
    /// above — combining that with an internal back-edge in one test
    /// previously produced a confounded, hard-to-diagnose result where the
    /// arming branch's own target math interfered with the back-edge's).
    ///
    /// Layout: entry_word is an ADD that's harmless on its first
    /// (external-entry) pass and overflows on its second (internal-back-edge)
    /// pass, once a runway word between them has advanced the operand;
    /// entry_word falls through into that runway, then an always-taken
    /// branch back to entry_word, whose delay slot is a nop.
    #[test]
    fn entry_word_faults_correctly_on_internal_back_edge_second_pass() {
        let entry_word: u16 = 10;
        let mut gpr = [0u64; 32];
        gpr[2] = 1; // r2: accumulator, ADD r2,r2,r3
        gpr[3] = 0x7FFF_FFFE; // r3: one below overflow

        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        // entry_word: ADD r2,r2,r3 -> 1 + 0x7FFFFFFE = 0x7FFFFFFF, harmless
        // on pass 1 (external entry); overflows on pass 2 (internal back-edge)
        // once r3 has advanced below.
        page_words[entry_word as usize] = make_r(crate::mips_isa::OP_SPECIAL, 2, 3, 2, 0, crate::mips_isa::FUNCT_ADD);
        page_words[entry_word as usize + 1] = make_i(crate::mips_isa::OP_ADDIU, 3, 3, 1); // runway: r3 += 1
        page_words[entry_word as usize + 2] = make_i(crate::mips_isa::OP_BEQ, 0, 0, 0xFFFD); // BEQ r0,r0,-3 -> target = entry_word (back-edge)
        page_words[entry_word as usize + 3] = 0; // delay slot (nop)

        let page: Vec<(u16, u32)> = (entry_word..entry_word + 4)
            .map(|w| (w, page_words[w as usize]))
            .collect();
        let pc = 0xFFFF_FFFF_8000_0000u64 + (entry_word as u64) * 4;

        let interp = run_interpreter_page(&page, gpr, pc, 5);

        let page_base = (pc & !(PAGE_SIZE as u64 - 1)) as u32;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, entry_word, page_base, 3);
        assert!(non_empty, "entry instruction must not be excluded");
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, entry_word, true, true)
            .expect("region must be compilable for this test to be meaningful");

        let (exec, _mem) = seeded_executor(gpr, pc);
        let mut exec = Box::new(exec);
        exec.install_jit_hooks();
        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        std::mem::forget(codegen);
        let jit = CoreSnapshot::capture(&exec.core);

        assert_eq!(jit, interp, "JIT and interpreter diverged: entry_word's fault on its 2nd (internal-back-edge) pass must match the interpreter exactly");
        assert_eq!(jit.cp0_epc, pc, "EPC must be entry_word's own address on this, its 2nd (internal-back-edge) visit");
        assert_eq!(jit.cp0_cause & crate::mips_core::CAUSE_BD, 0, "BD must be clear -- entry_word was reached via an ordinary internal branch edge on this pass, never a delay slot");
    }

    #[test]
    fn skip_entry_preamble_true_still_checks_on_an_internal_back_edge_into_entry() {
        // The flip side of the test above: skip_entry_preamble only ever
        // affects the path reached directly from the function's real entry
        // (entry_block). Every *other* head in the region -- including one
        // that later branches back to entry_word -- is never exempt: its
        // own preamble runs unconditionally, same as always. Layout:
        // entry_word (ADDIU r2,r2,1) falls through into a branch
        // (BEQ r0,r0,-1, always taken, target = word+1+(-1) = entry_word)
        // with a nop delay slot. With a pending interrupt seeded from the
        // very start, entry_word's own (skipped) preamble must not catch
        // it -- its semantics run for real, r2 becomes 1 -- but the
        // branch's own preamble (never skip-exempt) must catch it on the
        // very first visit to *that* word, bailing before the branch's
        // condition is even evaluated (so the taken back-edge into
        // entry_word never actually happens on this call).
        let word: u16 = 10;
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        page_words[word as usize] = make_i(crate::mips_isa::OP_ADDIU, 2, 2, 1); // ADDIU r2,r2,1 (entry)
        page_words[word as usize + 1] = make_i(crate::mips_isa::OP_BEQ, 0, 0, 0xFFFF); // BEQ r0,r0,-1 -> target = word
        page_words[word as usize + 2] = 0; // delay slot (nop)

        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, word, 0, 2);
        assert!(non_empty);
        let mut instrs_owned = *walked;

        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word, false, true)
            .expect("region must compile");

        let pc = 0xFFFF_FFFF_8000_0000u64 + (word as u64) * 4;
        let (mut exec, mem) = seeded_executor([0u64; 32], pc);
        exec.core.hot.interrupts.store(1, std::sync::atomic::Ordering::Relaxed); // pending from the start
        let status = unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        assert_eq!(exec.core.gpr[2], 1, "entry_word's own skipped preamble must not have bailed -- its semantics ran for real");
        assert_eq!(status, crate::mips_exec::EXEC_COMPLETE);
        assert_eq!(exec.core.pc, pc + 4, "the branch's own (never skip-exempt) preamble must catch the pending interrupt and bail at its own word, before ever taking the back-edge into entry_word");
    }

    #[test]
    fn sequential_at_last_word_falls_through_to_next_page_not_same_page() {
        // A plain (non-branch) instruction at the page's last word (offset
        // 0xFFC, word 1023) falls through to word 1024 — conceptually "next
        // page, word 0". Regression: emit_exit_block_body computed the exit
        // pc as `vbase | (word_offset * 4)`; at word_offset=1024 that's
        // `vbase | 0x1000`, and bit 12 of vbase is always 0 (page-aligned) so
        // OR looks like it should carry — but page_base already has bit 12
        // set whenever the *previous* page's own address does, and more
        // fundamentally OR is not addition: `vbase | PAGE_SIZE` is only
        // `vbase + PAGE_SIZE` when vbase's PAGE_SIZE bit is 0, which is not
        // guaranteed for every page. Found via jitv2_verify against a real
        // IRIX 5.3 boot trace: a Sequential instruction at 0x...9fc0fffc
        // landed the JIT back on page 0x9fc0f000 instead of advancing to
        // 0x9fc10000.
        let last_word = (ENTRIES_PER_PAGE - 1) as u16;
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        page_words[last_word as usize] = 0; // sll r0,r0,0 (nop, Sequential)

        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, last_word, 0, usize::MAX);
        assert!(non_empty);
        let mut instrs_owned = *walked;

        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, last_word, true, false)
            .expect("region must compile");

        // Deliberately pick a page whose base already has PAGE_SIZE's bit
        // set in the byte immediately below it (0x...f000 -> the 0x1000 bit
        // is part of the *next* page's address, not this one's own bits) so
        // a latent OR-vs-ADD bug can't hide behind a page_base that happens
        // to make bor and iadd agree.
        let pc = 0xFFFF_FFFF_9FC0_F000u64 + (last_word as u64) * 4; // = 0x...9fc0fffc
        let gpr = [0u64; 32];
        let (exec, _mem) = seeded_executor(gpr, pc);
        let mut exec = Box::new(exec);
        let status = unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        assert_eq!(status, crate::mips_exec::EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0xFFFF_FFFF_9FC1_0000u64, "must land on the next page, not stay on this one");
    }

    #[test]
    fn nested_regjump_in_delay_slot_compiles_and_innermost_transfer_wins() {
        // "Unusual but legal" on real hardware (a branch/jump's own delay
        // slot is itself a branch/jump) and already supported by the
        // interpreter's nested branch_delay — the analyzer walks the whole
        // slot-chain (`visit_slot`'s recursion) and codegen inlines each
        // nested level via `emit_nested_branch_slot`/`emit_nested_regjump_slot`.
        // Regression: emit_slot_semantics didn't know a nested slot could
        // itself terminate the Cranelift block (every nested branch/regjump
        // arm exits directly, per §6.1.4 — there's no other block in this
        // minimal region for it to jump into), so the outer instruction's
        // own condition-test/exit code kept being emitted unconditionally
        // afterward, tripping Cranelift's verifier ("a terminator
        // instruction was encountered before the end of block").
        //
        // Layout: BEQ r0,r0,5 (always taken) -> slot is JR ra -> JR's own
        // slot is ADDU r10,r8,r9. Real MIPS nested-delay-slot semantics:
        // the *innermost* dispatched transfer wins (branch_delay's target
        // gets overwritten by whichever nested branch_delay call happens
        // last) — so JR ra's target must be what pc ends up at, not BEQ's
        // own (word+6) target, even though BEQ is unconditionally taken.
        // The ADDU must still execute exactly once (JR's mandatory slot).
        let word: u16 = 10;
        let branch_raw = make_i(crate::mips_isa::OP_BEQ, 0, 0, 5);
        let slot_raw = make_r(crate::mips_isa::OP_SPECIAL, 31, 0, 0, 0, crate::mips_isa::FUNCT_JR);
        let inner_slot_raw = make_r(crate::mips_isa::OP_SPECIAL, 8, 9, 10, 0, crate::mips_isa::FUNCT_ADDU);

        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        page_words[word as usize] = branch_raw;
        page_words[word as usize + 1] = slot_raw;
        page_words[word as usize + 2] = inner_slot_raw;

        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, word, 0, 1);
        assert!(non_empty);
        let mut instrs_owned = *walked;

        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word, false, false)
            .expect("nested branch-in-delay-slot region must compile without a verifier error");

        let mut gpr = [0u64; 32];
        gpr[8] = 100;
        gpr[9] = 23;
        gpr[31] = 0xFFFF_FFFF_8000_9000;
        let pc = 0xFFFF_FFFF_8000_0000u64 + (word as u64) * 4;
        let (exec, _mem) = seeded_executor(gpr, pc);
        let mut exec = Box::new(exec);
        let status = unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        assert_eq!(status, crate::mips_exec::EXEC_COMPLETE);
        assert_eq!(exec.core.pc, 0xFFFF_FFFF_8000_9000u64, "innermost (JR ra) transfer must win over the outer BEQ's own target");
        assert_eq!(exec.core.gpr[10], 123, "JR's own mandatory slot (the ADDU) must still execute exactly once");
        assert!(!exec.core.in_delay_slot, "in_delay_slot must be restored/cleared, not left set from the nested dispatch");
    }

    #[test]
    fn self_loop_with_decrement_in_delay_slot_converges_natively() {
        // BGTZ r2,-1 / slot SUBU r2,r2,r3: a tight decrement loop whose
        // taken target is its own word — analyzer::visit's "already
        // visited" no-op for a back-edge to an in-progress word reuses the
        // *same* Cranelift block (§2.2/§3.1 "loops stay native"), so the
        // compiled function runs the whole loop to convergence in one call
        // rather than one iteration per call. This is intentional per the
        // design (not a jitv2_verify-checkable shape — see
        // rules/jitv2/codegen-gotchas.md's self-loop note) but must still
        // produce the architecturally correct final state: same as running
        // the interpreter for the same number of iterations by hand.
        let word: u16 = 10;
        let branch_raw = make_i(crate::mips_isa::OP_BGTZ, 2, 0, 0xFFFF); // BGTZ r2,-1
        let slot_raw = make_r(crate::mips_isa::OP_SPECIAL, 2, 3, 2, 0, crate::mips_isa::FUNCT_SUBU);

        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        page_words[word as usize] = branch_raw;
        page_words[word as usize + 1] = slot_raw;

        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, word, 0, 1);
        assert!(non_empty);
        let mut instrs_owned = *walked;

        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word, false, false)
            .expect("self-loop region must compile");

        // r2=5, r3=2: BGTZ tests r2>0 *before* the slot's decrement runs
        // (real MIPS delay-slot semantics — the condition reads pre-slot
        // state), so the loop keeps taking the branch through r2 =
        // 5,3,1,-1 (decrementing once per pass, condition checked against
        // the value from the *previous* pass) until r2<=0 is seen: exactly
        // 4 decrements, ending at r2 = -3. Cross-checked by hand against
        // exec_bgtz/exec_subu's semantics, not just "whatever the JIT
        // happens to produce" — a change to this expected value should be
        // treated as a red flag, not silently accepted.
        let mut gpr = [0u64; 32];
        gpr[2] = 5;
        gpr[3] = 2;
        let pc = 0xFFFF_FFFF_8000_0000u64 + (word as u64) * 4;
        let (exec, _mem) = seeded_executor(gpr, pc);
        let mut exec = Box::new(exec);
        let status = unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        assert_eq!(status, crate::mips_exec::EXEC_COMPLETE);
        assert_eq!(exec.core.gpr[2] as i64, -3, "native convergence must match hand-computed interpreter-equivalent result");
    }

    #[test]
    fn jalr_matches_interpreter_and_writes_rd() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8000_5000;
        // rd=8: JALR's link register is whatever `rd` decodes to, NOT
        // always r31 (unlike J/JAL/BLTZAL/BGEZAL) — pick a non-r31 register
        // specifically to catch a codegen bug that hardcodes r31.
        assert_branch_matches_interpreter(regjump_layout(crate::mips_isa::FUNCT_JALR, 1, 8), gpr);
    }

    #[test]
    fn jalr_link_register_is_rd_not_always_r31() {
        let pc = 0xFFFF_FFFF_8000_0000u64;
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8000_5000;
        let jit = run_jit_page(&regjump_layout(crate::mips_isa::FUNCT_JALR, 1, 8), gpr, pc, 0, 1, &[])
            .expect("JALR region must compile");
        assert_eq!(jit.gpr[8], pc + 8, "rd (r8) must hold the link address");
        assert_eq!(jit.gpr[31], 0, "r31 must be untouched when rd != 31");
    }

    #[test]
    fn jr_returning_to_ra_matches_interpreter() {
        // The common `jr ra` return idiom.
        let mut gpr = [0u64; 32];
        gpr[31] = 0xFFFF_FFFF_8000_9000;
        assert_branch_matches_interpreter(regjump_layout(crate::mips_isa::FUNCT_JR, 31, 0), gpr);
    }

    // ---- Batch 5: 64-bit ALU ops ------

    #[test]
    fn daddu_dsubu_match_interpreter() {
        alu_rrr_case(crate::mips_isa::FUNCT_DADDU, u64::MAX, 1); // full 64-bit wraparound
        alu_rrr_case(crate::mips_isa::FUNCT_DADDU, 100, 200);
        alu_rrr_case(crate::mips_isa::FUNCT_DSUBU, 0, 1); // full 64-bit wraparound
        alu_rrr_case(crate::mips_isa::FUNCT_DSUBU, 200, 100);
    }

    #[test]
    fn dadd_no_overflow_matches_interpreter() {
        alu_rrr_case(crate::mips_isa::FUNCT_DADD, 100, 200);
    }

    #[test]
    fn dadd_overflow_traps_and_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0x7FFF_FFFF_FFFF_FFFF; // i64::MAX
        gpr[2] = 1;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_DADD);
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let interp = run_interpreter(instr, gpr, pc, &[]);
        let jit = run_jit(instr, gpr, pc, (pc as u16 / 4) & 0x3FF, &[])
            .expect("DADD must be compilable for this test to be meaningful");
        assert_eq!(jit, interp);
        assert_ne!(jit.pc, pc, "64-bit overflow must vector via handle_exception");
    }

    #[test]
    fn dsub_no_overflow_matches_interpreter() {
        alu_rrr_case(crate::mips_isa::FUNCT_DSUB, 200, 100);
    }

    #[test]
    fn dsub_overflow_traps_and_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0x8000_0000_0000_0000; // i64::MIN
        gpr[2] = 1;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_DSUB);
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let interp = run_interpreter(instr, gpr, pc, &[]);
        let jit = run_jit(instr, gpr, pc, (pc as u16 / 4) & 0x3FF, &[])
            .expect("DSUB must be compilable for this test to be meaningful");
        assert_eq!(jit, interp);
        assert_ne!(jit.pc, pc, "64-bit overflow must vector via handle_exception");
    }

    #[test]
    fn dsll_dsrl_dsra_match_interpreter() {
        for sa in [0u32, 1, 31, 63] {
            shift_imm_case(crate::mips_isa::FUNCT_DSLL, 0x8765_4321_FEDC_BA98, sa);
            shift_imm_case(crate::mips_isa::FUNCT_DSRL, 0x8765_4321_FEDC_BA98, sa);
            shift_imm_case(crate::mips_isa::FUNCT_DSRA, 0x8765_4321_FEDC_BA98, sa);
        }
    }

    #[test]
    fn dsll32_dsrl32_dsra32_match_interpreter() {
        // sa field itself only spans 0..31; the "+32" is baked into the emitter.
        for sa in [0u32, 1, 15, 31] {
            shift_imm_case(crate::mips_isa::FUNCT_DSLL32, 0x8765_4321_FEDC_BA98, sa);
            shift_imm_case(crate::mips_isa::FUNCT_DSRL32, 0x8765_4321_FEDC_BA98, sa);
            shift_imm_case(crate::mips_isa::FUNCT_DSRA32, 0x8765_4321_FEDC_BA98, sa);
        }
    }

    #[test]
    fn dsllv_dsrlv_dsrav_match_interpreter() {
        for sa in [0u64, 1, 31, 63, 0xFFFF_FFFF_FFFF_FFFF] {
            // rs supplies the shift amount, masked to 6 bits (0x3F) — exercise
            // a value with high bits set to confirm the mask is applied.
            shift_var_case(crate::mips_isa::FUNCT_DSLLV, sa, 0x8765_4321_FEDC_BA98);
            shift_var_case(crate::mips_isa::FUNCT_DSRLV, sa, 0x8765_4321_FEDC_BA98);
            shift_var_case(crate::mips_isa::FUNCT_DSRAV, sa, 0x8765_4321_FEDC_BA98);
        }
    }

    // ---- Batch 4: mult/div/hi-lo ------

    #[test]
    fn mfhi_mflo_mthi_mtlo_match_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xDEAD_BEEF_1234_5678;
        for funct in [crate::mips_isa::FUNCT_MFHI, crate::mips_isa::FUNCT_MFLO] {
            let instr = make_r(crate::mips_isa::OP_SPECIAL, 0, 0, 3, 0, funct);
            assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
        }
        for funct in [crate::mips_isa::FUNCT_MTHI, crate::mips_isa::FUNCT_MTLO] {
            let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 0, 0, 0, funct);
            assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
        }
    }

    fn mult_case(funct: u32, rs_val: u64, rt_val: u64) {
        let mut gpr = [0u64; 32];
        gpr[1] = rs_val;
        gpr[2] = rt_val;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 0, 0, funct);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn mult_matches_interpreter() {
        mult_case(crate::mips_isa::FUNCT_MULT, 1000, 2000);
        mult_case(crate::mips_isa::FUNCT_MULT, 0xFFFF_FFFF, 0xFFFF_FFFF); // (-1)*(-1) = 1
        mult_case(crate::mips_isa::FUNCT_MULT, 0x7FFF_FFFF, 0x7FFF_FFFF); // large positive product
    }

    #[test]
    fn multu_matches_interpreter() {
        mult_case(crate::mips_isa::FUNCT_MULTU, 1000, 2000);
        mult_case(crate::mips_isa::FUNCT_MULTU, 0xFFFF_FFFF, 0xFFFF_FFFF);
        mult_case(crate::mips_isa::FUNCT_MULTU, 0x8000_0000, 2);
    }

    fn div_case(funct: u32, rs_val: u64, rt_val: u64) {
        let mut gpr = [0u64; 32];
        gpr[1] = rs_val;
        gpr[2] = rt_val;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 0, 0, funct);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn div_matches_interpreter() {
        div_case(crate::mips_isa::FUNCT_DIV, 100, 7);
        div_case(crate::mips_isa::FUNCT_DIV, 0xFFFF_FFFF_FFFF_FFFF, 3); // -1 / 3
        div_case(crate::mips_isa::FUNCT_DIV, 0xFFFF_FFFF_8000_0000, 0xFFFF_FFFF_FFFF_FFFF); // i32::MIN / -1
    }

    #[test]
    fn div_by_zero_is_noop_matches_interpreter() {
        div_case(crate::mips_isa::FUNCT_DIV, 100, 0);
        div_case(crate::mips_isa::FUNCT_DIVU, 100, 0);
    }

    #[test]
    fn divu_matches_interpreter() {
        div_case(crate::mips_isa::FUNCT_DIVU, 100, 7);
        div_case(crate::mips_isa::FUNCT_DIVU, 0xFFFF_FFFF, 3);
    }

    #[test]
    fn dmult_matches_interpreter() {
        // Same shape as MULT's test cases, but at 64-bit width — the
        // large-positive/-1-times--1 cases here genuinely produce a
        // nonzero hi half (unlike MULT, where every 32x32 product fits in
        // the lo half alone), exercising smulhi, not just imul.
        mult_case(crate::mips_isa::FUNCT_DMULT, 1_000_000_000_000, 2_000_000_000_000);
        mult_case(crate::mips_isa::FUNCT_DMULT, 0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF); // (-1)*(-1) = 1
        mult_case(crate::mips_isa::FUNCT_DMULT, 0x7FFF_FFFF_FFFF_FFFF, 0x7FFF_FFFF_FFFF_FFFF); // i64::MAX^2
        mult_case(crate::mips_isa::FUNCT_DMULT, 0xFFFF_FFFF_8000_0000, 0xFFFF_FFFF_8000_0000); // i64::MIN^2
    }

    #[test]
    fn dmultu_matches_interpreter() {
        mult_case(crate::mips_isa::FUNCT_DMULTU, 1_000_000_000_000, 2_000_000_000_000);
        mult_case(crate::mips_isa::FUNCT_DMULTU, 0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF);
        mult_case(crate::mips_isa::FUNCT_DMULTU, 0x8000_0000_0000_0000, 2);
    }

    #[test]
    fn ddiv_matches_interpreter() {
        div_case(crate::mips_isa::FUNCT_DDIV, 100, 7);
        div_case(crate::mips_isa::FUNCT_DDIV, 0xFFFF_FFFF_FFFF_FFFF, 3); // -1 / 3
    }

    #[test]
    fn ddiv_i64_min_over_neg1_is_noop_matches_interpreter() {
        // Unlike DIV's i32::MIN/-1 case (which computes a defined
        // wrapping-division result), exec_ddiv treats i64::MIN/-1 as a
        // no-op, same as divide-by-zero — this is the behavior
        // emit_ddiv_impl's overflow_block must reproduce (jump straight to
        // skip_block, no write to lo/hi at all), not DIV's "compute
        // MIN/wrap" pattern.
        div_case(crate::mips_isa::FUNCT_DDIV, 0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF);
    }

    #[test]
    fn ddiv_by_zero_is_noop_matches_interpreter() {
        div_case(crate::mips_isa::FUNCT_DDIV, 100, 0);
        div_case(crate::mips_isa::FUNCT_DDIVU, 100, 0);
    }

    #[test]
    fn ddivu_matches_interpreter() {
        div_case(crate::mips_isa::FUNCT_DDIVU, 100, 7);
        div_case(crate::mips_isa::FUNCT_DDIVU, 0xFFFF_FFFF_FFFF_FFFF, 3);
    }

    // ---- Batch 3: remaining loads/stores ------

    #[test]
    fn lb_lbu_sign_and_zero_extend_match_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0000;
        let lb = make_i(crate::mips_isa::OP_LB, 1, 2, 0);
        let lbu = make_i(crate::mips_isa::OP_LBU, 1, 2, 0);
        // High bit set -> LB sign-extends negative, LBU zero-extends.
        assert_jit_matches_interpreter_mem(lb, gpr, 0xFFFF_FFFF_8000_1000, &[(0xFFFF_FFFF_8010_0000, 0xFF)]);
        assert_jit_matches_interpreter_mem(lbu, gpr, 0xFFFF_FFFF_8000_1000, &[(0xFFFF_FFFF_8010_0000, 0xFF)]);
        gpr[1] = 0xFFFF_FFFF_8010_0000;
        assert_jit_matches_interpreter_mem(lb, gpr, 0xFFFF_FFFF_8000_1000, &[(0xFFFF_FFFF_8010_0000, 0x7F)]);
    }

    #[test]
    fn lh_lhu_sign_and_zero_extend_match_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0000;
        let lh = make_i(crate::mips_isa::OP_LH, 1, 2, 0);
        let lhu = make_i(crate::mips_isa::OP_LHU, 1, 2, 0);
        assert_jit_matches_interpreter_mem(lh, gpr, 0xFFFF_FFFF_8000_1000, &[(0xFFFF_FFFF_8010_0000, 0xFFFF)]);
        assert_jit_matches_interpreter_mem(lhu, gpr, 0xFFFF_FFFF_8000_1000, &[(0xFFFF_FFFF_8010_0000, 0xFFFF)]);
    }

    #[test]
    fn lwu_zero_extends_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0000;
        let instr = make_i(crate::mips_isa::OP_LWU, 1, 2, 0);
        assert_jit_matches_interpreter_mem(instr, gpr, 0xFFFF_FFFF_8000_1000, &[(0xFFFF_FFFF_8010_0000, 0x8000_0001)]);
    }

    #[test]
    fn ld_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0000;
        let instr = make_i(crate::mips_isa::OP_LD, 1, 2, 0);
        // mem_init writes 32-bit words (set_word); LD reads 8 bytes, so seed
        // both halves of the doubleword.
        assert_jit_matches_interpreter_mem(instr, gpr, 0xFFFF_FFFF_8000_1000,
            &[(0xFFFF_FFFF_8010_0000, 0xDEAD_BEEF), (0xFFFF_FFFF_8010_0004, 0xCAFE_BABE)]);
    }

    #[test]
    fn lwl_matches_interpreter_at_every_byte_offset() {
        // LWL merges the "left" (high-address) bytes of an unaligned word
        // into rt's existing low bytes — behavior genuinely differs at each
        // of the 4 possible byte_offset values (virt_addr & 3), unlike
        // every other load in this file, so all 4 need covering, not just
        // one representative case.
        let aligned = 0xFFFF_FFFF_8010_0000u64;
        for offset in 0..4u64 {
            let mut gpr = [0u64; 32];
            gpr[1] = aligned + offset; // base; imm=0
            gpr[2] = 0x1122_3344_5566_7788; // rt's pre-existing value, must partially survive
            let instr = make_i(crate::mips_isa::OP_LWL, 1, 2, 0);
            assert_jit_matches_interpreter_mem(instr, gpr, 0xFFFF_FFFF_8000_1000,
                &[(aligned, 0xDEAD_BEEF)]);
        }
    }

    #[test]
    fn lwr_matches_interpreter_at_every_byte_offset() {
        let aligned = 0xFFFF_FFFF_8010_0000u64;
        for offset in 0..4u64 {
            let mut gpr = [0u64; 32];
            gpr[1] = aligned + offset;
            gpr[2] = 0x1122_3344_5566_7788;
            let instr = make_i(crate::mips_isa::OP_LWR, 1, 2, 0);
            assert_jit_matches_interpreter_mem(instr, gpr, 0xFFFF_FFFF_8000_1000,
                &[(aligned, 0xDEAD_BEEF)]);
        }
    }

    #[test]
    fn lwl_lwr_combine_to_a_full_unaligned_word_load_matches_interpreter() {
        // The canonical compiler-generated idiom for an unaligned word
        // load: LWL rt,0(base) then LWR rt,3(base) reconstructs the full
        // word at `base` into rt regardless of base's own alignment —
        // exercises both emitters chained through the same register, the
        // way real generated code actually uses them (never just one in
        // isolation). Both engines are compared against each other, not a
        // hand-derived expected constant, to avoid this test's own byte
        // arithmetic being a second place to get the shift/mask direction
        // wrong in exactly the same way the emitter could.
        let base = 0xFFFF_FFFF_8010_0001u64; // deliberately unaligned
        let mut gpr = [0u64; 32];
        gpr[1] = base;
        let lwl = make_i(crate::mips_isa::OP_LWL, 1, 2, 0);
        let lwr = make_i(crate::mips_isa::OP_LWR, 1, 2, 3);
        let mem_init: &[(u64, u32)] = &[(0xFFFF_FFFF_8010_0000u64, 0xDEAD_BEEFu32), (0xFFFF_FFFF_8010_0004u64, 0xCAFE_BABEu32)];
        let pc = 0xFFFF_FFFF_8000_1000u64;

        let (mut interp_exec, mem) = seeded_executor_over(MockMemory::new_not_compilable(), gpr, pc);
        for &(addr, val) in mem_init { mem.set_word(addr, val); }
        interp_exec.exec(lwl);
        interp_exec.exec(lwr);
        let interp = CoreSnapshot::capture(&interp_exec.core);

        let page = vec![(0u16, lwl), (1u16, lwr)];
        let jit = run_jit_page(&page, gpr, pc, 0, 2, mem_init)
            .expect("LWL+LWR region must compile");
        assert_eq!(jit, interp, "LWL then LWR must reconstruct the full unaligned word identically on both engines");
    }

    #[test]
    fn ldl_matches_interpreter_at_every_byte_offset() {
        let aligned = 0xFFFF_FFFF_8010_0000u64;
        let mem_init = &[(aligned, 0xDEAD_BEEFu32), (aligned + 4, 0xCAFE_BABEu32)];
        for offset in 0..8u64 {
            let mut gpr = [0u64; 32];
            gpr[1] = aligned + offset;
            gpr[2] = 0x1122_3344_5566_7788;
            let instr = make_i(crate::mips_isa::OP_LDL, 1, 2, 0);
            assert_jit_matches_interpreter_mem(instr, gpr, 0xFFFF_FFFF_8000_1000, mem_init);
        }
    }

    #[test]
    fn ldr_matches_interpreter_at_every_byte_offset() {
        let aligned = 0xFFFF_FFFF_8010_0000u64;
        let mem_init = &[(aligned, 0xDEAD_BEEFu32), (aligned + 4, 0xCAFE_BABEu32)];
        for offset in 0..8u64 {
            let mut gpr = [0u64; 32];
            gpr[1] = aligned + offset;
            gpr[2] = 0x1122_3344_5566_7788;
            let instr = make_i(crate::mips_isa::OP_LDR, 1, 2, 0);
            assert_jit_matches_interpreter_mem(instr, gpr, 0xFFFF_FFFF_8000_1000, mem_init);
        }
    }

    /// Run a single store instruction through both engines from identical
    /// pre-seeded doubleword-aligned memory (`aligned8` = both the pre-seed
    /// address and what gets read back after) and assert both the register
    /// state AND the resulting memory bytes match — the plain
    /// `assert_jit_matches_interpreter`/`run_jit`-style helpers only ever
    /// compare `CoreSnapshot` (registers), which would miss a real bug in
    /// exactly the class of instruction being tested here: the entire point
    /// of SWL/SWR/SDL/SDR is *which bytes* get written, not any register
    /// effect (there is none — rt is read-only for these). Mirrors
    /// `run_interpreter`/`run_jit`'s exact construction, just keeping the
    /// `mem` handle alive afterward instead of discarding it.
    fn assert_masked_store_matches_interpreter(instr: u32, gpr: [u64; 32], pc: u64, aligned8: u64, seed: u64) {
        let mem_init = [(aligned8, (seed >> 32) as u32), (aligned8 + 4, seed as u32)];

        let (mut interp_exec, interp_mem) = seeded_executor_over(MockMemory::new_not_compilable(), gpr, pc);
        for &(addr, val) in &mem_init { interp_mem.set_word(addr, val); }
        interp_exec.exec(instr);
        let interp_snapshot = CoreSnapshot::capture(&interp_exec.core);
        let interp_result = ((interp_mem.get_word(aligned8) as u64) << 32) | interp_mem.get_word(aligned8 + 4) as u64;

        let word_offset: u16 = 0;
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[word_offset as usize] = instr;
        let page_base = (pc & !(PAGE_SIZE as u64 - 1)) as u32;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page, word_offset, page_base, 1);
        assert!(non_empty, "entry instruction must not be excluded — check the test's instruction encoding");
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word_offset, true, false)
            .expect("masked store must be compilable for this test to be meaningful");
        let (jit_exec, jit_mem) = seeded_executor(gpr, pc);
        let mut jit_exec = Box::new(jit_exec);
        for &(addr, val) in &mem_init { jit_mem.set_word(addr, val); }
        jit_exec.install_jit_hooks();
        unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        std::mem::forget(codegen);
        let jit_snapshot = CoreSnapshot::capture(&jit_exec.core);
        let jit_result = ((jit_mem.get_word(aligned8) as u64) << 32) | jit_mem.get_word(aligned8 + 4) as u64;

        assert_eq!(jit_snapshot, interp_snapshot, "register state diverged for instr=0x{:08x}", instr);
        assert_eq!(jit_result, interp_result, "resulting doubleword diverged for instr=0x{:08x}: jit=0x{:016x} interp=0x{:016x}", instr, jit_result, interp_result);
    }

    #[test]
    fn swl_matches_interpreter_at_every_byte_offset() {
        let aligned8 = 0xFFFF_FFFF_8010_0000u64;
        let seed = 0x1122_3344_5566_7788u64;
        for offset in 0..4u64 {
            let mut gpr = [0u64; 32];
            gpr[1] = aligned8 + offset;
            gpr[2] = 0xAABB_CCDD;
            let instr = make_i(crate::mips_isa::OP_SWL, 1, 2, 0);
            assert_masked_store_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000u64, aligned8, seed);
        }
    }

    #[test]
    fn swr_matches_interpreter_at_every_byte_offset() {
        let aligned8 = 0xFFFF_FFFF_8010_0000u64;
        let seed = 0x1122_3344_5566_7788u64;
        for offset in 0..4u64 {
            let mut gpr = [0u64; 32];
            gpr[1] = aligned8 + offset;
            gpr[2] = 0xAABB_CCDD;
            let instr = make_i(crate::mips_isa::OP_SWR, 1, 2, 0);
            assert_masked_store_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000u64, aligned8, seed);
        }
    }

    #[test]
    fn sdl_matches_interpreter_at_every_byte_offset() {
        let aligned8 = 0xFFFF_FFFF_8010_0000u64;
        let seed = 0x1122_3344_5566_7788u64;
        for offset in 0..8u64 {
            let mut gpr = [0u64; 32];
            gpr[1] = aligned8 + offset;
            gpr[2] = 0xAABB_CCDD_EEFF_0011;
            let instr = make_i(crate::mips_isa::OP_SDL, 1, 2, 0);
            assert_masked_store_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000u64, aligned8, seed);
        }
    }

    #[test]
    fn sdr_matches_interpreter_at_every_byte_offset() {
        let aligned8 = 0xFFFF_FFFF_8010_0000u64;
        let seed = 0x1122_3344_5566_7788u64;
        for offset in 0..8u64 {
            let mut gpr = [0u64; 32];
            gpr[1] = aligned8 + offset;
            gpr[2] = 0xAABB_CCDD_EEFF_0011;
            let instr = make_i(crate::mips_isa::OP_SDR, 1, 2, 0);
            assert_masked_store_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000u64, aligned8, seed);
        }
    }

    #[test]
    fn swl_swr_combine_to_a_full_unaligned_word_store_matches_interpreter() {
        // The canonical compiler-generated idiom for an unaligned word
        // store: SWL rt,0(base) then SWR rt,3(base) writes the full word at
        // `base` regardless of base's own alignment — exercises both
        // emitters chained through the same shared memory the way real
        // generated code actually uses them.
        let aligned8 = 0xFFFF_FFFF_8010_0000u64;
        let seed = 0x1122_3344_5566_7788u64;
        let base = aligned8 + 1; // deliberately unaligned
        let mut gpr = [0u64; 32];
        gpr[1] = base;
        gpr[2] = 0xDEAD_BEEF;
        let swl = make_i(crate::mips_isa::OP_SWL, 1, 2, 0);
        let swr = make_i(crate::mips_isa::OP_SWR, 1, 2, 3);
        let pc = 0xFFFF_FFFF_8000_1000u64;

        let mem_init = [(aligned8, (seed >> 32) as u32), (aligned8 + 4, seed as u32)];

        let (mut interp_exec, interp_mem) = seeded_executor_over(MockMemory::new_not_compilable(), gpr, pc);
        for &(addr, val) in &mem_init { interp_mem.set_word(addr, val); }
        interp_exec.exec(swl);
        interp_exec.exec(swr);
        let interp_snapshot = CoreSnapshot::capture(&interp_exec.core);
        let interp_result = ((interp_mem.get_word(aligned8) as u64) << 32) | interp_mem.get_word(aligned8 + 4) as u64;

        let page = [(0u16, swl), (1u16, swr)];
        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(w, r) in &page { page_words[w as usize] = r; }
        let page_base = (pc & !(PAGE_SIZE as u64 - 1)) as u32;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, 2);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
            .expect("SWL+SWR region must compile");
        let (jit_exec, jit_mem) = seeded_executor(gpr, pc);
        let mut jit_exec = Box::new(jit_exec);
        for &(addr, val) in &mem_init { jit_mem.set_word(addr, val); }
        jit_exec.install_jit_hooks();
        unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        std::mem::forget(codegen);
        let jit_snapshot = CoreSnapshot::capture(&jit_exec.core);
        let jit_result = ((jit_mem.get_word(aligned8) as u64) << 32) | jit_mem.get_word(aligned8 + 4) as u64;

        assert_eq!(jit_snapshot, interp_snapshot);
        assert_eq!(jit_result, interp_result, "SWL then SWR must write the same bytes on both engines: jit=0x{:016x} interp=0x{:016x}", jit_result, interp_result);
    }

    #[test]
    #[cfg(feature = "mips4")]
    fn movz_matches_interpreter_taken_and_not_taken() {
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_MOVZ);
        for rt in [0u64, 1, 0xFFFF_FFFF_FFFF_FFFF] {
            let mut gpr = [0u64; 32];
            gpr[1] = 0xDEAD_BEEF_1234_5678;
            gpr[2] = rt;
            gpr[3] = 0x1111_1111_1111_1111; // pre-existing rd, must survive when not taken
            assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
        }
    }

    #[test]
    #[cfg(feature = "mips4")]
    fn movn_matches_interpreter_taken_and_not_taken() {
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_MOVN);
        for rt in [0u64, 1, 0xFFFF_FFFF_FFFF_FFFF] {
            let mut gpr = [0u64; 32];
            gpr[1] = 0xDEAD_BEEF_1234_5678;
            gpr[2] = rt;
            gpr[3] = 0x1111_1111_1111_1111;
            assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
        }
    }

    #[test]
    #[cfg(feature = "mips4")]
    fn movz_writing_r0_is_a_noop() {
        let mut gpr = [0u64; 32];
        gpr[1] = 5;
        gpr[2] = 0; // condition true
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 0, 0, crate::mips_isa::FUNCT_MOVZ);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    #[cfg(feature = "mips4")]
    fn movci_matches_interpreter_across_all_cc_and_tf_combinations() {
        for cc in 0u32..8 {
            for tf in [false, true] {
                for cc_actual_value in [false, true] {
                    let mut gpr = [0u64; 32];
                    gpr[1] = 0xCAFE_BABE_0000_0001;
                    gpr[3] = 0x2222_2222_2222_2222;
                    // rd=3, rs=1, cc field in bits [20:18], tf in bit 16.
                    let instr = crate::mips_isa::OP_SPECIAL << 26
                        | (1 << 21) // rs
                        | (0 << 16) // rt bit0 = tf, overwritten below
                        | (3 << 11) // rd
                        | crate::mips_isa::FUNCT_MOVCI;
                    let instr = (instr & !(0x7 << 18)) | (cc << 18);
                    let instr = (instr & !(1 << 16)) | ((tf as u32) << 16);

                    let (mut interp_exec, _) = seeded_executor_over(MockMemory::new_not_compilable(), gpr, 0xFFFF_FFFF_8000_1000);
                    interp_exec.core.fpu_fcsr = 0;
                    interp_exec.core.set_fpu_cc(cc, cc_actual_value);
                    interp_exec.exec(instr);
                    let interp_snapshot = CoreSnapshot::capture(&interp_exec.core);

                    let word_offset = 0u16;
                    let pc = 0xFFFF_FFFF_8000_1000u64;
                    let mut page = [0u32; ENTRIES_PER_PAGE];
                    page[word_offset as usize] = instr;
                    let page_base = (pc & !(PAGE_SIZE as u64 - 1)) as u32;
                    let mut analyzer = Analyzer::new();
                    let (walked, non_empty) = analyzer.walk_bounded(&page, word_offset, page_base, 1);
                    assert!(non_empty);
                    let mut instrs_owned = *walked;
                    let mut codegen = Codegen::new();
                    let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word_offset, true, false)
                        .expect("MOVCI must be compilable for this test to be meaningful");
                    let (jit_exec, _) = seeded_executor(gpr, pc);
                    let mut jit_exec = Box::new(jit_exec);
                    jit_exec.core.fpu_fcsr = 0;
                    jit_exec.core.set_fpu_cc(cc, cc_actual_value);
                    jit_exec.install_jit_hooks();
                    unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
                    std::mem::forget(codegen);
                    let jit_snapshot = CoreSnapshot::capture(&jit_exec.core);

                    assert_eq!(jit_snapshot, interp_snapshot,
                        "MOVCI diverged for cc={} tf={} cc_actual={}", cc, tf, cc_actual_value);
                }
            }
        }
    }

    /// MOVCF.fmt (FMOVCF.s/FMOVCF.d): same cc/tf/cc_actual sweep as
    /// `movci_matches_interpreter_across_all_cc_and_tf_combinations`, but
    /// gating an FPR-to-FPR copy (`fs`=rd-position, `fd`=sa-position) instead
    /// of a GPR write, and exercised across both FR modes since fs/fd
    /// register addressing differs between them (see `emit_read_fpr_l`/
    /// `emit_write_fpr_l`'s FrMode handling).
    #[cfg(feature = "mips4")]
    fn fmovcf_case(fmt: u32, fr1: bool) {
        for cc in 0u32..8 {
            for tf in [false, true] {
                for cc_actual_value in [false, true] {
                    let mut fpr = [0u64; 32];
                    fpr[1] = 0xCAFE_BABE_1234_5678; // fs
                    fpr[3] = 0x2222_2222_2222_2222; // pre-existing fd, must survive when not taken
                    // fs=rd-position(1), fd=sa-position(3); cc/tf packed into
                    // the rt-position field: bits[20:18]=cc, bit16=tf.
                    let cc_tf = (cc << 2) | (tf as u32);
                    let instr = make_r(crate::mips_isa::OP_COP1, fmt, cc_tf, 1, 3, crate::mips_isa::FUNCT_FMOVCF);

                    let pc = 0xFFFF_FFFF_8000_1000u64;
                    let word_offset = (pc as u16 / 4) & 0x3FF;

                    let (mut interp_exec, _) = fpu_seeded_executor([0u64; 32], fpr, pc, fr1);
                    interp_exec.core.fpu_fcsr = 0;
                    interp_exec.core.set_fpu_cc(cc, cc_actual_value);
                    interp_exec.exec(instr);
                    let interp_snapshot = CoreSnapshot::capture(&interp_exec.core);

                    let mut page = [0u32; ENTRIES_PER_PAGE];
                    page[word_offset as usize] = instr;
                    let mut analyzer = Analyzer::new();
                    let (walked, non_empty) = analyzer.walk_bounded(&page, word_offset, 0, 1);
                    assert!(non_empty);
                    let mut instrs_owned = *walked;
                    let mut codegen = Codegen::new();
                    let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word_offset, fr1, false)
                        .expect("FMOVCF must be compilable for this test to be meaningful");

                    let (jit_exec, _) = fpu_seeded_executor([0u64; 32], fpr, pc, fr1);
                    let mut jit_exec = Box::new(jit_exec);
                    jit_exec.core.fpu_fcsr = 0;
                    jit_exec.core.set_fpu_cc(cc, cc_actual_value);
                    jit_exec.install_jit_hooks();
                    unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
                    std::mem::forget(codegen);
                    let jit_snapshot = CoreSnapshot::capture(&jit_exec.core);

                    assert_eq!(jit_snapshot, interp_snapshot,
                        "FMOVCF diverged for fmt={} fr1={} cc={} tf={} cc_actual={}", fmt, fr1, cc, tf, cc_actual_value);
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "mips4")]
    fn fmovcf_s_matches_interpreter_across_all_cc_and_tf_combinations() {
        fmovcf_case(crate::mips_isa::RS_S, false);
        fmovcf_case(crate::mips_isa::RS_S, true);
    }

    #[test]
    #[cfg(feature = "mips4")]
    fn fmovcf_d_matches_interpreter_across_all_cc_and_tf_combinations() {
        fmovcf_case(crate::mips_isa::RS_D, false);
        fmovcf_case(crate::mips_isa::RS_D, true);
    }

    fn trap_case_rr(funct: u32, rs_val: u64, rt_val: u64) {
        let mut gpr = [0u64; 32];
        gpr[1] = rs_val;
        gpr[2] = rt_val;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 0, 0, funct);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn tge_matches_interpreter_taken_and_not_taken() {
        trap_case_rr(crate::mips_isa::FUNCT_TGE, 5, 3); // taken: 5 >= 3
        trap_case_rr(crate::mips_isa::FUNCT_TGE, 3, 5); // not taken
        trap_case_rr(crate::mips_isa::FUNCT_TGE, 5, 5); // taken: equal
        trap_case_rr(crate::mips_isa::FUNCT_TGE, 0xFFFF_FFFF_FFFF_FFFF, 0); // -1 < 0 signed: not taken
    }

    #[test]
    fn tgeu_matches_interpreter_taken_and_not_taken() {
        trap_case_rr(crate::mips_isa::FUNCT_TGEU, 5, 3);
        trap_case_rr(crate::mips_isa::FUNCT_TGEU, 3, 5);
        trap_case_rr(crate::mips_isa::FUNCT_TGEU, 0xFFFF_FFFF_FFFF_FFFF, 0); // unsigned: taken
    }

    #[test]
    fn tlt_matches_interpreter_taken_and_not_taken() {
        trap_case_rr(crate::mips_isa::FUNCT_TLT, 3, 5); // taken
        trap_case_rr(crate::mips_isa::FUNCT_TLT, 5, 3); // not taken
        trap_case_rr(crate::mips_isa::FUNCT_TLT, 0xFFFF_FFFF_FFFF_FFFF, 0); // -1 < 0 signed: taken
    }

    #[test]
    fn tltu_matches_interpreter_taken_and_not_taken() {
        trap_case_rr(crate::mips_isa::FUNCT_TLTU, 3, 5);
        trap_case_rr(crate::mips_isa::FUNCT_TLTU, 5, 3);
        trap_case_rr(crate::mips_isa::FUNCT_TLTU, 0xFFFF_FFFF_FFFF_FFFF, 0); // unsigned: not taken
    }

    #[test]
    fn teq_matches_interpreter_taken_and_not_taken() {
        trap_case_rr(crate::mips_isa::FUNCT_TEQ, 7, 7);
        trap_case_rr(crate::mips_isa::FUNCT_TEQ, 7, 8);
    }

    #[test]
    fn tne_matches_interpreter_taken_and_not_taken() {
        trap_case_rr(crate::mips_isa::FUNCT_TNE, 7, 8);
        trap_case_rr(crate::mips_isa::FUNCT_TNE, 7, 7);
    }

    fn trap_case_ri(rt: u32, rs_val: u64, imm: u16) {
        let mut gpr = [0u64; 32];
        gpr[1] = rs_val;
        let instr = make_i(crate::mips_isa::OP_REGIMM, 1, rt, imm);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn tgei_matches_interpreter_taken_and_not_taken() {
        trap_case_ri(crate::mips_isa::RT_TGEI, 5, 3);
        trap_case_ri(crate::mips_isa::RT_TGEI, 3, 5);
        trap_case_ri(crate::mips_isa::RT_TGEI, 0xFFFF_FFFF_FFFF_FFFF, 0); // -1 >= 0 signed: not taken
    }

    #[test]
    fn tgeiu_matches_interpreter_taken_and_not_taken() {
        trap_case_ri(crate::mips_isa::RT_TGEIU, 5, 3);
        trap_case_ri(crate::mips_isa::RT_TGEIU, 3, 5);
        // imm is sign-extended to -1 (0xFFFF...FFFF) then compared unsigned:
        // rs=5 < 0xFFFF...FFFF unsigned -> not taken.
        trap_case_ri(crate::mips_isa::RT_TGEIU, 5, 0xFFFF);
    }

    #[test]
    fn tlti_matches_interpreter_taken_and_not_taken() {
        trap_case_ri(crate::mips_isa::RT_TLTI, 3, 5);
        trap_case_ri(crate::mips_isa::RT_TLTI, 5, 3);
        trap_case_ri(crate::mips_isa::RT_TLTI, 0xFFFF_FFFF_FFFF_FFFF, 0); // -1 < 0 signed: taken
    }

    #[test]
    fn tltiu_matches_interpreter_taken_and_not_taken() {
        trap_case_ri(crate::mips_isa::RT_TLTIU, 3, 5);
        trap_case_ri(crate::mips_isa::RT_TLTIU, 5, 3);
        trap_case_ri(crate::mips_isa::RT_TLTIU, 5, 0xFFFF); // 5 < 0xFFFF...FFFF unsigned: taken
    }

    #[test]
    fn teqi_matches_interpreter_taken_and_not_taken() {
        trap_case_ri(crate::mips_isa::RT_TEQI, 7, 7);
        trap_case_ri(crate::mips_isa::RT_TEQI, 7, 8);
    }

    #[test]
    fn tnei_matches_interpreter_taken_and_not_taken() {
        trap_case_ri(crate::mips_isa::RT_TNEI, 7, 8);
        trap_case_ri(crate::mips_isa::RT_TNEI, 7, 7);
    }

    #[test]
    fn trap_taken_vectors_via_handle_exception_and_matches_interpreter() {
        // Confirms the trapping path actually diverts PC through the
        // exception vector (Cause/EPC/Status.EXL all captured by
        // CoreSnapshot) rather than just returning the same jit/interp
        // equality trivially because neither engine did anything.
        let mut gpr = [0u64; 32];
        gpr[1] = 7;
        gpr[2] = 7;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 0, 0, crate::mips_isa::FUNCT_TEQ);
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let interp = run_interpreter(instr, gpr, pc, &[]);
        let jit = run_jit(instr, gpr, pc, (pc as u16 / 4) & 0x3FF, &[])
            .expect("TEQ must be compilable for this test to be meaningful");
        assert_eq!(jit, interp);
        assert_ne!(jit.pc, pc, "trap must vector via handle_exception");
    }

    #[test]
    fn sync_is_a_true_noop_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0x1234;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 0, 0, 0, 0, crate::mips_isa::FUNCT_SYNC);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    #[cfg(feature = "mips4")]
    fn pref_is_a_true_noop_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0000;
        let instr = make_i(crate::mips_isa::OP_PREF, 1, 0, 0);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn sb_sh_sd_match_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0000;
        gpr[2] = 0xFFFF_FFFF_FFFF_FFFF;
        for op in [crate::mips_isa::OP_SB, crate::mips_isa::OP_SH, crate::mips_isa::OP_SD] {
            let instr = make_i(op, 1, 2, 0);
            assert_jit_matches_interpreter_mem(instr, gpr, 0xFFFF_FFFF_8000_1000, &[]);
        }
    }

    #[test]
    fn lh_unaligned_raises_adel_and_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8010_0001; // odd address, misaligned for halfword
        let instr = make_i(crate::mips_isa::OP_LH, 1, 2, 0);
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let interp = run_interpreter(instr, gpr, pc, &[]);
        let jit = run_jit(instr, gpr, pc, (pc as u16 / 4) & 0x3FF, &[])
            .expect("LH must be compilable for this test to be meaningful");
        assert_eq!(jit, interp);
        assert_ne!(jit.pc, pc);
    }

    // ---- Batch 1: ALU reg-reg ops + shifts ------

    fn alu_rrr_case(funct: u32, rs_val: u64, rt_val: u64) {
        let mut gpr = [0u64; 32];
        gpr[1] = rs_val;
        gpr[2] = rt_val;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, funct);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn subu_matches_interpreter() {
        alu_rrr_case(crate::mips_isa::FUNCT_SUBU, 5, 20); // underflow case too
        alu_rrr_case(crate::mips_isa::FUNCT_SUBU, 0x8000_0000, 1); // sign-extension boundary
    }

    #[test]
    fn and_or_xor_nor_match_interpreter() {
        for funct in [crate::mips_isa::FUNCT_AND, crate::mips_isa::FUNCT_OR, crate::mips_isa::FUNCT_XOR, crate::mips_isa::FUNCT_NOR] {
            alu_rrr_case(funct, 0xFFFF_FFFF_0000_0000, 0x0000_FFFF_FFFF_0000);
            alu_rrr_case(funct, 0, 0);
            alu_rrr_case(funct, u64::MAX, 0);
        }
    }

    #[test]
    fn slt_sltu_match_interpreter() {
        // Signed vs. unsigned comparison must diverge for this pair: as i64,
        // -1 < 1; as u64, u64::MAX > 1 -- SLT and SLTU must disagree here.
        alu_rrr_case(crate::mips_isa::FUNCT_SLT, u64::MAX, 1);
        alu_rrr_case(crate::mips_isa::FUNCT_SLTU, u64::MAX, 1);
        alu_rrr_case(crate::mips_isa::FUNCT_SLT, 1, 1);
        alu_rrr_case(crate::mips_isa::FUNCT_SLTU, 1, 1);
    }

    fn shift_imm_case(funct: u32, rt_val: u64, sa: u32) {
        let mut gpr = [0u64; 32];
        gpr[2] = rt_val;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 0, 2, 3, sa, funct);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn sll_srl_sra_match_interpreter() {
        for sa in [0u32, 1, 15, 31] {
            shift_imm_case(crate::mips_isa::FUNCT_SLL, 0x8765_4321_FEDC_BA98, sa);
            shift_imm_case(crate::mips_isa::FUNCT_SRL, 0x8765_4321_FEDC_BA98, sa);
            shift_imm_case(crate::mips_isa::FUNCT_SRA, 0x8765_4321_FEDC_BA98, sa);
        }
    }

    #[test]
    fn sll_zero_is_true_nop() {
        // raw == 0 (sll r0, r0, 0) is the canonical NOP encoding.
        shift_imm_case(crate::mips_isa::FUNCT_SLL, 0, 0);
    }

    fn shift_var_case(funct: u32, rs_val: u64, rt_val: u64) {
        let mut gpr = [0u64; 32];
        gpr[1] = rs_val;
        gpr[2] = rt_val;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, funct);
        assert_jit_matches_interpreter(instr, gpr, 0xFFFF_FFFF_8000_1000);
    }

    #[test]
    fn add_no_overflow_matches_interpreter() {
        alu_rrr_case(crate::mips_isa::FUNCT_ADD, 10, 20);
    }

    #[test]
    fn add_overflow_traps_and_matches_interpreter() {
        // 0x7FFFFFFF + 1 overflows a 32-bit signed add -> EXC_OV, delivered
        // via handle_exception (§4.2) -- pc must vector, EPC/Cause must match.
        let mut gpr = [0u64; 32];
        gpr[1] = 0x7FFF_FFFF;
        gpr[2] = 1;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_ADD);
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let interp = run_interpreter(instr, gpr, pc, &[]);
        let jit = run_jit(instr, gpr, pc, (pc as u16 / 4) & 0x3FF, &[])
            .expect("ADD must be compilable for this test to be meaningful");
        assert_eq!(jit, interp);
        assert_ne!(jit.pc, pc, "overflow must vector via handle_exception");
        assert_eq!(jit.gpr[3], 0, "rd must be untouched when the add traps");
    }

    #[test]
    fn sub_no_overflow_matches_interpreter() {
        alu_rrr_case(crate::mips_isa::FUNCT_SUB, 20, 5);
    }

    #[test]
    fn sub_overflow_traps_and_matches_interpreter() {
        // i32::MIN - 1 overflows a 32-bit signed sub -> EXC_OV.
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8000_0000; // sign-extended i32::MIN in a GPR
        gpr[2] = 1;
        let instr = make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_SUB);
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let interp = run_interpreter(instr, gpr, pc, &[]);
        let jit = run_jit(instr, gpr, pc, (pc as u16 / 4) & 0x3FF, &[])
            .expect("SUB must be compilable for this test to be meaningful");
        assert_eq!(jit, interp);
        assert_ne!(jit.pc, pc, "overflow must vector via handle_exception");
    }

    #[test]
    fn overflow_in_delay_slot_traps_with_epc_and_bd_matching_interpreter() {
        // Regression test: an exception raised by the delay slot itself
        // (not the branch/jump) must vector with EPC pointing at the BRANCH
        // (not the slot) and Cause.BD set — exactly what
        // MipsCore::deliver_exception computes from `in_delay_slot`. The JIT
        // never dispatches the slot as a separate step (it's inlined into
        // the branch/jump's own compiled unit, §6.1.4), so codegen has to
        // communicate "currently executing a slot" to handle_exception_fn
        // through core.in_delay_slot (set directly by emit_slot_semantics —
        // the same MipsCore field the interpreter's own dispatch loop uses,
        // no separate JIT-only copy) — this is the case that
        // catches it getting left false (EPC would wrongly point at the slot
        // itself and Cause.BD would wrongly stay clear).
        // Hand-rolled setup (not run_jit_page): that helper always compiles
        // with page_base=0 regardless of pc, which is harmless for every
        // other test in this file (none of them depend on absolute-address
        // correctness) but would make emit_slot_semantics's core.pc write
        // land on a tiny word_addr(0, 1)=4 instead of this branch's real
        // page — silently correct-looking (deliver_exception still runs and
        // sets *something*) but wrong, exactly the class of bug this test
        // exists to catch. page_base must match pc's real page here.
        let mut gpr = [0u64; 32];
        gpr[1] = 0x7FFF_FFFF;
        gpr[2] = 1;
        let page = vec![
            (0u16, make_i(crate::mips_isa::OP_BEQ, 0, 0, 5)), // always taken, target = 0+1+5 = word 6 (unreachable — slot traps first)
            (1u16, make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_ADD)), // delay slot: overflows
        ];
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let page_base = pc as u32; // pc is already page-aligned in this test

        let interp = run_interpreter_page(&page, gpr, pc, 2);

        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, raw) in &page { page_words[word as usize] = raw; }
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, 2);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
            .expect("BEQ+ADD region must compile even though the slot traps");

        let (mut jit_exec, jit_mem) = seeded_executor(gpr, pc);
        for &(word, raw) in &page { jit_mem.set_word(pc + (word as u64) * 4, raw); }
        jit_exec.install_jit_hooks();
        unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        std::mem::forget(codegen);
        let jit = CoreSnapshot::capture(&jit_exec.core);

        assert_eq!(jit, interp, "EPC/Cause/pc must match the interpreter exactly");
        assert_eq!(jit.cp0_epc, pc, "EPC must point at the branch, not the delay slot");
        assert_ne!(jit.cp0_cause & crate::mips_core::CAUSE_BD, 0, "Cause.BD must be set — exception raised from a delay slot");
        assert_eq!(jit.gpr[3], 0, "rd must be untouched when the slot's add traps");
    }

    /// Regression/coverage test for the analyzer's "§6.1.4 dual semantics"
    /// case: a single word inside one compiled region that is BOTH (a)
    /// inlined as another branch's mandatory delay slot (`emit_slot_semantics`,
    /// a self-contained copy that brackets its own `core.pc`/`in_delay_slot`
    /// and restores them afterward) AND (b) independently reachable as a
    /// genuine internal branch target with its own standalone head block
    /// (`analyzer::visit`'s "promotion" — `is_slot_only` starts `true` from
    /// the first role, flips to `false` once a second, real branch-target
    /// role is discovered for the same offset). These are two entirely
    /// separate Cranelift emission sites for the same architectural
    /// instruction, each with its own independent (and independently
    /// correct-looking) `core.pc`/`in_delay_slot` bracketing — this test
    /// exists because that design was previously unverified end-to-end (no
    /// prior test constructed both roles landing on the same word in one
    /// region).
    ///
    /// Layout: word 0 is a taken BEQ whose delay slot is word 1 (an ADDIU,
    /// harmless — proves the inlined-slot copy ran). Word 0's own taken
    /// target is word 3, an unconditional BEQ back to word 1 -- landing on
    /// word 1 a *second* time, this time via `block_for_word`/
    /// `emit_target_edge`'s internal-jump path into word 1's own independent
    /// head block (word 1 is promoted: both an inlined slot AND a real
    /// head). Word 1 is actually an ADD that's harmless on its first
    /// (inlined-slot) pass and overflows on its second (independent-head,
    /// reached-as-branch-target) pass, once a runway between the two visits
    /// (word 3's own delay slot, word 4) has advanced the operand -- proving
    /// the second visit's fault gets ordinary (non-delay-slot) EPC/BD, not
    /// anything leaked from the first visit's slot-bracketing.
    #[test]
    fn word_both_inlined_delay_slot_and_independent_branch_target_matches_interpreter() {
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let page_base = pc as u32; // pc is already page-aligned in this test
        let mut gpr = [0u64; 32];
        gpr[2] = 1; // r2: accumulator, ADD r2,r2,r3
        gpr[3] = 0x7FFF_FFFE; // r3: one below overflow

        let page = vec![
            (0u16, make_i(crate::mips_isa::OP_BEQ, 0, 0, 2)), // taken, target = 0+1+2 = word 3
            // delay slot (word 1) -- inlined-slot pass: ADD r2,r2,r3 -> 0x7FFFFFFF, harmless
            (1u16, make_r(crate::mips_isa::OP_SPECIAL, 2, 3, 2, 0, crate::mips_isa::FUNCT_ADD)),
            (3u16, make_i(crate::mips_isa::OP_BEQ, 0, 0, 0xFFFD)), // taken, target = 3+1+(-3) = word 1 (independent branch target)
            (4u16, make_i(crate::mips_isa::OP_ADDIU, 3, 3, 1)), // word 3's own delay slot: r3 += 1, so word 1's 2nd pass overflows
        ];

        // 5 dispatches: word0 BEQ, word1 slot (1st pass), word3 BEQ, word4
        // slot, word1 again (2nd pass, faults).
        let interp = run_interpreter_page(&page, gpr, pc, 5);

        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, raw) in &page { page_words[word as usize] = raw; }
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, 3);
        assert!(non_empty, "entry instruction must not be excluded");
        // Confirm the dual-semantics promotion actually happened as this
        // test's premise requires -- word 1 must be a real head (not
        // slot-only) with its own edges, or this test isn't exercising what
        // it claims to.
        assert!(walked[1].visited && !walked[1].is_slot_only, "word 1 must be promoted to a real head (reached both as word 0's slot and word 3's branch target) for this test to be meaningful");
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
            .expect("region must be compilable for this test to be meaningful");

        let (mut exec, mem) = seeded_executor(gpr, pc);
        for &(word, raw) in &page { mem.set_word(pc + (word as u64) * 4, raw); }
        exec.install_jit_hooks();
        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        std::mem::forget(codegen);
        let jit = CoreSnapshot::capture(&exec.core);

        assert_eq!(jit, interp, "JIT and interpreter diverged: word 1's fault on its 2nd (independent-branch-target) pass must match the interpreter exactly, not anything left over from its 1st (inlined-delay-slot) pass");
        assert_eq!(jit.cp0_epc, pc + 4, "EPC must be word 1's own address on this, its 2nd (independent-branch-target) visit -- not word 0's address (which is what Cause.BD-set delay-slot EPC math would wrongly compute)");
        assert_eq!(jit.cp0_cause & crate::mips_core::CAUSE_BD, 0, "BD must be clear -- word 1 was reached as an ordinary internal branch target on this pass, not a delay slot");
    }

    /// Mirror of `word_both_inlined_delay_slot_and_independent_branch_target_matches_interpreter`
    /// above, faulting on the OPPOSITE occurrence: this time word 1's FIRST
    /// (inlined-delay-slot) pass overflows, and its second
    /// (independent-branch-target) pass is never reached at all (the fault
    /// vectors away before the taken branch at word 0 -- word 3, the branch
    /// that would have taken it back to word 1 a second time -- ever runs).
    /// Isolates `exception_other_word_block`'s `bd=true` literal path
    /// (`emit_slot_semantics` sets `ctx.bd = true` before calling into the
    /// slot's own semantics emitter) -- the sibling test above never
    /// actually exercises `bd=true` reaching `emit_exception_exit` at all
    /// (its slot pass is deliberately harmless), so without this test,
    /// `exception_other_word_block`'s `bd` param could regress to always
    /// writing `false` (or never be wired up at all) without any test
    /// catching it, since EPC/BD would still happen to come out right for
    /// the (only ever tested) `bd=false` occurrence.
    #[test]
    fn word_both_inlined_delay_slot_and_independent_branch_target_faults_on_slot_pass_matches_interpreter() {
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let page_base = pc as u32; // pc is already page-aligned in this test
        let mut gpr = [0u64; 32];
        gpr[2] = 0x7FFF_FFFF; // r2: i32::MAX
        gpr[3] = 1; // r3: ADD r2,r2,r3 overflows immediately, on the 1st (inlined-slot) pass

        let page = vec![
            (0u16, make_i(crate::mips_isa::OP_BEQ, 0, 0, 2)), // taken, target = 0+1+2 = word 3
            // delay slot (word 1) -- inlined-slot pass: ADD r2,r2,r3 overflows here, immediately
            (1u16, make_r(crate::mips_isa::OP_SPECIAL, 2, 3, 2, 0, crate::mips_isa::FUNCT_ADD)),
            (3u16, make_i(crate::mips_isa::OP_BEQ, 0, 0, 0xFFFD)), // taken, target = word 1 -- never reached, the slot faults first
        ];

        // 2 dispatches: word0 BEQ, word1 slot (1st pass, faults immediately) --
        // word 3's own branch never runs.
        let interp = run_interpreter_page(&page, gpr, pc, 2);

        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, raw) in &page { page_words[word as usize] = raw; }
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, 3);
        assert!(non_empty, "entry instruction must not be excluded");
        assert!(walked[1].visited && !walked[1].is_slot_only, "word 1 must be promoted to a real head (reached both as word 0's slot and word 3's branch target) for this test to be meaningful");
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
            .expect("region must be compilable for this test to be meaningful");

        let (mut exec, mem) = seeded_executor(gpr, pc);
        for &(word, raw) in &page { mem.set_word(pc + (word as u64) * 4, raw); }
        exec.install_jit_hooks();
        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        std::mem::forget(codegen);
        let jit = CoreSnapshot::capture(&exec.core);

        assert_eq!(jit, interp, "JIT and interpreter diverged: word 1's fault on its 1st (inlined-delay-slot) pass must match the interpreter exactly");
        assert_eq!(jit.cp0_epc, pc, "EPC must be the BRANCH's address (word 0, pc - 4), not the slot's own address -- this fault happened while word 1 was inlined as word 0's delay slot");
        assert_ne!(jit.cp0_cause & crate::mips_core::CAUSE_BD, 0, "BD must be set -- word 1 was executing as a delay slot on this, its 1st (inlined-slot) pass");
        assert_eq!(jit.gpr[2], 0x7FFF_FFFF, "rd must be untouched when the slot's own add traps");
    }

    /// Counterpart to `overflow_in_delay_slot_traps_with_epc_and_bd_matching_interpreter`
    /// for the *not-taken* arm of a non-annulling conditional branch. Per the
    /// MIPS spec (confirmed against an independent reference, not just
    /// MAME's r4000.cpp core, which only tracks delay-slot state on the
    /// taken path — a narrow deviation from spec, not something to mirror
    /// here) the delay slot always executes, taken or not, and Cause.BD is
    /// positional (any word immediately after a branch/jump), not outcome-
    /// dependent. `codegen.rs`'s `emit_slot_semantics` already runs
    /// unconditionally, before the branch's own `brif`, for every
    /// non-annulling branch shape (see that function and
    /// `emit_nested_branch_slot`'s `_ if !branch.annul` arm) — this proves it
    /// against the interpreter's own (`mips_exec.rs::handle_branch_not_taken`)
    /// matching fix.
    #[test]
    fn overflow_in_delay_slot_not_taken_traps_with_epc_and_bd_matching_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0x7FFF_FFFF;
        gpr[2] = 1;
        let page = vec![
            (0u16, make_i(crate::mips_isa::OP_BEQ, 1, 0, 5)), // r1 != r0 -> not taken
            (1u16, make_r(crate::mips_isa::OP_SPECIAL, 1, 2, 3, 0, crate::mips_isa::FUNCT_ADD)), // delay slot: overflows regardless
        ];
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let page_base = pc as u32;

        let interp = run_interpreter_page(&page, gpr, pc, 2);

        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, raw) in &page { page_words[word as usize] = raw; }
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, 2);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
            .expect("BEQ+ADD region must compile even though the slot traps");

        let (mut jit_exec, jit_mem) = seeded_executor(gpr, pc);
        for &(word, raw) in &page { jit_mem.set_word(pc + (word as u64) * 4, raw); }
        jit_exec.install_jit_hooks();
        unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        std::mem::forget(codegen);
        let jit = CoreSnapshot::capture(&jit_exec.core);

        assert_eq!(jit, interp, "EPC/Cause/pc must match the interpreter exactly");
        assert_eq!(jit.cp0_epc, pc, "EPC must point at the branch, not the delay slot, even though not taken");
        assert_ne!(jit.cp0_cause & crate::mips_core::CAUSE_BD, 0, "Cause.BD must be set — the slot always executes, taken or not");
        assert_eq!(jit.gpr[3], 0, "rd must be untouched when the slot's add traps");
    }

    /// Counterpart proving the opposite for an *annulling* Likely branch: on
    /// the not-taken arm, the slot must never execute at all (annulled, per
    /// spec — `handle_branch_likely_skip`'s plain pc+=8, no delay-slot
    /// dispatch), so a would-be-overflowing ADD in the slot must never fire
    /// and `in_delay_slot`/Cause.BD/rd must all read as if the slot were
    /// never there.
    #[test]
    fn likely_branch_not_taken_annuls_slot_no_trap() {
        let mut gpr = [0u64; 32];
        gpr[1] = 1; // BEQL r1,r0 -> not equal -> not taken -> annulled
        gpr[2] = 0x7FFF_FFFF;
        gpr[3] = 1; // would overflow if the slot ever ran
        let page = vec![
            (0u16, make_i(crate::mips_isa::OP_BEQL, 1, 0, 5)),
            (1u16, make_r(crate::mips_isa::OP_SPECIAL, 2, 3, 4, 0, crate::mips_isa::FUNCT_ADD)), // would overflow if executed
        ];
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let page_base = pc as u32;

        let interp = run_interpreter_page(&page, gpr, pc, 1); // annulled: one dispatch is the whole story, no separate slot step

        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(word, raw) in &page { page_words[word as usize] = raw; }
        let mut analyzer = Analyzer::new();
        // max_instrs=1 (not 2, unlike the taken/overflow tests above): the
        // not-taken/annulled arm is the one this test actually exercises, and
        // with budget for a second head, word 2 (implicit NOP past the slot)
        // gets walked as a real second head — continuing straight into it as
        // compiled code — instead of the clean bail-to-interpreter this test
        // wants to isolate. See walk_bounded_budget_excludes_delay_slot: a
        // budget of exactly 1 still gets the branch's own mandatory slot for
        // free, just nothing beyond it.
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, 0, page_base, 1);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, 0, true, false)
            .expect("BEQL+ADD region must compile");

        let (mut jit_exec, jit_mem) = seeded_executor(gpr, pc);
        for &(word, raw) in &page { jit_mem.set_word(pc + (word as u64) * 4, raw); }
        jit_exec.install_jit_hooks();
        unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        std::mem::forget(codegen);
        let jit_in_delay_slot = jit_exec.core.in_delay_slot;
        let jit = CoreSnapshot::capture(&jit_exec.core);

        assert_eq!(jit, interp, "annulled not-taken Likely branch must match the interpreter exactly");
        assert_eq!(jit.pc, pc + 8, "not taken lands past the annulled slot, word+2");
        assert!(!jit_in_delay_slot, "annulled slot must never set in_delay_slot");
        assert_eq!(jit.cp0_cause & crate::mips_core::CAUSE_BD, 0, "Cause.BD must stay clear — no exception, and the slot never ran");
        assert_eq!(jit.gpr[4], 0, "annulled slot's ADD must never have executed");
    }

    #[test]
    fn sllv_srlv_srav_match_interpreter() {
        for sa in [0u64, 1, 15, 31, 63, 0xFFFF_FFFF] {
            // rs supplies the shift amount, masked to 5 bits by the emitter —
            // exercise values above 31 to confirm the mask is applied.
            shift_var_case(crate::mips_isa::FUNCT_SLLV, sa, 0x8765_4321_FEDC_BA98);
            shift_var_case(crate::mips_isa::FUNCT_SRLV, sa, 0x8765_4321_FEDC_BA98);
            shift_var_case(crate::mips_isa::FUNCT_SRAV, sa, 0x8765_4321_FEDC_BA98);
        }
    }

    // ---- CP1 (FPU): batch F1 arithmetic — ADD.S/ADD.D ------

    #[test]
    fn add_s_matches_interpreter_fr1() {
        let mut fpr = [0u64; 32];
        fpr[1] = (2.5f32).to_bits() as u64;
        fpr[2] = (1.25f32).to_bits() as u64;
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 2, 1, 3, crate::mips_isa::FUNCT_FADD);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);
    }

    #[test]
    fn add_d_matches_interpreter_fr1() {
        let mut fpr = [0u64; 32];
        fpr[1] = (2.5f64).to_bits();
        fpr[2] = (1.25f64).to_bits();
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_D, 2, 1, 3, crate::mips_isa::FUNCT_FADD);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);
    }

    #[test]
    fn add_s_matches_interpreter_fr0() {
        // FR=0: single-format registers pack two per 64-bit slot (odd reg =
        // upper half). fs=0 (even, low half of slot 0), ft=1 (odd, upper
        // half of slot 0), fd=2 (even, low half of slot 2).
        let mut fpr = [0u64; 32];
        fpr[0] = ((1.25f32).to_bits() as u64) | (((2.5f32).to_bits() as u64) << 32);
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 1, 0, 2, crate::mips_isa::FUNCT_FADD);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, false);
    }

    #[test]
    fn add_d_matches_interpreter_fr0() {
        // FR=0 double: register numbers are forced even (reg & !1).
        let mut fpr = [0u64; 32];
        fpr[0] = (2.5f64).to_bits();
        fpr[2] = (1.25f64).to_bits();
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_D, 2, 0, 4, crate::mips_isa::FUNCT_FADD);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, false);
    }

    #[test]
    fn add_s_nan_matches_interpreter() {
        // NaN propagation + FCSR sticky-flag accumulation (invalid-op path
        // isn't triggered by a plain add of two NaNs, but the result and
        // FCSR flag state must still match exactly).
        let mut fpr = [0u64; 32];
        fpr[1] = f32::NAN.to_bits() as u64;
        fpr[2] = (1.0f32).to_bits() as u64;
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 2, 1, 3, crate::mips_isa::FUNCT_FADD);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);
    }

    #[test]
    fn add_s_without_cu1_traps_and_matches_interpreter() {
        // CU1 clear -> cpu_unusable -> EXC_CPU, delivered via handle_exception
        // (§4.2) — must match the interpreter's vectoring exactly, same as
        // the ADEL/overflow exception tests for integer ops.
        let mut fpr = [0u64; 32];
        fpr[1] = (1.0f32).to_bits() as u64;
        fpr[2] = (1.0f32).to_bits() as u64;
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 2, 1, 3, crate::mips_isa::FUNCT_FADD);

        let pc = 0xFFFF_FFFF_8000_1000u64;
        let word_offset = (pc as u16 / 4) & 0x3FF;
        let (mut interp_exec, _mem) = fpu_seeded_executor([0u64; 32], fpr, pc, true);
        interp_exec.core.cp0_status = 0; // CU1 clear
        interp_exec.exec(instr);
        let interp = CoreSnapshot::capture(&interp_exec.core);

        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[word_offset as usize] = instr;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page, word_offset, 0, 1);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word_offset, true, false)
            .expect("ADD.S must be compilable even though this test exercises the CU1-clear path");
        // fpu_seeded_executor's default MockMemory::new_not_compilable() is
        // wrong here: the entry guard's bail now calls through
        // core.interp_fallback_fn (emit_interp_fallback_exit), a real fetch+
        // decode+dispatch of whatever's actually at core.pc — it needs a
        // memory backend real fetch_instr can read from, same as any other
        // interpreter dispatch, not the "always denylisted" stub built only
        // for tests that never leave the JIT (see that stub's own doc
        // comment). Whether the page is denylisted is irrelevant to this
        // path (only exec_decoded's JIT gate consults it) — this just needs
        // page[] to be readable.
        let (mut jit_exec, jit_mem) = seeded_executor_over(MockMemory::new(), [0u64; 32], pc);
        jit_exec.core.fpr = fpr;
        jit_exec.core.cp0_status = 0; // CU1 clear
        jit_exec.update_fpr_mode();
        // pc is kseg0 (unmapped-cached): real physical address is
        // pc & 0x1FFFFFFF, not word-index*4 — see debug_cu1_guard_isolated's
        // sibling comment for the full story (interp_fallback_fn's real
        // fetch_instr needs the instruction at the actual translated address).
        let phys_base = (pc & 0x1FFFFFFF) as u64;
        for (i, &w) in page.iter().enumerate() {
            if w != 0 { jit_mem.set_word(phys_base + (i as u64) * 4, w); }
        }
        jit_exec.install_jit_hooks();
        // The entry guard now forces real forward progress via
        // interp_fallback_fn instead of a plain bail (see
        // emit_interp_fallback_exit's doc comment for why a plain bail can't
        // — the JIT gate would just call this identical compiled function
        // again, forever, without cpu_unusable ever actually running) — one
        // call is the complete dispatch, matching a single interpreter step.
        // emit_materialize_cpu_unusable delegates to emit_exception_exit,
        // which always returns EXEC_COMPLETE once core.pc/cp0_cause/cp0_epc
        // are already set to the vectored state — nothing further for the
        // caller to do (same contract as every other JIT-raised exception).
        // The real assertion is the CoreSnapshot comparison below.
        let _status = unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        std::mem::forget(codegen);
        let jit = CoreSnapshot::capture(&jit_exec.core);

        assert_eq!(jit, interp);
        assert_ne!(jit.pc, pc, "CU1-clear must vector via handle_exception (cpu_unusable)");
    }

    fn fbinop_case(funct: u32, fmt: u32, fs_val: f64, ft_val: f64) {
        let mut fpr = [0u64; 32];
        if fmt == crate::mips_isa::RS_S {
            fpr[1] = (fs_val as f32).to_bits() as u64;
            fpr[2] = (ft_val as f32).to_bits() as u64;
        } else {
            fpr[1] = (fs_val).to_bits();
            fpr[2] = (ft_val).to_bits();
        }
        let instr = make_r(crate::mips_isa::OP_COP1, fmt, 2, 1, 3, funct);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);
    }

    #[test]
    fn sub_s_d_match_interpreter() {
        fbinop_case(crate::mips_isa::FUNCT_FSUB, crate::mips_isa::RS_S, 5.0, 3.0);
        fbinop_case(crate::mips_isa::FUNCT_FSUB, crate::mips_isa::RS_D, 5.0, 3.0);
        fbinop_case(crate::mips_isa::FUNCT_FSUB, crate::mips_isa::RS_S, 3.0, 5.0); // negative result
    }

    #[test]
    fn mul_s_d_match_interpreter() {
        fbinop_case(crate::mips_isa::FUNCT_FMUL, crate::mips_isa::RS_S, 2.5, 4.0);
        fbinop_case(crate::mips_isa::FUNCT_FMUL, crate::mips_isa::RS_D, 2.5, 4.0);
    }

    #[test]
    fn div_s_d_match_interpreter() {
        fbinop_case(crate::mips_isa::FUNCT_FDIV, crate::mips_isa::RS_S, 10.0, 4.0);
        fbinop_case(crate::mips_isa::FUNCT_FDIV, crate::mips_isa::RS_D, 10.0, 4.0);
    }

    #[test]
    fn div_by_zero_fpu_matches_interpreter() {
        // FP divide-by-zero: unlike integer DIV (no-op on zero divisor),
        // this raises FCSR.Z and returns +/-Infinity per IEEE-754 — must
        // match the interpreter's FCSR flag accumulation exactly.
        fbinop_case(crate::mips_isa::FUNCT_FDIV, crate::mips_isa::RS_S, 1.0, 0.0);
        fbinop_case(crate::mips_isa::FUNCT_FDIV, crate::mips_isa::RS_D, 1.0, 0.0);
    }

    fn funop_case(funct: u32, fmt: u32, fs_val: f64) {
        let mut fpr = [0u64; 32];
        if fmt == crate::mips_isa::RS_S {
            fpr[1] = (fs_val as f32).to_bits() as u64;
        } else {
            fpr[1] = fs_val.to_bits();
        }
        let instr = make_r(crate::mips_isa::OP_COP1, fmt, 0, 1, 3, funct);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);
    }

    #[test]
    fn sqrt_s_d_match_interpreter() {
        funop_case(crate::mips_isa::FUNCT_FSQRT, crate::mips_isa::RS_S, 16.0);
        funop_case(crate::mips_isa::FUNCT_FSQRT, crate::mips_isa::RS_D, 2.0);
    }

    #[test]
    fn sqrt_of_negative_matches_interpreter() {
        // Invalid operation (NaN result + FCSR.V) — bug-for-bug parity check.
        funop_case(crate::mips_isa::FUNCT_FSQRT, crate::mips_isa::RS_S, -4.0);
    }

    #[test]
    fn abs_s_d_match_interpreter() {
        funop_case(crate::mips_isa::FUNCT_FABS, crate::mips_isa::RS_S, -3.5);
        funop_case(crate::mips_isa::FUNCT_FABS, crate::mips_isa::RS_D, -3.5);
        funop_case(crate::mips_isa::FUNCT_FABS, crate::mips_isa::RS_S, 3.5);
    }

    #[test]
    fn neg_s_d_match_interpreter() {
        funop_case(crate::mips_isa::FUNCT_FNEG, crate::mips_isa::RS_S, 3.5);
        funop_case(crate::mips_isa::FUNCT_FNEG, crate::mips_isa::RS_D, -3.5);
    }

    #[test]
    fn mov_s_d_match_interpreter() {
        funop_case(crate::mips_isa::FUNCT_FMOV, crate::mips_isa::RS_S, 3.5);
        funop_case(crate::mips_isa::FUNCT_FMOV, crate::mips_isa::RS_D, 3.5);
    }

    #[test]
    fn abs_neg_mov_do_not_touch_fcsr() {
        // These three never call fpu_update_fcsr (interpreter doesn't
        // either) — confirm no spurious FCSR flags appear even for inputs
        // that WOULD raise flags under add/sub/mul/div (e.g. abs(NaN)).
        let mut fpr = [0u64; 32];
        fpr[1] = f32::NAN.to_bits() as u64;
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 0, 1, 3, crate::mips_isa::FUNCT_FABS);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);
    }

    // ---- CP1: batch F2 moves — MFC1/DMFC1/CFC1/MTC1/DMTC1/CTC1 ------

    #[test]
    fn mfc1_matches_interpreter() {
        let mut fpr = [0u64; 32];
        fpr[5] = 0xFFFF_FFFF_DEAD_BEEFu64; // high bits must be discarded, low 32 sign-extended
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_MFC1, 3, 5, 0, 0);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);
    }

    #[test]
    fn dmfc1_matches_interpreter() {
        let mut fpr = [0u64; 32];
        fpr[5] = 0xFFFF_FFFF_DEAD_BEEFu64;
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_DMFC1, 3, 5, 0, 0);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);
    }

    #[test]
    fn mtc1_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[3] = 0xFFFF_FFFF_DEAD_BEEFu64; // MTC1 truncates to low 32 bits
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_MTC1, 3, 5, 0, 0);
        assert_fpu_matches_interpreter(instr, gpr, [0u64; 32], true);
    }

    #[test]
    fn dmtc1_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[3] = 0xFFFF_FFFF_DEAD_BEEFu64;
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_DMTC1, 3, 5, 0, 0);
        assert_fpu_matches_interpreter(instr, gpr, [0u64; 32], true);
    }

    #[test]
    fn mtc1_dmtc1_fr0_preserve_other_half() {
        // FR=0: MTC1 into an odd register must not clobber the even
        // register sharing its 64-bit slot.
        let mut gpr = [0u64; 32];
        gpr[3] = 0xDEAD_BEEFu64;
        let mut fpr = [0u64; 32];
        fpr[0] = 0x1234_5678_0000_0000u64; // pre-existing even-half (reg 0) content
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_MTC1, 3, 1, 0, 0); // fs=1 (odd)
        assert_fpu_matches_interpreter(instr, gpr, fpr, false);
    }

    #[test]
    fn cfc1_fcsr_matches_interpreter() {
        let mut fpr = [0u64; 32];
        // Seed FCSR indirectly isn't possible via fpr[]; use a fresh core
        // and check reg 0 (FIR, read-only, nonzero default) instead, which
        // exercises the same read-path dispatch.
        let _ = &mut fpr;
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_CFC1, 3, 0, 0, 0); // fs=0 (FIR)
        assert_fpu_matches_interpreter(instr, [0u64; 32], [0u64; 32], true);
    }

    #[test]
    fn ctc1_fexr_fenr_match_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[3] = 0x1234;
        let instr_fexr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_CTC1, 3, 26, 0, 0);
        assert_fpu_matches_interpreter(instr_fexr, gpr, [0u64; 32], true);
        let instr_fenr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_CTC1, 3, 28, 0, 0);
        assert_fpu_matches_interpreter(instr_fenr, gpr, [0u64; 32], true);
    }

    #[test]
    fn ctc1_fcsr_writes_rounding_mode_matches_interpreter() {
        let mut gpr = [0u64; 32];
        gpr[3] = 0x2; // RM = 2 (round toward +Infinity), no cause/enable bits set
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_CTC1, 3, 31, 0, 0);
        assert_fpu_matches_interpreter(instr, gpr, [0u64; 32], true);
    }

    #[test]
    fn ctc1_fcsr_pending_exception_traps_and_matches_interpreter() {
        // Cause bit V (invalid) set together with its Enable bit -> CTC1
        // must immediately raise EXC_FPE (the "recheck after write" path),
        // not just accumulate the flag silently.
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let word_offset = (pc as u16 / 4) & 0x3FF;
        let mut gpr = [0u64; 32];
        // FCSR: Cause.V (bit 16) set, Enable.V (bit 7) set.
        gpr[3] = (1 << 16) | (1 << 7);
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_CTC1, 3, 31, 0, 0);

        let interp = run_interpreter_fpu(instr, gpr, [0u64; 32], pc, true);
        let jit = run_jit_fpu(instr, gpr, [0u64; 32], pc, word_offset, true)
            .expect("CTC1 must be compilable");
        assert_eq!(jit, interp);
        assert_ne!(jit.pc, pc, "pending-cause-vs-enabled CTC1 write must vector via handle_exception");
    }

    #[test]
    fn cfc1_ctc1_fccr_roundtrip_matches_interpreter() {
        // Write FCCR (reg 25) then read it back — exercises fcsr_with_fccr
        // scatter and fccr_from_fcsr gather together.
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let mut gpr = [0u64; 32];
        gpr[3] = 0b1010_1010; // arbitrary 8-bit condition-code pattern
        let ctc1 = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_CTC1, 3, 25, 0, 0);
        assert_fpu_matches_interpreter(ctc1, gpr, [0u64; 32], true);

        let cfc1 = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_CFC1, 4, 25, 0, 0);
        // Chain: write via CTC1 first, then read back via CFC1 in the same
        // seeded state (both engines start from the post-CTC1 FCSR).
        let (mut interp_exec, _m) = fpu_seeded_executor(gpr, [0u64; 32], pc, true);
        interp_exec.exec(ctc1);
        interp_exec.exec(cfc1);
        let interp = CoreSnapshot::capture(&interp_exec.core);
        assert_eq!(interp.gpr[4] as u32, gpr[3] as u32 & 0xFF, "sanity: interpreter roundtrips FCCR");
    }

    // ---- CP1: FPU load/store (LWC1/LDC1/SWC1/SDC1) ------
    //
    // Architecturally plain memory ops (separate top-level opcodes, not
    // OP_COP1-encoded), but routed through lookup_cp1_semantics so the
    // region-wide CU1/FR guard's single trigger check still catches them —
    // see codegen.rs's lookup_cp1_semantics doc comment.

    #[test]
    fn lwc1_matches_interpreter_fr0_and_fr1() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8000_2000; // base
        let instr = make_i(crate::mips_isa::OP_LWC1, 1, 2, 0x10); // ft=2, offset=0x10
        let mem_init = &[(0xFFFF_FFFF_8000_2010u64, 0x3F80_0000u32)]; // 1.0f32 bits
        assert_fpu_matches_interpreter_mem(instr, gpr, [0u64; 32], false, mem_init);
        assert_fpu_matches_interpreter_mem(instr, gpr, [0u64; 32], true, mem_init);
    }

    #[test]
    fn ldc1_matches_interpreter_fr0_and_fr1() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8000_2000;
        let instr = make_i(crate::mips_isa::OP_LDC1, 1, 2, 0x18); // ft=2, offset=0x18
        // 2.5f64 bits, high/low words as they'd sit in big-endian-addressed
        // memory (MockMemory::set_word writes one 32-bit word at a time;
        // read_data<8> assembles the full 64-bit value from two such words —
        // matches how emit_ldc1/exec_ldc1 both read via the same read64_fn).
        let bits = (2.5f64).to_bits();
        let mem_init = &[
            (0xFFFF_FFFF_8000_2018u64, (bits >> 32) as u32),
            (0xFFFF_FFFF_8000_201Cu64, bits as u32),
        ];
        assert_fpu_matches_interpreter_mem(instr, gpr, [0u64; 32], false, mem_init);
        assert_fpu_matches_interpreter_mem(instr, gpr, [0u64; 32], true, mem_init);
    }

    #[test]
    fn swc1_matches_interpreter_fr0_and_fr1() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8000_2000;
        let mut fpr = [0u64; 32];
        fpr[2] = (1.5f32).to_bits() as u64;
        let instr = make_i(crate::mips_isa::OP_SWC1, 1, 2, 0x20); // ft=2, offset=0x20
        assert_fpu_matches_interpreter(instr, gpr, fpr, false);
        assert_fpu_matches_interpreter(instr, gpr, fpr, true);
    }

    #[test]
    fn sdc1_matches_interpreter_fr0_and_fr1() {
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8000_2000;
        let mut fpr = [0u64; 32];
        fpr[2] = (4.25f64).to_bits();
        let instr = make_i(crate::mips_isa::OP_SDC1, 1, 2, 0x28); // ft=2, offset=0x28
        assert_fpu_matches_interpreter(instr, gpr, fpr, false);
        assert_fpu_matches_interpreter(instr, gpr, fpr, true);
    }

    #[test]
    fn lwc1_without_cu1_traps_and_matches_interpreter() {
        // Region-wide CU1 guard must fire for a region containing ONLY
        // LWC1 — the exact gap this instruction's implementation had to
        // close in lookup_cp1_semantics's dispatch (see that function's
        // doc comment): if LWC1 had instead been routed through
        // lookup_semantics, has_fpu's single check would never trigger and
        // this test would silently execute the load instead of trapping.
        // fpu_seeded_executor always sets CU1 (it's meant for FPU-op
        // tests), so both engines here clear it again immediately after
        // seeding, before dispatch/compile ever runs.
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let word_offset = (pc as u16 / 4) & 0x3FF;
        let mut gpr = [0u64; 32];
        gpr[1] = 0xFFFF_FFFF_8000_2000;
        let instr = make_i(crate::mips_isa::OP_LWC1, 1, 2, 0);
        let mem_word = (0xFFFF_FFFF_8000_2000u64, 0x3F80_0000u32);

        let (mut interp_exec, mem) = fpu_seeded_executor(gpr, [0u64; 32], pc, true);
        interp_exec.core.cp0_status &= !crate::mips_core::STATUS_CU1;
        mem.set_word(mem_word.0, mem_word.1);
        interp_exec.exec(instr);
        let interp = CoreSnapshot::capture(&interp_exec.core);

        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[word_offset as usize] = instr;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page, word_offset, 0, 1);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        // compile_region has no cp0_status input — the CU1 guard it emits
        // is a runtime check baked into the compiled body, so this must
        // compile successfully regardless of live CU1 state at test time.
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word_offset, true, false)
            .expect("LWC1 must be compilable regardless of live CU1 state");
        let (jit_exec, jit_mem) = fpu_seeded_executor(gpr, [0u64; 32], pc, true);
        let mut jit_exec = Box::new(jit_exec);
        jit_exec.core.cp0_status &= !crate::mips_core::STATUS_CU1;
        jit_mem.set_word(mem_word.0, mem_word.1);
        jit_exec.install_jit_hooks();
        unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        std::mem::forget(codegen);
        let jit = CoreSnapshot::capture(&jit_exec.core);

        assert_eq!(jit, interp, "CU1-clear LWC1 must trap identically on both engines");
        assert_ne!(jit.pc, pc, "CU1-clear must vector via handle_exception, not silently load");
    }

    // ---- CP1: batch F3 conversions ------

    #[test]
    fn cvt_d_s_and_s_d_match_interpreter() {
        let mut fpr = [0u64; 32];
        fpr[1] = (3.5f32).to_bits() as u64;
        let cvt_d_s = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 0, 1, 2, crate::mips_isa::FUNCT_FCVT_D);
        assert_fpu_matches_interpreter(cvt_d_s, [0u64; 32], fpr, true);

        let mut fpr_d = [0u64; 32];
        fpr_d[1] = (3.5f64).to_bits();
        let cvt_s_d = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_D, 0, 1, 2, crate::mips_isa::FUNCT_FCVT_S);
        assert_fpu_matches_interpreter(cvt_s_d, [0u64; 32], fpr_d, true);
    }

    #[test]
    fn cvt_s_d_precision_loss_matches_interpreter() {
        // A double value with more precision than f32 can hold — narrowing
        // rounding behavior must match Rust's `as f32` cast exactly.
        let mut fpr = [0u64; 32];
        fpr[1] = (std::f64::consts::PI).to_bits();
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_D, 0, 1, 2, crate::mips_isa::FUNCT_FCVT_S);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);
    }

    /// FCSR's Inexact bits (Cause bit 12, Flag bit 2) for ROUND/TRUNC/CEIL/
    /// FLOOR/CVT-to-int ops: masked out of the comparison below.
    /// `MipsExecutor::fpu_update_fcsr`'s Inexact reporting for these ops
    /// depends on whether the interpreter's own Rust `.round()`/`.trunc()`/
    /// etc. happen to lower to a host FP instruction that touches MXCSR —
    /// which differs between debug and release Rust builds (`.round()`
    /// sets Precision/Inexact in debug builds via a real ROUNDSS-class
    /// lowering, but not in release builds, where trunc/ceil/floor never
    /// set it in either profile). That's an LLVM codegen artifact, not a
    /// specified MIPS/FCSR behavior — the JIT (via emit_round_and_convert's
    /// explicit clear before the final int conversion) deliberately never
    /// reports Inexact for this op class, matching the *stable* behavior
    /// (release-build interpreter, and trunc/ceil/floor in any profile)
    /// rather than chasing whichever the current interpreter build happens
    /// to produce for `.round()` specifically. See
    /// rules/jitv2/codegen-gotchas.md.
    const FCSR_INEXACT_MASK: u32 = (1 << 12) | (1 << 2);

    fn cvt_to_int_case(funct: u32, fmt: u32, fs_val: f64) {
        let mut fpr = [0u64; 32];
        if fmt == crate::mips_isa::RS_S {
            fpr[1] = (fs_val as f32).to_bits() as u64;
        } else {
            fpr[1] = fs_val.to_bits();
        }
        let instr = make_r(crate::mips_isa::OP_COP1, fmt, 0, 1, 2, funct);
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let word_offset = (pc as u16 / 4) & 0x3FF;
        let interp = run_interpreter_fpu(instr, [0u64; 32], fpr, pc, true);
        let jit = run_jit_fpu(instr, [0u64; 32], fpr, pc, word_offset, true)
            .expect("conversion must be compilable for this test to be meaningful");
        let mut interp_masked = interp.clone();
        let mut jit_masked = jit.clone();
        interp_masked.fpu_fcsr &= !FCSR_INEXACT_MASK;
        jit_masked.fpu_fcsr &= !FCSR_INEXACT_MASK;
        assert_eq!(jit_masked, interp_masked, "JIT and interpreter diverged (ignoring the build-dependent Inexact bit) for instr=0x{:08x}", instr);
    }

    #[test]
    fn cvt_w_s_d_match_interpreter() {
        cvt_to_int_case(crate::mips_isa::FUNCT_FCVT_W, crate::mips_isa::RS_S, 3.7);
        cvt_to_int_case(crate::mips_isa::FUNCT_FCVT_W, crate::mips_isa::RS_D, 3.7);
        cvt_to_int_case(crate::mips_isa::FUNCT_FCVT_W, crate::mips_isa::RS_S, -3.7);
    }

    #[test]
    fn cvt_l_s_d_match_interpreter() {
        cvt_to_int_case(crate::mips_isa::FUNCT_FCVT_L, crate::mips_isa::RS_S, 3.7);
        cvt_to_int_case(crate::mips_isa::FUNCT_FCVT_L, crate::mips_isa::RS_D, 3.7);
    }

    #[test]
    fn round_half_away_from_zero_matches_interpreter() {
        // The exact case that distinguishes Rust's round() (half away from
        // zero) from Cranelift's default `nearest` (half to even): 2.5 and
        // -2.5 must both round AWAY from zero (3.0, -3.0), not to the
        // nearest even integer (2.0, -2.0).
        cvt_to_int_case(crate::mips_isa::FUNCT_FROUND_W, crate::mips_isa::RS_S, 2.5);
        cvt_to_int_case(crate::mips_isa::FUNCT_FROUND_W, crate::mips_isa::RS_S, -2.5);
        cvt_to_int_case(crate::mips_isa::FUNCT_FROUND_W, crate::mips_isa::RS_S, 3.5);
        cvt_to_int_case(crate::mips_isa::FUNCT_FROUND_W, crate::mips_isa::RS_D, 4.5);
    }

    #[test]
    fn trunc_ceil_floor_w_match_interpreter() {
        for val in [3.7, -3.7, 3.2, -3.2] {
            cvt_to_int_case(crate::mips_isa::FUNCT_FTRUNC_W, crate::mips_isa::RS_S, val);
            cvt_to_int_case(crate::mips_isa::FUNCT_FCEIL_W, crate::mips_isa::RS_S, val);
            cvt_to_int_case(crate::mips_isa::FUNCT_FFLOOR_W, crate::mips_isa::RS_S, val);
        }
    }

    #[test]
    fn trunc_ceil_floor_l_match_interpreter() {
        for val in [3.7, -3.7] {
            cvt_to_int_case(crate::mips_isa::FUNCT_FTRUNC_L, crate::mips_isa::RS_D, val);
            cvt_to_int_case(crate::mips_isa::FUNCT_FCEIL_L, crate::mips_isa::RS_D, val);
            cvt_to_int_case(crate::mips_isa::FUNCT_FFLOOR_L, crate::mips_isa::RS_D, val);
        }
    }

    #[test]
    fn cvt_to_int_overflow_saturates_matches_interpreter() {
        // MIPS's fixed poison pattern on float->int overflow: INT_MAX
        // (0x7FFFFFFF/0x7FFFFFFFFFFFFFFF) for positive overflow, INT_MIN for
        // negative — see `cvt_f32/64_to_int_by_value` in mips_exec.rs, the
        // single function both engines call through `cvt_to_int_and_commit`.
        // Also sets FCSR.V/Cause, which `cvt_to_int_case`'s snapshot compare
        // covers alongside the destination register value.
        cvt_to_int_case(crate::mips_isa::FUNCT_FCVT_W, crate::mips_isa::RS_S, 1.0e20);
        cvt_to_int_case(crate::mips_isa::FUNCT_FCVT_W, crate::mips_isa::RS_S, -1.0e20);

        // cvt_to_int_case only checks JIT == interpreter, which a shared bug
        // in cvt_to_int_and_commit would pass right through — pin the actual
        // MIPS poison values against the interpreter directly too.
        let mut fpr = [0u64; 32];
        fpr[1] = (1.0e20f32).to_bits() as u64;
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 0, 1, 2, crate::mips_isa::FUNCT_FCVT_W);
        let pos = run_interpreter_fpu(instr, [0u64; 32], fpr, 0xFFFF_FFFF_8000_1000u64, true);
        assert_eq!(pos.fpr[2] as u32, i32::MAX as u32, "positive overflow must poison to INT_MAX");
        assert_ne!(pos.fpu_fcsr & 0x0001_0000, 0, "overflow must raise FCSR Cause.V (bit 16)");

        fpr[1] = (-1.0e20f32).to_bits() as u64;
        let neg = run_interpreter_fpu(instr, [0u64; 32], fpr, 0xFFFF_FFFF_8000_1000u64, true);
        assert_eq!(neg.fpr[2] as u32, i32::MIN as u32, "negative overflow must poison to INT_MIN");
        assert_ne!(neg.fpu_fcsr & 0x0001_0000, 0, "overflow must raise FCSR Cause.V (bit 16)");
    }

    #[test]
    fn cvt_to_int_nan_matches_interpreter() {
        // MIPS's poison pattern for NaN input is INT_MAX (same as positive
        // overflow) — see `cvt_to_int_overflow_saturates_matches_interpreter`.
        let mut fpr = [0u64; 32];
        fpr[1] = f32::NAN.to_bits() as u64;
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 0, 1, 2, crate::mips_isa::FUNCT_FCVT_W);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);

        let nan = run_interpreter_fpu(instr, [0u64; 32], fpr, 0xFFFF_FFFF_8000_1000u64, true);
        assert_eq!(nan.fpr[2] as u32, i32::MAX as u32, "NaN must poison to INT_MAX");
        assert_ne!(nan.fpu_fcsr & 0x0001_0000, 0, "NaN input must raise FCSR Cause.V (bit 16)");
    }

    fn cvt_from_int_case(funct: u32, fmt: u32, val: i64) {
        let mut fpr = [0u64; 32];
        if fmt == crate::mips_isa::RS_W {
            fpr[1] = (val as i32 as u32) as u64;
        } else {
            fpr[1] = val as u64;
        }
        let instr = make_r(crate::mips_isa::OP_COP1, fmt, 0, 1, 2, funct);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);
    }

    #[test]
    fn cvt_s_w_d_w_match_interpreter() {
        cvt_from_int_case(crate::mips_isa::FUNCT_FCVT_S, crate::mips_isa::RS_W, 42);
        cvt_from_int_case(crate::mips_isa::FUNCT_FCVT_D, crate::mips_isa::RS_W, 42);
        cvt_from_int_case(crate::mips_isa::FUNCT_FCVT_S, crate::mips_isa::RS_W, -42);
    }

    #[test]
    fn cvt_s_l_d_l_match_interpreter() {
        cvt_from_int_case(crate::mips_isa::FUNCT_FCVT_S, crate::mips_isa::RS_L, 123456789);
        cvt_from_int_case(crate::mips_isa::FUNCT_FCVT_D, crate::mips_isa::RS_L, 123456789);
        cvt_from_int_case(crate::mips_isa::FUNCT_FCVT_D, crate::mips_isa::RS_L, -123456789);
    }

    // ---- CP1: batch F4 compares — C.cond.fmt ------

    fn fcc_case(funct: u32, fmt: u32, cc: u32, fs_val: f64, ft_val: f64) {
        let mut fpr = [0u64; 32];
        if fmt == crate::mips_isa::RS_S {
            fpr[1] = (fs_val as f32).to_bits() as u64;
            fpr[2] = (ft_val as f32).to_bits() as u64;
        } else {
            fpr[1] = fs_val.to_bits();
            fpr[2] = ft_val.to_bits();
        }
        let instr = make_r(crate::mips_isa::OP_COP1, fmt, 2, 1, cc, funct);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);
    }

    #[test]
    fn c_eq_s_d_match_interpreter() {
        fcc_case(crate::mips_isa::FUNCT_FC_EQ, crate::mips_isa::RS_S, 0, 1.0, 1.0);
        fcc_case(crate::mips_isa::FUNCT_FC_EQ, crate::mips_isa::RS_S, 0, 1.0, 2.0);
        fcc_case(crate::mips_isa::FUNCT_FC_EQ, crate::mips_isa::RS_D, 0, 1.0, 1.0);
    }

    #[test]
    fn c_lt_le_ult_ule_match_interpreter() {
        for (a, b) in [(1.0, 2.0), (2.0, 1.0), (1.0, 1.0)] {
            fcc_case(crate::mips_isa::FUNCT_FC_OLT, crate::mips_isa::RS_S, 0, a, b);
            fcc_case(crate::mips_isa::FUNCT_FC_OLE, crate::mips_isa::RS_S, 0, a, b);
            fcc_case(crate::mips_isa::FUNCT_FC_ULT, crate::mips_isa::RS_S, 0, a, b);
            fcc_case(crate::mips_isa::FUNCT_FC_ULE, crate::mips_isa::RS_S, 0, a, b);
        }
    }

    #[test]
    fn c_all_16_conditions_match_interpreter() {
        let conds = [
            crate::mips_isa::FUNCT_FC_F, crate::mips_isa::FUNCT_FC_UN, crate::mips_isa::FUNCT_FC_EQ,
            crate::mips_isa::FUNCT_FC_UEQ, crate::mips_isa::FUNCT_FC_OLT, crate::mips_isa::FUNCT_FC_ULT,
            crate::mips_isa::FUNCT_FC_OLE, crate::mips_isa::FUNCT_FC_ULE, crate::mips_isa::FUNCT_FC_SF,
            crate::mips_isa::FUNCT_FC_NGLE, crate::mips_isa::FUNCT_FC_SEQ, crate::mips_isa::FUNCT_FC_NGL,
            crate::mips_isa::FUNCT_FC_LT, crate::mips_isa::FUNCT_FC_NGE, crate::mips_isa::FUNCT_FC_LE,
            crate::mips_isa::FUNCT_FC_NGT,
        ];
        for funct in conds {
            fcc_case(funct, crate::mips_isa::RS_S, 0, 2.0, 3.0);
        }
    }

    #[test]
    fn c_cc_index_nonzero_matches_interpreter() {
        // cc=1..7 map to different FCSR bits (24+cc) than cc0's bit 23 —
        // exercise a nonzero index specifically.
        fcc_case(crate::mips_isa::FUNCT_FC_EQ, crate::mips_isa::RS_S, 3, 5.0, 5.0);
        fcc_case(crate::mips_isa::FUNCT_FC_EQ, crate::mips_isa::RS_S, 7, 5.0, 6.0);
    }

    #[test]
    fn c_un_with_nan_matches_interpreter() {
        // UN (unordered) is non-signaling — NaN operands must not raise
        // FCSR.V, just evaluate the (true) condition.
        let mut fpr = [0u64; 32];
        fpr[1] = f32::NAN.to_bits() as u64;
        fpr[2] = (1.0f32).to_bits() as u64;
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 2, 1, 0, crate::mips_isa::FUNCT_FC_UN);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);
    }

    #[test]
    fn c_signaling_nan_raises_v_matches_interpreter() {
        // SEQ (signaling, funct bit 3 set) with a NaN operand must set
        // FCSR.V (Cause+Flag) but NOT raise EXC_FPE since EV is clear by
        // default — confirms the "set V, check EV, don't raise" path.
        let mut fpr = [0u64; 32];
        fpr[1] = f32::NAN.to_bits() as u64;
        fpr[2] = (1.0f32).to_bits() as u64;
        let instr = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 2, 1, 0, crate::mips_isa::FUNCT_FC_SEQ);
        assert_fpu_matches_interpreter(instr, [0u64; 32], fpr, true);
    }

    #[test]
    fn c_signaling_nan_with_ev_enabled_traps_and_matches_interpreter() {
        // Same as above but with FCSR.EV (Enable.V) pre-set via CTC1 first
        // — must raise EXC_FPE immediately, vectoring via handle_exception.
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let word_offset = (pc as u16 / 4) & 0x3FF;
        let mut fpr = [0u64; 32];
        fpr[1] = f32::NAN.to_bits() as u64;
        fpr[2] = (1.0f32).to_bits() as u64;
        let seq = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 2, 1, 0, crate::mips_isa::FUNCT_FC_SEQ);

        // Interpreter: CTC1 to set EV, then the signaling compare.
        let (mut interp_exec, _m) = fpu_seeded_executor([0u64; 32], fpr, pc, true);
        interp_exec.core.fpu_fcsr = 0x800; // EV bit
        interp_exec.exec(seq);
        let interp = CoreSnapshot::capture(&interp_exec.core);

        // JIT: same starting FCSR, compiled region is just the compare.
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[word_offset as usize] = seq;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page, word_offset, 0, 1);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, word_offset, true, false)
            .expect("C.SEQ.S must be compilable");
        let (mut jit_exec, _m2) = fpu_seeded_executor([0u64; 32], fpr, pc, true);
        jit_exec.core.fpu_fcsr = 0x800;
        jit_exec.install_jit_hooks();
        unsafe { jit_fn(&mut jit_exec.core as *mut MipsCore) };
        std::mem::forget(codegen);
        let jit = CoreSnapshot::capture(&jit_exec.core);

        assert_eq!(jit, interp);
        assert_ne!(jit.pc, pc, "signaling NaN with EV enabled must vector via handle_exception");
    }

    /// Isolated repro for a live-boot divergence caught by
    /// `MipsExecutor::lockstep_check_branch` (rules/jitv2/): `BNE $5,$0,+7`
    /// with `$5==0` (not taken) and a delay slot that's an ordinary ALU op
    /// (`LUI $1,0xc0`, not a NOP) — first seen at PROM pc=0xbfc00434 during a
    /// real boot. Drives the exact code path (`exec.exec(instr)` through
    /// `exec_decoded`'s `#[cfg(feature = "jitv2_lockstep")]` gate) rather
    /// than calling `Codegen`/`Analyzer` directly, unlike every other test in
    /// this file — the bug lives in `lockstep_check_branch` itself, not in
    /// codegen, so it only reproduces through the real dispatch gate.
    ///
    /// Uses a low identity-mapped pc (`0x1fc00434`, not the real kseg1
    /// `0xffffffffbfc00434`) deliberately: `MipsExecutor::exec` (this file's
    /// only entry point, and every other jitv2-lockstep test's) is documented
    /// as treating `core.pc` as already physical, true for every existing
    /// caller because they all stick to `PassthroughTlb`'s identity-mapped
    /// range — a kseg1 pc here would need translate_impl's kseg1 masking
    /// that `exec()` doesn't do, corrupting `self.pcp`'s PFN and making this
    /// test fail for a second, unrelated reason on top of the one it's
    /// trying to isolate.
    /// Run exactly one instruction (`instr` at `pc`) through the real dispatch
    /// gate under jitv2_lockstep: places `instr` at `pc` and a region-boundary
    /// sentinel right after it (both virtual and kseg-masked physical, so the
    /// gate's fetch + any fallback re-fetch see the same bytes) so the compiled
    /// region is exactly this one instruction, then dispatches. The lockstep
    /// brackets self-check `instr` against the interpreter as it runs. Replaces
    /// the old `exec.exec(instr)`-with-nothing-in-memory pattern, which relied
    /// on the removed standalone lockstep_check reading the decoded scratch
    /// directly; the redesigned gate fetches the region from memory instead.
    #[cfg(feature = "jitv2_lockstep")]
    fn ls_exec_one(exec: &mut MipsExecutor<PassthroughTlb, PassthroughCache>, mem: &MockMemory, pc: u64, instr: u32) {
        let word_offset = ((pc & 0xFFF) >> 2) as u16;
        let phys_base = pc & !(PAGE_SIZE as u64 - 1);
        let store = |off: u16, raw: u32| {
            mem.set_word(phys_base + (off as u64) * 4, raw);
            mem.set_word((phys_base & 0x1FFF_FFFF) + (off as u64) * 4, raw);
        };
        store(word_offset, instr);
        store(word_offset + 1, crate::mips_isa::JIT_REGION_BOUNDARY_SENTINEL);
        exec.exec(instr);
    }

    /// A lockstep divergence must PRINT a report and return a break signal
    /// (EXEC_BREAKPOINT) — NOT panic/abort the process (an extern "C" callback
    /// panic is a non-unwinding abort that loses the monitor and all state).
    /// `test_force_lockstep_divergence` drives `lockstep_compare` with a
    /// deliberately-mismatched reference and asserts it reports the divergence
    /// (returns true) without unwinding/aborting — the whole point of the
    /// print-and-break-into-monitor redesign.
    #[cfg(feature = "jitv2_lockstep")]
    #[test]
    fn lockstep_divergence_reports_and_breaks_without_panicking() {
        let (mut exec, _mem) = seeded_executor([0u64; 32], 0x1fc00434);
        let diverged = exec.test_force_lockstep_divergence();
        assert!(diverged, "a mismatched reference must be reported as a divergence");
    }

    #[cfg(feature = "jitv2_lockstep")]
    #[test]
    fn lockstep_branch_not_taken_with_alu_delay_slot() {
        let pc = 0x1fc00434u64;
        let branch = make_i(crate::mips_isa::OP_BNE, 5, 0, 7);
        let slot = 0x3c0100c0u32; // LUI $1, 0x00c0
        let word_offset = ((pc & 0xFFF) >> 2) as u16;

        let mut gpr = [0u64; 32];
        gpr[5] = 0; // not taken
        let (mut exec, mem) = seeded_executor(gpr, pc);
        let phys_base = pc & !(PAGE_SIZE as u64 - 1);
        mem.set_word(phys_base + (word_offset as u64) * 4, branch);
        mem.set_word(phys_base + ((word_offset + 1) as u64) * 4, slot);
        exec.install_jit_hooks();

        // Not asserting anything here beyond "doesn't panic" — lockstep_check
        // itself is the assertion (it panics internally on any JIT/interp
        // divergence, see MipsExecutor::lockstep_check_branch).
        exec.exec(branch);
    }

    /// Regression test: `lockstep_check_load_store`'s every early-return
    /// path (codegen gap, analyzer exclusion) used to leave `self.core`
    /// rolled back to *before* the interpreter's real dispatch (the
    /// pre-instruction `LockstepSnapshot` restored ahead of the JIT-probe
    /// attempt) instead of restoring the interpreter's real post-dispatch
    /// state, while still returning `Some(interp_status)` as if the
    /// dispatch had completed normally — the caller (`exec_decoded`) trusts
    /// that pairing as final and never re-dispatches, so `core.pc` silently
    /// never advanced. `CACHE` has no codegen emitter at all (`LockstepClass
    /// ::LoadStore` includes it, but `lookup_semantics`/`lookup_branch_or_jump`
    /// don't), so it always takes the earliest codegen-gap bail — hanging
    /// the CPU on the same `CACHE` instruction forever under
    /// `jitv2_lockstep` (observed live on a real IRIX boot: PROM's L1
    /// cache-init loop stuck re-executing one `CACHE Index_Store_Tag`
    /// indefinitely).
    #[cfg(feature = "jitv2_lockstep")]
    #[test]
    fn lockstep_load_store_codegen_gap_still_advances_pc() {
        let pc = 0x1fc00434u64;
        // CACHE op=Index_Store_Tag(2), cache=PD(1) -> rt field = (2<<2)|1 = 9; base=v0(2), offset=0
        let cache_instr = make_i(crate::mips_isa::OP_CACHE, 2, 9, 0);
        let (mut exec, mem) = seeded_executor([0u64; 32], pc);
        exec.install_jit_hooks();

        ls_exec_one(&mut exec, &mem, pc, cache_instr);
        assert_eq!(exec.core.pc, pc + 4, "CACHE (a codegen gap under jitv2_lockstep) must still advance pc — it must not be silently rolled back to pre-dispatch state");
    }

    /// Basic smoke test for `lockstep_check_fpu` (LockstepClass::Fpu — never
    /// wired into `lockstep_check`'s dispatch before now): a plain FADD.S,
    /// no delay slot, no memory involved, verifying the JIT/interpreter
    /// comparison actually engages for CP1 arithmetic and doesn't panic on
    /// a clean case. Not asserting anything beyond "doesn't panic" —
    /// lockstep_check itself is the assertion, same as the branch/load-store
    /// smoke tests above.
    #[cfg(feature = "jitv2_lockstep")]
    #[test]
    fn lockstep_fpu_add_s_matches_interpreter() {
        let pc = 0x1fc00434u64;
        // FADD.S fd=$f4, fs=$f0, ft=$f2 — even register numbers only: in the
        // default FR0 mode (STATUS_FR unset), single-precision registers
        // pack two per 64-bit fpr[] slot (read_fpr_w_fr0/write_fpr_w_fr0:
        // odd fs/ft/fd address the *upper* 32 bits of fpr[reg & !1], not
        // fpr[reg] directly) — even numbers keep this test's setup simple.
        let fadd_s = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 2, 0, 4, crate::mips_isa::FUNCT_FADD);
        let (mut exec, mem) = seeded_executor([0u64; 32], pc);
        exec.core.cp0_status |= crate::mips_core::STATUS_CU1;
        exec.core.fpr[0] = (1.5f32).to_bits() as u64;
        exec.core.fpr[2] = (2.25f32).to_bits() as u64;
        exec.update_fpr_mode();
        exec.install_jit_hooks();

        ls_exec_one(&mut exec, &mem, pc, fadd_s);
        assert_eq!(exec.core.pc, pc + 4);
        assert_eq!(f32::from_bits(exec.core.fpr[4] as u32), 3.75f32, "FADD.S result must still be correct — not just non-divergent");
    }

    /// Regression test: `lockstep_check_fpu`'s JIT probe compiles a region
    /// whose `emit_fpu_entry_guard` (present whenever the region has any CP1
    /// instruction) checks STATUS_CU1 itself and, if clear, now forces real
    /// forward progress via `core.interp_fallback_fn`
    /// (`emit_interp_fallback_exit`) instead of a plain bail — see that
    /// function's doc comment for why a plain bail can't actually make the
    /// interpreter's real `cpu_unusable` exception fire (the JIT dispatch
    /// gate would just re-call this identical compiled function forever).
    /// `lockstep_check_fpu` itself needs no special-casing for this: its own
    /// `jit.pc == before.pc` check only ever mattered for a genuine
    /// codegen-gap `None` (filtered earlier) or the FR-mismatch arm (still a
    /// plain `emit_bail`) — the CU1 arm now returns a real, already-vectored
    /// PC, which the bottom path's fresh interpreter comparison verifies
    /// against directly, same as any other real JIT result.
    #[cfg(feature = "jitv2_lockstep")]
    #[test]
    fn lockstep_fpu_cu1_unusable_matches_interpreter_no_false_divergence() {
        let pc = 0x1fc00434u64;
        let cfc1 = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_CFC1, 2, 31, 0, 0); // CFC1 $2, $31 (FCSR)
        let (mut exec, mem) = seeded_executor([0u64; 32], pc);
        // STATUS_CU1 deliberately left clear.
        // interp_fallback_fn does a real fetch from core.pc — unlike the old
        // plain-bail guard, the compiled region's own copy of `cfc1` (passed
        // directly to exec() below, never otherwise touching memory) isn't
        // enough on its own; the backing memory must hold the same word at
        // the real physical address for the fallback's own fetch to find it.
        mem.set_word((pc & 0x1FFFFFFF) as u64, cfc1);
        exec.install_jit_hooks();

        // Not asserting anything beyond "doesn't panic" — lockstep_check
        // itself is the assertion (see lockstep_check_fpu's jit.pc ==
        // before.pc guard). Confirm the real, expected outcome too: a
        // genuine EXC_CPU delivered by the interpreter, not silently eaten.
        ls_exec_one(&mut exec, &mem, pc, cfc1);
        assert_ne!(exec.core.pc, pc, "CU1-unusable CFC1 must still vector via handle_exception");
        assert_eq!((exec.core.cp0_cause >> 2) & 0x1F, crate::mips_exec::EXC_CPU);
    }

    /// Regression test: `CVT.W.S` on an already-integer-valued source
    /// (`79.0`) must not set FCSR's Inexact flag — observed live on a real
    /// IRIX boot as a `jitv2_lockstep` FPU divergence: `jit.fcsr=0x1`
    /// (correct, no Inexact) vs `interp.fcsr=0x1005` (Inexact spuriously
    /// set) for exactly this input. Root cause: `exec_fcvt_w_s`'s two-step
    /// `fs_val.round() as i32` implementation used to read the *host* FPU's
    /// Precision sticky flag, which the intermediate `.round()` call (a
    /// real SSE `ROUNDSS` instruction on x86-64) can set even when rounding
    /// an already-integer value — not what MIPS specifies for this
    /// instruction. Fixed by computing Inexact by value instead (does the
    /// int result convert back to a different float than the original
    /// source?), passed via `fpu_update_fcsr_with_inexact_override` — see
    /// `exec_fround_l_s`'s doc comment for the full reasoning.
    #[cfg(feature = "jitv2_lockstep")]
    #[test]
    fn lockstep_fpu_cvt_w_s_exact_integer_value_no_spurious_inexact() {
        let pc = 0x1fc00434u64;
        // CVT.W.S fd=$f18, fs=$f16 — matches the real fs/fd register numbers
        // from the live-boot divergence (both even, so FR0 packing is a
        // non-issue here regardless).
        let cvt_w_s = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 0, 16, 18, crate::mips_isa::FUNCT_FCVT_W);
        let (mut exec, mem) = seeded_executor([0u64; 32], pc);
        exec.core.cp0_status |= crate::mips_core::STATUS_CU1;
        exec.core.fpr[16] = (79.0f32).to_bits() as u64;
        exec.update_fpr_mode();
        exec.install_jit_hooks();

        // lockstep_check itself is the primary assertion (panics internally
        // on any JIT/interp divergence) — confirm the actual FCSR outcome
        // too, not just "didn't diverge", so a future regression that makes
        // both engines agreeably wrong wouldn't slip through silently.
        ls_exec_one(&mut exec, &mem, pc, cvt_w_s);
        assert_eq!(exec.core.pc, pc + 4);
        assert_eq!(exec.core.fpr[18] as i32, 79, "CVT.W.S of 79.0 must produce the integer 79");
        assert_eq!(exec.core.fpu_fcsr & 0x1004, 0, "CVT.W.S of an exact integer value must not set FCSR Inexact (Cause bit 12 or Flag bit 2)");
    }

    /// Counterpart proving Inexact still fires correctly when the source
    /// genuinely isn't an integer, so the fix above isn't just "never sets
    /// Inexact" — `TRUNC.W.S` on `3.7` truncates to `3`, which does differ
    /// from the original source, and MIPS spec says that must set Inexact.
    #[cfg(feature = "jitv2_lockstep")]
    #[test]
    fn lockstep_fpu_trunc_w_s_non_integer_value_sets_inexact() {
        let pc = 0x1fc00434u64;
        let trunc_w_s = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 0, 16, 18, crate::mips_isa::FUNCT_FTRUNC_W);
        let (mut exec, mem) = seeded_executor([0u64; 32], pc);
        exec.core.cp0_status |= crate::mips_core::STATUS_CU1;
        exec.core.fpr[16] = (3.7f32).to_bits() as u64;
        exec.update_fpr_mode();
        exec.install_jit_hooks();

        ls_exec_one(&mut exec, &mem, pc, trunc_w_s);
        assert_eq!(exec.core.pc, pc + 4);
        assert_eq!(exec.core.fpr[18] as i32, 3, "TRUNC.W.S of 3.7 must produce the integer 3");
        assert_ne!(exec.core.fpu_fcsr & 0x1004, 0, "TRUNC.W.S of a non-integer value must set FCSR Inexact");
    }

    /// Live-boot regression: `CVT.W.D $f4, $f6` on `65535.5` with FCSR.RM=1
    /// (round-toward-zero) diverged jit(65536, RM-blind round-half-away-
    /// from-zero) vs interp(65535, RM-aware truncation) — traced to
    /// `f64::round()` itself being empirically sensitive to the host's live
    /// MXCSR rounding-control bits on this build (contrary to its documented
    /// fixed-immediate `ROUNDSD` encoding), so the two engines silently
    /// disagreed on ROUND's own always-round-half-away-from-zero contract
    /// depending on ambient host state. Fixed by routing both engines
    /// through a portable, bit-manipulation-only rounding primitive
    /// (`round_f64_to_int_mode`/`emit_round_to_int_mode`) with no hardware
    /// rounding instruction anywhere, plus making plain CVT.W/L honor
    /// FCSR.RM dynamically (previously a separate, already-documented spec
    /// gap — [[project_fpu_rounding_spec_gap]] — closed in the same pass
    /// since it's the same root cause).
    #[cfg(feature = "jitv2_lockstep")]
    #[test]
    fn lockstep_fpu_cvt_w_d_65535_5_honors_fcsr_rm_toward_zero() {
        let pc = 0x1fc00434u64;
        let cvt_w_d = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_D, 0, 6, 4, crate::mips_isa::FUNCT_FCVT_W);
        let (mut exec, mem) = seeded_executor([0u64; 32], pc);
        exec.core.cp0_status |= crate::mips_core::STATUS_CU1;
        exec.core.fpr[6] = 65535.5f64.to_bits();
        exec.core.fpu_fcsr = 1; // RM = round toward zero
        exec.update_fpr_mode();
        exec.install_jit_hooks();

        ls_exec_one(&mut exec, &mem, pc, cvt_w_d);
        assert_eq!(exec.core.pc, pc + 4);
        assert_eq!(exec.core.fpr[4] as i32, 65535, "CVT.W.D under FCSR.RM=toward-zero must truncate, not round-half-away-from-zero");
    }

    /// Sweep all four FCSR.RM values through CVT.W.D on the same tie value —
    /// `lockstep_check_fpu`'s internal assertion is the primary check (JIT
    /// vs interpreter agreement for every mode), the explicit expected
    /// values additionally confirm both engines are agreeably *correct*,
    /// not just agreeably consistent.
    #[cfg(feature = "jitv2_lockstep")]
    #[test]
    fn lockstep_fpu_cvt_w_d_all_rounding_modes() {
        let pc = 0x1fc00434u64;
        let cvt_w_d = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_D, 0, 6, 4, crate::mips_isa::FUNCT_FCVT_W);
        // (fcsr_rm, expected CVT.W.D(65535.5))
        let cases: [(u32, i32); 4] = [
            (0, 65536), // nearest-even: tie between odd 65535 and even 65536 -> even
            (1, 65535), // toward zero: truncate
            (2, 65536), // toward +inf
            (3, 65535), // toward -inf
        ];
        for (rm, expected) in cases {
            let (mut exec, mem) = seeded_executor([0u64; 32], pc);
            exec.core.cp0_status |= crate::mips_core::STATUS_CU1;
            exec.core.fpr[6] = 65535.5f64.to_bits();
            exec.core.fpu_fcsr = rm;
            exec.update_fpr_mode();
            exec.install_jit_hooks();

            ls_exec_one(&mut exec, &mem, pc, cvt_w_d);
            assert_eq!(exec.core.pc, pc + 4);
            assert_eq!(exec.core.fpr[4] as i32, expected, "CVT.W.D(65535.5) under FCSR.RM={rm} expected {expected}");
        }
    }

    /// Live-boot regression #2: `CVT.W.D $f10, $f10` on `-0.9757914543151855`
    /// under FCSR.RM=0 (nearest-even) diverged jit=0 vs interp=-1 (correct
    /// is -1: |x| is closer to 1 than to 0). This exercises the `exp < 0`
    /// regime of `emit_round_to_int_mode` (magnitude in [0.5, 1.0)) that
    /// `lockstep_fpu_cvt_w_d_all_rounding_modes` never touched (that test
    /// only used 65535.5, magnitude >> 1).
    #[cfg(feature = "jitv2_lockstep")]
    #[test]
    fn lockstep_fpu_cvt_w_d_magnitude_between_half_and_one() {
        let pc = 0x1fc00434u64;
        let cvt_w_d = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_D, 0, 10, 10, crate::mips_isa::FUNCT_FCVT_W);
        // (source, fcsr_rm, expected)
        let cases: [(f64, u32, i32); 12] = [
            (-0.9757914543151855, 0, -1),
            (0.9757914543151855, 0, 1),
            (-0.6, 0, -1),
            (0.6, 0, 1),
            (-0.6, 1, 0),  // toward zero
            (0.6, 1, 0),
            (-0.6, 2, 0),  // toward +inf
            (0.6, 2, 1),
            (-0.6, 3, -1), // toward -inf
            (0.6, 3, 0),
            (-0.5, 0, 0),  // exact half, nearest-even ties to 0
            (0.5, 0, 0),
        ];
        for (src, rm, expected) in cases {
            let (mut exec, mem) = seeded_executor([0u64; 32], pc);
            exec.core.cp0_status |= crate::mips_core::STATUS_CU1;
            exec.core.fpr[10] = src.to_bits();
            exec.core.fpu_fcsr = rm;
            exec.update_fpr_mode();
            exec.install_jit_hooks();

            ls_exec_one(&mut exec, &mem, pc, cvt_w_d);
            assert_eq!(exec.core.pc, pc + 4);
            assert_eq!(exec.core.fpr[10] as i32, expected, "CVT.W.D({src}) under FCSR.RM={rm} expected {expected}");
        }
    }

    /// ROUND.W.S must always round-half-away-from-zero-to-nearest-even
    /// regardless of the live FCSR.RM — unlike CVT.W/L, ROUND/TRUNC/CEIL/
    /// FLOOR ignore FCSR.RM entirely and always use their own fixed mode.
    /// Exercised under a non-default RM specifically because that's exactly
    /// the ambient condition that exposed the host-rounding-instruction
    /// MXCSR sensitivity bug in the first place.
    #[cfg(feature = "jitv2_lockstep")]
    #[test]
    fn lockstep_fpu_round_w_s_ignores_fcsr_rm() {
        let pc = 0x1fc00434u64;
        let round_w_s = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 0, 16, 18, crate::mips_isa::FUNCT_FROUND_W);
        let (mut exec, mem) = seeded_executor([0u64; 32], pc);
        exec.core.cp0_status |= crate::mips_core::STATUS_CU1;
        exec.core.fpr[16] = 65535.5f32.to_bits() as u64;
        exec.core.fpu_fcsr = 1; // RM = toward zero -- must NOT affect ROUND.W.S
        exec.update_fpr_mode();
        exec.install_jit_hooks();

        ls_exec_one(&mut exec, &mem, pc, round_w_s);
        assert_eq!(exec.core.pc, pc + 4);
        assert_eq!(exec.core.fpr[18] as i32, 65536, "ROUND.W.S ties-to-even must still round to 65536 regardless of FCSR.RM");
    }

    /// TRUNC/CEIL/FLOOR must likewise ignore the live FCSR.RM (fixed mode
    /// wins over whatever the dynamic register says) — swept under RM=2
    /// (toward +inf) so a bug that accidentally let FCSR.RM leak into these
    /// fixed-mode handlers would show up as a wrong answer here.
    #[cfg(feature = "jitv2_lockstep")]
    #[test]
    fn lockstep_fpu_trunc_ceil_floor_ignore_fcsr_rm() {
        let pc = 0x1fc00434u64;
        let (mut exec, mem) = seeded_executor([0u64; 32], pc);
        exec.core.cp0_status |= crate::mips_core::STATUS_CU1;
        exec.core.fpr[16] = (2.5f32).to_bits() as u64;
        exec.core.fpu_fcsr = 2; // RM = toward +inf -- must not affect any of these
        exec.update_fpr_mode();
        exec.install_jit_hooks();

        let trunc_w_s = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 0, 16, 18, crate::mips_isa::FUNCT_FTRUNC_W);
        ls_exec_one(&mut exec, &mem, pc, trunc_w_s);
        assert_eq!(exec.core.fpr[18] as i32, 2, "TRUNC.W.S(2.5) must truncate to 2 regardless of FCSR.RM");

        exec.core.pc = pc;
        let ceil_w_s = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 0, 16, 18, crate::mips_isa::FUNCT_FCEIL_W);
        ls_exec_one(&mut exec, &mem, pc, ceil_w_s);
        assert_eq!(exec.core.fpr[18] as i32, 3, "CEIL.W.S(2.5) must round up to 3 regardless of FCSR.RM");

        exec.core.pc = pc;
        let floor_w_s = make_r(crate::mips_isa::OP_COP1, crate::mips_isa::RS_S, 0, 16, 18, crate::mips_isa::FUNCT_FFLOOR_W);
        exec.exec(floor_w_s);
        assert_eq!(exec.core.fpr[18] as i32, 2, "FLOOR.W.S(2.5) must round down to 2 regardless of FCSR.RM");
    }

    // ---- Interpreter-fallback-inside-a-region tests --------------------
    //
    // SPEC (not yet implemented): an analyzer-`Excluded` instruction in the
    // middle of a region must NOT end the region. Instead codegen keeps the
    // region going and, at the excluded word, emits: the normal per-instruction
    // int-check/cycle preamble, then a call to `core.interp_fallback_fn`
    // (`interp_dispatch_one` — runs the excluded instruction through the real
    // interpreter handler), then:
    //   - if the returned status != EXEC_COMPLETE, return it directly (the
    //     handler already delivered whatever exception/fault it raised);
    //   - otherwise fall through to the successor word.
    //
    // The load-bearing insight: `interp_dispatch_one` leaves `core.pc` and
    // `core.in_delay_slot` exactly as the interpreter's own `step()` would —
    // i.e. authoritative external state — so the successor is in the *identical*
    // position to a fresh external entry and must materialize pc+bd from `core`
    // rather than trusting any compile-time constant. That is what makes a
    // fallback's successor "entry-like" and is the crux these tests pin down.
    //
    // Lockstep/trace verify can't reach this path (lockstep compiles one
    // instruction standalone and skips analyzer-excluded words; a boot trace
    // hits it only incidentally), so correctness rests entirely on these unit
    // tests — hence they are written as the spec, ahead of the implementation.
    // They are RED until the analyzer/codegen changes land.

    /// A benign `Excluded` instruction that the interpreter retires with a
    /// plain PC+4 (no exception, minimal privilege requirements): MTC0 to a
    /// CP0 register. `exec_mtc0` needs no kernel/CU0 check (unlike CACHE) and
    /// ends in `handle_exec_complete`, so it is the cleanest stand-in for "an
    /// unsupported-but-normally-retiring instruction" in a fall-through test.
    /// `rt`/`rd` are chosen so both engines apply the identical CP0 write.
    fn benign_excluded_mtc0() -> u32 {
        // MTC0 rt=1 -> rd=4 (CP0 Context): `write_cp0(4, ..)` is a plain field
        // assignment with no timer/scheduling side effects (unlike Count(9)/
        // Compare(11)), so it is a clean, effect-observable stand-in. Value
        // comes from gpr[1], set by the caller. Both engines write it
        // identically; `cp0_context` is outside `CoreSnapshot`, so it never
        // introduces a spurious divergence in the snapshot-comparison tests.
        make_r(crate::mips_isa::OP_COP0, crate::mips_isa::RS_MTC0, 1, 4, 0, 0)
    }

    /// TEST 1 — pc+bd left correct after a fallback.
    ///
    /// A lone excluded instruction as the region's single head. After the JIT
    /// runs it via the fallback path, `core.pc` must be advanced past it (PC+4)
    /// and `core.in_delay_slot` false — exactly what a bare interpreter step
    /// produces. This is Primitive A's core contract in isolation.
    #[test]
    fn fallback_leaves_pc_and_bd_like_interpreter() {
        let _fb = fallback_on_guard();
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let entry_word = ((pc & 0xFFF) / 4) as u16;
        let mut gpr = [0u64; 32];
        gpr[1] = 0x1234_5678;

        let page = [(entry_word, benign_excluded_mtc0())];
        // steps=1: MTC0 retires in a single interpreter dispatch (PC+4).
        // max_instrs=1: the excluded head is the whole region.
        assert_jit_matches_interpreter_page(&page, gpr, pc, entry_word, 1, 1);
    }

    /// TEST 2 — successor materializes pc+bd, does not assume them.
    ///
    /// Region `[normal, EXCLUDED, normal]`. The third word must derive its PC
    /// from live `core` state (which the fallback updated), not from its
    /// compile-time word constant. A happy-path boot would pass even if the
    /// successor wrongly trusted the constant (the two agree there), so the
    /// full three-instruction snapshot comparison — GPR effects of both normal
    /// instructions plus final PC — is what catches a successor that skips
    /// materialization. Interpreter dispatches: ADDIU, MTC0, ADDIU = 3 steps.
    #[test]
    fn successor_of_fallback_is_entry_like() {
        let _fb = fallback_on_guard();
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let entry_word = ((pc & 0xFFF) / 4) as u16;
        let mut gpr = [0u64; 32];
        gpr[1] = 0x40; // MTC0 source; also base for the ADDIUs' operands

        // word0: ADDIU r2 = r1 + 0x11   (pre-fallback normal instruction)
        // word1: MTC0  r1 -> CP0[9]     (excluded -> interpreter fallback)
        // word2: ADDIU r3 = r1 + 0x22   (post-fallback "entry-like" successor)
        let addiu0 = make_i(crate::mips_isa::OP_ADDIU, 1, 2, 0x11);
        let mtc0 = benign_excluded_mtc0();
        let addiu2 = make_i(crate::mips_isa::OP_ADDIU, 1, 3, 0x22);
        let page = [
            (entry_word, addiu0),
            (entry_word + 1, mtc0),
            (entry_word + 2, addiu2),
        ];
        assert_jit_matches_interpreter_page(&page, gpr, pc, entry_word, 3, 3);
    }

    /// TEST 2b — successor after fallback works when pc is NOT on page 0.
    ///
    /// Same shape as TEST 2 but at a high page base, guarding specifically
    /// against a successor that reconstructs its address from a zeroed/assumed
    /// vbase instead of the live `core.pc` the fallback left behind. (The
    /// harness already threads the real page_base through the analyzer; this
    /// makes the off-page-successor case an explicit, named regression.)
    #[test]
    fn successor_of_fallback_off_page_zero() {
        let _fb = fallback_on_guard();
        let pc = 0xFFFF_FFFF_8ABC_D000u64 + 0x40; // deep into a high page
        let entry_word = ((pc & 0xFFF) / 4) as u16;
        let mut gpr = [0u64; 32];
        gpr[1] = 0x7;

        let addiu0 = make_i(crate::mips_isa::OP_ADDIU, 1, 2, 0x11);
        let mtc0 = benign_excluded_mtc0();
        let addiu2 = make_i(crate::mips_isa::OP_ADDIU, 1, 3, 0x22);
        let page = [
            (entry_word, addiu0),
            (entry_word + 1, mtc0),
            (entry_word + 2, addiu2),
        ];
        assert_jit_matches_interpreter_page(&page, gpr, pc, entry_word, 3, 3);
    }

    /// TEST 3 — a non-EXEC_COMPLETE fallback short-circuits the region.
    ///
    /// The excluded instruction raises an exception (SYSCALL). The compiled
    /// unit must return the handler's status immediately and NOT run the
    /// successor — the successor's GPR side effect must be absent. Both engines
    /// end at the exception vector with matching EPC/Cause/Status; the
    /// successor ADDIU (word2) never executes on either side.
    ///
    /// Interpreter steps=2: word0 ADDIU retires, then SYSCALL vectors (one
    /// dispatch that ends in `handle_exception`); the interpreter's `core.pc`
    /// is left at the exception vector and the successor is never fetched — so
    /// the JIT, which returns right after the fallback, must match exactly.
    #[test]
    fn fallback_exception_status_short_circuits_successor() {
        let _fb = fallback_on_guard();
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let entry_word = ((pc & 0xFFF) / 4) as u16;
        let mut gpr = [0u64; 32];
        gpr[1] = 0x100;

        let addiu0 = make_i(crate::mips_isa::OP_ADDIU, 1, 2, 0x11);
        let syscall = make_r(crate::mips_isa::OP_SPECIAL, 0, 0, 0, 0, crate::mips_isa::FUNCT_SYSCALL);
        // Successor: if it ever runs, r3 becomes nonzero — the snapshot would
        // then diverge from the interpreter, which never reaches it.
        let addiu2 = make_i(crate::mips_isa::OP_ADDIU, 0, 3, 0x55);
        let page = [
            (entry_word, addiu0),
            (entry_word + 1, syscall),
            (entry_word + 2, addiu2),
        ];
        // steps=2: ADDIU + SYSCALL(-vectoring). max_instrs=3 so the walker is
        // free to (wrongly, pre-fix) try to include word2 — the test asserts
        // it must not run at execution time regardless of what was walked.
        assert_jit_matches_interpreter_page(&page, gpr, pc, entry_word, 2, 3);

        // Belt-and-suspenders: prove the successor's write is actually absent
        // in the JIT result (not merely that both engines agree). r3 must be 0.
        let jit = run_jit_page(&page, gpr, pc, entry_word, 3, &[])
            .expect("region containing a fallback must still compile");
        assert_eq!(jit.gpr[3], 0,
            "successor after an exception-raising fallback must not execute");
    }

    /// TEST 4 — a word reachable BOTH as a normal fallthrough AND as a
    /// fallback-successor executes correctly on both paths.
    ///
    /// This is the "two versions if necessary" case: a word that is entered
    /// once as an ordinary in-region fallthrough (pc+bd compile-time-known) and
    /// also as the successor of a fallback (pc+bd from `core`) needs an
    /// entry-like variant without breaking the ordinary-edge variant. Layout:
    ///
    ///   word0: BEQ r0,r0 -> word3     (taken; skips the excluded word)
    ///   word1: <delay slot: NOP/ADDIU>
    ///   word2: MTC0 (excluded)        (fallthrough into word3 via fallback)
    ///   word3: ADDIU r3 = r1 + 1      (target of the branch AND successor of
    ///                                  the word2 fallback)
    ///
    /// The branch path reaches word3 as a normal taken-target (never touching
    /// word2); a straight-line entry at word2 reaches word3 as a fallback
    /// successor. Both must land word3 with identical results. We run the two
    /// entries separately (entry_word=0 for the branch path, entry_word=2 for
    /// the fallback-successor path) and compare each against the interpreter.
    #[test]
    fn dual_reachable_successor_word_has_both_variants() {
        let _fb = fallback_on_guard();
        let pc_base = 0xFFFF_FFFF_8000_1000u64;
        let entry0 = ((pc_base & 0xFFF) / 4) as u16;
        let mut gpr = [0u64; 32];
        gpr[1] = 0x10;

        // BEQ r0,r0,+2 (from word0, delay slot word1, target word3).
        // word4 is the region-boundary sentinel terminating the region for BOTH
        // entry paths deterministically (no delay slot, never executed), so the
        // interpreter's step count and the JIT's run-to-exit stop at the same pc
        // (word4, just past word3).
        let beq = make_i(crate::mips_isa::OP_BEQ, 0, 0, 2);
        let nop = 0u32;
        let mtc0 = benign_excluded_mtc0();
        let addiu3 = make_i(crate::mips_isa::OP_ADDIU, 1, 3, 1);
        let page = [
            (entry0, beq),
            (entry0 + 1, nop),
            (entry0 + 2, mtc0),
            (entry0 + 3, addiu3),
            (entry0 + 4, crate::mips_isa::JIT_REGION_BOUNDARY_SENTINEL),
        ];

        // Path A: enter at the branch. Interpreter dispatches: BEQ (arms delay),
        // NOP slot (retires -> target word3), ADDIU3 = 3 dispatches, landing at
        // word4 (the boundary, never executed). JIT runs to the same boundary.
        assert_jit_matches_interpreter_page(&page, gpr, pc_base, entry0, 3, 5);

        // Path B: enter straight-line at the excluded word. Interpreter: MTC0
        // (fallback), ADDIU3 = 2 dispatches, landing at word4. Same word3,
        // reached the other way; same boundary exit.
        let pc_b = pc_base + 2 * 4;
        assert_jit_matches_interpreter_page(&page, gpr, pc_b, entry0 + 2, 2, 5);
    }

    /// TEST 5 — the int-check fires on the fallback instruction itself.
    ///
    /// The fallback word must run its own per-instruction int-check before
    /// calling the interpreter — the property that distinguishes it from
    /// today's entry instruction, which relies on `step()` having already
    /// checked. We enter the region **directly at the excluded word** with an
    /// interrupt already pending, so the only thing that can stop the fallback
    /// from running is the fallback word's own preamble. The compiled unit must
    /// bail at the excluded word's own PC (interrupt still pending, for the
    /// interpreter's `step()` to deliver) and the fallback's side effect (the
    /// CP0 write) must be absent.
    ///
    /// (A mid-region variant — interrupt latched only after a preceding word
    /// retires — can't be expressed against a single native region call, which
    /// runs start-to-finish atomically; entering at the excluded word isolates
    /// the fallback-word preamble as the sole gate, which is exactly the
    /// property under test.)
    ///
    /// Uses the executor path directly (not `assert_jit_matches_interpreter_page`,
    /// whose interpreter side has no interrupt-delivery step wired) so we can
    /// assert the precise exit PC and the absence of the fallback's effect.
    #[test]
    fn fallback_performs_int_check_before_running() {
        let _fb = fallback_on_guard();
        // Excluded word deliberately NOT at word 0, so "bailed at entry" and
        // "bailed at the excluded word" are distinguishable PCs.
        let excluded_word: u16 = 5;
        let page_base_v = 0xFFFF_FFFF_8000_1000u64;
        let pc = page_base_v + (excluded_word as u64) * 4; // enter at the excluded word
        let mut gpr = [0u64; 32];
        gpr[1] = 0xDEAD_BEEF;

        let mtc0 = benign_excluded_mtc0();
        let page = [(excluded_word, mtc0)];

        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(w, raw) in &page { page_words[w as usize] = raw; }
        let page_base = (pc & !(PAGE_SIZE as u64 - 1)) as u32;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, excluded_word, page_base, 1);
        assert!(non_empty, "a lone excluded word must be a compilable (fallback) region");
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, excluded_word, true, false)
            .expect("region containing a fallback must still compile");

        let (exec, mem) = seeded_executor(gpr, pc);
        let mut exec = Box::new(exec);
        for &(w, raw) in &page { mem.set_word(page_base as u64 + (w as u64) * 4, raw); }
        exec.install_jit_hooks();
        // Latch a pending interrupt so the fallback word's own preamble bails
        // before the fallback runs. The preamble predicate mirrors step()'s
        // own `hot.interrupts` load.
        exec.core.hot.interrupts.store(1 << 10, std::sync::atomic::Ordering::Relaxed);
        // benign_excluded_mtc0 writes gpr[1] into CP0 Context (reg 4). Seed a
        // distinct sentinel so a run of the fallback would be unmistakable.
        exec.core.cp0_context = 0;
        let cp0_before = exec.core.cp0_context; // MTC0 target; must stay unchanged

        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        std::mem::forget(codegen);

        let excluded_pc = page_base_v | ((excluded_word as u64) * 4);
        assert_eq!(exec.core.pc, excluded_pc,
            "region must bail at the excluded word's own PC so the interpreter delivers the pending IRQ");
        assert_eq!(exec.core.cp0_context, cp0_before,
            "the excluded instruction's CP0 write must not have run — int-check must precede the fallback");
    }

    /// TEST 7 — a fallback that MOVES PC (returns EXEC_COMPLETE but pc !=
    /// fallback_pc + 4) must NOT fall through to the compile-time successor.
    ///
    /// The successor-is-entry-like rule ("if fallback returns EXEC_COMPLETE,
    /// continue to the next word") is only safe when the fallback actually
    /// advanced PC by one word. Some Excluded instructions retire with
    /// EXEC_COMPLETE yet relocate PC: ERET (-> EPC), BC1 (taken CP1 branch).
    /// For those the successor block's compile-time PC assumption is wrong, and
    /// blindly running it would execute the wrong instruction with the wrong
    /// architectural PC. The fallback continuation must therefore check
    /// `core.pc == fallback_pc + 4` and, if not, return EXEC_COMPLETE so the
    /// interpreter re-dispatches at wherever PC now points.
    ///
    /// The entry instruction path does NOT already cover this: it only checks
    /// `core.in_delay_slot` (the foreign-slot case), and it never runs a prior
    /// instruction that could have moved PC — so this is genuinely new to the
    /// fallback continuation, not inherited from entry semantics.
    ///
    /// ERET is the cleanest vehicle: Excluded, returns EXEC_COMPLETE (via
    /// `exec_complete_pc_set`), and sets `core.pc = EPC`. We enter at the ERET
    /// word with EPC pointed at a distinct address and a successor that writes a
    /// GPR; the unit must exit with `core.pc == EPC` and the successor's write
    /// absent.
    #[test]
    fn fallback_that_moves_pc_does_not_run_successor() {
        let _fb = fallback_on_guard();
        let excluded_word: u16 = 3;
        let page_base_v = 0xFFFF_FFFF_8000_1000u64;
        let pc = page_base_v + (excluded_word as u64) * 4; // enter at the ERET word
        let epc = 0xFFFF_FFFF_9000_2000u64; // distinct target, different page
        let gpr = [0u64; 32];

        // word3: ERET (excluded; retires EXEC_COMPLETE, pc <- EPC)
        // word4: ADDIU r5 = r0 + 0x77  (must NOT run — PC moved to EPC)
        let eret = make_r(crate::mips_isa::OP_COP0, crate::mips_isa::RS_TLB, 0, 0, 0, crate::mips_isa::FUNCT_ERET);
        let addiu_succ = make_i(crate::mips_isa::OP_ADDIU, 0, 5, 0x77);
        let page = [(excluded_word, eret), (excluded_word + 1, addiu_succ)];

        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(w, raw) in &page { page_words[w as usize] = raw; }
        let page_base = (pc & !(PAGE_SIZE as u64 - 1)) as u32;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, excluded_word, page_base, 2);
        assert!(non_empty, "a region entered at an excluded (fallback) word must compile");
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, excluded_word, true, false)
            .expect("region containing a fallback must still compile");

        let (exec, mem) = seeded_executor(gpr, pc);
        let mut exec = Box::new(exec);
        // interp_dispatch_one re-fetches the fallback word from the bus, which
        // sees the *physical* (kseg-masked) address — store code there, not at
        // the unmasked virtual page base (a native-emitter region never
        // re-fetches, so run_jit_page's unmasked store is only harmless there).
        let phys_base = (page_base & 0x1FFF_FFFF) as u64;
        for &(w, raw) in &page { mem.set_word(phys_base + (w as u64) * 4, raw); }
        exec.install_jit_hooks();
        // Put the CPU at exception level with EPC set so ERET returns to EPC.
        // A reset core has STATUS_ERL set (mips_core.rs's reset:
        // BEV|ERL) — exec_eret checks ERL *first* and would return to
        // ErrorEPC(0), so clear ERL and set EXL to exercise the EPC path.
        exec.core.cp0_status &= !crate::mips_core::STATUS_ERL;
        exec.core.cp0_status |= crate::mips_core::STATUS_EXL;
        exec.core.cp0_epc = epc;

        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        std::mem::forget(codegen);

        assert_eq!(exec.core.pc, epc,
            "ERET fallback must leave pc at EPC, and the unit must exit there — not fall through to the successor");
        assert_eq!(exec.core.gpr[5], 0,
            "successor must not run after a fallback that relocated pc (pc != fallback_pc+4)");
    }

    /// TEST 8 — LL/SC CAS loop verbatim from a live IRIX boot that faulted with
    /// fallback ON (`compare_and_swap_ptr`, reached via mutex_lock). LL and SC
    /// are both `Classify::Excluded` -> fallback heads, and the loop has a
    /// backward branch (`beq v0,zero,-5`) landing on the LL fallback head, plus
    /// a forward `bne` exit and a `jr ra` exit — a fallback-density/structure
    /// the earlier synthetic tests never covered. Runs the "swap succeeds first
    /// try" path (LL loads the expected value, BNE not taken, SC stores, BEQ not
    /// taken, JR exits) through both engines and asserts identical state. This
    /// is the reduced repro for the live s3-corruption regression.
    #[test]
    fn ll_sc_cas_loop_matches_interpreter_with_fallback() {
        let _fb = fallback_on_guard();
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let entry0 = ((pc & 0xFFF) / 4) as u16;

        // Data word CAS operates on. Same base/mem_init convention as the LW/SW
        // equiv tests (proven to translate correctly under PassthroughTlb).
        let data_vaddr = 0xFFFF_FFFF_8010_0000u64;
        let mut gpr = [0u64; 32];
        gpr[4] = data_vaddr;   // a0 = &word
        gpr[5] = 0;            // a1 = expected (compare value); memory holds 0 too
        gpr[6] = 0x1234_5678;  // a2 = new value to store
        gpr[31] = pc + 0x400;  // ra = off-region return target

        // Verbatim encodings from the boot trace (compare_and_swap_ptr):
        let ll   = 0xc08c0000u32; // ll   t4, 0(a0)      Excluded -> fallback
        let bne  = 0x15850006u32; // bne  t4, a1, +6     (not taken when equal)
        let or_  = 0x00c01025u32; // or   v0, a2, zero   (bne's delay slot)
        let sc   = 0xe0820000u32; // sc   v0, 0(a0)      Excluded -> fallback
        let beq  = 0x1040fffbu32; // beq  v0, zero, -5   (retry; not taken on success)
        let nop  = 0x00000000u32;
        let jr   = 0x03e00008u32; // jr   ra
        let page = [
            (entry0,     ll),
            (entry0 + 1, bne),
            (entry0 + 2, or_),
            (entry0 + 3, sc),
            (entry0 + 4, beq),
            (entry0 + 5, nop),
            (entry0 + 6, jr),
            (entry0 + 7, nop), // jr's delay slot
        ];

        // Interpreter dispatch count for the success path: ll(1), bne(2, arms
        // delay), or-slot(3), sc(4), beq(5, arms delay), nop-slot(6), jr(7,
        // arms delay), nop-slot(8) = 8. JIT runs to the jr exit.
        let mem_init = &[(data_vaddr, 0u32)];
        let interp = run_interpreter_page(&page, gpr, pc, 8);
        let jit = run_jit_page(&page, gpr, pc, entry0, 8, mem_init)
            .expect("LL/SC CAS region must compile with fallback on");
        assert_eq!(jit, interp,
            "JIT (fallback on) diverged from interpreter on the LL/SC CAS loop");
    }

    /// TEST 8c — the FULL live repro: caller -> mutex_lock -> compare_and_swap_ptr,
    /// three functions on three different physical pages, driven through the real
    /// `step()` dispatch gate so every cross-page jal/jr flows through the
    /// interpreter exactly as it does live. compare_and_swap_ptr is the LL/SC
    /// CAS loop (fallback heads); mutex_lock and the caller hold callee-saved
    /// registers (s3, ra, sp) live across the calls. Reproduces the exact
    /// structure of the live s3-corruption panic and asserts the JIT (fallback
    /// on, inline compile) matches a pure interpreter run bit-for-bit.
    #[cfg(not(feature = "lightning"))]
    #[test]
    fn full_mutex_lock_cas_call_chain_matches_interpreter() {
        let _fb = fallback_on_guard();

        // Real kseg0 virtual addresses from the boot trace, so cross-page
        // classification (J/JAL absolute target, RegJump return) matches live.
        let cas_base    = 0xFFFF_FFFF_8000_FE00u64; // compare_and_swap_ptr
        let mutex_base  = 0xFFFF_FFFF_800E_A388u64; // mutex_lock
        let caller_base = 0xFFFF_FFFF_8023_7350u64; // synthetic caller

        // compare_and_swap_ptr (verbatim).
        let cas = [
            (cas_base + 0x00, 0xc08c0000u32), // ll   t4, 0(a0)
            (cas_base + 0x04, 0x15850006u32), // bne  t4, a1, +6  -> cas+0x20 (off-region here)
            (cas_base + 0x08, 0x00c01025u32), // or   v0, a2, zero (slot)
            (cas_base + 0x0c, 0xe0820000u32), // sc   v0, 0(a0)
            (cas_base + 0x10, 0x1040fffbu32), // beq  v0, zero, -5 (retry -> cas)
            (cas_base + 0x14, 0x00000000u32), // nop
            (cas_base + 0x18, 0x03e00008u32), // jr   ra
            (cas_base + 0x1c, 0x00000000u32), // nop (slot)
            // bne's off-region target (cas+0x20): return-ish path — just jr ra.
            (cas_base + 0x20, 0x03e00008u32), // jr ra
            (cas_base + 0x24, 0x00000000u32), // nop (slot)
        ];

        // mutex_lock (verbatim, jal target rewritten to our cas_base's low 28b —
        // JAL is absolute-in-256MB, and cas_base is in the same 256MB region).
        let jal_cas = 0x0c00_0000u32 | (((cas_base & 0x0FFF_FFFF) >> 2) as u32);
        let mutex = [
            // Real code has `lw a2, -24552(zero)` (a kernel global); that
            // address doesn't map in this harness and would fault both engines
            // into the exception vector (uncharted all-zero memory) — noise for
            // this test. a2 only feeds cas's `or v0,a2,zero` store value, which
            // is irrelevant to equivalence, so substitute `or a2,zero,zero`.
            (mutex_base + 0x00, 0x00003025u32), // or a2, zero, zero  (was: lw a2,-24552(zero))
            (mutex_base + 0x04, 0x27bdffe0u32), // addiu sp, sp, -32
            (mutex_base + 0x08, 0xffa40000u32), // sd a0, 0(sp)
            (mutex_base + 0x0c, 0xffa50008u32), // sd a1, 8(sp)
            (mutex_base + 0x10, 0xffbf0010u32), // sd ra, 16(sp)
            (mutex_base + 0x14, jal_cas),       // jal compare_and_swap_ptr
            (mutex_base + 0x18, 0x00002825u32), // or a1, zero, zero (slot)
            (mutex_base + 0x1c, 0x10400004u32), // beq v0, zero, +4 -> mutex+0x30
            (mutex_base + 0x20, 0xdfa40000u32), // ld a0, 0(sp)
            (mutex_base + 0x24, 0xdfbf0010u32), // ld ra, 16(sp)
            (mutex_base + 0x28, 0x03e00008u32), // jr ra
            (mutex_base + 0x2c, 0x27bd0020u32), // addiu sp, sp, 32 (slot)
            (mutex_base + 0x30, 0x03e00008u32), // jr ra (the beq-taken path)
            (mutex_base + 0x34, 0x27bd0020u32), // addiu sp, sp, 32 (slot)
        ];

        // Synthetic caller: mirrors the trace's shape around 0x88237354 — set up
        // a1, jal mutex_lock, then the faulting `lw t6, 84(s3)`, then a clean
        // sentinel exit (jr ra to a fixed return address the driver stops at).
        let jal_mutex = 0x0c00_0000u32 | (((mutex_base & 0x0FFF_FFFF) >> 2) as u32);
        let caller = [
            (caller_base + 0x00, 0x24050014u32), // addiu a1, zero, 20
            (caller_base + 0x04, 0x02c02025u32), // or a0, s6, zero
            (caller_base + 0x08, jal_mutex),     // jal mutex_lock
            (caller_base + 0x0c, 0x24050018u32), // addiu a1, zero, 24 (slot)
            (caller_base + 0x10, 0x8e6e0054u32), // lw t6, 84(s3)   <-- the live fault site
            (caller_base + 0x14, 0x03e00008u32), // jr ra  (return to sentinel)
            (caller_base + 0x18, 0x00000000u32), // nop (slot)
        ];

        let mut code: Vec<(u64, u32)> = Vec::new();
        code.extend_from_slice(&cas);
        code.extend_from_slice(&mutex);
        code.extend_from_slice(&caller);

        // Data: a valid stack, a valid s3 with [s3+84] populated, and the mutex
        // word cas operates on. The `lw a2,-24552(zero)` global cas reads is
        // left unset (reads 0 on both engines — its exact value is irrelevant to
        // equivalence, only that both see the same thing).
        let stack_top = 0xFFFF_FFFF_8020_0000u64;
        let s3_ptr    = 0xFFFF_FFFF_8021_0000u64;
        let mutex_wrd = 0xFFFF_FFFF_8022_0000u64;
        let data = [
            (s3_ptr + 84, 0x0000_0042u32),  // [s3+84] — the lw the fault site reads
            (mutex_wrd, 0x0000_0000u32),    // mutex starts unlocked (== a1 expected)
        ];

        let mut gpr = [0u64; 32];
        gpr[19] = s3_ptr;                 // s3 (callee-saved, live across calls)
        gpr[22] = mutex_wrd;              // s6 -> a0 (the mutex pointer)
        gpr[29] = stack_top;              // sp
        gpr[31] = 0xFFFF_FFFF_8000_0000u64; // ra sentinel (top-level return target)

        let pc = caller_base;
        // Generous step budget; both engines quiesce at the ra sentinel (an
        // all-zero page — they'll just spin on NOPs identically past that, which
        // is fine since we compare final state after a fixed count).
        let steps = 60;
        let interp = run_multipage(&code, &data, gpr, pc, steps, false);
        let jit    = run_multipage(&code, &data, gpr, pc, steps, true);
        assert_eq!(jit, interp,
            "JIT (fallback on) diverged from interpreter on the full mutex_lock/CAS call chain");
    }

    /// BC1 (branch on CP1 condition) is `Classify::Excluded`, so with fallback
    /// on it's a fallback head — AND it's a *branch*: running it through the
    /// interpreter arms a delay slot. Its successor (the slot) must be treated
    /// as an entry-like word (honor the pending `delay_slot_target`), or the
    /// branch transfer is silently dropped. Drives a region entered at the BC1
    /// fallback through the real `step()` gate and compares JIT vs interpreter,
    /// including the target transfer, for taken / not-taken-non-likely /
    /// likely-annul. `run_multipage` seeds no FPU state, so this builds the
    /// executors directly to set CU1 + a condition code.
    #[cfg(not(feature = "lightning"))]
    fn assert_bc1_fallback_matches(bc1: u32, cc0: bool, steps: usize) {
        let pc = 0xFFFF_FFFF_8000_1000u64;
        // word0: BC1 cc0 +2 (fallback, branch). target = word0+1+2 = word3.
        // word1: ADDIU r1,r0,0x11   BC1's delay slot (runs unless likely-annulled)
        // word2: ADDIU r2,r0,0x22   not-taken fallthrough
        // word3: ADDIU r3,r0,0x33   taken target
        // word4: JR ra              shared exit — both paths funnel here
        // word5: NOP                jr's delay slot
        // Not-taken (word2) falls through to word3 then word4; taken jumps to
        // word3 then word4 — both quiesce at `ra`, so a fixed step count lands
        // both engines in the same place regardless of which arm ran. The
        // difference the test actually checks is which of r1/r2/r3 got set.
        let addiu1 = make_i(crate::mips_isa::OP_ADDIU, 0, 1, 0x11);
        let addiu2 = make_i(crate::mips_isa::OP_ADDIU, 0, 2, 0x22);
        let addiu3 = make_i(crate::mips_isa::OP_ADDIU, 0, 3, 0x33);
        let jr = make_r(crate::mips_isa::OP_SPECIAL, 31, 0, 0, 0, crate::mips_isa::FUNCT_JR);
        // `ra` lands on a `jr $ra`-to-SELF (ra points at the jr itself), whose
        // delay slot is a nop. This pins pc at `ra` and, being a RegJump (always
        // a region boundary), exits the JIT region every dispatch — so `step()`
        // returns each time and the fixed step count lands both engines pinned
        // at `ra` identically, instead of spinning forward through NOP memory at
        // different rates (a BEQ self-loop, by contrast, compiles to an infinite
        // in-region loop the JIT never exits within one step() — it hangs).
        let ra = pc + 0x400;
        let code = [
            (pc + 0x00, bc1),
            (pc + 0x04, addiu1),
            (pc + 0x08, addiu2),
            (pc + 0x0c, addiu3),
            (pc + 0x10, jr),
            (pc + 0x14, 0),
            (ra + 0x00, jr),  // jr $ra, and $ra == ra -> self loop, exits region each dispatch
            (ra + 0x04, 0),   // jr's delay slot
        ];

        let build = |jit: bool| -> CoreSnapshot {
            let mem = if jit { MockMemory::new() } else { MockMemory::new_not_compilable() };
            let mut gpr = [0u64; 32];
            gpr[31] = ra;
            let (exec, mem) = seeded_executor_over(mem, gpr, pc);
            let mut exec = Box::new(exec);
            let store = |vaddr: u64, val: u32| { mem.set_word(vaddr, val); mem.set_word(vaddr & 0x1FFF_FFFF, val); };
            for &(vaddr, raw) in &code { store(vaddr, raw); }
            exec.core.cp0_status |= crate::mips_core::STATUS_CU1;
            exec.core.set_fpu_cc(0, cc0);
            exec.update_fpr_mode();
            if jit { exec.jitv2_inline_compile = true; exec.install_jit_hooks(); }
            // Step until BOTH engines settle at `ra` with no transfer pending
            // (pc == ra && !in_delay_slot) — a fixed step count would capture
            // the two engines at different sub-steps of `ra`'s jr-self 2-dispatch
            // cycle (jr then slot), a quiescence artifact, not a real divergence.
            // `steps` is a safety cap.
            for _ in 0..steps {
                if exec.core.pc == ra && !exec.core.in_delay_slot { break; }
                exec.step();
            }
            CoreSnapshot::capture(&exec.core)
        };
        let interp = build(false);
        let jit = build(true);
        assert_eq!(jit, interp,
            "BC1 fallback (cc0={}) diverged from interpreter — the delay-slot transfer must be preserved", cc0);
    }

    #[cfg(not(feature = "lightning"))]
    #[test]
    fn bc1_fallback_taken_matches_interpreter() {
        let _fb = fallback_on_guard();
        // BC1T cc0, +2 (target = word0+1+2 = word3). cc0=true -> taken.
        let bc1t = 0x45010002u32;
        assert_bc1_fallback_matches(bc1t, true, 12);
    }

    #[cfg(not(feature = "lightning"))]
    #[test]
    fn bc1_fallback_not_taken_matches_interpreter() {
        let _fb = fallback_on_guard();
        // BC1T cc0, +2. cc0=false -> not taken (non-likely): slot still runs.
        let bc1t = 0x45010002u32;
        assert_bc1_fallback_matches(bc1t, false, 12);
    }

    #[cfg(not(feature = "lightning"))]
    #[test]
    fn bc1_fallback_likely_annul_matches_interpreter() {
        let _fb = fallback_on_guard();
        // BC1TL cc0, +2 (likely). cc0=false -> not taken -> slot ANNULLED (pc+8).
        // rt = (cc<<2)|(likely<<1)|tf = (0<<2)|(1<<1)|1 = 3.
        let bc1tl = (crate::mips_isa::OP_COP1 << 26) | (crate::mips_isa::RS_BC1 << 21) | (3u32 << 16) | 2;
        assert_bc1_fallback_matches(bc1tl, false, 12);
    }

    /// TEST 8b — a backward branch whose TARGET is a fallback head, taken
    /// multiple times (the CAS retry loop's structure: `beq ... , -N` landing
    /// on the LL fallback). Exercises re-entering a fallback head via an
    /// in-region back-edge repeatedly — distinct from TEST 8's single pass.
    /// Uses a plain countdown so the iteration count is deterministic (no
    /// SC-fail nondeterminism needed): word0 is a benign excluded MTC0
    /// (fallback head + loop top), word1 decrements the counter, word2 is a
    /// backward BNE to word0 while the counter is nonzero.
    #[test]
    fn backward_branch_target_is_fallback_head_looped() {
        let _fb = fallback_on_guard();
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let entry0 = ((pc & 0xFFF) / 4) as u16;
        let mut gpr = [0u64; 32];
        gpr[1] = 3; // loop counter

        // word0: MTC0 (Excluded -> fallback head; also the loop-top / branch
        //        target). Benign write to CP0 Context (reg 4).
        // word1: addiu r1, r1, -1   (decrement counter)
        // word2: bne  r1, zero, -3  (target = 2+1+(-3) = word0)
        // word3: nop (bne's delay slot)
        // word4: region-boundary sentinel (clean end)
        let mtc0 = benign_excluded_mtc0();
        let dec  = make_i(crate::mips_isa::OP_ADDIU, 1, 1, 0xFFFF); // -1
        let bne  = make_i(crate::mips_isa::OP_BNE, 1, 0, (-3i16) as u16);
        let page = [
            (entry0,     mtc0),
            (entry0 + 1, dec),
            (entry0 + 2, bne),
            (entry0 + 3, 0),
            (entry0 + 4, crate::mips_isa::JIT_REGION_BOUNDARY_SENTINEL),
        ];

        // Interpreter dispatches per iteration: mtc0(1) + dec(2) + bne(3, arms
        // delay) + nop-slot(4) = 4. r1: 3->2->1->0; the bne re-takes while
        // r1!=0 (2 retakes) then falls through on r1==0. 3 iterations * 4 = 12,
        // landing at word4 (the boundary, never executed).
        let interp = run_interpreter_page(&page, gpr, pc, 12);
        let jit = run_jit_page(&page, gpr, pc, entry0, /*max_instrs=*/8, &[])
            .expect("looping-fallback region must compile with fallback on");
        assert_eq!(jit, interp,
            "JIT (fallback on) diverged on a backward branch into a fallback head");
    }

    /// TEST 7b — the PC-advance check must compare against THIS instruction's
    /// page, not the post-fallback page. Regression for a real bug: step (4)
    /// originally computed the expected-successor address from a fresh
    /// `emit_word_addr(word+1)`, which reads live `core.pc` — already relocated
    /// by the fallback. If the fallback moves pc to a *different page* whose
    /// in-page offset happens to equal `(word+1)*4`, that stale-vbase compare
    /// would spuriously match and wrongly run the successor. Here ERET lands on
    /// EPC on a different page with low bits `== (excluded_word+1)*4`; the
    /// unit must still detect "pc moved" (via `own_pc + 4`) and exit at EPC
    /// without running the successor. TEST 7 passed only because its EPC low
    /// bits differed from the successor offset, so it never exercised this.
    #[test]
    fn fallback_pc_move_to_aliasing_offset_on_other_page() {
        let _fb = fallback_on_guard();
        let excluded_word: u16 = 3;
        let page_base_v = 0xFFFF_FFFF_8000_1000u64;
        let pc = page_base_v + (excluded_word as u64) * 4;
        // EPC: a DIFFERENT page, but low bits == (excluded_word+1)*4 — i.e. the
        // exact in-page offset the buggy stale-vbase compare would alias to.
        let epc = 0xFFFF_FFFF_9000_0000u64 | (((excluded_word as u64) + 1) * 4);
        let gpr = [0u64; 32];

        let eret = make_r(crate::mips_isa::OP_COP0, crate::mips_isa::RS_TLB, 0, 0, 0, crate::mips_isa::FUNCT_ERET);
        let addiu_succ = make_i(crate::mips_isa::OP_ADDIU, 0, 5, 0x77);
        let page = [(excluded_word, eret), (excluded_word + 1, addiu_succ)];

        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        for &(w, raw) in &page { page_words[w as usize] = raw; }
        let page_base = (pc & !(PAGE_SIZE as u64 - 1)) as u32;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, excluded_word, page_base, 2);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        let mut codegen = Codegen::new();
        let jit_fn: JitFn = codegen.compile_region(&mut instrs_owned, excluded_word, true, false)
            .expect("region containing a fallback must still compile");

        let (exec, mem) = seeded_executor(gpr, pc);
        let mut exec = Box::new(exec);
        let phys_base = (page_base & 0x1FFF_FFFF) as u64;
        for &(w, raw) in &page { mem.set_word(phys_base + (w as u64) * 4, raw); }
        exec.install_jit_hooks();
        exec.core.cp0_status &= !crate::mips_core::STATUS_ERL;
        exec.core.cp0_status |= crate::mips_core::STATUS_EXL;
        exec.core.cp0_epc = epc;

        unsafe { jit_fn(&mut exec.core as *mut MipsCore) };
        std::mem::forget(codegen);

        assert_eq!(exec.core.pc, epc,
            "must exit at EPC even though EPC's in-page offset aliases the successor's");
        assert_eq!(exec.core.gpr[5], 0,
            "successor must not run — the aliasing offset must not fool the PC-advance check");
    }

    /// TEST 6 — an excluded instruction in a branch's DELAY SLOT.
    ///
    /// This is the genuinely-new placement: a slot can't "fall through to the
    /// next block" (its contract is to continue into the branch's own target
    /// logic in the same block — see `emit_slot_semantics`). The eventual
    /// implementation must handle it like a nested-branch slot (exit the
    /// function after the fallback), OR — for a first cut — deliberately keep
    /// treating an excluded slot as a hard region boundary (declining to
    /// compile the branch), which is rare and loses little.
    ///
    /// Marked `#[ignore]` for now: it documents the slot case as a known,
    /// separately-scoped follow-up rather than asserting a behavior the first
    /// implementation cut is expected to provide. Remove the ignore when the
    /// slot-position fallback lands. Encodes the interpreter's own reference
    /// result via the executor so it's ready to compare once un-ignored.
    #[test]
    #[ignore = "delay-slot-position fallback is a separately-scoped follow-up"]
    fn fallback_in_delay_slot() {
        let _fb = fallback_on_guard();
        let pc = 0xFFFF_FFFF_8000_1000u64;
        let entry_word = ((pc & 0xFFF) / 4) as u16;
        let mut gpr = [0u64; 32];
        gpr[1] = 0x10;

        // word0: BEQ r0,r0,+2 (target word3)
        // word1: MTC0 (excluded) IN THE DELAY SLOT
        // word3: ADDIU r3 = r1 + 1 (branch target)
        let beq = make_i(crate::mips_isa::OP_BEQ, 0, 0, 2);
        let mtc0 = benign_excluded_mtc0();
        let addiu3 = make_i(crate::mips_isa::OP_ADDIU, 1, 3, 1);
        let page = [
            (entry_word, beq),
            (entry_word + 1, mtc0),
            (entry_word + 3, addiu3),
        ];
        // Interpreter: BEQ (arms delay), MTC0 slot (retires, advances to
        // target), ADDIU3 = 3 dispatches.
        assert_jit_matches_interpreter_page(&page, gpr, pc, entry_word, 3, 4);
    }
}
