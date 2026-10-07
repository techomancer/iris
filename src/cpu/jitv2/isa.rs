//! The ISA level jitv2 compiles for, as a *runtime* property of the CPU
//! model rather than a build-time one.
//!
//! ## Why this exists
//!
//! The interpreter has always gated MIPS IV on `C::MIPS4`, an associated
//! const on the `CpuModel` trait: `R4400Cache` sets it false, `R5000Cache`
//! sets it true, as does R10000. Earlier jitv2 gated on the `mips4` Cargo
//! feature, and read `C::MIPS4` exactly zero times.
//!
//! Those are different axes and could only agree by coincidence. The CPU is
//! a **runtime** choice (`cpu = "r5000"` in a config), and one binary serves
//! every model, so a `mips4` build pointed at an R4400 config had jitv2
//! executing `MOVZ`/`COP1X` where the interpreter raised Reserved
//! Instruction — a real divergence, and exactly what `jitv2_lockstep` exists
//! to catch. Conversely, the old default build ran an R5000
//! with MIPS IV compilation switched off, costing ~20% of integer throughput
//! (measured on IP28 / R10000, now a runtime machine profile).
//!
//! ## How it is set
//!
//! It is not read from here on any path that compiles guest code. The
//! `CpuModel`'s `MIPS4` const travels as data: `MipsExecutor::new` hands it
//! to its own inline `Analyzer` and to the compile pool
//! (`CompileQueue::set_isa`), each worker builds an `Analyzer::with_isa`, and
//! `Analyzer` passes it through `Budget` into `classify` and
//! `opcode_support::has_emitter`, the single point that asks the question.
//!
//! What remains here is the **default** for the things that have no CPU to
//! ask: `Analyzer::new()`, `CompileQueue::new()`, and the offline tools that
//! replay a recorded trace. That default is MIPS IV: only a MIPS III model
//! (the R4400) turns it off.
//!
//! This replaced a process-global that `MipsExecutor::new` published under
//! `cfg(not(test))`, so the one line that mattered was compiled out of every
//! test build and deleting it left the suite green.
//! `an_executor_walks_at_its_own_models_isa_level` is the test that could not
//! be written before.

use std::sync::atomic::{AtomicBool, Ordering};

/// Default ISA for tools and constructors without a CPU model. Live executor
/// compilation passes its own model's ISA directly to each analyzer/worker.
static MIPS4: AtomicBool = AtomicBool::new(true);

/// Default ISA for callers without an explicit CPU model. This global is not
/// the ISA selector for live executor compilation.
#[inline]
pub fn mips4_enabled() -> bool {
    MIPS4.load(Ordering::Relaxed)
}

/// Set the default ISA used by constructors/tools without a CPU model.
/// Live executors pass `C::MIPS4` directly instead of publishing here.
///
/// This does not update existing analyzers, workers, or compiled regions.
/// Set it before constructing a tool that relies on the default; tests use
/// `test_isa` to restore the previous value.
pub fn set_mips4(on: bool) {
    MIPS4.store(on, Ordering::Relaxed);
}

/// Set the ISA level for the duration of a test, restoring it on drop.
///
/// The flag is a process global, and the test harness runs tests in
/// parallel, so a test that flipped it unguarded would corrupt whatever else
/// happened to be compiling a region at the time — an intermittent failure of
/// exactly the kind that is miserable to track down. Every test that changes
/// the ISA level must go through this, which serialises them against each
/// other on a single mutex.
#[cfg(test)]
pub fn test_isa(on: bool) -> IsaGuard {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A panic inside a guarded test poisons the mutex; the ISA level is not
    // the thing under test in that case, so recover rather than cascade the
    // failure into every later test.
    let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev = mips4_enabled();
    set_mips4(on);
    IsaGuard { prev, _lock: lock }
}

#[cfg(test)]
pub struct IsaGuard {
    prev: bool,
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl Drop for IsaGuard {
    fn drop(&mut self) {
        set_mips4(self.prev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::mips_cache_v2::{CpuModel, PassthroughCache, PassthroughCacheM4,
                               R4400Cache, R5000Cache};

    /// The models' own `MIPS4` consts are what the interpreter gates on, so
    /// this pins the mapping jitv2 now shares with it. If someone flips one
    /// of these, both engines change together — which is the entire point of
    /// moving the gate onto this axis.
    #[test]
    fn model_isa_levels_are_what_the_interpreter_gates_on() {
        assert!(!<R4400Cache as CpuModel>::MIPS4, "R4400 is MIPS III");
        assert!(<R5000Cache as CpuModel>::MIPS4, "R5000 is MIPS IV");
    }

    /// Verify explicit updates to the default ISA for both polarities. Live
    /// executors pass their model directly; the wiring test below covers that.
    #[test]
    fn publishes_the_cpu_models_isa_level_to_jitv2() {
        let _isa = test_isa(false);

        set_mips4(<PassthroughCacheM4 as CpuModel>::MIPS4);
        assert!(mips4_enabled(), "a MIPS IV model must enable MIPS IV compilation");

        set_mips4(<PassthroughCache as CpuModel>::MIPS4);
        assert!(!mips4_enabled(), "a MIPS III model must disable it again");
    }

    /// The wiring test the old design could not have.
    ///
    /// jitv2 used to read the ISA level from a process global that
    /// `MipsExecutor::new` published under `cfg(not(test))` — so the line
    /// that mattered was compiled out of every test build, and deleting it
    /// left the suite green. The level now travels as data, which means this
    /// can assert the thing that actually matters: an executor built for a
    /// model hands that model's `MIPS4` to the analyzer its compiles walk
    /// with.
    #[test]
    fn an_executor_walks_at_its_own_models_isa_level() {
        use crate::cpu::mips_cache_v2::{CpuModel, PassthroughCache, PassthroughCacheM4};
        use crate::cpu::mips_exec::{MipsCpuConfig, MipsExecutor};
        use crate::cpu::mips_tlb::PassthroughTlb;
        use crate::dev::mem::Memory;
        use crate::traits::BusDevice;
        use std::sync::Arc;

        let cfg = MipsCpuConfig::indy();

        let bus: Arc<dyn BusDevice> = Arc::new(Memory::new(1));
        let mips3: MipsExecutor<PassthroughTlb, PassthroughCache> =
            MipsExecutor::new(bus, PassthroughTlb::default(), &cfg);
        assert!(!<PassthroughCache as CpuModel>::MIPS4, "fixture must be a MIPS III model");
        assert!(
            !mips3.jitv2_inline_analyzer.mips4(),
            "a MIPS III executor must walk at MIPS III, whatever the build was configured with",
        );

        let bus: Arc<dyn BusDevice> = Arc::new(Memory::new(1));
        let mips4: MipsExecutor<PassthroughTlb, PassthroughCacheM4> =
            MipsExecutor::new(bus, PassthroughTlb::default(), &cfg);
        assert!(<PassthroughCacheM4 as CpuModel>::MIPS4, "fixture must be a MIPS IV model");
        assert!(
            mips4.jitv2_inline_analyzer.mips4(),
            "a MIPS IV executor must walk at MIPS IV",
        );
    }

    /// The guard must restore whatever was there before, or one test's ISA
    /// level leaks into every later one.
    #[test]
    fn test_isa_guard_restores_the_previous_level() {
        let before = mips4_enabled();
        {
            let _isa = test_isa(!before);
            assert_eq!(mips4_enabled(), !before);
        }
        assert_eq!(mips4_enabled(), before);
    }
}
