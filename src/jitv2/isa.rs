//! The ISA level jitv2 compiles for, as a *runtime* property of the CPU
//! model rather than a build-time one.
//!
//! ## Why this exists
//!
//! The interpreter has always gated MIPS IV on `C::MIPS4`, an associated
//! const on the `CpuModel` trait: `R4400Cache` sets it false, `R5000Cache`
//! sets it true. jitv2 instead gated on the `mips4` cargo
//! feature, and read `C::MIPS4` exactly zero times.
//!
//! Those are different axes and could only agree by coincidence. The CPU is
//! a **runtime** choice (`cpu = "r5000"` in a config), and one binary serves
//! every model, so a `mips4` build pointed at an R4400 config had jitv2
//! executing `MOVZ`/`COP1X` where the interpreter raised Reserved
//! Instruction — a real divergence, and exactly what `jitv2_lockstep` exists
//! to catch. The reverse is the common case: a default build runs an R5000
//! with MIPS IV compilation switched off, worth ~20% of integer throughput
//! (measured on our IP28 / R10000, which comes in a later pull request).
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

/// MIPS IV until a CPU is constructed; `set_mips4` then publishes the real
/// model's value.
static MIPS4: AtomicBool = AtomicBool::new(true);

/// True when the running CPU model implements MIPS IV — i.e. when
/// `C::MIPS4` is set for the model the guest was configured with.
#[inline]
pub fn mips4_enabled() -> bool {
    MIPS4.load(Ordering::Relaxed)
}

/// Publish the CPU model's ISA level. Called from `MipsExecutor::new` with
/// `C::MIPS4`.
///
/// Changing this after regions have been compiled would leave code around
/// that was built for the other ISA level, so it is only ever called during
/// CPU construction, before any compilation can have happened.
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
    use crate::mips_cache_v2::{CpuModel, PassthroughCache, PassthroughCacheM4,
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

    /// `MipsExecutor::new` publishes `C::MIPS4` here. It cannot do that under
    /// `cfg(test)` (see the comment at that call site — the harness builds
    /// several models concurrently), so this covers the publish itself for
    /// both polarities.
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
        use crate::mips_cache_v2::{CpuModel, PassthroughCache, PassthroughCacheM4};
        use crate::mips_exec::{MipsCpuConfig, MipsExecutor};
        use crate::mips_tlb::PassthroughTlb;
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
