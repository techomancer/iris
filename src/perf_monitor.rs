//! Aggregate performance snapshot for the telnet monitor (`perf snapshot`).

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::dev::ng1::rex3::Rex3;
use crate::traits::Device;

pub struct PerfMonitor {
    cpu_running: Arc<AtomicBool>,
    /// Pointer into the CPU's `MipsCore.hot.cycles` word — see
    /// `CyclesPtr`/`Hot::cycles`'s doc comments. Obtained from
    /// `MipsCpu::cycles_ptr()`, which has the same "stable from construction
    /// onward" guarantee `interrupts_ptr` does.
    cycles: crate::cpu::mips_core::CyclesPtr,
    /// Every processor's cycle word on a machine with more than one (the
    /// first is `cycles`); empty on a one-processor machine.
    cpus: Vec<crate::cpu::mips_core::CyclesPtr>,
    fasttick: Arc<AtomicU64>,
    rex3: Option<Arc<Rex3>>,
}

impl PerfMonitor {
    pub fn new(
        cpu_running: Arc<AtomicBool>,
        cycles: crate::cpu::mips_core::CyclesPtr,
        fasttick: Arc<AtomicU64>,
        rex3: Option<Arc<Rex3>>,
    ) -> Arc<Self> {
        Arc::new(Self { cpu_running, cycles, cpus: Vec::new(), fasttick, rex3 })
    }

    /// As [`new`](Self::new) for a machine with several processors: the
    /// snapshot also prints each one's retired cycles, so two snapshots give
    /// each processor's rate. The first counter alone hides an idle processor
    /// spinning beside a busy one.
    pub fn with_cpus(
        cpu_running: Arc<AtomicBool>,
        cpus: Vec<crate::cpu::mips_core::CyclesPtr>,
        fasttick: Arc<AtomicU64>,
    ) -> Arc<Self> {
        assert!(!cpus.is_empty(), "a machine has a processor");
        Arc::new(Self { cpu_running, cycles: cpus[0], cpus, fasttick, rex3: None })
    }
}

impl Device for PerfMonitor {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { false }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![(
            "perf".to_string(),
            "Performance snapshot: perf snapshot".to_string(),
        )]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn Write + Send>) -> Result<(), String> {
        if cmd != "perf" {
            return Err(format!("Unknown command: {}", cmd));
        }
        match args.first().copied().unwrap_or("") {
            "snapshot" => self.write_snapshot(&mut writer),
            _ => Err("Usage: perf snapshot".to_string()),
        }
    }
}

impl PerfMonitor {
    fn write_snapshot(&self, writer: &mut dyn Write) -> Result<(), String> {
        writeln!(writer, "=== IRIS perf snapshot ===").map_err(|e| e.to_string())?;
        writeln!(
            writer,
            "CPU running: {}  cycles: {}  fastticks: {}",
            self.cpu_running.load(Ordering::Relaxed),
            self.cycles.get(),
            self.fasttick.load(Ordering::Relaxed),
        ).map_err(|e| e.to_string())?;
        if self.cpus.len() > 1 {
            let each: Vec<String> = self.cpus.iter().map(|c| c.retired().to_string()).collect();
            writeln!(writer, "CPU cycles: {}", each.join(" ")).map_err(|e| e.to_string())?;
        }
        writeln!(writer, "{}", crate::thread_affinity::status_line()).map_err(|e| e.to_string())?;

        if let Some(rex) = &self.rex3 {
            // The per-GO counters and DIAG_LOOP_* bits are compiled out under
            // `lightning` (see the `rexdiag` feature). Say so rather than print
            // the frozen zeros they would otherwise report as measurements.
            #[cfg(not(feature = "rexdiag"))]
            writeln!(
                writer,
                "REX3 GO: counters disabled in this build (rexdiag off/lightning)  gfifo {}/{}",
                rex.gfifo.len(), crate::dev::ng1::rex3::GFIFO_DEPTH,
            ).map_err(|e| e.to_string())?;

            #[cfg(feature = "rexdiag")]
            {
            let jit = rex.jit_go_count.load(Ordering::Relaxed);
            let interp = rex.interp_go_count.load(Ordering::Relaxed);
            let total = jit + interp;
            let pct = if total > 0 { jit * 100 / total } else { 0 };
            let diag = rex.diag.load(Ordering::Relaxed);
            writeln!(
                writer,
                "REX3 GO: {} total  JIT {}%  gfifo {}/{}",
                total, pct, rex.gfifo.len(), crate::dev::ng1::rex3::GFIFO_DEPTH,
            ).map_err(|e| e.to_string())?;
            writeln!(
                writer,
                "REX3 diag: {:#018x}  gfxbusy: {}",
                diag,
                rex.gfxbusy.load(Ordering::Relaxed),
            ).map_err(|e| e.to_string())?;
            }
            #[cfg(feature = "rex-jit")]
            if let Some(ref jit) = rex.rex_jit {
                writeln!(writer, "REX3 JIT cache: {} shaders", jit.shader_list().len()).map_err(|e| e.to_string())?;
            }
        } else {
            writeln!(writer, "REX3: not present").map_err(|e| e.to_string())?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::mips_core::CyclesPtr;

    fn snapshot(m: &PerfMonitor) -> String {
        let mut out = Vec::new();
        m.write_snapshot(&mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn a_snapshot_names_every_processors_cycles() {
        let words = [5u64, 7, 11, 13];
        let cpus = words.iter().map(|w| CyclesPtr::new(w)).collect();
        let m = PerfMonitor::with_cpus(Arc::new(AtomicBool::new(true)), cpus, Arc::new(AtomicU64::new(0)));
        let s = snapshot(&m);
        assert!(s.contains("cycles: 5 "), "{s}");
        assert!(s.contains("CPU cycles: 5 7 11 13\n"), "{s}");
    }

    #[test]
    fn a_one_processor_snapshot_has_no_per_processor_line() {
        let word = 3u64;
        let m = PerfMonitor::new(Arc::new(AtomicBool::new(true)), CyclesPtr::new(&word), Arc::new(AtomicU64::new(0)), None);
        assert!(!snapshot(&m).contains("CPU cycles"));
    }
}
