//! Aggregate performance snapshot for the telnet monitor (`perf snapshot`).

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::hal2::Hal2;
use crate::rex3::Rex3;
use crate::traits::Device;

pub struct PerfMonitor {
    cpu_running: Arc<AtomicBool>,
    /// Pointer into the CPU's `MipsCore.hot.cycles` word — see
    /// `CyclesPtr`/`Hot::cycles`'s doc comments. Obtained from
    /// `MipsCpu::cycles_ptr()`, which has the same "stable from construction
    /// onward" guarantee `interrupts_ptr` does.
    cycles: crate::mips_core::CyclesPtr,
    fasttick: Arc<AtomicU64>,
    rex3: Option<Arc<Rex3>>,
    hal2: Option<Arc<Hal2>>,
}

impl PerfMonitor {
    pub fn new(
        cpu_running: Arc<AtomicBool>,
        cycles: crate::mips_core::CyclesPtr,
        fasttick: Arc<AtomicU64>,
        rex3: Option<Arc<Rex3>>,
        hal2: Option<Arc<Hal2>>,
    ) -> Arc<Self> {
        Arc::new(Self { cpu_running, cycles, fasttick, rex3, hal2 })
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
        writeln!(writer, "{}", crate::thread_affinity::status_line()).map_err(|e| e.to_string())?;

        if let Some(rex) = &self.rex3 {
            let jit = rex.jit_go_count.load(Ordering::Relaxed);
            let interp = rex.interp_go_count.load(Ordering::Relaxed);
            let total = jit + interp;
            let pct = if total > 0 { jit * 100 / total } else { 0 };
            let diag = rex.diag.load(Ordering::Relaxed);
            writeln!(
                writer,
                "REX3 GO: {} total  JIT {}%  gfifo {}/{}  simd_fills {}",
                total, pct, rex.gfifo.len(), crate::rex3::GFIFO_DEPTH,
                rex.simd_fill_rows.load(Ordering::Relaxed),
            ).map_err(|e| e.to_string())?;
            writeln!(
                writer,
                "REX3 diag: {:#018x}  gfxbusy: {}",
                diag,
                rex.gfxbusy.load(Ordering::Relaxed),
            ).map_err(|e| e.to_string())?;
            #[cfg(feature = "rex-jit")]
            if let Some(ref jit) = rex.rex_jit {
                writeln!(writer, "REX3 JIT cache: {} shaders", jit.shader_list().len()).map_err(|e| e.to_string())?;
            }
        } else {
            writeln!(writer, "REX3: not present").map_err(|e| e.to_string())?;
        }

        if let Some(hal2) = &self.hal2 {
            writeln!(
                writer,
                "HAL2 underruns: {}  codec A: hptimer",
                hal2.underrun_count(),
            ).map_err(|e| e.to_string())?;
        } else {
            writeln!(writer, "HAL2: disabled (no_audio)").map_err(|e| e.to_string())?;
        }

        Ok(())
    }
}
