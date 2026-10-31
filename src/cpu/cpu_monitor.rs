//! The monitor's view of the machine's CPUs: one object in front of however
//! many there are. It is what the monitor talks to; a CPU never registers
//! itself.
//!
//! With one CPU every command goes straight to it. With more, a command goes
//! to the selected CPU, except the few that mean the whole machine:
//!
//! | Form | Goes to |
//! |---|---|
//! | `regs`, `step`, `mem` ... | the selected CPU |
//! | `start`, `stop`, `status` | every CPU |
//! | `bp`, `b`, `bd`, `be`, `bb` | every CPU, so a breakpoint means the same on all |
//! | `cpu <n> <command>` | CPU n alone, whatever the command |
//! | `r<n>`, `s<n>` | `cpu <n> regs`, `cpu <n> step` |
//! | `cpu <n>` | selects CPU n |
//! | `cpus` | lists them |
//!
//! Stepping one CPU: `stop` holds them all, then `cpu 1 step` (or `s1`) moves
//! CPU 1 while the others stay where they are.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::cpu::mips_exec::CpuDevice;
use crate::traits::Device;

/// Commands that mean every CPU when no CPU is named.
const ON_ALL: &[&str] = &["start", "stop", "bp", "b", "bd", "be", "bb"];
/// Commands where every CPU answers, each under its own heading.
const FROM_EACH: &[&str] = &["status"];

/// Where one command line goes.
#[derive(Debug, PartialEq, Eq)]
enum Route<'a> {
    /// `cpus`.
    List,
    /// `cpu <n>` alone.
    Select(usize),
    /// One CPU.
    One(usize, &'a str, Vec<&'a str>),
    /// Every CPU; the selected one's answer is the one shown.
    All(&'a str, Vec<&'a str>),
    /// Every CPU, each answer shown.
    Each(&'a str, Vec<&'a str>),
}

/// `r3` is `Some(("regs", 3))`: the per-CPU spellings.
fn per_cpu_alias(cmd: &str) -> Option<(&'static str, usize)> {
    let name = match cmd.as_bytes().first()? {
        b'r' => "regs",
        b's' => "step",
        _ => return None,
    };
    Some((name, cmd[1..].parse().ok()?))
}

fn route<'a>(cmd: &'a str, args: &[&'a str], count: usize, selected: usize) -> Result<Route<'a>, String> {
    let in_range = |n: usize| {
        if n < count {
            Ok(n)
        } else {
            Err(format!("no CPU {n}: this machine has {count}"))
        }
    };
    if cmd == "cpus" {
        return Ok(Route::List);
    }
    if let Some((name, n)) = per_cpu_alias(cmd) {
        return Ok(Route::One(in_range(n)?, name, args.to_vec()));
    }
    let (cmd, args) = match (cmd, args.first().map(|a| a.parse::<usize>())) {
        ("cpu", Some(Ok(n))) => {
            let n = in_range(n)?;
            return Ok(match args.get(1) {
                None => Route::Select(n),
                Some(c) => Route::One(n, c, args[2..].to_vec()),
            });
        }
        // `cpu regs` is `regs`; a bare `cpu` is the CPU's own usage message.
        ("cpu", Some(Err(_))) => (args[0], &args[1..]),
        _ => (cmd, args),
    };
    Ok(if ON_ALL.contains(&cmd) {
        Route::All(cmd, args.to_vec())
    } else if FROM_EACH.contains(&cmd) {
        Route::Each(cmd, args.to_vec())
    } else {
        Route::One(selected, cmd, args.to_vec())
    })
}

/// A writer several CPUs can answer through in turn.
#[derive(Clone)]
struct SharedWriter(Arc<Mutex<Box<dyn Write + Send>>>);

impl Write for SharedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().flush()
    }
}

pub struct CpuMonitor {
    cpus: Vec<Arc<dyn CpuDevice>>,
    selected: AtomicUsize,
}

impl CpuMonitor {
    pub fn new(cpus: Vec<Arc<dyn CpuDevice>>) -> Self {
        assert!(!cpus.is_empty(), "a machine has at least one CPU");
        Self { cpus, selected: AtomicUsize::new(0) }
    }

    fn list(&self, w: &mut dyn Write) {
        let selected = self.selected.load(Ordering::Relaxed);
        for (i, cpu) in self.cpus.iter().enumerate() {
            let mark = if i == selected { '*' } else { ' ' };
            let state = if cpu.clone().as_device().is_running() { "running" } else { "stopped" };
            let _ = writeln!(w, "{mark} cpu {i}: {state}");
        }
    }
}

impl Device for CpuMonitor {
    /// The CPUs run on their own; this object only answers the monitor.
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool {
        self.cpus.iter().any(|c| c.clone().as_device().is_running())
    }
    fn get_clock(&self) -> u64 {
        self.cpus[0].clone().as_device().get_clock()
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        let mut c = self.cpus[0].monitor_commands();
        c.push(("cpus".into(), "List the CPUs; `cpu <n>` selects one, `cpu <n> <command>` addresses one".into()));
        for i in 0..self.cpus.len() {
            c.push((format!("r{i}"), format!("Registers of CPU {i} (cpu {i} regs)")));
            c.push((format!("s{i}"), format!("Step CPU {i} alone (cpu {i} step)")));
        }
        c
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn Write + Send>) -> Result<(), String> {
        let selected = self.selected.load(Ordering::Relaxed);
        match route(cmd, args, self.cpus.len(), selected)? {
            Route::List => {
                self.list(&mut w);
                Ok(())
            }
            Route::Select(n) => {
                self.selected.store(n, Ordering::Relaxed);
                self.list(&mut w);
                Ok(())
            }
            Route::One(n, c, a) => self.cpus[n].monitor_execute(c, &a, w),
            Route::All(c, a) => {
                for (i, cpu) in self.cpus.iter().enumerate() {
                    if i != selected {
                        cpu.monitor_execute(c, &a, Box::new(std::io::sink()))?;
                    }
                }
                self.cpus[selected].monitor_execute(c, &a, w)
            }
            Route::Each(c, a) if self.cpus.len() == 1 => self.cpus[0].monitor_execute(c, &a, w),
            Route::Each(c, a) => {
                let shared = SharedWriter(Arc::new(Mutex::new(w)));
                for (i, cpu) in self.cpus.iter().enumerate() {
                    let mut out = shared.clone();
                    let _ = writeln!(out, "cpu {i}:");
                    cpu.monitor_execute(c, &a, Box::new(shared.clone()))?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_command_goes_to_the_selected_cpu() {
        assert_eq!(route("regs", &[], 2, 1), Ok(Route::One(1, "regs", vec![])));
        assert_eq!(route("mem", &["0x80000000"], 2, 0), Ok(Route::One(0, "mem", vec!["0x80000000"])));
    }

    #[test]
    fn the_cpu_prefix_without_a_number_changes_nothing() {
        assert_eq!(route("cpu", &["regs"], 1, 0), route("regs", &[], 1, 0));
        assert_eq!(route("cpu", &["stop"], 2, 0), Ok(Route::All("stop", vec![])));
    }

    #[test]
    fn a_bare_cpu_reaches_the_cpu_for_its_usage_message() {
        assert_eq!(route("cpu", &[], 1, 0), Ok(Route::One(0, "cpu", vec![])));
    }

    #[test]
    fn a_number_addresses_that_cpu_alone_even_for_machine_wide_commands() {
        assert_eq!(route("cpu", &["1", "step", "5"], 2, 0), Ok(Route::One(1, "step", vec!["5"])));
        assert_eq!(route("cpu", &["1", "stop"], 2, 0), Ok(Route::One(1, "stop", vec![])));
        assert_eq!(route("cpu", &["1"], 2, 0), Ok(Route::Select(1)));
    }

    #[test]
    fn the_short_forms_name_a_cpu() {
        assert_eq!(route("r1", &[], 2, 0), Ok(Route::One(1, "regs", vec![])));
        assert_eq!(route("s0", &["3"], 2, 1), Ok(Route::One(0, "step", vec!["3"])));
    }

    #[test]
    fn names_that_only_look_like_the_short_forms_are_left_alone() {
        for c in ["r", "s", "regs", "step", "status", "start", "stop", "run", "rx"] {
            assert!(per_cpu_alias(c).is_none(), "{c}");
        }
    }

    #[test]
    fn a_cpu_that_is_not_there_is_refused() {
        assert!(route("r2", &[], 2, 0).is_err());
        assert!(route("cpu", &["2"], 2, 0).is_err());
        assert!(route("cpu", &["2", "regs"], 2, 0).is_err());
    }

    #[test]
    fn machine_wide_commands_reach_every_cpu() {
        assert_eq!(route("bp", &["add", "0x100"], 2, 1), Ok(Route::All("bp", vec!["add", "0x100"])));
        assert_eq!(route("status", &[], 2, 1), Ok(Route::Each("status", vec![])));
    }
}
