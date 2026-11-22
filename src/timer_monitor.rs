//! The monitor's `timers` command: what the `hptimer` worker and the device
//! register paths cost in wakes and host clock reads. Two samples give rates.

use std::io::Write;
use std::sync::Arc;

use crate::hptimer::TimerManager;
use crate::traits::Device;

pub struct TimersMonitor {
    timers: Arc<TimerManager>,
}

impl TimersMonitor {
    pub fn new(timers: Arc<TimerManager>) -> Arc<Self> {
        Arc::new(Self { timers })
    }
}

impl Device for TimersMonitor {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { false }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("timers".to_string(), "Timer worker wakes and passes, register-path host clock reads: timers".to_string())]
    }

    fn execute_command(&self, cmd: &str, _args: &[&str], mut w: Box<dyn Write + Send>) -> Result<(), String> {
        if cmd != "timers" {
            return Err(format!("Unknown command: {}", cmd));
        }
        let (wakes, scans) = self.timers.counters();
        writeln!(
            w,
            "timers: host_ns {} wakes {} scans {} register_clock_reads {}",
            crate::dev::deadline::host_ns(),
            wakes,
            scans,
            crate::dev::deadline::register_clock_reads(),
        ).map_err(|e| e.to_string())
    }
}
