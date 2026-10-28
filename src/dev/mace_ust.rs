//! MACE's UST counter: the O2's guest-visible time base.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Mutex;

use crate::traits::{BusDevice, BusRead32, BusRead64, BUS_OK};
use std::io::Write;
use crate::traits::{Device, Resettable, Saveable};
use crate::snapshot::{get_field, hex_u64, toml_u64};

/// MACE's UST/MSC counter: unadjusted system time, free-running.
///
/// Driven by executed instructions, not by reads.
///
/// Advancing on read was enough to stop delay loops hanging, but it makes
/// duration meaningless: a loop that reads twice as often "takes" twice as
/// long. Anything that decodes a *pulse width* — 1-Wire above all — needs
/// elapsed time to mean something, so the harness advances this from its step
/// count and the rate below fixes the relationship.
pub struct MaceUst {
    ticks: AtomicU64,
    /// When set, the clock is the host's: microseconds since this instant,
    /// read live. The 1-Wire ID chip's slots are tens of microseconds, so a
    /// clock stepped from another thread cannot time them.
    host: std::sync::OnceLock<std::time::Instant>,
    /// Added to the host clock, so a restored machine's time runs on from
    /// where it was saved rather than from when this process started.
    base: std::sync::atomic::AtomicI64,
    /// The guest's own view of time ([`MaceUst::guest_us`]), in host
    /// microseconds: the host time at the last guest clock read, the host
    /// time the guest clock has dropped (stalls during 1-Wire work), and
    /// until when the 1-Wire bus counts as busy. Outside 1-Wire work a read
    /// is two loads and a store; `onewire` serialises the reads inside it,
    /// which only the PROM makes.
    last: AtomicU64,
    lost: AtomicU64,
    onewire_until: AtomicU64,
    onewire: Mutex<()>,
}

/// While the PROM bit-banges 1-Wire, the longest step the guest clock takes
/// between two of its reads. The PROM times each slot by spinning on the
/// clock; a host stall in the middle (a JIT pause, a busy host) would land in
/// the slot. 8 us is far above one spin of the loop and far below the 15 us
/// a 1-Wire "1" may last.
const GUEST_CLOCK_STEP_CAP_US: u64 = 8;

/// How long after the last 1-Wire access the bus counts as busy.
const ONEWIRE_BUSY_US: u64 = 5000;

/// UST ticks per emulated instruction, as a fraction: `TICKS_NUM / TICKS_DEN`.
///
/// The PROM's delay loops are written in microseconds, and its 1-Wire slots
/// are tens of microseconds, so a tick is taken to be 1 µs. At roughly 100
/// emulated instructions per microsecond a bring-up run stays fast while pulse
/// widths keep their proportions.
pub const UST_TICKS_NUM: u64 = 1;

/// Nanoseconds per count of MACE's UST counter.
pub const MACE_UST_PERIOD_NS: u64 = 960;
pub const UST_TICKS_DEN: u64 = 100;

impl Default for MaceUst {
    fn default() -> Self { Self::new() }
}

impl MaceUst {
    pub fn new() -> Self {
        Self {
            ticks: AtomicU64::new(0),
            host: std::sync::OnceLock::new(),
            base: std::sync::atomic::AtomicI64::new(0),
            last: AtomicU64::new(0),
            lost: AtomicU64::new(0),
            onewire_until: AtomicU64::new(0),
            onewire: Mutex::new(()),
        }
    }

    /// Run from the host clock from now on (a machine, not the harness).
    pub fn use_host_clock(&self) {
        let _ = self.host.set(std::time::Instant::now());
    }

    /// Set the clock to `us` microseconds, for a machine that keeps time by
    /// the host clock rather than by instructions run.
    pub fn set_us(&self, us: u64) {
        self.ticks.store(us, Relaxed);
    }

    /// Advance the clock to match `instructions` executed so far.
    pub fn advance_to(&self, instructions: u64) {
        self.ticks.store(instructions * UST_TICKS_NUM / UST_TICKS_DEN, Relaxed);
    }

    /// The counter as MACE presents it: one count per 960 ns (the 33.3 MHz
    /// PCI clock over 32, spec 7.2.2), which IRIX converts back at 960 ns a
    /// count. Counted in microseconds it ran 4% slow against the audio the
    /// codec plays, and programs that time sound by UST drifted.
    pub fn counter(&self) -> u64 {
        self.guest_us() * 1000 / MACE_UST_PERIOD_NS
    }

    /// The time the guest reads (UST, CRIME's timer): the host's, less what
    /// host stalls cost while the PROM was bit-banging 1-Wire. Then each step
    /// between two guest reads counts at most [`GUEST_CLOCK_STEP_CAP_US`],
    /// so after a stall the PROM goes on spinning to its own deadline and
    /// every slot comes out as wide as it meant. Timed on the host clock, a
    /// stall inside a "1" read as a "0" (or a reset), and on some boots the
    /// PROM reported `ds2502_read_rom failed` and IRIX had no Ethernet
    /// address. Capping a device-side clock alone was not enough: a stall
    /// early in a "0" slot let the PROM see its deadline pass and release,
    /// and the slot read short. Outside 1-Wire work nothing is dropped, and
    /// the guest's time never runs backwards.
    pub fn guest_us(&self) -> u64 {
        let now = self.now();
        if now >= self.onewire_until.load(Relaxed) {
            self.last.store(now, Relaxed);
            return now - self.lost.load(Relaxed);
        }
        let _one = self.onewire.lock().unwrap();
        let last = self.last.load(Relaxed);
        if last != 0 {
            let step = now.saturating_sub(last);
            if step > GUEST_CLOCK_STEP_CAP_US {
                self.lost.fetch_add(step - GUEST_CLOCK_STEP_CAP_US, Relaxed);
            }
        }
        self.last.store(now, Relaxed);
        now - self.lost.load(Relaxed)
    }

    /// The PROM touched the 1-Wire bus: hold stalls out of the guest's clock
    /// for a while.
    pub fn onewire_busy(&self) {
        let now = self.now();
        self.onewire_until.store(now + ONEWIRE_BUSY_US, Relaxed);
    }

    pub fn now(&self) -> u64 {
        match self.host.get() {
            Some(t0) => (t0.elapsed().as_micros() as i64
                + self.base.load(std::sync::atomic::Ordering::Relaxed)) as u64,
            None => self.ticks.load(Relaxed),
        }
    }
}

impl Device for MaceUst {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("ust".to_string(), "MACE UST: ust status".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "ust" {
            return Err("Command not found".to_string());
        }
        match args.first().copied().unwrap_or("status") {
            "status" => {
                // Read without advancing the guest clock (guest_us does).
                let now = self.now();
                let lost = self.lost.load(Relaxed);
                writeln!(w, "UST: {} clock, now {now} us, guest {} us ({lost} us held out for 1-Wire), counter {:#x} ({} ns a count)",
                    if self.host.get().is_some() { "host" } else { "instruction" },
                    now - lost, (now - lost) * 1000 / MACE_UST_PERIOD_NS, MACE_UST_PERIOD_NS).ok();
                Ok(())
            }
            _ => Err("Usage: ust status".to_string()),
        }
    }
}

/// The clock as it stood: the raw time, what 1-Wire work has held out of the
/// guest's view, and the bus's busy window, so the guest's time neither jumps
/// nor runs backwards across a restore.
impl Saveable for MaceUst {
    fn save_state(&self) -> toml::Value {
        let now = self.now();
        let mut t = toml::map::Map::new();
        t.insert("now".into(), hex_u64(now));
        t.insert("last".into(), hex_u64(self.last.load(Relaxed)));
        t.insert("lost".into(), hex_u64(self.lost.load(Relaxed)));
        t.insert("onewire_until".into(), hex_u64(self.onewire_until.load(Relaxed)));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let u = |k: &str| get_field(v, k).and_then(toml_u64).unwrap_or(0);
        let now = u("now");
        self.ticks.store(now, Relaxed);
        let elapsed = self.host.get().map_or(0, |t0| t0.elapsed().as_micros() as i64);
        self.base.store(now as i64 - elapsed, std::sync::atomic::Ordering::Relaxed);
        self.last.store(u("last"), Relaxed);
        self.lost.store(u("lost"), Relaxed);
        self.onewire_until.store(u("onewire_until"), Relaxed);
        Ok(())
    }
}

impl Resettable for MaceUst {
    /// The counter is free-running from power-up; nothing a reset clears.
    fn power_on(&self) {}
}

impl BusDevice for MaceUst {
    fn read32(&self, addr: u64) -> BusRead32 {
        let t = self.counter();
        // 64-bit counter presented as two 32-bit halves.
        BusRead32::ok(if addr & 4 == 0 { (t >> 32) as u32 } else { t as u32 })
    }
    fn read64(&self, _addr: u64) -> BusRead64 { BusRead64::ok(self.counter()) }
    fn write32(&self, _a: u64, _v: u32) -> u32 { BUS_OK }
    fn write64(&self, _a: u64, _v: u64) -> u32 { BUS_OK }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_round_trip() {
        let src = MaceUst::new();
        src.set_us(5_000_000);
        src.last.store(4_999_000, Relaxed);
        src.lost.store(1234, Relaxed);
        src.onewire_until.store(5_003_000, Relaxed);
        let v1 = src.save_state();

        let dst = MaceUst::new();
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "MaceUst save_state mismatch after load_state round-trip");
    }

    /// On the host clock, a restored machine's time carries on from the
    /// saved value even though this process's clock started later.
    #[test]
    fn a_restored_host_clock_runs_on_from_the_saved_time() {
        let src = MaceUst::new();
        src.set_us(7_000_000_000);
        let v = src.save_state();
        let dst = MaceUst::new();
        dst.use_host_clock();
        dst.load_state(&v).expect("load_state");
        let t = dst.now();
        assert!((7_000_000_000..7_001_000_000).contains(&t), "restored clock at {t}");
    }
}
