//! Bare-metal test device: guest console, machine-state dump, process exit code.
//!
//! Default-off (`--test-device`). It exists so a self-checking bare-metal test
//! binary can report to the host without the SCC: a byte at a time to stdout, a
//! full machine-state dump to a JSON file, and a real process exit code instead
//! of a serial string CI has to match.
//!
//! **Address**: GIO64 expansion slot 0 (`0x1F400000`, 64 KB of the 2 MB
//! aperture). An Indy has one GIO64 expansion connector and a stock machine
//! ships with nothing in it, so IRIS routes the whole slot to the GIO bus-error
//! (timeout) handler — the device replaces "nothing answers here" with itself
//! and displaces no real hardware. It cannot collide with the IP22/IP24 map:
//! graphics live in the *graphics* slot at 0x1F000000, the second Newport head
//! and IMPACT/MGRAS take expansion slot 1 at 0x1F600000, the MC is at
//! 0x1FA00000, HPC3 at 0x1FB80000 and the PROM at 0x1FC00000. The one thing
//! that does claim this slot is the `ultra64` dev board, which is why enabling
//! both is refused.
//!
//! The PROM's own slot-0 probe was measured at physical `0x1F46A07C`, outside
//! the 64 KB this device decodes, so POST still sees a GIO timeout there even
//! with the device enabled.
//!
//! The guest detects the device by reading `SIGNATURE`; on real hardware the
//! empty slot times out, so a suite falls back to SCC-only output.
//!
//! **Two output modes.** `new()` is the `iris --test-device` process: bytes to
//! stdout, dumps to a file, `EXIT` ends the process. `new_embedded()` is for a
//! host that links IRIS as a library and runs a bare-metal image in-process
//! (`crate::bench_runner`, and the GUI behind it): bytes to a buffer the
//! embedder drains, no dump file, and `EXIT` calls a hook instead of killing
//! the host application.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::mips_core::MipsCore;
use crate::traits::{BusDevice, BusRead32, BusRead8, Device, Resettable, Saveable, BUS_OK};

/// GIO64 expansion slot 0. 64 KB is decoded; registers repeat every 16 bytes.
pub const TEST_DEV_BASE: u32 = 0x1F400000;
pub const TEST_DEV_SIZE: u32 = 0x0001_0000;

pub const REG_SIGNATURE: u32 = 0x00;
pub const REG_PUTC: u32 = 0x04;
pub const REG_DUMP: u32 = 0x08;
pub const REG_EXIT: u32 = 0x0C;
/// Host monotonic nanoseconds since the device was created. Reading LO latches
/// the whole 64-bit value; HI returns the high word of that same latch, so a
/// guest doing LO-then-HI never sees a torn count. Benchmarks need a time base
/// the emulator cannot distort: CP0 Count is a *virtual* clock derived from a
/// calibrated `count_hz` (mips_core.rs), so timing anything with it measures
/// the timer model as much as the workload.
pub const REG_HOST_NS_LO: u32 = 0x10;
pub const REG_HOST_NS_HI: u32 = 0x14;
/// Guest instructions retired (`MipsCore::hot.cycles`), same latch protocol.
/// Advanced once per retired instruction by both the interpreter and jitv2
/// (`emit_increment_cycles`), so it is directly comparable across engines —
/// which is what makes "guest MIPS" a meaningful per-kernel number.
pub const REG_ICOUNT_LO: u32 = 0x18;
pub const REG_ICOUNT_HI: u32 = 0x1C;
/// Capability bitmask, so a guest built against a newer header can still run
/// on an older emulator: read it, and only use what it advertises.
pub const REG_CAPS: u32 = 0x20;
/// Run configuration, read once at guest startup. A bare-metal image loaded
/// with `--load-elf` has no argv and no environment, so this register is how
/// the host asks for a shorter run: see `RunConfig`. Zero — the value an
/// emulator without the capability returns — means "everything, full length",
/// so a guest that reads it unconditionally still behaves.
pub const REG_RUN_CONFIG: u32 = 0x24;

/// Registers decode within this many bytes and the window repeats across the
/// whole 64 KB the device claims. Was 16 before the clock/icount registers.
pub const REG_WINDOW: u32 = 0x40;

/// Reads back at `REG_SIGNATURE` — "IRIS" in ASCII.
pub const SIGNATURE: u32 = 0x4952_4953;

/// `REG_CAPS` bit 0: `REG_HOST_NS_*` and `REG_ICOUNT_*` are present.
pub const CAP_TIMEBASE: u32 = 1 << 0;
/// `REG_CAPS` bit 1: `REG_RUN_CONFIG` is present and meaningful.
pub const CAP_RUN_CONFIG: u32 = 1 << 1;

/// What `REG_RUN_CONFIG` carries, packed into one word:
///
/// ```text
///   31            16 15   12 11             0
///   +---------------+-------+---------------+
///   |     groups    |repeats|    time_pct   |
///   +---------------+-------+---------------+
/// ```
///
/// **Every field means "unrestricted" when zero**, so the word reading back as
/// 0 — on an emulator that predates the register, or on a run that never set
/// it — is exactly the behaviour the suite had before it existed. That is the
/// whole reason for the encoding: a guest can read it unconditionally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunConfig {
    /// The suite's own group mask — `BG_INT`, `BG_FPU`, … from
    /// `bench/harness/benchlib.h`, which the guest ANDs against each kernel's
    /// declared group. 0 selects all of them.
    pub groups: u16,
    /// Per-kernel target time as a percentage of the harness default. 0 means
    /// unchanged; the guest clamps against its own floor. Capped at 12 bits.
    pub time_pct: u16,
    /// Timed passes per kernel. 0 means the harness default (best of two —
    /// the slow sample is host scheduling noise, since the guest does a fixed
    /// amount of work either way). Capped at 4 bits.
    pub repeats: u8,
}

impl RunConfig {
    pub const ALL: Self = Self { groups: 0, time_pct: 0, repeats: 0 };

    pub const fn to_word(self) -> u32 {
        ((self.groups as u32) << 16)
            | (((self.repeats as u32) & 0xF) << 12)
            | (self.time_pct as u32 & 0xFFF)
    }

    pub const fn from_word(w: u32) -> Self {
        Self {
            groups: (w >> 16) as u16,
            repeats: ((w >> 12) & 0xF) as u8,
            time_pct: (w & 0xFFF) as u16,
        }
    }
}

/// Where `REG_DUMP` writes go when no path is configured.
pub const DEFAULT_DUMP_PATH: &str = "iris-testdev-dump.json";

/// Called on `REG_EXIT` in embedded mode, on the CPU thread, from inside the
/// guest's store. It must not block on anything that thread owns — signal and
/// return; see `TestDevice::exit`.
pub type ExitHook = Box<dyn Fn(u32) + Send + Sync>;

/// Where `PUTC` bytes go.
enum Console {
    /// The process's own stdout, flushed per line.
    Stdout,
    /// A buffer the embedder drains. Unbounded: a bare-metal image prints
    /// kilobytes, and truncating the guest's own report to save memory would
    /// lose the part the host is there to parse.
    Buffer(std::sync::Arc<Mutex<Vec<u8>>>),
}

pub struct TestDevice {
    /// Dump file path, or `None` in embedded mode, where `DUMP` is a counted
    /// no-op — a sandboxed host has nowhere to put the file and the embedded
    /// users (benchmarks) never ask for one. Never defaults to a bare
    /// CWD-relative name in a sandboxed app, where the CWD is `/` and
    /// unwritable — the caller passes an absolute path there.
    dump_path: Option<std::path::PathBuf>,
    out: Console,
    on_exit: Option<ExitHook>,
    /// Published to the guest at `REG_RUN_CONFIG`.
    run_config: AtomicU32,
    /// Set by `attach_core`. Only read from the CPU thread, which is the thread
    /// that issues the store that lands here, so there is no race — the same
    /// process-lifetime argument as `MipsCpu`'s `cycles_ptr`/`interrupts_ptr`.
    core: Mutex<Option<CorePtr>>,
    dumps: AtomicU32,
    last_tag: AtomicU64,
    chars: AtomicU64,
    running: AtomicBool,
    /// Origin for `REG_HOST_NS_*`. Only deltas are meaningful to the guest, so
    /// where zero lands does not matter — only that it never moves.
    epoch: std::time::Instant,
    /// Latched 64-bit values, published by a read of the LO half.
    host_ns_latch: AtomicU64,
    icount_latch: AtomicU64,
}

/// Raw pointer to the executor's `MipsCore`, valid for the process lifetime.
struct CorePtr(*const MipsCore);
// SAFETY: the pointee outlives the device and is only dereferenced from the CPU
// thread while that thread is inside its own store instruction.
unsafe impl Send for CorePtr {}
unsafe impl Sync for CorePtr {}

impl TestDevice {
    pub fn new(dump_path: impl Into<std::path::PathBuf>) -> Self {
        Self::build(Some(dump_path.into()), Console::Stdout, None, RunConfig::ALL)
    }

    /// In-process mode: guest console into `sink`, `EXIT` into `on_exit`, no
    /// dump file, and `cfg` published at `REG_RUN_CONFIG`.
    pub fn new_embedded(
        sink: std::sync::Arc<Mutex<Vec<u8>>>,
        on_exit: ExitHook,
        cfg: RunConfig,
    ) -> Self {
        Self::build(None, Console::Buffer(sink), Some(on_exit), cfg)
    }

    fn build(
        dump_path: Option<std::path::PathBuf>,
        out: Console,
        on_exit: Option<ExitHook>,
        cfg: RunConfig,
    ) -> Self {
        Self {
            dump_path,
            out,
            on_exit,
            run_config: AtomicU32::new(cfg.to_word()),
            core: Mutex::new(None),
            dumps: AtomicU32::new(0),
            last_tag: AtomicU64::new(0),
            chars: AtomicU64::new(0),
            running: AtomicBool::new(false),
            epoch: std::time::Instant::now(),
            host_ns_latch: AtomicU64::new(0),
            icount_latch: AtomicU64::new(0),
        }
    }

    /// Point the device at the CPU state it dumps. Called once at wiring time.
    pub fn attach_core(&self, core: *const MipsCore) {
        *self.core.lock() = Some(CorePtr(core));
    }

    pub fn dump_path(&self) -> Option<&std::path::Path> {
        self.dump_path.as_deref()
    }

    pub fn dumps_written(&self) -> u32 {
        self.dumps.load(Ordering::Relaxed)
    }

    fn putc(&self, byte: u8) {
        match &self.out {
            Console::Stdout => {
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(&[byte]);
                // Flush per line so a hung test still shows everything it printed.
                if byte == b'\n' {
                    let _ = out.flush();
                }
            }
            // No per-line flush to do: the embedder reads the buffer whenever
            // it likes, and a partial line is visible the moment it lands.
            Console::Buffer(buf) => buf.lock().push(byte),
        }
        self.chars.fetch_add(1, Ordering::Relaxed);
    }

    /// Write the full machine state as JSON. `tag` distinguishes dump points, so
    /// a test can dump more than once; the file is overwritten each time.
    fn dump(&self, tag: u32) {
        self.last_tag.store(tag as u64, Ordering::Relaxed);
        // Embedded: nothing to write to. Still counted, so `testdev` and the
        // save state report what the guest asked for.
        let Some(path) = self.dump_path.clone() else {
            self.dumps.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let guard = self.core.lock();
        let Some(CorePtr(ptr)) = guard.as_ref() else {
            eprintln!("test device: DUMP with no CPU attached");
            return;
        };
        // SAFETY: see CorePtr — the CPU thread is the one storing to this register.
        let core = unsafe { &**ptr };
        let json = dump_json(core, tag);
        drop(guard);

        match std::fs::write(&path, json) {
            Ok(()) => {
                self.dumps.fetch_add(1, Ordering::Relaxed);
                eprintln!("test device: dump {} → {}", tag, path.display());
            }
            Err(e) => eprintln!("test device: dump {} to {}: {}", tag, path.display(), e),
        }
    }

    /// The guest is done. Standalone, that means ending the process with its
    /// exit code; embedded, it means telling the host and *returning*.
    ///
    /// Returning is safe, and is the reason this needs no parking or signalling
    /// on the CPU thread: every guest that has a test device reaches this
    /// through `testdev_exit()`, which spins forever afterwards
    /// (`cpu-tests/harness/console.c`) precisely because a bare-metal image has
    /// nowhere to return to. So the store completes, the CPU thread carries on
    /// looping in guest code that does nothing, and the host stops the machine
    /// from its own thread whenever it gets to it. A hook that blocked here
    /// instead would deadlock `Machine::stop`, which joins this thread.
    fn exit(&self, code: u32) {
        if let Some(hook) = &self.on_exit {
            hook(code);
            return;
        }
        let _ = std::io::stdout().flush();
        eprintln!("test device: guest requested exit({})", code as u8);
        std::process::exit(code as u8 as i32)
    }
}

/// Full architectural state as JSON: GPRs, HI/LO, PC, CP0 by name, FPRs + FCSR/FIR.
pub fn dump_json(core: &MipsCore, tag: u32) -> String {
    let hex64 = |v: u64| format!("\"{:#018x}\"", v);
    let hex32 = |v: u32| format!("\"{:#010x}\"", v);
    let list = |vals: &[u64]| {
        vals.iter().map(|v| hex64(*v)).collect::<Vec<_>>().join(", ")
    };

    // PRId imp field: 0x04 = R4000/R4400, 0x23 = R5000.
    let cpu = match (core.cp0_prid >> 8) & 0xFF {
        0x04 => "R4400",
        0x23 => "R5000",
        _ => "unknown",
    };

    let mut s = String::with_capacity(4096);
    s.push_str("{\n");
    s.push_str(&format!("  \"tag\": {},\n", tag));
    s.push_str(&format!("  \"cpu\": \"{}\",\n", cpu));
    s.push_str(&format!("  \"pc\": {},\n", hex64(core.pc)));
    s.push_str(&format!("  \"hi\": {},\n", hex64(core.hi)));
    s.push_str(&format!("  \"lo\": {},\n", hex64(core.lo)));
    s.push_str(&format!("  \"gpr\": [{}],\n", list(&core.gpr)));
    s.push_str(&format!("  \"fpr\": [{}],\n", list(&core.fpr)));
    s.push_str(&format!("  \"fcsr\": {},\n", hex32(core.fpu_fcsr)));
    s.push_str(&format!("  \"fir\": {},\n", hex32(core.fpu_fir)));
    s.push_str("  \"cp0\": {\n");
    let cp0: [(&str, String); 24] = [
        ("Index",    hex32(core.cp0_index)),
        ("Random",   hex32(core.cp0_random)),
        ("EntryLo0", hex64(core.cp0_entrylo0)),
        ("EntryLo1", hex64(core.cp0_entrylo1)),
        ("Context",  hex64(core.cp0_context)),
        ("PageMask", hex64(core.cp0_pagemask)),
        ("Wired",    hex32(core.cp0_wired)),
        ("BadVAddr", hex64(core.cp0_badvaddr)),
        ("Count",    hex64(core.cp0_count)),
        ("EntryHi",  hex64(core.cp0_entryhi)),
        ("Compare",  hex64(core.cp0_compare)),
        ("Status",   hex32(core.cp0_status)),
        ("Cause",    hex32(core.cp0_cause)),
        ("EPC",      hex64(core.cp0_epc)),
        ("PRId",     hex32(core.cp0_prid)),
        ("Config",   hex32(core.cp0_config)),
        ("LLAddr",   hex32(core.cp0_lladdr)),
        ("WatchLo",  hex32(core.cp0_watchlo)),
        ("WatchHi",  hex32(core.cp0_watchhi)),
        ("XContext", hex64(core.cp0_xcontext)),
        ("ECC",      hex32(core.cp0_ecc)),
        ("CacheErr", hex32(core.cp0_cacheerr)),
        ("TagLo",    hex32(core.cp0_taglo)),
        ("TagHi",    hex32(core.cp0_taghi)),
    ];
    let body: Vec<String> = cp0.iter().map(|(n, v)| format!("    \"{}\": {}", n, v)).collect();
    s.push_str(&body.join(",\n"));
    s.push_str("\n  }\n}\n");
    s
}

impl TestDevice {
    /// Nanoseconds since `epoch`, latched so the HI read that follows sees the
    /// same sample.
    fn latch_host_ns(&self) -> u64 {
        let ns = self.epoch.elapsed().as_nanos() as u64;
        self.host_ns_latch.store(ns, Ordering::Relaxed);
        ns
    }

    /// Retired guest instructions, latched the same way. Zero when no core is
    /// attached (a guest reading a flat zero learns the counter is unusable
    /// rather than getting a plausible-looking wrong number).
    fn latch_icount(&self) -> u64 {
        let guard = self.core.lock();
        let n = match guard.as_ref() {
            // SAFETY: see CorePtr — the CPU thread is the one issuing this load.
            Some(CorePtr(ptr)) => unsafe { (**ptr).hot.cycles },
            None => 0,
        };
        drop(guard);
        self.icount_latch.store(n, Ordering::Relaxed);
        n
    }

    fn read_reg(&self, off: u32) -> u32 {
        match off {
            REG_SIGNATURE => SIGNATURE,
            REG_HOST_NS_LO => self.latch_host_ns() as u32,
            REG_HOST_NS_HI => (self.host_ns_latch.load(Ordering::Relaxed) >> 32) as u32,
            REG_ICOUNT_LO => self.latch_icount() as u32,
            REG_ICOUNT_HI => (self.icount_latch.load(Ordering::Relaxed) >> 32) as u32,
            REG_CAPS => CAP_TIMEBASE | CAP_RUN_CONFIG,
            REG_RUN_CONFIG => self.run_config.load(Ordering::Relaxed),
            _ => 0,
        }
    }
}

impl BusDevice for TestDevice {
    fn read32(&self, addr: u32) -> BusRead32 {
        BusRead32::ok(self.read_reg((addr - TEST_DEV_BASE) & (REG_WINDOW - 1) & !3))
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        match (addr - TEST_DEV_BASE) & (REG_WINDOW - 1) & !3 {
            REG_PUTC => self.putc(val as u8),
            REG_DUMP => self.dump(val),
            REG_EXIT => self.exit(val),
            _ => {}
        }
        BUS_OK
    }

    fn read8(&self, addr: u32) -> BusRead8 {
        // Big-endian lane select within the containing word: byte 0 is the high
        // byte. Reading the low byte of a latching register still latches, since
        // the whole word is materialized to pick the lane out of.
        let off = (addr - TEST_DEV_BASE) & (REG_WINDOW - 1);
        let word = self.read_reg(off & !3);
        BusRead8::ok((word >> (8 * (3 - (off & 3)))) as u8)
    }

    fn write8(&self, addr: u32, val: u8) -> u32 {
        // A byte store to any lane of a register acts on that register, so
        // `sb` to PUTC works without the guest building a whole word.
        match (addr - TEST_DEV_BASE) & (REG_WINDOW - 1) & !3 {
            REG_PUTC => self.putc(val),
            REG_DUMP => self.dump(val as u32),
            REG_EXIT => self.exit(val as u32),
            _ => {}
        }
        BUS_OK
    }
}

impl Device for TestDevice {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) { self.running.store(false, Ordering::Relaxed); }
    fn start(&self) { self.running.store(true, Ordering::Relaxed); }
    fn is_running(&self) -> bool { self.running.load(Ordering::Relaxed) }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("testdev".to_string(), "Test device status: testdev".to_string())]
    }

    fn execute_command(&self, cmd: &str, _args: &[&str], mut writer: Box<dyn Write + Send>) -> Result<(), String> {
        if cmd != "testdev" {
            return Err("Command not found".to_string());
        }
        writeln!(writer, "test device @ {:#010x}  signature {:#010x}", TEST_DEV_BASE, SIGNATURE).unwrap();
        match &self.dump_path {
            Some(p) => writeln!(writer, "  dump file : {}", p.display()).unwrap(),
            None => writeln!(writer, "  dump file : (embedded — DUMP is a no-op)").unwrap(),
        }
        writeln!(writer, "  dumps     : {}  (last tag {})",
                 self.dumps.load(Ordering::Relaxed), self.last_tag.load(Ordering::Relaxed)).unwrap();
        writeln!(writer, "  chars out : {}", self.chars.load(Ordering::Relaxed)).unwrap();
        Ok(())
    }
}

impl Resettable for TestDevice {
    fn power_on(&self) {
        self.dumps.store(0, Ordering::Relaxed);
        self.last_tag.store(0, Ordering::Relaxed);
        self.chars.store(0, Ordering::Relaxed);
        self.host_ns_latch.store(0, Ordering::Relaxed);
        self.icount_latch.store(0, Ordering::Relaxed);
    }
}

impl Saveable for TestDevice {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::value::Table::new();
        t.insert("dumps".into(), toml::Value::Integer(self.dumps.load(Ordering::Relaxed) as i64));
        t.insert("last_tag".into(), toml::Value::Integer(self.last_tag.load(Ordering::Relaxed) as i64));
        t.insert("chars".into(), toml::Value::Integer(self.chars.load(Ordering::Relaxed) as i64));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let get = |k: &str| v.get(k).and_then(|x| x.as_integer()).unwrap_or(0);
        self.dumps.store(get("dumps") as u32, Ordering::Relaxed);
        self.last_tag.store(get("last_tag") as u64, Ordering::Relaxed);
        self.chars.store(get("chars") as u64, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_reads_back_word_and_byte_wise() {
        let d = TestDevice::new("unused");
        assert_eq!(d.read32(TEST_DEV_BASE).data, SIGNATURE);
        // Repeats every REG_WINDOW bytes across the decoded window. (It was
        // every 16 until the clock/icount registers claimed 0x10..0x20.)
        assert_eq!(d.read32(TEST_DEV_BASE + REG_WINDOW).data, SIGNATURE);
        let bytes: Vec<u8> = (0..4).map(|i| d.read8(TEST_DEV_BASE + i).data).collect();
        assert_eq!(&bytes, b"IRIS", "signature is big-endian ASCII");
    }

    #[test]
    fn caps_advertises_the_timebase() {
        let d = TestDevice::new("unused");
        assert_eq!(d.read32(TEST_DEV_BASE + REG_CAPS).data & CAP_TIMEBASE, CAP_TIMEBASE);
    }

    #[test]
    fn run_config_round_trips_and_defaults_to_unrestricted() {
        // The property the guest depends on: a zero word is "run everything,
        // full length", so reading the register unconditionally is safe.
        assert_eq!(RunConfig::from_word(0), RunConfig::ALL);
        assert_eq!(RunConfig::ALL.to_word(), 0);

        for c in [
            RunConfig { groups: 0xBEEF, time_pct: 30, repeats: 1 },
            RunConfig { groups: 1, time_pct: 4095, repeats: 15 },
            RunConfig { groups: 0, time_pct: 100, repeats: 2 },
        ] {
            assert_eq!(RunConfig::from_word(c.to_word()), c, "{:?} did not round-trip", c);
        }

        // Fields must not bleed into each other.
        let c = RunConfig { groups: 0xFFFF, time_pct: 0, repeats: 0 };
        assert_eq!(c.to_word(), 0xFFFF_0000);
    }

    #[test]
    fn the_run_config_register_reads_back_what_it_was_built_with() {
        let want = RunConfig { groups: 0b101, time_pct: 30, repeats: 1 };
        let d = TestDevice::new_embedded(
            std::sync::Arc::new(Mutex::new(Vec::new())),
            Box::new(|_| {}),
            want,
        );
        assert_eq!(d.read32(TEST_DEV_BASE + REG_CAPS).data & CAP_RUN_CONFIG, CAP_RUN_CONFIG);
        assert_eq!(RunConfig::from_word(d.read32(TEST_DEV_BASE + REG_RUN_CONFIG).data), want);
    }

    #[test]
    fn embedded_mode_captures_output_and_reports_exit_without_killing_the_process() {
        let sink = std::sync::Arc::new(Mutex::new(Vec::new()));
        let seen = std::sync::Arc::new(AtomicU32::new(u32::MAX));
        let d = {
            let seen = seen.clone();
            TestDevice::new_embedded(
                sink.clone(),
                Box::new(move |c| seen.store(c, Ordering::Relaxed)),
                RunConfig::ALL,
            )
        };

        for b in b"hi\n" { d.write32(TEST_DEV_BASE + REG_PUTC, *b as u32); }
        assert_eq!(&*sink.lock(), b"hi\n");

        // DUMP has nowhere to go and must not try: still counted, no file, and
        // above all no panic on a device with no dump path.
        d.write32(TEST_DEV_BASE + REG_DUMP, 3);
        assert_eq!(d.dumps_written(), 1);
        assert!(d.dump_path().is_none());

        // The point of the whole exercise: EXIT returns instead of ending the
        // host process, so this test can observe it at all.
        d.write32(TEST_DEV_BASE + REG_EXIT, 7);
        assert_eq!(seen.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn host_ns_latches_so_lo_then_hi_cannot_tear() {
        let d = TestDevice::new("unused");
        let lo = d.read32(TEST_DEV_BASE + REG_HOST_NS_LO).data;
        let hi = d.read32(TEST_DEV_BASE + REG_HOST_NS_HI).data;
        let first = ((hi as u64) << 32) | lo as u64;
        // HI on its own never re-samples: read it again and it is the same half
        // of the same latch, however much time has passed in between.
        assert_eq!(d.read32(TEST_DEV_BASE + REG_HOST_NS_HI).data, hi);

        // A fresh LO read advances (the clock is monotonic and this is not
        // instantaneous, but do not assume it ticked — assert non-regression).
        let lo2 = d.read32(TEST_DEV_BASE + REG_HOST_NS_LO).data;
        let hi2 = d.read32(TEST_DEV_BASE + REG_HOST_NS_HI).data;
        let second = ((hi2 as u64) << 32) | lo2 as u64;
        assert!(second >= first, "host clock went backwards: {} -> {}", first, second);
    }

    #[test]
    fn icount_reports_retired_instructions_and_zero_with_no_core() {
        let d = TestDevice::new("unused");
        // No core attached: a flat zero, not a plausible-looking wrong number.
        assert_eq!(d.read32(TEST_DEV_BASE + REG_ICOUNT_LO).data, 0);
        assert_eq!(d.read32(TEST_DEV_BASE + REG_ICOUNT_HI).data, 0);

        let mut core = MipsCore::new();
        core.hot.cycles = 0x1_2345_6789;
        d.attach_core(&core as *const MipsCore);
        let lo = d.read32(TEST_DEV_BASE + REG_ICOUNT_LO).data;
        let hi = d.read32(TEST_DEV_BASE + REG_ICOUNT_HI).data;
        assert_eq!(((hi as u64) << 32) | lo as u64, 0x1_2345_6789);
    }

    #[test]
    fn byte_reads_pick_the_big_endian_lane_of_any_register() {
        let d = TestDevice::new("unused");
        let mut core = MipsCore::new();
        core.hot.cycles = 0x0000_0000_AABB_CCDD;
        d.attach_core(&core as *const MipsCore);
        let bytes: Vec<u8> = (0..4)
            .map(|i| d.read8(TEST_DEV_BASE + REG_ICOUNT_LO + i).data)
            .collect();
        assert_eq!(&bytes, &[0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn dump_json_covers_the_whole_architectural_state() {
        let mut core = MipsCore::new();
        core.pc = 0xFFFF_FFFF_8810_0000;
        core.gpr[31] = 0xDEAD_BEEF;
        core.hi = 1;
        core.lo = 2;
        core.fpr[7] = 0x4000_0000_0000_0000;
        core.fpu_fcsr = 0x0080_0000;
        core.cp0_status = 0x3400_0000;
        core.cp0_epc = 0x8810_0004;

        let j = dump_json(&core, 42);
        assert!(j.contains("\"tag\": 42"));
        assert!(j.contains("\"pc\": \"0xffffffff88100000\""));
        assert!(j.contains("\"hi\": \"0x0000000000000001\""));
        assert!(j.contains("0x00000000deadbeef"), "gpr[31] must be present");
        assert!(j.contains("0x4000000000000000"), "fpr[7] must be present");
        assert!(j.contains("\"fcsr\": \"0x00800000\""));
        assert_eq!(j.matches("0x").count() >= 32 + 32, true, "all 64 registers present");
        for reg in ["Status", "Cause", "EPC", "BadVAddr", "Config", "PRId", "EntryHi", "EntryLo0",
                    "EntryLo1", "PageMask", "Index", "Random", "Wired", "Context", "XContext",
                    "Count", "Compare", "WatchLo", "WatchHi", "LLAddr", "TagLo", "TagHi"] {
            assert!(j.contains(&format!("\"{}\"", reg)), "cp0 {} missing", reg);
        }
        // The R4400/R5000 identification comes from PRId, not a build flag.
        assert!(j.contains("\"cpu\": \"R4400\"") || j.contains("\"cpu\": \"R5000\""));
    }

    #[test]
    fn dump_writes_a_file_and_counts() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let path = std::env::temp_dir().join(format!("iris-testdev-{}.json", nanos));
        let core = MipsCore::new();
        let d = TestDevice::new(&path);
        d.attach_core(&core as *const MipsCore);

        d.write32(TEST_DEV_BASE + REG_DUMP, 7);
        assert_eq!(d.dumps_written(), 1);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"tag\": 7"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_load_round_trip() {
        let core = MipsCore::new();
        let a = TestDevice::new("unused");
        a.attach_core(&core as *const MipsCore);
        a.write32(TEST_DEV_BASE + REG_PUTC, b'x' as u32);
        a.last_tag.store(3, Ordering::Relaxed);
        a.dumps.store(2, Ordering::Relaxed);

        let saved = a.save_state();
        let b = TestDevice::new("unused");
        b.load_state(&saved).unwrap();
        assert_eq!(b.save_state(), saved, "save → load → save must round-trip");

        b.power_on();
        assert_eq!(b.dumps_written(), 0, "power_on clears counters");
    }
}
