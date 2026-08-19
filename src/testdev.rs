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
//! The guest detects the device by reading `SIGNATURE`; on real hardware the
//! empty slot times out, so a suite falls back to SCC-only output.

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

/// Reads back at `REG_SIGNATURE` — "IRIS" in ASCII.
pub const SIGNATURE: u32 = 0x4952_4953;

/// Where `REG_DUMP` writes go when no path is configured.
pub const DEFAULT_DUMP_PATH: &str = "iris-testdev-dump.json";

pub struct TestDevice {
    /// Dump file path. Never defaults to a bare CWD-relative name in a
    /// sandboxed app, where the CWD is `/` and unwritable — the caller passes
    /// an absolute path there.
    dump_path: std::path::PathBuf,
    /// Set by `attach_core`. Only read from the CPU thread, which is the thread
    /// that issues the store that lands here, so there is no race — the same
    /// process-lifetime argument as `MipsCpu`'s `cycles_ptr`/`interrupts_ptr`.
    core: Mutex<Option<CorePtr>>,
    dumps: AtomicU32,
    last_tag: AtomicU64,
    chars: AtomicU64,
    running: AtomicBool,
}

/// Raw pointer to the executor's `MipsCore`, valid for the process lifetime.
struct CorePtr(*const MipsCore);
// SAFETY: the pointee outlives the device and is only dereferenced from the CPU
// thread while that thread is inside its own store instruction.
unsafe impl Send for CorePtr {}
unsafe impl Sync for CorePtr {}

impl TestDevice {
    pub fn new(dump_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            dump_path: dump_path.into(),
            core: Mutex::new(None),
            dumps: AtomicU32::new(0),
            last_tag: AtomicU64::new(0),
            chars: AtomicU64::new(0),
            running: AtomicBool::new(false),
        }
    }

    /// Point the device at the CPU state it dumps. Called once at wiring time.
    pub fn attach_core(&self, core: *const MipsCore) {
        *self.core.lock() = Some(CorePtr(core));
    }

    pub fn dump_path(&self) -> &std::path::Path {
        &self.dump_path
    }

    pub fn dumps_written(&self) -> u32 {
        self.dumps.load(Ordering::Relaxed)
    }

    fn putc(&self, byte: u8) {
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(&[byte]);
        // Flush per line so a hung test still shows everything it printed.
        if byte == b'\n' {
            let _ = out.flush();
        }
        self.chars.fetch_add(1, Ordering::Relaxed);
    }

    /// Write the full machine state as JSON. `tag` distinguishes dump points, so
    /// a test can dump more than once; the file is overwritten each time.
    fn dump(&self, tag: u32) {
        self.last_tag.store(tag as u64, Ordering::Relaxed);
        let guard = self.core.lock();
        let Some(CorePtr(ptr)) = guard.as_ref() else {
            eprintln!("test device: DUMP with no CPU attached");
            return;
        };
        // SAFETY: see CorePtr — the CPU thread is the one storing to this register.
        let core = unsafe { &**ptr };
        let json = dump_json(core, tag);
        drop(guard);

        match std::fs::write(&self.dump_path, json) {
            Ok(()) => {
                self.dumps.fetch_add(1, Ordering::Relaxed);
                eprintln!("test device: dump {} → {}", tag, self.dump_path.display());
            }
            Err(e) => eprintln!("test device: dump {} to {}: {}", tag, self.dump_path.display(), e),
        }
    }

    /// Terminate the emulator with the guest's exit code. Stdout is flushed
    /// first so buffered `PUTC` output is never lost.
    fn exit(&self, code: u32) -> ! {
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

impl BusDevice for TestDevice {
    fn read32(&self, addr: u32) -> BusRead32 {
        match (addr - TEST_DEV_BASE) & 0xF {
            REG_SIGNATURE => BusRead32::ok(SIGNATURE),
            _ => BusRead32::ok(0),
        }
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        match (addr - TEST_DEV_BASE) & 0xF {
            REG_PUTC => self.putc(val as u8),
            REG_DUMP => self.dump(val),
            REG_EXIT => self.exit(val),
            _ => {}
        }
        BUS_OK
    }

    fn read8(&self, addr: u32) -> BusRead8 {
        // Byte reads of SIGNATURE, big-endian: byte 0 is the high byte.
        let off = (addr - TEST_DEV_BASE) & 0xF;
        if off < 4 {
            return BusRead8::ok((SIGNATURE >> (8 * (3 - off))) as u8);
        }
        BusRead8::ok(0)
    }

    fn write8(&self, addr: u32, val: u8) -> u32 {
        // A byte store to any lane of a register acts on that register, so
        // `sb` to PUTC works without the guest building a whole word.
        match (addr - TEST_DEV_BASE) & 0xC {
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
        writeln!(writer, "  dump file : {}", self.dump_path.display()).unwrap();
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
        // Repeats every 16 bytes across the decoded window.
        assert_eq!(d.read32(TEST_DEV_BASE + 0x10).data, SIGNATURE);
        let bytes: Vec<u8> = (0..4).map(|i| d.read8(TEST_DEV_BASE + i).data).collect();
        assert_eq!(&bytes, b"IRIS", "signature is big-endian ASCII");
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
