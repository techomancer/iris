use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use parking_lot::Mutex;
use crate::traits::Device;

/// The bit of event `name` in a module's `EVENTS` table; a name the table lacks fails the build.
pub const fn event(names: &[&str], name: &str) -> u32 {
    let mut i = 0;
    while i < names.len() {
        if names[i].len() == name.len() {
            let (a, b) = (names[i].as_bytes(), name.as_bytes());
            let mut j = 0;
            while j < a.len() && a[j] == b[j] { j += 1; }
            if j == a.len() { return 1 << i; }
        }
        i += 1;
    }
    panic!("no such event")
}

/// One module's trace state: a bit per named event (all clear is off) and its file sink.
struct Point {
    name:   &'static str,
    events: &'static [&'static str],
    on:     AtomicU32,
    file:   Mutex<Option<(String, BufWriter<File>)>>,
}

impl Point {
    const fn new(name: &'static str, events: &'static [&'static str]) -> Self {
        Self { name, events, on: AtomicU32::new(0), file: Mutex::new(None) }
    }
}

// Each module: its name, then its event names (none for a module that is simply on or off).
macro_rules! catalogue {
    ($($(#[$doc:meta])* $id:ident $name:literal $events:expr,)*) => {
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        #[repr(usize)]
        pub enum LogModule { $($(#[$doc])* $id,)* }

        impl LogModule {
            pub const ALL: &'static [LogModule] = &[$(LogModule::$id),*];
        }

        static POINTS: [Point; LogModule::ALL.len()] = [$(Point::new($name, $events)),*];
    };
}

catalogue! {
    Net "net" &[],
    Hpc3 "hpc3" &[],
    Seeq "seeq" &[],
    Hal2 "hal2" &[],
    Mc "mc" &[],
    Rex3 "rex3" &[],
    Mips "mips" crate::cpu::mips_exec::EVENTS,
    Ioc "ioc" &[],
    Scsi "scsi" &[],
    Pdma "pdma" crate::dev::hpc3::PDMA_EVENTS,
    Vino "vino" &[],
    Dcb "dcb" &[],
    Vc2 "vc2" &[],
    Cmap "cmap" &[],
    Xmap "xmap" &[],
    Bt445 "bt445" &[],
    Scc "scc" &[],
    Ps2 "ps2" &[],
    Rtc "rtc" &[],
    Eeprom "eeprom" &[],
    L1i "l1i" crate::cpu::mips_cache_v2::EVENTS,  // L1 instruction cache: fetch hit/miss, fill, cache ops
    L1d "l1d" crate::cpu::mips_cache_v2::EVENTS,  // L1 data cache: read/write hit/miss, fill, cache ops
    L2c "l2c" crate::cpu::mips_cache_v2::EVENTS,  // L2 unified cache: fill, writeback, invalidate, cache ops
    Ultra "ultra" &[],  // N64 dev board: shm reads/writes, interrupts, register traffic
    /// Second WD33C93A (fullhouse/Indigo2 SCSI controller 1). Kept as a
    /// distinct module (rather than folding into `Scsi`) so it can be
    /// toggled independently of controller 0.
    Scsi1 "scsi1" &[],
    /// HPC3-side motherboard EEPROM (NVRAM/env vars/MAC @ words 0x7D-0x7F,
    /// `nveeprom` command). Kept distinct from `Eeprom` (MC-side CPU
    /// daughtercard chip, `eeprom` command) — the two are separate physical
    /// parts on real hardware, see machine.rs's `eeprom_mc`/`eeprom_hpc3`.
    Nveeprom "nveeprom" &[],
    /// GR2 (XZ / Extreme) graphics: HQ2, GE7, RE3, VC1, XMAP5, Bt457.
    Gr2 "gr2" &[],
    /// The Octane's RAD1 PCI audio: rings, status DMA and register accesses
    /// the model does not support.
    Rad1 "rad1" &[],
    /// The Octane's VPro (Odyssey) graphics board: guest accesses the
    /// model does not support.
    Odyssey "odyssey" &[],
    Bedrock "bedrock" &[],
    /// The IP27 node controller: guest accesses the model does not support.
    Hub "hub" &[],
    /// PCI configuration cycles on every bus: who was addressed, what was
    /// read or written, and whether anything answered.
    Pci "pci" &[],
    /// The Octane's Bridge ASIC: register and translation-table accesses,
    /// and transfers a function started that reached no memory.
    Bridge "bridge" &[],
    /// The Octane's HEART: processor-interface register accesses (not
    /// counter reads), interrupt vectors raised and lowered, and bus
    /// accesses nothing answers.
    Heart "heart" &[],
    /// The NVM Express controller: register writes, controller state
    /// changes, and the commands it runs (events `state admin regs io`).
    Nvme "nvme" crate::dev::nvme::EVENTS,
    L1 "l1" &[],
    Usb "usb" &[],
    /// The IP27 module controller (ELSC and the Origin 2000 module), its
    /// configuration NVRAM and its I2C controller: protocol surprises and
    /// refused commands.
    Module "module" &[],
    /// The InfiniteReality (KONA) graphics pipe's host interface: guest
    /// accesses and command words the model does not support.
    Kona "kona" &[],
    /// The ICE1724 audio controller: accesses and values nothing
    /// establishes.
    Ice1724 "ice1724" &[],
    /// The physical bus: accesses that hit the black hole or a bus-error
    /// region, and in a developer build every access.
    Phys "phys" &[],
    /// The O2's audio: every register access and codec operation.
    O2Audio "o2audio" &[],
}

impl LogModule {
    fn point(self) -> &'static Point { &POINTS[self as usize] }

    pub fn name(self) -> &'static str { self.point().name }

    pub fn events(self) -> &'static [&'static str] { self.point().events }

    /// A mask as the names of its events (`on` for a module with none), or `-` when off.
    fn describe(self, mask: u32) -> String {
        if mask == 0 { return "-".to_string(); }
        if self.events().is_empty() { return "on".to_string(); }
        let named: Vec<&str> = self.events().iter().enumerate().filter(|(i, _)| mask >> i & 1 != 0).map(|(_, e)| *e).collect();
        if named.len() == self.events().len() { "all".to_string() } else { named.join("+") }
    }
}

/// True if `s` matches `p`, where `*` in `p` stands for any run of characters.
fn glob(p: &str, s: &str) -> bool {
    match p.split_once('*') {
        None => p == s,
        Some((head, rest)) => s.strip_prefix(head)
            .is_some_and(|s| (0..=s.len()).any(|i| s.is_char_boundary(i) && glob(rest, &s[i..]))),
    }
}

/// The bits of `names` a glob matches.
fn matching(p: &str, names: &[(&str, u32)]) -> u32 {
    names.iter().filter(|(n, _)| glob(p, n)).fold(0, |b, (_, x)| b | x)
}

/// The one pattern grammar for names with a bit each: comma-separated globs, `all` for the set
/// `all`, a leading `-` to remove. A term matching nothing is an error.
pub fn select(pattern: &str, names: &[(&str, u32)], all: u32) -> Result<u32, String> {
    let mut mask = 0;
    for term in pattern.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let (neg, body) = term.strip_prefix('-').map_or((false, term), |t| (true, t));
        let bits = if body == "all" { all } else { matching(body, names) };
        if bits == 0 {
            return Err(format!("'{term}' matches nothing (have: all, {})", names.iter().map(|n| n.0).collect::<Vec<_>>().join(", ")));
        }
        if neg { mask &= !bits } else { mask |= bits }
    }
    Ok(mask)
}

/// The points a pattern selects, as a mask per module: globs over `module.event`.
/// A module with no events is one point.
fn parse(pattern: &str) -> Result<Vec<(LogModule, u32)>, String> {
    let mut sel = vec![0u32; LogModule::ALL.len()];
    for term in pattern.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let (neg, body) = term.strip_prefix('-').map_or((false, term), |t| (true, t));
        let (mp, ep) = if body == "all" { ("*", "*") } else { body.split_once('.').unwrap_or((body, "*")) };
        let mut hit = false;
        for &m in LogModule::ALL {
            let events: Vec<(&str, u32)> = m.events().iter().enumerate().map(|(i, e)| (*e, 1 << i)).collect();
            let bits = if !glob(mp, m.name()) { 0 }
                else if events.is_empty() { (ep == "*") as u32 }
                else { matching(ep, &events) };
            hit |= bits != 0;
            if neg { sel[m as usize] &= !bits } else { sel[m as usize] |= bits }
        }
        if !hit { return Err(format!("No trace point matches '{term}'. Try `help log`")); }
    }
    Ok(LogModule::ALL.iter().map(|&m| (m, sel[m as usize])).filter(|&(_, b)| b != 0).collect())
}

// ── DevLog ───────────────────────────────────────────────────────────────────

/// Generic write+send sink used by DevLog. Holds either a TCP monitor connection
/// or a host stream like stderr (for diagnostic mode).
pub type DevLogWriter = Arc<Mutex<dyn Write + Send>>;

pub struct DevLog {
    /// All currently connected monitor clients (and any host sinks like stderr).
    /// Entries are pruned on write error.
    writers: Mutex<Vec<DevLogWriter>>,
}

pub static DEVLOG: DevLog = DevLog { writers: Mutex::new(Vec::new()) };

pub fn devlog() -> &'static DevLog { &DEVLOG }

impl DevLog {
    /// Register a generic Write+Send sink (e.g. stderr for diagnostic runs).
    pub fn add_sink(&self, sink: DevLogWriter) {
        self.writers.lock().push(sink);
    }

    /// Unregister a sink `add_sink` took, such as a monitor client that hung up.
    pub fn remove_sink(&self, sink: &DevLogWriter) {
        self.writers.lock().retain(|w| !Arc::ptr_eq(w, sink));
    }

    /// Turn the points a pattern names on or off; returns the modules it reached. A module
    /// left with none on also closes its file sink.
    pub fn select(&self, pattern: &str, on: bool) -> Result<Vec<LogModule>, String> {
        let sel = parse(pattern)?;
        for &(m, bits) in &sel {
            if on {
                m.point().on.fetch_or(bits, Ordering::Relaxed);
            } else if m.point().on.fetch_and(!bits, Ordering::Relaxed) & !bits == 0 {
                self.clear_file(m);
            }
        }
        Ok(sel.into_iter().map(|(m, _)| m).collect())
    }

    /// Turn on what `pattern` names, with output to stderr as well as the monitor.
    pub fn start(&self, pattern: &str) -> Result<(), String> {
        if pattern.is_empty() { return Ok(()); }
        self.select(pattern, true)?;
        static STDERR: std::sync::Once = std::sync::Once::new();
        STDERR.call_once(|| self.add_sink(Arc::new(Mutex::new(std::io::stderr()))));
        Ok(())
    }

    /// Open a file sink for a module.
    pub fn set_file(&self, m: LogModule, path: &str) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
        *m.point().file.lock() = Some((path.to_string(), BufWriter::new(file)));
        Ok(())
    }

    /// Close a module's file sink (output reverts to monitor broadcast).
    pub fn clear_file(&self, m: LogModule) {
        *m.point().file.lock() = None;
    }

    /// Write a log message unconditionally — bypasses the module enabled flag.
    /// Use for output that must interleave with dlog output (e.g. step Exec: lines).
    /// No-ops silently if there are no writers.
    pub fn write_unconditional(&self, args: fmt::Arguments) {
        let msg = fmt::format(args);
        let mut writers = self.writers.lock();
        writers.retain(|w| {
            let mut guard = w.lock();
            write!(guard, "{}\n", msg).is_ok() && guard.flush().is_ok()
        });
    }

    /// Write a log message for a module that is on. Out of line, so the
    /// formatting stays out of the hot function that logs.
    #[cold]
    #[inline(never)]
    pub fn write(&self, m: LogModule, args: fmt::Arguments) {
        let msg = fmt::format(args);

        // File sink takes priority — write there and return.
        if let Some((_, f)) = &mut *m.point().file.lock() {
            let _ = writeln!(f, "[{}] {}", m.name(), msg);
            let _ = f.flush();
            return;
        }

        // Broadcast to all connected monitor clients; prune dead connections.
        self.writers.lock().retain(|w| {
            let mut guard = w.lock();
            write!(guard, "[{}] {}\n", m.name(), msg).is_ok() && guard.flush().is_ok()
        });
    }
}

/// True if any event of the module is on. One relaxed load of a static.
#[inline(always)]
pub fn devlog_is_active(m: LogModule) -> bool {
    m.point().on.load(Ordering::Relaxed) != 0
}

/// The module's events that are on, as bits of its `EVENTS` table.
#[inline(always)]
pub fn devlog_mask(m: LogModule) -> u32 {
    m.point().on.load(Ordering::Relaxed)
}

/// True if the module's event `ev` (an `EVENTS` bit) is on.
#[inline(always)]
pub fn devlog_on(m: LogModule, ev: u32) -> bool {
    devlog_mask(m) & ev != 0
}

// ── Macros ───────────────────────────────────────────────────────────────────

/// Log to a module, or to one of its events (`dlog!(L1d, CACHE_LOG_MISS, ..)`). One relaxed
/// load when off; output goes to the monitor clients or the module's file sink.
#[macro_export]
macro_rules! dlog {
    ($mod:expr, $ev:ident, $($arg:tt)*) => {
        if $crate::devlog::devlog_on($mod, $ev) {
            $crate::devlog::DEVLOG.write($mod, format_args!($($arg)*));
        }
    };
    ($mod:expr, $($arg:tt)*) => {
        if $crate::devlog::devlog_is_active($mod) {
            $crate::devlog::DEVLOG.write($mod, format_args!($($arg)*));
        }
    };
}

/// Unconditional log — bypasses module enabled flag, goes to all monitor writers.
/// Use when output must interleave with dlog! output on the same connection.
/// No-ops if no writers are connected.
#[macro_export]
macro_rules! dlog_unconditional {
    ($($arg:tt)*) => {
        $crate::devlog::DEVLOG.write_unconditional(format_args!($($arg)*))
    };
}

/// Hot-path log. Compiles to nothing in non-developer (release) builds.
#[macro_export]
macro_rules! dlog_dev {
    ($($arg:tt)*) => {
        {
            #[cfg(feature = "developer")]
            $crate::dlog!($($arg)*);
        }
    };
}

// ── Device impl for monitor `log` command ────────────────────────────────────

/// The monitor's `log` command, over the global log state.
pub struct LogDevice;

impl Device for LogDevice {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { false }

    fn register_commands(&self) -> Vec<(String, String)> {
        let modules: Vec<&str> = LogModule::ALL.iter().map(|m| m.name()).collect();
        let events: String = LogModule::ALL.iter().filter(|m| !m.events().is_empty())
            .map(|m| format!("\n             {} events: {}", m.name(), m.events().join(" "))).collect();
        vec![(
            "log".to_string(),
            format!("[DEV] log <points> <on|off>  |  log <points> file <path|off>  |  log status\n\
             \x20             points: comma-separated globs over module.event, a bare module for all its events,\n\
             \x20             all for everything, a leading - to remove: heart,l1d.*,-l1d.hit\n\
             \x20             modules: {}{}\n\
             \x20             [DEV] = some of a module's messages appear only in a --features developer build", modules.join(" "), events),
        )]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "log" { return Err(format!("Unknown command: {}", cmd)); }
        let arg0 = args.get(0).copied().unwrap_or("");
        let dl = &DEVLOG;

        if arg0 == "status" {
            writeln!(writer, "{:<8} {:<24}  file", "module", "events").map_err(|e| e.to_string())?;
            writeln!(writer, "{}", "-".repeat(60)).map_err(|e| e.to_string())?;
            for &m in LogModule::ALL {
                let path = m.point().file.lock().as_ref().map(|(p, _)| p.clone()).unwrap_or_default();
                writeln!(writer, "{:<8} {:<24}  {}", m.name(), m.describe(devlog_mask(m)), path)
                    .map_err(|e| e.to_string())?;
            }
            writeln!(writer, "\nMonitor clients: {}", dl.writers.lock().len()).map_err(|e| e.to_string())?;
            return Ok(());
        }

        let arg2 = args.get(2).copied().unwrap_or("");
        match args.get(1).copied().unwrap_or("") {
            "on" => { dl.select(arg0, true)?; }
            "off" => { dl.select(arg0, false)?; }
            "file" if arg2 == "off" || arg2.is_empty() => {
                for (m, _) in parse(arg0)? { dl.clear_file(m); }
            }
            "file" => {
                if parse(arg0)?.len() != 1 {
                    return Err("log file takes one module".to_string());
                }
                for m in dl.select(arg0, true)? { dl.set_file(m, arg2)?; }
            }
            _ => return Err("Usage: log <points> <on|off|file <path|off>>  |  log status".to_string()),
        }
        writeln!(writer, "log {}", args.join(" ")).map_err(|e| e.to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picked(pattern: &str) -> Vec<(&'static str, u32)> {
        parse(pattern).unwrap().into_iter().map(|(m, b)| (m.name(), b)).collect()
    }

    /// Every module's name selects that module alone, and no two share one.
    #[test]
    fn every_module_name_selects_itself() {
        for (i, m) in LogModule::ALL.iter().enumerate() {
            assert_eq!(*m as usize, i);
            assert_eq!(picked(m.name()).iter().map(|p| p.0).collect::<Vec<_>>(), [m.name()]);
        }
    }

    /// The `log` command's help names every module `log` takes.
    #[test]
    fn the_log_help_names_every_module() {
        let help = LogDevice.register_commands().remove(0).1;
        let listed = help.lines().find_map(|l| l.trim().strip_prefix("modules: ")).expect("a modules line");
        let listed: Vec<&str> = listed.split_whitespace().collect();
        for m in LogModule::ALL {
            assert!(listed.contains(&m.name()), "{} is missing from the help", m.name());
        }
    }

    /// An event is its place in the module's table, and a mask names the events it holds.
    #[test]
    fn events_are_bits_by_name() {
        assert_eq!(event(&["a", "bc", "b"], "b"), 4);
        assert_eq!(LogModule::L1d.describe(event(LogModule::L1d.events(), "miss")), "miss");
        assert_eq!(LogModule::L1d.describe(0), "-");
        assert_eq!(LogModule::L1d.describe(0b111), "all");
        assert_eq!(LogModule::Heart.describe(1), "on");
    }

    /// The one pattern grammar: globs over `module.event`, `-` to remove, errors for a miss.
    #[test]
    fn a_pattern_selects_points_by_glob_and_minus() {
        assert_eq!(picked("heart"), [("heart", 1)]);
        assert_eq!(picked("l1d.miss"), [("l1d", 2)]);
        assert_eq!(picked("l1*.hit,-l1i.*"), [("l1d", 1)]);
        assert_eq!(picked("pdma.enet*"), [("pdma", 0b1100_0000_0000)]);
        assert_eq!(picked("all").len(), LogModule::ALL.len());
        assert_eq!(picked("net, -net"), []);
        assert!(parse("heart.irq").is_err(), "heart has no events");
        assert!(parse("nope").is_err());
    }

    /// `select` turns points on and off, one at a time.
    #[test]
    fn select_turns_points_on_and_off() {
        let dl = &DEVLOG;
        dl.select("pdma.hal0,pdma.enet1", true).unwrap();
        assert_eq!(devlog_mask(LogModule::Pdma), 1 | 1 << 11);
        dl.select("pdma.hal0", false).unwrap();
        assert!(devlog_on(LogModule::Pdma, 1 << 11) && !devlog_on(LogModule::Pdma, 1));
        dl.select("pdma", false).unwrap();
        assert!(!devlog_is_active(LogModule::Pdma));
    }

    /// Every pattern the docs show (`--log P`, `log = "P"`, `log P on|off`) parses.
    #[test]
    fn the_documented_patterns_parse() {
        let mut seen = 0;
        for doc in [include_str!("../debug.md"), include_str!("../HELP.md"), include_str!("../FEATURES.md")] {
            for line in doc.lines() {
                let mut found = Vec::new();
                if let Some((_, r)) = line.split_once("--log ") { found.extend(r.split_whitespace().next()); }
                if let Some(r) = line.trim_start_matches("# ").strip_prefix("log = \"") { found.extend(r.split('"').next()); }
                for (r, _) in line.match_indices("`log ") {
                    let t = line[r + 5..].split(|c: char| c == ' ' || c == '`').next().unwrap_or("");
                    if line[r + 5 + t.len()..].starts_with(" on") || line[r + 5 + t.len()..].starts_with(" off") { found.push(t); }
                }
                for p in found.into_iter().filter(|p| !p.starts_with('<')) {
                    let p = p.trim_matches(|c| c == '`' || c == ',');
                    assert!(parse(p).is_ok(), "{p:?}: {:?}", parse(p));
                    seen += 1;
                }
            }
        }
        assert!(seen >= 4, "found only {seen} patterns");
    }

    /// One file cannot take two modules: their writers would overwrite each other.
    #[test]
    fn a_file_takes_one_module() {
        let r = LogDevice.execute_command("log", &["net,heart", "file", "/nonexistent/x"], Box::new(std::io::sink()));
        assert!(r.unwrap_err().contains("one module"));
    }

    /// A monitor connection's log sink holds a handle on its socket. One left
    /// behind after the client hangs up is a file descriptor until a log line
    /// happens to fail on it, and with no module logging that is never: a
    /// harness that opens a connection per command runs out of descriptors.
    #[test]
    fn a_monitor_connection_takes_its_log_sink_away() {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        let dl = &DEVLOG;
        let monitor = Arc::new(crate::monitor::Monitor::new());
        monitor.clone().start_server("127.0.0.1:0".to_string());
        let addr = monitor.bound_addr().expect("bound");
        let before = dl.writers.lock().len();
        for _ in 0..20 {
            let mut s = TcpStream::connect(addr).unwrap();
            s.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
            s.write_all(b"quit\n").unwrap();
            let mut sink = Vec::new();
            s.read_to_end(&mut sink).expect("the monitor hung up");
        }
        monitor.shutdown();
        // Other tests' connections may come and go meanwhile; twenty left
        // behind cannot hide in that.
        let after = dl.writers.lock().len();
        assert!(after < before + 5, "{} sinks left by 20 connections", after - before);
    }
}
