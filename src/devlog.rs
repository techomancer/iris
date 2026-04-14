use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::net::TcpStream;
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use parking_lot::Mutex;
use crate::traits::Device;

// ── LogModule ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub enum LogModule {
    Net  = 0,
    Hpc3 = 1,
    Seeq = 2,
    Hal2 = 3,
    Mc   = 4,
    Rex3 = 5,
    Mips = 6,
    Ioc  = 7,
    Scsi = 8,
    Pdma = 9,
    Vino = 10,
    Dcb  = 11,
    Vc2  = 12,
    Cmap  = 13,
    Xmap  = 14,
    Bt445  = 15,
    Scc    = 16,
    Ps2    = 17,
    Rtc    = 18,
    Eeprom = 19,
}

impl LogModule {
    pub const COUNT: usize = 20;

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "net"  => Some(Self::Net),
            "hpc3" => Some(Self::Hpc3),
            "seeq" => Some(Self::Seeq),
            "hal2" => Some(Self::Hal2),
            "mc"   => Some(Self::Mc),
            "rex3" => Some(Self::Rex3),
            "mips" => Some(Self::Mips),
            "ioc"  => Some(Self::Ioc),
            "scsi" => Some(Self::Scsi),
            "pdma" => Some(Self::Pdma),
            "vino" => Some(Self::Vino),
            "dcb"  => Some(Self::Dcb),
            "vc2"  => Some(Self::Vc2),
            "cmap"  => Some(Self::Cmap),
            "xmap"  => Some(Self::Xmap),
            "bt445" => Some(Self::Bt445),
            "scc"    => Some(Self::Scc),
            "ps2"    => Some(Self::Ps2),
            "rtc"    => Some(Self::Rtc),
            "eeprom" => Some(Self::Eeprom),
            _        => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Net  => "net",
            Self::Hpc3 => "hpc3",
            Self::Seeq => "seeq",
            Self::Hal2 => "hal2",
            Self::Mc   => "mc",
            Self::Rex3 => "rex3",
            Self::Mips => "mips",
            Self::Ioc  => "ioc",
            Self::Scsi => "scsi",
            Self::Pdma => "pdma",
            Self::Vino => "vino",
            Self::Dcb  => "dcb",
            Self::Vc2  => "vc2",
            Self::Cmap  => "cmap",
            Self::Xmap  => "xmap",
            Self::Bt445  => "bt445",
            Self::Scc    => "scc",
            Self::Ps2    => "ps2",
            Self::Rtc    => "rtc",
            Self::Eeprom => "eeprom",
        }
    }

    pub fn all() -> &'static [LogModule] {
        &[
            Self::Net, Self::Hpc3, Self::Seeq, Self::Hal2, Self::Mc,
            Self::Rex3, Self::Mips, Self::Ioc, Self::Scsi, Self::Pdma,
            Self::Vino, Self::Dcb, Self::Vc2, Self::Cmap, Self::Xmap,
            Self::Bt445, Self::Scc, Self::Ps2, Self::Rtc, Self::Eeprom,
        ]
    }

    /// Describe a mask value in human-readable terms for this module.
    pub fn describe_mask(self, mask: u32) -> String {
        if mask == 0 { return "none".to_string(); }
        if mask == 0xFFFF_FFFF { return "all".to_string(); }
        match self {
            Self::Pdma => {
                let mut parts = Vec::new();
                if mask & 0x00FF != 0 { parts.push(format!("hal({:#04x})",  mask & 0x00FF)); }
                if mask & 0x0300 != 0 { parts.push(format!("scsi({:#04x})", mask & 0x0300)); }
                if mask & 0x0C00 != 0 { parts.push(format!("enet({:#04x})", mask & 0x0C00)); }
                let rest = mask & !0x0FFF;
                if rest != 0 { parts.push(format!("{:#010x}", rest)); }
                if parts.is_empty() { format!("{:#010x}", mask) } else { parts.join("+") }
            }
            Self::Mips => {
                let mut parts = Vec::new();
                if mask & 0x0001 != 0 { parts.push("insn"); }
                if mask & 0x0002 != 0 { parts.push("tlb"); }
                if mask & 0x0004 != 0 { parts.push("mem"); }
                let rest = mask & !0x0007;
                let mut s = parts.join("+");
                if rest != 0 { s.push_str(&format!("+{:#010x}", rest)); }
                if s.is_empty() { format!("{:#010x}", mask) } else { s }
            }
            _ => format!("{:#010x}", mask),
        }
    }

    /// Parse a category/mask string for this module.
    /// Returns None if the string is not recognized as a category for this module
    /// (caller should then try hex parse).
    pub fn parse_mask(self, s: &str) -> Option<u32> {
        match self {
            Self::Pdma => match s {
                "hal"  => Some(0x00FF),
                "scsi" => Some(0x0300),
                "enet" => Some(0x0C00),
                "on" | "all"    => Some(0xFFFF_FFFF),
                "off" | "none"  => Some(0x0000_0000),
                _ => u32::from_str_radix(s.trim_start_matches("0x"), 16).ok(),
            },
            Self::Mips => match s {
                "insn" => Some(0x0001),
                "tlb"  => Some(0x0002),
                "mem"  => Some(0x0004),
                "on" | "all"   => Some(0xFFFF_FFFF),
                "off" | "none" => Some(0x0000_0000),
                _ => u32::from_str_radix(s.trim_start_matches("0x"), 16).ok(),
            },
            _ => match s {
                "on" | "all"   => Some(0xFFFF_FFFF),
                "off" | "none" => Some(0x0000_0000),
                _ => u32::from_str_radix(s.trim_start_matches("0x"), 16).ok(),
            },
        }
    }
}

// ── ModuleLog ────────────────────────────────────────────────────────────────

struct ModuleLog {
    enabled:   AtomicBool,
    /// Per-module bitmask. Meaning is module-defined. 0 = nothing, !0 = everything by default.
    /// For Pdma: bit N = channel N enabled.
    mask:      AtomicU32,
    file_sink: Mutex<Option<BufWriter<File>>>,
    file_path: Mutex<Option<String>>,
}

impl ModuleLog {
    const fn new() -> Self {
        Self {
            enabled:   AtomicBool::new(false),
            mask:      AtomicU32::new(0xFFFF_FFFF),
            file_sink: Mutex::new(None),
            file_path: Mutex::new(None),
        }
    }
}

// ── DevLog ───────────────────────────────────────────────────────────────────

/// Generic write+send sink used by DevLog. Holds either a TCP monitor connection
/// or a host stream like stderr (for diagnostic mode).
pub type DevLogWriter = Arc<Mutex<dyn Write + Send>>;

pub struct DevLog {
    /// Fast-path gate: true if any module is enabled or any file sink is open.
    /// Checked before taking any lock. When false, dlog! is a two-load no-op.
    any_active: AtomicBool,
    /// All currently connected monitor clients (and any host sinks like stderr).
    /// Entries are pruned on write error.
    writers: Mutex<Vec<DevLogWriter>>,
    modules: [ModuleLog; LogModule::COUNT],
}

impl DevLog {
    fn new() -> Self {
        Self {
            any_active: AtomicBool::new(false),
            writers: Mutex::new(Vec::new()),
            modules: [
                ModuleLog::new(), ModuleLog::new(), ModuleLog::new(),
                ModuleLog::new(), ModuleLog::new(), ModuleLog::new(),
                ModuleLog::new(), ModuleLog::new(), ModuleLog::new(),
                ModuleLog::new(), ModuleLog::new(), ModuleLog::new(),
                ModuleLog::new(), ModuleLog::new(), ModuleLog::new(),
                ModuleLog::new(), ModuleLog::new(), // Bt445, Scc
                ModuleLog::new(), ModuleLog::new(), ModuleLog::new(), // Ps2, Rtc, Eeprom
            ],
        }
    }

    /// Register a new monitor connection. Called from monitor::handle_client at connect time.
    pub fn add_writer(&self, w: Arc<Mutex<BufWriter<TcpStream>>>) {
        self.writers.lock().push(w);
        // Writers alone don't activate log output — a module must be enabled too.
        // any_active is set when a module is enabled.
    }

    /// Register a generic Write+Send sink (e.g. stderr for diagnostic runs).
    pub fn add_sink(&self, sink: DevLogWriter) {
        self.writers.lock().push(sink);
    }

    /// Enable a module. Output will go to all connected monitor clients.
    pub fn enable(&self, m: LogModule) {
        self.modules[m as usize].enabled.store(true, Ordering::Relaxed);
        self.any_active.store(true, Ordering::Relaxed);
    }

    /// Disable a module. Also closes its file sink if open.
    pub fn disable(&self, m: LogModule) {
        self.modules[m as usize].enabled.store(false, Ordering::Relaxed);
        *self.modules[m as usize].file_sink.lock() = None;
        *self.modules[m as usize].file_path.lock() = None;
        self.recompute_any_active();
    }

    /// Set the bitmask for a module and enable it.
    pub fn set_mask(&self, m: LogModule, mask: u32) {
        self.modules[m as usize].mask.store(mask, Ordering::Relaxed);
        if mask != 0 {
            self.modules[m as usize].enabled.store(true, Ordering::Relaxed);
            self.any_active.store(true, Ordering::Relaxed);
        } else {
            self.modules[m as usize].enabled.store(false, Ordering::Relaxed);
            self.recompute_any_active();
        }
    }

    /// Open a file sink for a module and enable it.
    pub fn set_file(&self, m: LogModule, path: &str) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
        *self.modules[m as usize].file_sink.lock() = Some(BufWriter::new(file));
        *self.modules[m as usize].file_path.lock() = Some(path.to_string());
        self.modules[m as usize].enabled.store(true, Ordering::Relaxed);
        self.any_active.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Close a module's file sink (output reverts to monitor broadcast).
    pub fn clear_file(&self, m: LogModule) {
        *self.modules[m as usize].file_sink.lock() = None;
        *self.modules[m as usize].file_path.lock() = None;
        self.recompute_any_active();
    }

    /// Write a log message for a module. Called only when devlog_is_active() returned true.
    pub fn write(&self, m: LogModule, args: fmt::Arguments) {
        let ml = &self.modules[m as usize];
        let msg = fmt::format(args);

        // File sink takes priority — write there and return.
        {
            let mut sink = ml.file_sink.lock();
            if let Some(ref mut f) = *sink {
                let _ = writeln!(f, "[{}] {}", m.name(), msg);
                let _ = f.flush();
                return;
            }
        }

        // Broadcast to all connected monitor clients; prune dead connections.
        let mut writers = self.writers.lock();
        writers.retain(|w| {
            let mut guard = w.lock();
            write!(guard, "[{}] {}\n", m.name(), msg).is_ok() && guard.flush().is_ok()
        });
        if writers.is_empty() {
            self.recompute_any_active();
        }
    }

    fn recompute_any_active(&self) {
        let active = self.modules.iter().any(|ml| {
            ml.enabled.load(Ordering::Relaxed) || ml.file_sink.lock().is_some()
        });
        self.any_active.store(active, Ordering::Relaxed);
    }
}

// ── Global singleton ─────────────────────────────────────────────────────────

pub static DEVLOG: OnceLock<Arc<DevLog>> = OnceLock::new();

/// Call once at startup (in Machine::new). Returns the Arc for registering with monitor.
pub fn init_devlog() -> Arc<DevLog> {
    let dl = Arc::new(DevLog::new());
    let _ = DEVLOG.set(dl.clone());
    dl
}

/// Access the global DevLog. Panics if init_devlog() was not called.
pub fn devlog() -> &'static Arc<DevLog> {
    DEVLOG.get().expect("DevLog not initialized")
}

/// Fast-path enabled check used by dlog! macro. Two atomic loads, no allocation.
#[inline(always)]
pub fn devlog_is_active(m: LogModule) -> bool {
    let dl = match DEVLOG.get() { Some(d) => d, None => return false };
    dl.any_active.load(Ordering::Relaxed)
        && dl.modules[m as usize].enabled.load(Ordering::Relaxed)
}

/// Read the bitmask for a module. Used by per-channel/per-bit log gating.
#[inline(always)]
pub fn devlog_mask(m: LogModule) -> u32 {
    match DEVLOG.get() {
        Some(dl) => dl.modules[m as usize].mask.load(Ordering::Relaxed),
        None => 0,
    }
}

// ── Macros ───────────────────────────────────────────────────────────────────

/// Log to a module. Two atomic loads when idle — zero allocation, no lock.
/// Output goes to all connected monitor clients or the module's file sink.
#[macro_export]
macro_rules! dlog {
    ($mod:expr, $($arg:tt)*) => {
        if $crate::devlog::devlog_is_active($mod) {
            $crate::devlog::devlog().write($mod, format_args!($($arg)*));
        }
    };
}

/// Hot-path log. Compiles to nothing in non-developer (release) builds.
#[macro_export]
macro_rules! dlog_dev {
    ($mod:expr, $($arg:tt)*) => {
        {
            #[cfg(feature = "developer")]
            $crate::dlog!($mod, $($arg)*);
        }
    };
}

// ── Device impl for monitor `log` command ────────────────────────────────────

impl Device for DevLog {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { false }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![(
            "log".to_string(),
            "[DEV] log <module|all> <on|off>  |  log <module> mask <cat|hex>  |  log <module> file <path|off>  |  log status\n\
             \x20             modules: net hpc3 seeq hal2 mc rex3 mips ioc scsi pdma vino dcb vc2 cmap xmap bt445 scc ps2 rtc eeprom\n\
             \x20             pdma mask categories: hal enet scsi on/all off/none <hex>\n\
             \x20             mips mask categories: insn tlb mem on/all off/none <hex>\n\
             \x20             [DEV] = requires --features developer build to produce output".to_string(),
        )]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "log" { return Err(format!("Unknown command: {}", cmd)); }
        let arg0 = args.get(0).copied().unwrap_or("");

        if arg0 == "status" {
            writeln!(writer, "{:<8} {:>7}  {:<24}  file", "module", "enabled", "mask").map_err(|e| e.to_string())?;
            writeln!(writer, "{}", "-".repeat(60)).map_err(|e| e.to_string())?;
            for &m in LogModule::all() {
                let en   = self.modules[m as usize].enabled.load(Ordering::Relaxed);
                let mask = self.modules[m as usize].mask.load(Ordering::Relaxed);
                let path = self.modules[m as usize].file_path.lock().clone().unwrap_or_default();
                let mask_str = if en { m.describe_mask(mask) } else { "-".to_string() };
                writeln!(writer, "{:<8} {:>7}  {:<24}  {}",
                    m.name(), if en { "on" } else { "off" }, mask_str, path)
                    .map_err(|e| e.to_string())?;
            }
            let wcount = self.writers.lock().len();
            writeln!(writer, "\nMonitor clients: {}", wcount).map_err(|e| e.to_string())?;
            return Ok(());
        }

        let arg1 = args.get(1).copied().unwrap_or("");
        let arg2 = args.get(2).copied().unwrap_or("");

        let modules: Vec<LogModule> = if arg0 == "all" {
            LogModule::all().to_vec()
        } else {
            match LogModule::from_str(arg0) {
                Some(m) => vec![m],
                None => return Err(format!("Unknown module '{}'. Try: net hpc3 seeq hal2 mc rex3 mips ioc scsi pdma vino dcb vc2 cmap xmap bt445 scc ps2 rtc eeprom all", arg0)),
            }
        };

        match arg1 {
            "on" => {
                for m in &modules { self.enable(*m); }
                writeln!(writer, "log {} on", arg0).map_err(|e| e.to_string())?;
            }
            "off" => {
                for m in &modules { self.disable(*m); }
                writeln!(writer, "log {} off", arg0).map_err(|e| e.to_string())?;
            }
            "mask" => {
                if arg2.is_empty() {
                    return Err("Usage: log <module> mask <category|hex>".to_string());
                }
                for m in &modules {
                    let mask = m.parse_mask(arg2)
                        .ok_or_else(|| format!("Unknown mask '{}' for module {}", arg2, m.name()))?;
                    self.set_mask(*m, mask);
                }
                writeln!(writer, "log {} mask {}", arg0, arg2).map_err(|e| e.to_string())?;
            }
            "file" => {
                if arg2 == "off" || arg2.is_empty() {
                    for m in &modules { self.clear_file(*m); }
                    writeln!(writer, "log {} file closed", arg0).map_err(|e| e.to_string())?;
                } else {
                    for m in &modules { self.set_file(*m, arg2)?; }
                    writeln!(writer, "log {} -> {}", arg0, arg2).map_err(|e| e.to_string())?;
                }
            }
            _ => return Err("Usage: log <module|all> <on|off|mask <cat|hex>|file <path>>  |  log status".to_string()),
        }
        Ok(())
    }
}
