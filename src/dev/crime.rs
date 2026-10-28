//! CRIME's CPU interface on the O2: its interrupt controller and memory
//! bank control (the rendering engine is `o2gfx`).

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use crate::snapshot::{get_field, hex_u64, load_u64_slice, toml_u64, u64_slice_to_toml};
use crate::traits::{BusDevice, BusRead32, BusRead64, Device, Resettable, Saveable, BUS_OK};
use crate::ip32::*;

// ── CRIME registers ─────────────────────────────────────────────────────────
// Offsets follow the names NetBSD's crimereg.h uses, so the two can be read
// side by side. CRIME is a 64-bit register file: the PROM reaches it with
// `sd`/`ld` through uncached KSEG1 (0xb4000000), never 32-bit accesses.
pub mod crime_reg {
    pub const REVISION: u32 = 0x0000;
    pub const CONTROL: u32 = 0x0008;
    /// CONTROL bit 11, cold reset: software sets it to reset the system as
    /// at power-on, and the hardware clears it (CRIME 1.5 spec, table 5-9).
    /// NetBSD reboots this way, where IRIX and OpenBSD go back through the
    /// PROM.
    pub const CONTROL_COLD_RESET: u64 = 1 << 11;
    pub const INT_STAT: u32 = 0x0010;
    pub const INT_MASK: u32 = 0x0018;
    pub const INT_SOFT: u32 = 0x0020;
    pub const WATCHDOG: u32 = 0x0030;
    pub const TIME: u32 = 0x0038;
    /// Pending interrupts, and which of them are allowed through. NetBSD
    /// attaches CRIME as `platform.intr0`, so anything getting past the mask
    /// arrives at the CPU as IP2.
    pub const INTSTAT: u32 = 0x0010;
    pub const INTMASK: u32 = 0x0018;
    /// Software interrupts: bits 31:28 and 24:0 can be set from here.
    pub const SOFTINT: u32 = 0x0020;
    /// The hardware sources; writing 0 to a latched bit clears it.
    pub const HARDINT: u32 = 0x0028;
    pub const CPU_ERROR_ADDR: u32 = 0x0040;
    pub const CPU_ERROR_STAT: u32 = 0x0048;
    pub const CPU_ERROR_ENA: u32 = 0x0050;

    pub const MEM_CONTROL: u32 = 0x0200;
    pub const MEM_BANK_CTRL0: u32 = 0x0208;
    pub const MEM_REFRESH_CNTR: u32 = 0x0248;
    pub const MEM_ERROR_STAT: u32 = 0x0250;
    pub const MEM_ERROR_ADDR: u32 = 0x0258;
    pub const MEM_ERROR_ECC_SYN: u32 = 0x0260;
    pub const MEM_ERROR_ECC_CHK: u32 = 0x0268;
    pub const MEM_ERROR_ECC_REPL: u32 = 0x0270;
}

/// CRIME interrupt bits that latch (CRIME 1.5 spec): 30, 24:22, 19:16.
pub const CRIME_LATCHED_BITS: u64 = 0x41cf_0000;

/// CRIME revision as reported to the guest: CRIME 1.1 (ID 0xa1), the value
/// the PROM requires before it adds the graphics component. Every O2 has its
/// GBE, with or without a keyboard and monitor, so this is the only revision
/// modelled; a headless machine differs only in its console.
pub const CRIME_REV: u64 = 0x0000_0000_0000_00a1;

/// CRIME's interrupt controller: what the level-type inputs drive (INTSTAT's
/// hardware bits), INTMASK, SOFTINT, and the edge-triggered sources CRIME
/// holds until software writes a 0 to the bit in HARDINT: GBE's four timing
/// lines (16..19), the rendering engine's empty/full/idle edges (22..24) and
/// bit 30.
#[derive(Default)]
struct IntState {
    levels: u64,
    mask: u64,
    soft: u64,
    latched: u64,
}

impl IntState {
    fn intstat(&self) -> u64 {
        self.levels | self.soft | self.latched
    }

    fn pending(&self) -> bool {
        self.intstat() & self.mask != 0
    }
}

/// Words of `Crime::touched_bits`: a bit per 64-bit register and direction.
const TOUCHED_WORDS: usize = (CRIME_SIZE / 8 * 2 / 64) as usize;

/// Minimal CRIME: a 64-bit register file that remembers what it is told, with
/// the handful of reads the PROM cannot be fobbed off with zero for.
pub struct Crime {
    regs: Mutex<Vec<u64>>,
    /// Every distinct offset the guest has touched, in first-touch order, with
    /// whether it was read or written. This is the actual product of the first
    /// milestone: the list of CRIME registers POST depends on. `touched_bits`
    /// has a bit per register and direction, so an access already recorded
    /// costs one relaxed load and the list is locked only on a first touch.
    /// Not kept in a `lightning` build.
    trace: Mutex<Vec<(u32, bool)>>,
    touched_bits: [AtomicU64; TOUCHED_WORDS],
    /// The interrupt state, apart from `regs`. Every change to it, an input
    /// driven or a mask written, recomputes the output and hands it to IP2
    /// under this lock, so two threads cannot deliver it out of order.
    /// `regs` keeps no copy of INTSTAT, INTMASK or SOFTINT.
    int: Mutex<IntState>,
    /// The guest's last accesses to the interrupt registers (status, mask,
    /// HARDINT, SOFTINT): `(write, offset, value)`, newest last, for the
    /// control port's `crimelog` (chasing interrupt storms).
    /// Off until the monitor's `crime irqlog on`, and not kept in a
    /// `lightning` build: IRIX reads INTSTAT and writes INTMASK on every
    /// interrupt and every spl change.
    irqlog: Mutex<std::collections::VecDeque<(bool, u32, u64)>>,
    irqlog_on: AtomicBool,
    revision: std::sync::atomic::AtomicU64,
    /// The clock CRIME's free-running counter is derived from, so the PROM's
    /// delays take the time they ask for rather than a number of reads.
    ust: std::sync::Arc<MaceUst>,
    /// Subtracted from the counter, so a write sets what the next read sees.
    time_base: AtomicU64,
    /// CRIME's one output, to the CPU's IP2 (`connect_output`). Driven under
    /// `int`: computed and handed on as one step, a "low" and a "high" from
    /// two threads cannot cross and leave IP2 up with nothing behind it (the
    /// O2's input stall, when IRIX re-entered its CRIME dispatcher forever).
    out: std::sync::OnceLock<crate::irq::IrqLine>,
    /// Software asked for a cold reset (CONTROL bit 11); the machine takes
    /// it ([`Crime::take_reset_request`]).
    reset_request: std::sync::atomic::AtomicBool,
}

/// CRIME's counter runs at 66.67 MHz. The PROM's own delay routine computes
/// `66 * microseconds` before waiting on it, which is where this number comes
/// from — it is the firmware's arithmetic, not a datasheet's.
pub const CRIME_TICKS_PER_US: u64 = 66;

impl Crime {
    pub fn new(ust: std::sync::Arc<MaceUst>) -> Self {
        Self {
            regs: Mutex::new(vec![0u64; (CRIME_SIZE / 8) as usize]),
            trace: Mutex::new(Vec::new()),
            touched_bits: std::array::from_fn(|_| AtomicU64::new(0)),
            int: Mutex::new(IntState::default()),
            irqlog: Mutex::new(std::collections::VecDeque::new()),
            irqlog_on: AtomicBool::new(false),
            revision: std::sync::atomic::AtomicU64::new(CRIME_REV),
            ust,
            time_base: AtomicU64::new(0),
            out: std::sync::OnceLock::new(),
            reset_request: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn with_revision(ust: std::sync::Arc<MaceUst>, revision: u64) -> Self {
        let c = Self::new(ust);
        c.set_revision(revision);
        c
    }

    /// Override the revision CRIME reports; snapshots restore it.
    pub fn set_revision(&self, rev: u64) {
        self.revision.store(rev, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn revision(&self) -> u64 {
        self.revision.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Where CRIME's output goes: the CPU's IP2. Without one (the bring-up
    /// harness, which polls `interrupting`) nothing is pushed.
    pub fn connect_output(&self, line: crate::irq::IrqLine) {
        let _ = self.out.set(line);
    }

    pub fn has_output(&self) -> bool {
        self.out.get().is_some()
    }

    /// Hand IP2 the output as `s` stands, with `int` held. Returns it.
    fn drive(&self, s: &IntState) -> bool {
        let level = s.pending();
        if let Some(out) = self.out.get() {
            out.set(level);
        }
        level
    }

    /// Hand IP2 the output as it stands (after a snapshot restores the
    /// state, or a CPU reset re-wires IP2). Returns it.
    pub fn update_output(&self) -> bool {
        self.drive(&self.int.lock().unwrap())
    }

    /// Change the interrupt state and drive IP2 from the result.
    fn change(&self, f: impl FnOnce(&mut IntState)) {
        let mut s = self.int.lock().unwrap();
        f(&mut s);
        self.drive(&s);
    }

    /// INTSTAT: the level sources, the latched ones and the software ones.
    pub(crate) fn intstat(&self) -> u64 {
        self.int.lock().unwrap().intstat()
    }

    /// True when something is pending that the mask lets through.
    pub fn interrupting(&self) -> bool {
        self.int.lock().unwrap().pending()
    }

    /// The register file as the guest sees it, the interrupt registers
    /// included: for the monitor and snapshots.
    fn regs_view(&self) -> Vec<u64> {
        let mut r = self.regs.lock().unwrap().clone();
        let s = self.int.lock().unwrap();
        r[(crime_reg::INTSTAT / 8) as usize] = s.levels;
        r[(crime_reg::INTMASK / 8) as usize] = s.mask;
        r[(crime_reg::SOFTINT / 8) as usize] = s.soft;
        r
    }

    /// The raw counter, before the offset a write installs.
    fn raw_time(&self) -> u64 {
        self.ust.guest_us().wrapping_mul(CRIME_TICKS_PER_US)
    }

    #[inline]
    fn note(&self, off: u32, write: bool) {
        #[cfg(not(feature = "lightning"))]
        {
            let bit = (off / 8) * 2 + write as u32;
            let (word, mask) = (&self.touched_bits[(bit / 64) as usize], 1u64 << (bit % 64));
            if word.load(Ordering::Relaxed) & mask == 0 && word.fetch_or(mask, Ordering::Relaxed) & mask == 0 {
                self.trace.lock().unwrap().push((off, write));
            }
        }
        #[cfg(feature = "lightning")]
        let _ = (off, write);
    }

    /// Registers touched so far, in first-touch order: `(offset, was_write)`.
    pub fn touched(&self) -> Vec<(u32, bool)> {
        self.trace.lock().unwrap().clone()
    }

    #[inline]
    fn log_irq(&self, write: bool, off: u32, val: u64) {
        #[cfg(not(feature = "lightning"))]
        if self.irqlog_on.load(Ordering::Relaxed)
            && matches!(off, crime_reg::INTSTAT | crime_reg::INTMASK | crime_reg::HARDINT | crime_reg::SOFTINT)
        {
            let mut l = self.irqlog.lock().unwrap();
            if l.len() >= 8192 {
                l.pop_front();
            }
            l.push_back((write, off, val));
        }
        #[cfg(feature = "lightning")]
        let _ = (write, off, val);
    }

    /// Start or stop recording interrupt-register accesses. Starting clears
    /// what an earlier recording left. A `lightning` build records nothing.
    pub fn set_irqlog(&self, on: bool) -> Result<(), String> {
        if cfg!(feature = "lightning") {
            return Err("crime: a lightning build does not record the interrupt log".into());
        }
        if on {
            self.irqlog.lock().unwrap().clear();
        }
        self.irqlog_on.store(on, Ordering::Relaxed);
        Ok(())
    }

    /// The interrupt-register log, one access per line, oldest first.
    pub fn irqlog_text(&self) -> String {
        self.irqlog.lock().unwrap().iter().map(|&(w, off, v)| format!("{} {:#04x} {:#018x}\n", if w { 'W' } else { 'R' }, off, v)).collect()
    }

    pub(crate) fn load(&self, off: u32) -> u64 {
        let v = self.load_inner(off);
        self.log_irq(false, off, v);
        v
    }

    /// A register's value, as a read would return it, without recording
    /// the access.
    pub(crate) fn load_inner(&self, off: u32) -> u64 {
        match off {
            // Identity. Returning zero here is the one thing guaranteed to make
            // a PROM decide it is running on hardware it does not recognise.
            crime_reg::REVISION => self.revision(),
            crime_reg::INTSTAT => self.intstat(),
            crime_reg::INTMASK => self.int.lock().unwrap().mask,
            crime_reg::SOFTINT => self.int.lock().unwrap().soft,
            crime_reg::HARDINT => {
                let s = self.int.lock().unwrap();
                s.levels | s.latched
            }
            // A free-running microsecond-ish counter. POST loops on this to time
            // out, so a constant would hang it forever.
            crime_reg::TIME => {
                let base = self.time_base.load(Ordering::Relaxed);
                self.raw_time().wrapping_sub(base) & 0x0000_ffff_ffff_ffff
            }
            _ => self.regs.lock().unwrap()[(off / 8) as usize],
        }
    }

    fn store(&self, off: u32, val: u64) {
        self.log_irq(true, off, val);
        self.store_inner(off, val);
    }

    /// Whether software has asked for a cold reset since the last call.
    pub fn take_reset_request(&self) -> bool {
        self.reset_request.swap(false, std::sync::atomic::Ordering::AcqRel)
    }

    fn store_inner(&self, off: u32, val: u64) {
        match off {
            crime_reg::CONTROL if val & crime_reg::CONTROL_COLD_RESET != 0 => {
                self.regs.lock().unwrap()[(off / 8) as usize] = val & !crime_reg::CONTROL_COLD_RESET;
                self.reset_request.store(true, std::sync::atomic::Ordering::Release);
                return;
            }
            // Read-only: the sources set it, not the CPU.
            crime_reg::INTSTAT => return,
            crime_reg::HARDINT => return self.change(|s| s.latched &= val | !CRIME_LATCHED_BITS),
            crime_reg::SOFTINT => return self.change(|s| s.soft = val & 0xf1ff_ffff),
            crime_reg::INTMASK => return self.change(|s| s.mask = val),
            _ => {}
        }
        if off == crime_reg::TIME {
            // The PROM zeroes the counter and then waits for it to reach a
            // deadline, so a write has to set where it counts from.
            self.time_base.store(self.raw_time().wrapping_sub(val), Ordering::Relaxed);
        }
        self.regs.lock().unwrap()[(off / 8) as usize] = val;
    }
}

/// CRIME collects every interrupt on the O2: MACE's (0..15, the ISA
/// peripherals among them through MACE's own status), GBE's (16..19) and
/// the rendering engine's. The edge-type inputs latch until software clears
/// them in HARDINT; the rest follow their level.
impl crate::irq::InterruptController for Crime {
    fn set_line(&self, line: u32, level: bool) {
        let bit = 1u64 << line;
        self.change(|s| {
            if bit & CRIME_LATCHED_BITS != 0 {
                if level {
                    s.latched |= bit;
                }
            } else if level {
                s.levels |= bit;
            } else {
                s.levels &= !bit;
            }
        });
    }
}

impl Device for Crime {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("crime".to_string(), "CRIME: crime status | crime regs | crime irqlog [on|off|n] | crime touched".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "crime" {
            return Err("Command not found".to_string());
        }
        let regs = self.regs_view();
        let reg = |off: u32| regs[(off / 8) as usize];
        match args.first().copied().unwrap_or("status") {
            "status" => {
                let (stat, mask) = (self.intstat(), reg(crime_reg::INTMASK));
                writeln!(w, "CRIME revision {:#x}", self.revision()).ok();
                writeln!(w, "  interrupts: status {stat:#010x} mask {mask:#010x} pending {:#010x} latched {:#010x} soft {:#010x}",
                    stat & mask, self.int.lock().unwrap().latched, reg(crime_reg::SOFTINT)).ok();
                writeln!(w, "  control {:#x} time {:#x}", reg(crime_reg::CONTROL), self.load_inner(crime_reg::TIME)).ok();
                for i in 0..8u32 {
                    writeln!(w, "  bank {i}: ctrl {:#06x}", reg(crime_reg::MEM_BANK_CTRL0 + 8 * i)).ok();
                }
            }
            "regs" => {
                for (i, &v) in regs.iter().enumerate() {
                    if v != 0 {
                        writeln!(w, "  {:#06x}: {v:#018x}", i * 8).ok();
                    }
                }
            }
            "irqlog" if matches!(args.get(1).copied(), Some("on" | "off")) => {
                let on = args[1] == "on";
                self.set_irqlog(on)?;
                writeln!(w, "  interrupt-register log {}", if on { "recording" } else { "stopped" }).ok();
            }
            "irqlog" => {
                if !self.irqlog_on.load(Ordering::Relaxed) {
                    writeln!(w, "  (not recording; `crime irqlog on` starts)").ok();
                }
                let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(32);
                let log = self.irqlog_text();
                let lines: Vec<&str> = log.lines().collect();
                for l in &lines[lines.len().saturating_sub(n)..] {
                    writeln!(w, "  {l}").ok();
                }
            }
            "touched" => {
                for (off, write) in self.touched() {
                    writeln!(w, "  {off:#06x} {}", if write { "W" } else { "R" }).ok();
                }
            }
            _ => return Err("Usage: crime status | regs | irqlog [on|off|n] | touched".to_string()),
        }
        Ok(())
    }
}

impl Resettable for Crime {
    /// Registers, latched interrupts and the counter's offset back to zero;
    /// the revision is the chip's and stays.
    fn power_on(&self) {
        self.regs.lock().unwrap().fill(0);
        self.change(|s| *s = IntState::default());
        self.time_base.store(0, Ordering::Relaxed);
        self.irqlog.lock().unwrap().clear();
    }
}

/// The registers, the latched edges, the revision and the counter's offset
/// (against the UST, which restores its own time); loading hands the output
/// to IP2 again.
impl Saveable for Crime {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("regs".into(), u64_slice_to_toml(&self.regs_view()));
        t.insert("latched".into(), hex_u64(self.int.lock().unwrap().latched));
        t.insert("revision".into(), hex_u64(self.revision()));
        t.insert("time_base".into(), hex_u64(self.time_base.load(Ordering::Relaxed)));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut int = IntState { latched: get_field(v, "latched").and_then(toml_u64).unwrap_or(0), ..Default::default() };
        if let Some(r) = get_field(v, "regs") {
            let mut regs = self.regs.lock().unwrap();
            load_u64_slice(r, &mut regs);
            let mut take = |off: u32| std::mem::take(&mut regs[(off / 8) as usize]);
            int.levels = take(crime_reg::INTSTAT);
            int.mask = take(crime_reg::INTMASK);
            int.soft = take(crime_reg::SOFTINT);
        }
        *self.int.lock().unwrap() = int;
        if let Some(rev) = get_field(v, "revision").and_then(toml_u64) {
            self.set_revision(rev);
        }
        self.time_base.store(get_field(v, "time_base").and_then(toml_u64).unwrap_or(0), Ordering::Relaxed);
        self.update_output();
        Ok(())
    }
}

impl BusDevice for Crime {
    fn read64(&self, addr: u64) -> BusRead64 {
        let off = (addr as u32) & (CRIME_SIZE - 1) & !7;
        self.note(off, false);
        BusRead64::ok(self.load(off))
    }

    fn write64(&self, addr: u64, val: u64) -> u32 {
        let off = (addr as u32) & (CRIME_SIZE - 1) & !7;
        self.note(off, true);
        self.store(off, val);
        BUS_OK
    }

    // CRIME is a 64-bit register file. The PROM only ever uses sd/ld, but a
    // 32-bit access should read the correct half rather than fail, so a stray
    // one shows up in the trace instead of as a bus error we would then have to
    // chase.
    fn read32(&self, addr: u64) -> BusRead32 {
        let q = self.read64(addr & !7).data;
        let hi = (addr & 4) == 0;
        BusRead32::ok(if hi { (q >> 32) as u32 } else { q as u32 })
    }

    fn write32(&self, addr: u64, val: u32) -> u32 {
        let off = (addr as u32) & (CRIME_SIZE - 1) & !7;
        let cur = self.load_inner(off);
        let merged = if (addr & 4) == 0 {
            (cur & 0x0000_0000_ffff_ffff) | ((val as u64) << 32)
        } else {
            (cur & 0xffff_ffff_0000_0000) | val as u64
        };
        self.note(off, true);
        self.store(off, merged);
        BUS_OK
    }
}

/// CRIME as the bus sees it: its registers at 0x14000000 and its rendering
/// engine at 0x15000000, one device that decodes which an access is for.
/// A write to a memory bank control moves the bank in `ram` at once, since
/// CRIME is the memory controller too.
pub struct CrimeSpace {
    pub regs: std::sync::Arc<Crime>,
    pub re: crate::dev::o2gfx::CrimeRe,
    ram: std::sync::Arc<crate::dev::crime_mem::O2Memory>,
}

impl CrimeSpace {
    pub fn new(regs: std::sync::Arc<Crime>, ram: std::sync::Arc<crate::dev::crime_mem::O2Memory>) -> Self {
        Self { re: crate::dev::o2gfx::CrimeRe::new(ram.clone()), regs, ram }
    }

    /// Whether CRIME answers `addr`.
    pub fn contains(addr: u32) -> bool {
        Self::in_regs(addr) || Self::in_re(addr)
    }

    pub fn in_regs(addr: u32) -> bool {
        (CRIME_BASE..CRIME_BASE + CRIME_SIZE).contains(&addr)
    }

    pub fn in_re(addr: u32) -> bool {
        (CRIME_RE_BASE..CRIME_RE_BASE + CRIME_RE_SIZE).contains(&addr)
    }

    fn regs_wrote(&self, addr: u32) {
        let off = (addr - CRIME_BASE) & !7;
        if (crime_reg::MEM_BANK_CTRL0..crime_reg::MEM_BANK_CTRL0 + 8 * 8).contains(&off) {
            let v = self.regs.load_inner(off);
            self.ram.set_ctrl(((off - crime_reg::MEM_BANK_CTRL0) / 8) as usize, v);
        }
    }
}

/// CRIME's registers and the rendering engine. A load sets the memory
/// banks again from the restored bank controls, so RAM appears where the
/// saved machine had it.
impl Saveable for CrimeSpace {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("crime".into(), self.regs.save_state());
        t.insert("re".into(), self.re.save_state());
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        if let Some(c) = get_field(v, "crime") {
            self.regs.load_state(c)?;
        }
        for bank in 0..8 {
            self.regs_wrote(CRIME_BASE + crime_reg::MEM_BANK_CTRL0 + 8 * bank);
        }
        if let Some(re) = get_field(v, "re") {
            self.re.load_state(re)?;
        }
        Ok(())
    }
}

// Narrow accesses reach both as the 32-bit word they fall in: neither has a
// byte-wide register, and the engine's busy status passes through.
/// CRIME's registers and its rendering engine are smaller than a slot of
/// the bus's decode table, so the space takes whole slots and answers the
/// rest as the bus does where nothing is: reads find zero, writes fail.
impl BusDevice for CrimeSpace {
    fn read8(&self, addr: u64) -> crate::traits::BusRead8 {
        let r = self.read32(addr & !3);
        crate::traits::BusRead8 { status: r.status, data: r.data as u8 }
    }
    fn read16(&self, addr: u64) -> crate::traits::BusRead16 {
        let r = self.read32(addr & !3);
        crate::traits::BusRead16 { status: r.status, data: r.data as u16 }
    }
    fn read32(&self, addr: u64) -> BusRead32 {
        if Self::in_re(addr as u32) { self.re.read32(addr) }
        else if Self::in_regs(addr as u32) { self.regs.read32(addr) }
        else { BusRead32::ok(0) }
    }
    fn read64(&self, addr: u64) -> BusRead64 {
        if Self::in_re(addr as u32) { self.re.read64(addr) }
        else if Self::in_regs(addr as u32) { self.regs.read64(addr) }
        else { BusRead64::ok(0) }
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        self.write32(addr & !3, val as u32)
    }
    fn write16(&self, addr: u64, val: u16) -> u32 {
        self.write32(addr & !3, val as u32)
    }
    fn write32(&self, addr: u64, val: u32) -> u32 {
        if Self::in_re(addr as u32) {
            return self.re.write32(addr, val);
        }
        if !Self::in_regs(addr as u32) {
            return crate::traits::BUS_ERR;
        }
        let r = self.regs.write32(addr, val);
        self.regs_wrote(addr as u32);
        r
    }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        if Self::in_re(addr as u32) {
            return self.re.write64(addr, val);
        }
        if !Self::in_regs(addr as u32) {
            return crate::traits::BUS_ERR;
        }
        let r = self.regs.write64(addr, val);
        self.regs_wrote(addr as u32);
        r
    }
}
