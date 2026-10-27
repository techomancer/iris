//! CRIME's CPU interface on the O2: its interrupt controller and memory
//! bank control (the rendering engine is `o2gfx`).

use std::io::Write;
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

/// Minimal CRIME: a 64-bit register file that remembers what it is told, with
/// the handful of reads the PROM cannot be fobbed off with zero for.
pub struct Crime {
    regs: Mutex<Vec<u64>>,
    /// Every distinct offset the guest has touched, in first-touch order, with
    /// whether it was read or written. This is the actual product of the first
    /// milestone: the list of CRIME registers POST depends on.
    trace: Mutex<Vec<(u32, bool)>>,
    /// Values written, in order, as `(offset, value)` — the bank programming
    /// is only meaningful if you can see what it was told.
    writes: Mutex<Vec<(u32, u64)>>,
    /// Edge-triggered sources CRIME holds until software writes a 0 to the
    /// bit in HARDINT: GBE's four timing lines (16..19), the rendering
    /// engine's empty/full/idle edges (22..24) and bit 30.
    latched: Mutex<u64>,
    /// The guest's last accesses to the interrupt registers (status, mask,
    /// HARDINT, SOFTINT): `(write, offset, value)`, newest last, for the
    /// control port's `crimelog` (chasing interrupt storms).
    irqlog: Mutex<std::collections::VecDeque<(bool, u32, u64)>>,
    revision: std::sync::atomic::AtomicU64,
    /// The clock CRIME's free-running counter is derived from, so the PROM's
    /// delays take the time they ask for rather than a number of reads.
    ust: std::sync::Arc<MaceUst>,
    /// Subtracted from the counter, so a write sets what the next read sees.
    time_base: Mutex<u64>,
    /// CRIME's one output, to the CPU's IP2 (`connect_output`), the level
    /// last handed to it, and the lock that makes computing and handing on
    /// one step. Unlocked, one thread could record "low" while another
    /// raised IP2, after which every update saw no change and IP2 stayed up
    /// with nothing behind it: IRIX re-entered its CRIME dispatcher forever
    /// and the clock interrupt starved (the O2's input stall).
    out: std::sync::OnceLock<crate::irq::IrqLine>,
    out_level: std::sync::atomic::AtomicBool,
    out_lock: Mutex<()>,
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
            writes: Mutex::new(Vec::new()),
            latched: Mutex::new(0),
            irqlog: Mutex::new(std::collections::VecDeque::new()),
            revision: std::sync::atomic::AtomicU64::new(CRIME_REV),
            ust,
            time_base: Mutex::new(0),
            out: std::sync::OnceLock::new(),
            out_level: std::sync::atomic::AtomicBool::new(false),
            out_lock: Mutex::new(()),
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

    /// Hand the output on if it changed. Whoever changes CRIME's state calls
    /// this after the change, so a caller that finds no change can leave it
    /// to them; a change is computed and handed on under the lock.
    pub fn update_output(&self) -> bool {
        use std::sync::atomic::Ordering;
        let Some(out) = self.out.get() else { return self.interrupting() };
        if self.interrupting() == self.out_level.load(Ordering::Acquire) {
            return self.out_level.load(Ordering::Acquire);
        }
        let _held = self.out_lock.lock().unwrap();
        let level = self.interrupting();
        if self.out_level.swap(level, Ordering::AcqRel) != level {
            out.set(level);
        }
        level
    }

    /// Raise or lower one of CRIME's interrupt inputs.
    pub fn set_int(&self, bit: u32, on: bool) {
        let mut r = self.regs.lock().unwrap();
        let i = (crime_reg::INTSTAT / 8) as usize;
        if on {
            r[i] |= 1u64 << bit;
        } else {
            r[i] &= !(1u64 << bit);
        }
    }

    /// Latch an edge-triggered source until software clears it in HARDINT.
    pub fn latch_int(&self, bit: u32) {
        *self.latched.lock().unwrap() |= (1u64 << bit) & CRIME_LATCHED_BITS;
    }

    /// INTSTAT: the level sources, the latched ones and the software ones.
    pub(crate) fn intstat(&self) -> u64 {
        let r = self.regs.lock().unwrap();
        r[(crime_reg::INTSTAT / 8) as usize] | r[(crime_reg::SOFTINT / 8) as usize] | *self.latched.lock().unwrap()
    }

    /// True when something is pending that the mask lets through.
    pub fn interrupting(&self) -> bool {
        let mask = self.regs.lock().unwrap()[(crime_reg::INTMASK / 8) as usize];
        self.intstat() & mask != 0
    }

    /// The raw counter, before the offset a write installs.
    fn raw_time(&self) -> u64 {
        self.ust.guest_us().wrapping_mul(CRIME_TICKS_PER_US)
    }

    fn note(&self, off: u32, write: bool) {
        let mut t = self.trace.lock().unwrap();
        if !t.iter().any(|&(o, w)| o == off && w == write) {
            t.push((off, write));
        }
    }

    /// Registers touched so far, in first-touch order: `(offset, was_write)`.
    pub fn touched(&self) -> Vec<(u32, bool)> {
        self.trace.lock().unwrap().clone()
    }

    fn log_irq(&self, write: bool, off: u32, val: u64) {
        if matches!(off, crime_reg::INTSTAT | crime_reg::INTMASK | crime_reg::HARDINT | crime_reg::SOFTINT) {
            let mut l = self.irqlog.lock().unwrap();
            if l.len() >= 8192 {
                l.pop_front();
            }
            l.push_back((write, off, val));
        }
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

    fn load_inner(&self, off: u32) -> u64 {
        match off {
            // Identity. Returning zero here is the one thing guaranteed to make
            // a PROM decide it is running on hardware it does not recognise.
            crime_reg::REVISION => self.revision(),
            crime_reg::INTSTAT => self.intstat(),
            crime_reg::HARDINT => self.regs.lock().unwrap()[(crime_reg::INTSTAT / 8) as usize] | *self.latched.lock().unwrap(),
            // A free-running microsecond-ish counter. POST loops on this to time
            // out, so a constant would hang it forever.
            crime_reg::TIME => {
                let base = *self.time_base.lock().unwrap();
                self.raw_time().wrapping_sub(base) & 0x0000_ffff_ffff_ffff
            }
            _ => self.regs.lock().unwrap()[(off / 8) as usize],
        }
    }

    fn store(&self, off: u32, val: u64) {
        self.log_irq(true, off, val);
        self.store_inner(off, val);
        if matches!(off, crime_reg::INTMASK | crime_reg::SOFTINT | crime_reg::HARDINT) {
            self.update_output();
        }
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
            crime_reg::HARDINT => {
                let mut l = self.latched.lock().unwrap();
                *l &= val | !CRIME_LATCHED_BITS;
                return;
            }
            crime_reg::SOFTINT => {
                self.regs.lock().unwrap()[(off / 8) as usize] = val & 0xf1ff_ffff;
                return;
            }
            _ => {}
        }
        if off == crime_reg::TIME {
            // The PROM zeroes the counter and then waits for it to reach a
            // deadline, so a write has to set where it counts from.
            *self.time_base.lock().unwrap() = self.raw_time().wrapping_sub(val);
        }
        self.regs.lock().unwrap()[(off / 8) as usize] = val;
        let mut w = self.writes.lock().unwrap();
        if w.len() < 256 {
            w.push((off, val));
        }
    }

    /// Values written, in order.
    pub fn written(&self) -> Vec<(u32, u64)> {
        self.writes.lock().unwrap().clone()
    }
}

/// CRIME collects every interrupt on the O2: MACE's (0..15, the ISA
/// peripherals among them through MACE's own status), GBE's (16..19) and
/// the rendering engine's. The edge-type inputs latch until software clears
/// them in HARDINT; the rest follow their level.
impl crate::irq::InterruptController for Crime {
    fn set_line(&self, line: u32, level: bool) {
        if (1u64 << line) & CRIME_LATCHED_BITS != 0 {
            if level {
                self.latch_int(line);
            }
        } else {
            self.set_int(line, level);
        }
        self.update_output();
    }
}

impl Device for Crime {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("crime".to_string(), "CRIME: crime status | crime regs | crime irqlog [n] | crime touched".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "crime" {
            return Err("Command not found".to_string());
        }
        let reg = |off: u32| self.regs.lock().unwrap()[(off / 8) as usize];
        match args.first().copied().unwrap_or("status") {
            "status" => {
                let (stat, mask) = (self.intstat(), reg(crime_reg::INTMASK));
                writeln!(w, "CRIME revision {:#x}", self.revision()).ok();
                writeln!(w, "  interrupts: status {stat:#010x} mask {mask:#010x} pending {:#010x} latched {:#010x} soft {:#010x}",
                    stat & mask, *self.latched.lock().unwrap(), reg(crime_reg::SOFTINT)).ok();
                writeln!(w, "  control {:#x} time {:#x}", reg(crime_reg::CONTROL), self.load_inner(crime_reg::TIME)).ok();
                for i in 0..8u32 {
                    writeln!(w, "  bank {i}: ctrl {:#06x}", reg(crime_reg::MEM_BANK_CTRL0 + 8 * i)).ok();
                }
            }
            "regs" => {
                for (i, &v) in self.regs.lock().unwrap().iter().enumerate() {
                    if v != 0 {
                        writeln!(w, "  {:#06x}: {v:#018x}", i * 8).ok();
                    }
                }
            }
            "irqlog" => {
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
            _ => return Err("Usage: crime status | regs | irqlog [n] | touched".to_string()),
        }
        Ok(())
    }
}

impl Resettable for Crime {
    /// Registers, latched interrupts and the counter's offset back to zero;
    /// the revision is the chip's and stays.
    fn power_on(&self) {
        self.regs.lock().unwrap().fill(0);
        *self.latched.lock().unwrap() = 0;
        *self.time_base.lock().unwrap() = 0;
        self.irqlog.lock().unwrap().clear();
    }
}

/// The registers, the latched edges, the revision and the counter's offset
/// (against the UST, which restores its own time); loading hands the output
/// to IP2 again.
impl Saveable for Crime {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("regs".into(), u64_slice_to_toml(&self.regs.lock().unwrap()));
        t.insert("latched".into(), hex_u64(*self.latched.lock().unwrap()));
        t.insert("revision".into(), hex_u64(self.revision()));
        t.insert("time_base".into(), hex_u64(*self.time_base.lock().unwrap()));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        if let Some(r) = get_field(v, "regs") {
            load_u64_slice(r, &mut self.regs.lock().unwrap());
        }
        *self.latched.lock().unwrap() = get_field(v, "latched").and_then(toml_u64).unwrap_or(0);
        if let Some(rev) = get_field(v, "revision").and_then(toml_u64) {
            self.set_revision(rev);
        }
        *self.time_base.lock().unwrap() = get_field(v, "time_base").and_then(toml_u64).unwrap_or(0);
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
        let cur = self.load(off);
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
            let v = self.regs.load(off);
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
impl BusDevice for CrimeSpace {
    fn read8(&self, addr: u64) -> crate::traits::BusRead8 {
        let r = if Self::in_re(addr as u32) { self.re.read32(addr & !3) } else { self.regs.read32(addr & !3) };
        crate::traits::BusRead8 { status: r.status, data: r.data as u8 }
    }
    fn read16(&self, addr: u64) -> crate::traits::BusRead16 {
        let r = if Self::in_re(addr as u32) { self.re.read32(addr & !3) } else { self.regs.read32(addr & !3) };
        crate::traits::BusRead16 { status: r.status, data: r.data as u16 }
    }
    fn read32(&self, addr: u64) -> BusRead32 {
        if Self::in_re(addr as u32) { self.re.read32(addr) } else { self.regs.read32(addr) }
    }
    fn read64(&self, addr: u64) -> BusRead64 {
        if Self::in_re(addr as u32) { self.re.read64(addr) } else { self.regs.read64(addr) }
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
        let r = self.regs.write32(addr, val);
        self.regs_wrote(addr as u32);
        r
    }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        if Self::in_re(addr as u32) {
            return self.re.write64(addr, val);
        }
        let r = self.regs.write64(addr, val);
        self.regs_wrote(addr as u32);
        r
    }
}
