//! The O2's DS1687 real-time clock and NVRAM.

use std::sync::Mutex;

use crate::traits::{BusDevice, BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK};
use std::io::Write;
use crate::traits::{Device, Resettable, Saveable};
use crate::snapshot::{get_field, load_u8_slice, u8_slice_to_toml};

/// A Dallas-style RTC with battery-backed NVRAM.
///
/// Registers are reached with MACE's ISA spacing, `(reg << 8) + 7`. The low
/// registers are the clock; everything from [`NVRAM_FIRST`] up is storage that
/// survives power loss on a real machine — which is precisely why firmware
/// trusts it, and why handing back zeros is not neutral.
pub struct MaceRtc {
    cells: Mutex<Vec<u8>>,
    /// The clock: guest seconds since 1970 at a host instant, so it runs.
    /// A kernel that calibrates its CPU clock against the RTC waits for the
    /// seconds to change, and a stopped clock leaves it waiting for ever
    /// (OpenBSD, straight after autoconfiguration).
    clock: Mutex<(i64, std::time::Instant)>,
    /// The century while SET is up, as the guest last wrote it (binary).
    /// SET freezes the century with the other time bytes, and clearing SET
    /// commits all of them together: IRIX writes the century between the
    /// year and the weekday.
    held_century: Mutex<Option<i64>>,
    /// Set when the guest writes PAB in extended control register 4A: on an
    /// O2 that is software power-off (the RTC lets go of the supply). Both
    /// NetBSD and OpenBSD power down this way; without it OpenBSD retries
    /// for ever, printing "Power supply did not shut down".
    powered_off: std::sync::atomic::AtomicBool,
}

/// The DS1687's clock registers and the bits of register B that shape them.
mod rtc_reg {
    pub const SEC: usize = 0x00;
    pub const MIN: usize = 0x02;
    pub const HOUR: usize = 0x04;
    pub const WDAY: usize = 0x06;
    pub const MDAY: usize = 0x07;
    pub const MONTH: usize = 0x08;
    pub const YEAR: usize = 0x09;
    pub const A: usize = 0x0a;
    /// Register A's bank select: with it set, 0x40-0x7f are the extended
    /// registers, and 0x48 is the century.
    pub const A_DV0: u8 = 0x10;
    pub const CENTURY: usize = 0x48;
    /// Extended control register 4A (bank 1), and its power-active-bar bit.
    pub const XCTRL4A: usize = 0x4a;
    pub const XCTRL4A_PAB: u8 = 0x08;
    pub const B: usize = 0x0b;
    pub const B_SET: u8 = 0x80;
    pub const B_BINARY: u8 = 0x04;
    pub const B_24H: u8 = 0x02;
    pub const TIME: [usize; 7] = [SEC, MIN, HOUR, WDAY, MDAY, MONTH, YEAR];
}

/// Days since 1970-01-01 for a proleptic Gregorian date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// The date for a count of days since 1970-01-01: (year, month, day).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + if m <= 2 { 1 } else { 0 }, m, d)
}

impl MaceRtc {
    /// First register that is general-purpose NVRAM rather than clock state.
    pub const NVRAM_FIRST: usize = 0x0e;
    pub const REGS: usize = 0x100;

    pub fn new() -> Self {
        let mut c = vec![0u8; Self::REGS];
        // A plausible stopped-but-valid clock. Register D bit 7 is the
        // valid-RAM-and-time flag: firmware that sees it clear concludes the
        // battery died, which is its own diagnostic path.
        c[0x0d] = 0x80;
        // BCD, 24-hour: what SGI firmware leaves the part in.
        c[rtc_reg::B] = rtc_reg::B_24H;
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64).unwrap_or(0);
        Self {
            cells: Mutex::new(c),
            clock: Mutex::new((now, std::time::Instant::now())),
            held_century: Mutex::new(None),
            powered_off: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Whether the guest has switched the machine off.
    pub fn powered_off(&self) -> bool {
        self.powered_off.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Guest time now, in seconds since 1970.
    fn now(&self) -> i64 {
        let (secs, at) = *self.clock.lock().unwrap();
        secs + at.elapsed().as_secs() as i64
    }

    /// The clock registers' value for `t`, in the mode register B selects.
    fn field(&self, reg: usize, t: i64, b: u8) -> u8 {
        let days = t.div_euclid(86400);
        let secs = t.rem_euclid(86400);
        let (y, m, d) = civil_from_days(days);
        let mut hour = (secs / 3600) as u8;
        let mut pm = 0u8;
        if reg == rtc_reg::HOUR && b & rtc_reg::B_24H == 0 {
            pm = if hour >= 12 { 0x80 } else { 0 };
            hour %= 12;
            if hour == 0 { hour = 12; }
        }
        let v = match reg {
            rtc_reg::SEC => (secs % 60) as u8,
            rtc_reg::MIN => (secs / 60 % 60) as u8,
            rtc_reg::HOUR => hour,
            // 1 = Sunday; 1970-01-01 was a Thursday.
            rtc_reg::WDAY => ((days + 4).rem_euclid(7) + 1) as u8,
            rtc_reg::MDAY => d as u8,
            rtc_reg::MONTH => m as u8,
            _ => y.rem_euclid(100) as u8,
        };
        let v = if b & rtc_reg::B_BINARY != 0 { v } else { (v / 10) << 4 | v % 10 };
        v | pm
    }

    /// Restart the clock from what the guest has written into the time
    /// registers. A two-digit year below 70 is taken as 20xx.
    fn set_from_cells(&self) {
        self.set_from_cells_in(None)
    }

    /// The same, with the century given (a write to the century register).
    fn set_from_cells_in(&self, century: Option<i64>) {
        let c = self.cells.lock().unwrap();
        let b = c[rtc_reg::B];
        let get = |r: usize| -> i64 {
            let raw = c[r] & if r == rtc_reg::HOUR && b & rtc_reg::B_24H == 0 { 0x7f } else { 0xff };
            (if b & rtc_reg::B_BINARY != 0 { raw } else { (raw >> 4) * 10 + (raw & 0x0f) }) as i64
        };
        let mut hour = get(rtc_reg::HOUR);
        if b & rtc_reg::B_24H == 0 {
            hour %= 12;
            if c[rtc_reg::HOUR] & 0x80 != 0 { hour += 12; }
        }
        let yy = get(rtc_reg::YEAR);
        let year = match century {
            Some(c) => c * 100 + yy,
            None if yy < 70 => 2000 + yy,
            None => 1900 + yy,
        };
        let (month, mday) = (get(rtc_reg::MONTH).clamp(1, 12), get(rtc_reg::MDAY).clamp(1, 31));
        let t = days_from_civil(year, month, mday) * 86400
            + hour * 3600 + get(rtc_reg::MIN) * 60 + get(rtc_reg::SEC);
        drop(c);
        *self.clock.lock().unwrap() = (t, std::time::Instant::now());
    }

    fn reg_of(addr: u32) -> Option<usize> {
        if addr & 7 != 7 {
            return None;
        }
        Some(((addr >> 8) & 0xff) as usize)
    }

    pub fn peek(&self, reg: usize) -> u8 {
        self.cells.lock().unwrap()[reg & 0xff]
    }
    pub fn poke(&self, reg: usize, val: u8) {
        self.cells.lock().unwrap()[reg & 0xff] = val;
    }
}

impl Device for MaceRtc {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("rtc".to_string(), "DS1687 RTC: rtc status | rtc dump | rtc r <reg> | rtc w <reg> <val>".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "rtc" {
            return Err("Command not found".to_string());
        }
        let num = |s: Option<&&str>| -> Result<usize, String> {
            let s = s.ok_or("rtc: missing number")?;
            usize::from_str_radix(s.trim_start_matches("0x"), 16).map_err(|_| format!("rtc: {s:?} is not a hex number"))
        };
        match args.first().copied().unwrap_or("status") {
            "status" => {
                let t = self.now();
                let (y, m, d) = civil_from_days(t.div_euclid(86400));
                let s = t.rem_euclid(86400);
                writeln!(w, "RTC: {y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} (guest), register B {:#04x} D {:#04x}{}",
                    s / 3600, s / 60 % 60, s % 60, self.peek(rtc_reg::B), self.peek(0x0d),
                    if self.powered_off() { ", powered off" } else { "" }).ok();
            }
            "dump" => {
                for (i, row) in self.cells.lock().unwrap().chunks(16).enumerate() {
                    let hex: Vec<String> = row.iter().map(|b| format!("{b:02x}")).collect();
                    writeln!(w, "  {:02x}: {}", i * 16, hex.join(" ")).ok();
                }
            }
            "r" => {
                let r = num(args.get(1))?;
                writeln!(w, "  {r:02x}: {:02x}", self.peek(r)).ok();
            }
            "w" => {
                let (r, v) = (num(args.get(1))?, num(args.get(2))?);
                self.poke(r, v as u8);
                writeln!(w, "  {r:02x}: {:02x}", self.peek(r)).ok();
            }
            _ => return Err("Usage: rtc status | dump | r <reg> | w <reg> <val>".to_string()),
        }
        Ok(())
    }
}

impl Resettable for MaceRtc {
    /// Battery-backed: the clock and NVRAM live through a reset. Only the
    /// power-off request is forgotten.
    fn power_on(&self) {
        self.powered_off.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The cells, the clock as guest seconds and a century written under SET:
/// a restored machine's clock runs on from the moment it was saved, and a
/// clock write the save interrupted completes when SET comes down.
impl Saveable for MaceRtc {
    fn save_state(&self) -> toml::Value {
        let mut tbl = toml::map::Map::new();
        tbl.insert("cells".into(), u8_slice_to_toml(&self.cells.lock().unwrap()));
        tbl.insert("time".into(), toml::Value::Integer(self.now()));
        if let Some(c) = *self.held_century.lock().unwrap() {
            tbl.insert("held_century".into(), toml::Value::Integer(c));
        }
        tbl.insert("powered_off".into(), toml::Value::Boolean(self.powered_off()));
        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        if let Some(c) = get_field(v, "cells") {
            load_u8_slice(c, &mut self.cells.lock().unwrap());
        }
        if let Some(t) = get_field(v, "time").and_then(|t| t.as_integer()) {
            *self.clock.lock().unwrap() = (t, std::time::Instant::now());
        }
        *self.held_century.lock().unwrap() = get_field(v, "held_century").and_then(|c| c.as_integer());
        let off = get_field(v, "powered_off").and_then(|b| b.as_bool()).unwrap_or(false);
        self.powered_off.store(off, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

impl Default for MaceRtc {
    fn default() -> Self { Self::new() }
}

impl BusDevice for MaceRtc {
    fn read8(&self, addr: u64) -> BusRead8 {
        let Some(r) = Self::reg_of(addr as u32) else { return BusRead8::ok(0) };
        let b = self.peek(rtc_reg::B);
        if r == rtc_reg::CENTURY && self.peek(rtc_reg::A) & rtc_reg::A_DV0 != 0 {
            let c = match *self.held_century.lock().unwrap() {
                Some(c) if b & rtc_reg::B_SET != 0 => c as u8,
                _ => (civil_from_days(self.now().div_euclid(86400)).0 / 100) as u8,
            };
            return BusRead8::ok(if b & rtc_reg::B_BINARY != 0 { c } else { (c / 10) << 4 | c % 10 });
        }
        // While SET is up the clock is frozen at what is being written.
        if rtc_reg::TIME.contains(&r) && b & rtc_reg::B_SET == 0 {
            return BusRead8::ok(self.field(r, self.now(), b));
        }
        BusRead8::ok(self.peek(r))
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        let Some(r) = Self::reg_of(addr as u32) else { return BUS_OK };
        let b = self.peek(rtc_reg::B);
        if r == rtc_reg::XCTRL4A && self.peek(rtc_reg::A) & rtc_reg::A_DV0 != 0 {
            if val & rtc_reg::XCTRL4A_PAB != 0 {
                self.powered_off.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            return BUS_OK;
        }
        if r == rtc_reg::CENTURY && self.peek(rtc_reg::A) & rtc_reg::A_DV0 != 0 {
            let c = if b & rtc_reg::B_BINARY != 0 { val } else { (val >> 4) * 10 + (val & 0x0f) };
            if b & rtc_reg::B_SET != 0 {
                // Held with the other time bytes until SET comes down.
                *self.held_century.lock().unwrap() = Some(c as i64);
                return BUS_OK;
            }
            // The century, in the extended bank: keep the rest of the time.
            let t = self.now();
            for &f in &rtc_reg::TIME {
                let v = self.field(f, t, b);
                self.poke(f, v);
            }
            self.set_from_cells_in(Some(c as i64));
            return BUS_OK;
        }
        if r == rtc_reg::B && b & rtc_reg::B_SET == 0 && val & rtc_reg::B_SET != 0 {
            // SET going up: freeze the time registers at the current time,
            // so fields the guest does not rewrite keep their values.
            let t = self.now();
            for &f in &rtc_reg::TIME {
                let v = self.field(f, t, b);
                self.poke(f, v);
            }
            // A century the guest does not rewrite is left to the year's
            // pivot, as before.
            *self.held_century.lock().unwrap() = None;
        }
        self.poke(r, val);
        if r == rtc_reg::B && b & rtc_reg::B_SET != 0 && val & rtc_reg::B_SET == 0 {
            // SET coming down: the clock runs on from what was written.
            let century = self.held_century.lock().unwrap().take();
            self.set_from_cells_in(century);
        } else if rtc_reg::TIME.contains(&r) && b & rtc_reg::B_SET == 0 {
            // A field written without SET: freeze the others at the current
            // time first, then take the whole lot as the new time.
            let t = self.now();
            for &f in &rtc_reg::TIME {
                if f != r {
                    let v = self.field(f, t, b);
                    self.poke(f, v);
                }
            }
            self.set_from_cells();
        }
        BUS_OK
    }
    fn read16(&self, a: u64) -> BusRead16 { BusRead16::ok(self.read8(a | 7).data as u16) }
    fn read32(&self, a: u64) -> BusRead32 { BusRead32::ok(self.read8(a | 7).data as u32) }
    fn read64(&self, a: u64) -> BusRead64 { BusRead64::ok(self.read8(a | 7).data as u64) }
    fn write16(&self, a: u64, v: u16) -> u32 { self.write8(a | 7, v as u8) }
    fn write32(&self, a: u64, v: u32) -> u32 { self.write8(a | 7, v as u8) }
    fn write64(&self, a: u64, v: u64) -> u32 { self.write8(a | 7, v as u8) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The RTC runs, from host time, and a guest that sets it with SET moves it.
    #[test]
    fn the_rtc_runs_and_can_be_set() {
        let r = MaceRtc::new();
        let reg = |n: u32| ((n << 8) | 7) as u64;
        // Host time on power-up: some year from 2026 on, in BCD.
        let yy = r.read8(reg(9)).data;
        assert!(yy >= 0x26 && yy & 0x0f < 10, "year {yy:#x}");
        // Set 1999-12-31 23:59:58 under SET, binary mode, 24-hour.
        r.write8(reg(0x0b), 0x80 | 0x04 | 0x02);
        for (n, v) in [(0u32, 58u8), (2, 59), (4, 23), (7, 31), (8, 12), (9, 99)] {
            r.write8(reg(n), v);
        }
        r.write8(reg(0x0b), 0x04 | 0x02);
        assert_eq!(r.read8(reg(9)).data, 99);
        assert_eq!(r.read8(reg(6)).data, 6, "a Friday");
        // Three seconds on, it is the year 2000.
        {
            let mut c = r.clock.lock().unwrap();
            c.1 -= std::time::Duration::from_secs(3);
        }
        assert_eq!((r.read8(reg(9)).data, r.read8(reg(8)).data, r.read8(reg(7)).data), (0, 1, 1));
        assert_eq!(r.read8(reg(0)).data, 1);
        // And BCD again on request.
        r.write8(reg(0x0b), 0x02);
        assert_eq!(r.read8(reg(8)).data, 0x01);
        // The century, in the extended bank (register A bit 4).
        r.write8(reg(0x0a), 0x10);
        assert_eq!(r.read8(reg(0x48)).data, 0x20);
        r.write8(reg(0x48), 0x19);
        r.write8(reg(0x0a), 0x00);
        assert_eq!(r.read8(reg(9)).data, 0x00, "1900: same two digits");
        r.write8(reg(0x0a), 0x10);
        assert_eq!(r.read8(reg(0x48)).data, 0x19);
        assert_eq!(days_from_civil(2000, 1, 1), 10957);
        // Software power-off: PAB in extended control register 4A.
        assert!(!r.powered_off());
        r.write8(reg(0x0a), 0x10);
        r.write8(reg(0x4a), 0x08);
        assert!(r.powered_off(), "PAB switches the machine off");
        assert_eq!(civil_from_days(10957), (2000, 1, 1));
    }

    /// SET holds the century with the other time bytes. IRIX's clock write
    /// raises SET, writes the seconds through the year, then the century,
    /// then the weekday, and clears SET; the clock must then run from the
    /// date written, not from the time SET went up.
    #[test]
    fn a_century_written_under_set_waits_for_set_to_clear() {
        let r = MaceRtc::new();
        let reg = |n: u32| ((n << 8) | 7) as u64;
        r.write8(reg(0x0a), 0x30);
        r.write8(reg(0x0b), 0x80 | 0x04 | 0x02);
        for (n, v) in [(0u32, 0u8), (2, 0), (4, 19), (7, 1), (8, 6), (9, 4), (0x48, 20), (6, 3)] {
            r.write8(reg(n), v);
        }
        assert_eq!(r.read8(reg(0x48)).data, 20, "the held century reads back under SET");
        r.write8(reg(0x0b), 0x04 | 0x02);
        assert_eq!(r.now(), 1_086_116_400, "2004-06-01 19:00:00 UTC");
        assert_eq!((r.read8(reg(9)).data, r.read8(reg(8)).data, r.read8(reg(7)).data), (4, 6, 1));
        assert_eq!(r.read8(reg(0x48)).data, 20);
        // A century of 19 under SET lands in the 1900s.
        r.write8(reg(0x0b), 0x80 | 0x04 | 0x02);
        r.write8(reg(9), 99);
        r.write8(reg(0x48), 19);
        r.write8(reg(0x0b), 0x04 | 0x02);
        assert_eq!(civil_from_days(r.now().div_euclid(86400)).0, 1999);
    }

    /// Set 2004-06-01 19:00:00 UTC the way IRIX does, century included.
    fn set_2004(r: &MaceRtc) {
        let reg = |n: u32| ((n << 8) | 7) as u64;
        r.write8(reg(0x0a), 0x30);
        r.write8(reg(0x0b), 0x80 | 0x04 | 0x02);
        for (n, v) in [(0u32, 0u8), (2, 0), (4, 19), (7, 1), (8, 6), (9, 4), (0x48, 20), (6, 3)] {
            r.write8(reg(n), v);
        }
        r.write8(reg(0x0b), 0x04 | 0x02);
    }

    /// A snapshot taken mid-write, between the century and SET clearing,
    /// completes the write after a restore.
    #[test]
    fn a_snapshot_keeps_a_century_written_under_set() {
        let reg = |n: u32| ((n << 8) | 7) as u64;
        let src = MaceRtc::new();
        src.write8(reg(0x0a), 0x30);
        src.write8(reg(0x0b), 0x80 | 0x04 | 0x02);
        for (n, v) in [(0u32, 0u8), (2, 0), (4, 19), (7, 1), (8, 6), (9, 70), (0x48, 20)] {
            src.write8(reg(n), v);
        }
        let dst = MaceRtc::new();
        dst.load_state(&src.save_state()).unwrap();
        dst.write8(reg(0x0b), 0x04 | 0x02);
        assert_eq!(dst.now(), 3_168_874_800, "2070-06-01 19:00:00 UTC, not 1970");
        // A save with no write in progress restores none.
        let after = MaceRtc::new();
        after.load_state(&dst.save_state()).unwrap();
        assert_eq!(*after.held_century.lock().unwrap(), None);
    }

    /// A snapshot after a date set restores the set date, not host time.
    #[test]
    fn a_snapshot_keeps_a_set_date() {
        let src = MaceRtc::new();
        set_2004(&src);
        let dst = MaceRtc::new();
        dst.load_state(&src.save_state()).unwrap();
        assert!((1_086_116_400..=1_086_116_402).contains(&dst.now()));
        assert_eq!(civil_from_days(dst.now().div_euclid(86400)), (2004, 6, 1));
    }

    #[test]
    fn save_load_round_trip() {
        let src = MaceRtc::new();
        src.poke(0x40, 0xa5);
        src.poke(0xff, 0x5a);
        *src.clock.lock().unwrap() = (946_684_798, std::time::Instant::now());
        src.powered_off.store(true, std::sync::atomic::Ordering::Relaxed);
        let v1 = src.save_state();

        let dst = MaceRtc::new();
        dst.load_state(&v1).expect("load_state");
        assert_eq!(dst.now(), 946_684_798, "the clock resumes from the saved time");
        assert!(dst.powered_off());
        assert_eq!(v1, dst.save_state(), "MaceRtc save_state mismatch after load_state round-trip");
    }
}
