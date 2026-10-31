//! DS1744-compatible timekeeping SRAM: 32 KiB of battery-backed bytes whose
//! last eight are a BCD calendar.
//!
//! The Tezro's IO9 maps one in the IOC4's BAR (byte-bus
//! select 0 of the IOC4). The part reads like the Fuel's M48T35 [`Timekeeper`] at a
//! glance, and has the same geometry and clock offsets, but differs where
//! software looks: the control register carries the century beside the
//! write and read bits, the day register holds a weekday that advances
//! with the calendar, an oscillator-stop bit rides in the seconds, a commit
//! with an impossible date is refused, and a test-mode bit makes the
//! seconds' low bit run at 1024 Hz. Those are a different calendar rather
//! than a flag on the M48T35's, so this is its own model.
//!
//! The guest's write bit stages a calendar image and the clearing of it
//! commits it; the read bit freezes the visible calendar while the real one
//! runs on.
//!
//! # What is approximated here
//!
//! * The fitted IO9 part and its physical address wiring are unknown. The
//!   32 KiB geometry is the profile's choice.
//! * With a fresh-year base of 1968 the guest's year digits count from that
//!   year, as its driver expects: the host date is shifted into 2000 to
//!   2099 and the weekday comes from the unshifted date, so the weekday
//!   follows the host's real calendar while the digits follow the shifted
//!   one.
//! * Host time and a backing file stand in for the oscillator and battery.
//!   A restart restores a completed calendar and lets host time pass; an
//!   unfinished write or read transaction is not restored.
//!
//! Ported from hw/rtc/ds1742.c in jamesbraid/qemu-sgi-origami.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

use crate::traits::{BusDevice, BusRead8, BUS_ERR, BUS_OK};

pub const SIZE: usize = 0x8000;
/// The control register, first of the eight calendar bytes.
pub const CONTROL: usize = 0x7ff8;
const WRITE: u8 = 0x80;
const READ: u8 = 0x40;
/// Seconds bit 7: the oscillator is stopped.
const OSCILLATOR_STOP: u8 = 0x80;
/// Day bit 6: test mode; bit 7: the battery flag, read-only and always set.
const TEST_MODE: u8 = 0x40;
const BATTERY_FLAG: u8 = 0x80;
const CLOCK_REF_BYTES: usize = 16;
const CLOCK_REF_MAGIC: &[u8; 8] = b"DS1742R1";
/// Test mode flips the seconds' low bit every 976562 ns.
const TEST_BIT_NS: i64 = 976_562;

/// How the guest's year digits relate to the host's calendar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YearBase {
    /// The host's own year.
    Gregorian,
    /// The year digits count from this year: the host date is shifted into
    /// 2000 to 2099. Only 1968 is accepted.
    From(u32),
}

fn bcd(v: i64) -> u8 {
    ((v / 10) * 16 + v % 10) as u8
}

fn from_bcd(v: u8) -> i64 {
    i64::from((v >> 4) * 10 + (v & 15))
}

fn bcd_in(v: u8, lo: i64, hi: i64) -> bool {
    v & 15 < 10 && v >> 4 < 10 && (lo..=hi).contains(&from_bcd(v))
}

fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn month_days(year: i64, month: i64) -> i64 {
    match month {
        2 => 28 + i64::from(is_leap(year)),
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days since 1970-01-01 of a Gregorian date.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The Gregorian date of a day count since 1970-01-01.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Day number of `seconds`, rounding down.
fn day_index(seconds: i64) -> i64 {
    seconds.div_euclid(86_400)
}

/// The 1-based weekday (Sunday is 1) of a count of days since 1970-01-01,
/// a Thursday.
fn weekday_of_day(days: i64) -> u8 {
    (days + 4).rem_euclid(7) as u8 + 1
}

struct State {
    ram: Vec<u8>,
    control: u8,
    /// Test-mode flag and weekday as last written (FT and bits 2:0).
    weekday: u8,
    staged: [u8; 8],
    frozen: [u8; 8],
    frozen_valid: bool,
    /// Unix seconds of the calendar when it was last set.
    epoch: i64,
    /// Clock reading (ns) when it was last set.
    reference_ns: i64,
    stopped: bool,
    file: Option<PathBuf>,
    /// A backing write failed; the next equal write retries it.
    storage_failed: bool,
}

/// The nanosecond clock the calendar runs on.
type Clock = Arc<dyn Fn() -> i64 + Send + Sync>;

pub struct Ds1744 {
    st: Mutex<State>,
    clock: Clock,
}

impl Ds1744 {
    /// A part whose calendar starts at the host's date and time.
    pub fn new(base: YearBase) -> Result<Ds1744, String> {
        let host = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| e.to_string())?.as_secs() as i64;
        let born = Instant::now();
        Ds1744::with_clock(host, base, Arc::new(move || born.elapsed().as_nanos() as i64))
    }

    /// A part seeded with `host_unix_secs` and running on `clock`, a count of
    /// nanoseconds. Tests supply a clock they step.
    pub fn with_clock(host_unix_secs: i64, base: YearBase, clock: Clock) -> Result<Ds1744, String> {
        let mut epoch = host_unix_secs;
        let (year, month, day) = civil_from_days(day_index(host_unix_secs));
        let mut control = bcd(year / 100);
        let weekday = weekday_of_day(day_index(host_unix_secs));
        match base {
            YearBase::Gregorian => {}
            YearBase::From(1968) => {
                let base = 1968;
                if !(base..=base + 99).contains(&year) {
                    return Err(format!("ds1744: fresh-year-base={base} cannot seed host year {year}"));
                }
                let shifted = 2000 + (year - base) % 100;
                // A leap day in a shifted year without one is the next day.
                epoch = days_from_civil(shifted, month, day) * 86_400 + host_unix_secs.rem_euclid(86_400);
                control = bcd(shifted / 100);
            }
            YearBase::From(other) => return Err(format!("ds1744: fresh-year-base must be 0 or 1968, not {other}")),
        }
        let now = clock();
        Ok(Ds1744 {
            st: Mutex::new(State {
                ram: vec![0; CONTROL],
                control,
                weekday,
                staged: [0; 8],
                frozen: [0; 8],
                frozen_valid: false,
                epoch,
                reference_ns: now,
                stopped: false,
                file: None,
                storage_failed: false,
            }),
            clock,
        })
    }

    fn now_ns(&self) -> i64 {
        (self.clock)()
    }

    fn seconds(&self, st: &State) -> i64 {
        st.epoch + if st.stopped { 0 } else { (self.now_ns() - st.reference_ns) / 1_000_000_000 }
    }

    /// The calendar bytes as the part would show them now.
    fn image(&self, st: &State) -> [u8; 8] {
        let now = self.seconds(st);
        let (year, month, day) = civil_from_days(day_index(now));
        let sec = now.rem_euclid(86_400);
        let weekday = ((i64::from(st.weekday & 7) - 1 + day_index(now) - day_index(st.epoch)).rem_euclid(7)) as u8;
        [
            (st.control & 0xc0) | bcd(year / 100),
            bcd(sec % 60) | if st.stopped { OSCILLATOR_STOP } else { 0 },
            bcd(sec / 60 % 60),
            bcd(sec / 3600),
            BATTERY_FLAG | (st.weekday & TEST_MODE) | (weekday + 1),
            bcd(day),
            bcd(month),
            bcd(year % 100),
        ]
    }

    /// Make the staged image the calendar, if it is a date. A refusal leaves
    /// the calendar as it was.
    fn commit(&self, st: &mut State, century: u8) -> bool {
        let s = st.staged;
        let valid = bcd_in(century, 0, 39)
            && bcd_in(s[1] & 0x7f, 0, 59)
            && bcd_in(s[2], 0, 59)
            && bcd_in(s[3], 0, 23)
            && bcd_in(s[4] & 7, 1, 7)
            && bcd_in(s[5], 1, 31)
            && bcd_in(s[6], 1, 12)
            && bcd_in(s[7], 0, 99);
        if !valid {
            crate::dlog!(crate::devlog::LogModule::Rtc, "DS1744: invalid calendar write");
            return false;
        }
        let year = from_bcd(century) * 100 + from_bcd(s[7]);
        let (month, day) = (from_bcd(s[6]), from_bcd(s[5]));
        if day > month_days(year, month) {
            crate::dlog!(crate::devlog::LogModule::Rtc, "DS1744: unsupported calendar date {year:04}-{month:02}-{day:02}");
            return false;
        }
        st.epoch = days_from_civil(year, month, day) * 86_400
            + from_bcd(s[3]) * 3600
            + from_bcd(s[2]) * 60
            + from_bcd(s[1] & 0x7f);
        st.reference_ns = self.now_ns();
        st.weekday = s[4] & (TEST_MODE | 7);
        st.stopped = s[1] & OSCILLATOR_STOP != 0;
        true
    }

    // ----- backing file -----

    /// Back the part with a 32 KiB file and a 16-byte clock reference beside
    /// it (`<path>.clock`). A missing file starts blank and is created; a file
    /// of another size is refused. A saved calendar is restored and lets the
    /// host time since it was saved pass; a blank one keeps the seed.
    pub fn set_nvram_file(&self, path: &Path) -> Result<(), String> {
        let clock_path = PathBuf::from(format!("{}.clock", path.display()));
        let image = match std::fs::read(path) {
            Ok(v) if v.len() == SIZE => Some(v),
            Ok(_) => return Err("DS1744: backing image must contain exactly 32768 bytes".into()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.to_string()),
        };
        let reference = match std::fs::read(&clock_path) {
            Ok(v) if v.len() >= CLOCK_REF_BYTES => Some(v),
            Ok(_) => return Err("DS1744: clock reference is shorter than 16 bytes".into()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.to_string()),
        };
        let mut st = self.st.lock();
        if let Some(image) = image {
            st.ram.copy_from_slice(&image[..CONTROL]);
            let cal: [u8; 8] = image[CONTROL..].try_into().expect("eight calendar bytes");
            if [cal[1], cal[2], cal[3], cal[5], cal[6], cal[7]].iter().any(|&b| b != 0) {
                self.restore_clock(&mut st, cal, reference.as_deref());
            }
        }
        st.file = Some(path.to_path_buf());
        self.save_clock(&mut st);
        Ok(())
    }

    fn restore_clock(&self, st: &mut State, cal: [u8; 8], reference: Option<&[u8]>) {
        st.staged = cal;
        if !self.commit(st, cal[0] & 0x3f) {
            crate::dlog!(crate::devlog::LogModule::Rtc, "DS1744: invalid saved calendar; using the seed");
            return;
        }
        st.control = cal[0] & 0x3f;
        let mut elapsed = 0;
        if !st.stopped {
            if let Some(r) = reference.filter(|r| &r[..8] == CLOCK_REF_MAGIC) {
                let saved = i64::from_le_bytes(r[8..16].try_into().expect("eight bytes"));
                elapsed = (unix_now() - saved).max(0);
            }
        }
        if elapsed != 0 {
            let days = day_index(st.epoch + elapsed) - day_index(st.epoch);
            st.weekday = (st.weekday & TEST_MODE) | (((i64::from(st.weekday & 7) - 1 + days % 7) % 7) as u8 + 1);
            st.epoch += elapsed;
        }
    }

    /// Write the whole image and the clock reference. Errors are reported
    /// and remembered so an equal write retries them.
    fn save_clock(&self, st: &mut State) {
        let Some(path) = st.file.clone() else { return };
        let mut image = st.ram.clone();
        image.extend_from_slice(&self.image(st));
        let mut reference = CLOCK_REF_MAGIC.to_vec();
        reference.extend_from_slice(&unix_now().to_le_bytes());
        let clock_path = format!("{}.clock", path.display());
        let result = std::fs::write(&path, &image).and_then(|()| std::fs::write(clock_path, &reference));
        st.storage_failed = result.is_err();
        if let Err(e) = result {
            log::error!("DS1744: could not save {}: {e}", path.display());
        }
    }

    /// Save the calendar now, as a stop or shutdown does.
    pub fn flush(&self) {
        let mut st = self.st.lock();
        self.save_clock(&mut st);
    }

    fn read(&self, a: usize) -> u8 {
        let st = self.st.lock();
        if a < CONTROL {
            return st.ram[a];
        }
        let image = self.image(&st);
        if a == CONTROL {
            return if st.control & WRITE != 0 {
                st.control
            } else if st.frozen_valid {
                (st.control & 0xc0) | (st.frozen[0] & 0x3f)
            } else {
                image[0]
            };
        }
        let i = a - CONTROL;
        let mut value = if st.control & WRITE != 0 {
            st.staged[i]
        } else if st.frozen_valid {
            st.frozen[i]
        } else {
            image[i]
        };
        if i == 4 {
            value |= BATTERY_FLAG;
        }
        if i == 1 && st.weekday & TEST_MODE != 0 && !st.stopped {
            value = (value & !1) | ((self.now_ns() / TEST_BIT_NS) & 1) as u8;
        }
        value
    }

    fn write(&self, a: usize, value: u8) {
        let mut st = self.st.lock();
        if a < CONTROL {
            if st.ram[a] == value && !st.storage_failed {
                return;
            }
            st.ram[a] = value;
            if let Some(path) = &st.file {
                use std::io::{Seek, SeekFrom, Write};
                let result = std::fs::OpenOptions::new().write(true).open(path).and_then(|mut f| {
                    f.seek(SeekFrom::Start(a as u64))?;
                    f.write_all(&[value])
                });
                st.storage_failed = result.is_err();
                if let Err(e) = result {
                    log::error!("DS1744: could not save SRAM byte at {a:#x}: {e}");
                }
            }
            return;
        }
        if a != CONTROL {
            if st.control & WRITE != 0 {
                st.staged[a - CONTROL] = value;
            } else {
                crate::dlog!(crate::devlog::LogModule::Rtc, "DS1744: calendar write with W clear at {a:#x}");
            }
            return;
        }
        if st.control & (WRITE | READ) == 0 {
            st.control = self.image(&st)[0];
        }
        let old = st.control;
        if value & READ != 0 && old & READ == 0 {
            st.frozen = self.image(&st);
            st.frozen_valid = true;
        } else if value & READ == 0 {
            st.frozen_valid = false;
        }
        if value & WRITE != 0 && old & WRITE == 0 {
            st.staged = self.image(&st);
        }
        if value & WRITE == 0 && old & WRITE != 0 {
            let valid = self.commit(&mut st, value & 0x3f);
            st.control = (value & 0xc0) | if valid { value & 0x3f } else { old & 0x3f };
            if valid {
                self.save_clock(&mut st);
            }
            return;
        }
        // The control rules preserve the century while W or R is set.
        st.control = (value & 0xc0) | (old & 0x3f);
    }
}

fn unix_now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

impl BusDevice for Ds1744 {
    fn read8(&self, addr: u64) -> BusRead8 {
        match usize::try_from(addr) {
            Ok(a) if a < SIZE => BusRead8 { status: BUS_OK, data: self.read(a) },
            _ => BusRead8::err(),
        }
    }

    fn write8(&self, addr: u64, val: u8) -> u32 {
        match usize::try_from(addr) {
            Ok(a) if a < SIZE => {
                self.write(a, val);
                BUS_OK
            }
            _ => BUS_ERR,
        }
    }
}

impl Drop for Ds1744 {
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(test)]
#[path = "ds1744_tests.rs"]
mod tests;
