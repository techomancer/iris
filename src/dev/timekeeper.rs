//! The Fuel M48T35 Timekeeper exposes 32 KiB of byte-wide NVRAM and calendar
//! registers at the end of that window. The guest reads the clock independently
//! of processor byte order, and software can stage clock changes through the
//! control register before committing them.
//!
//! # What is approximated here
//!
//! Host time and a persisted sidecar stand in for the battery-backed clock.
//! The model does not reproduce the physical oscillator or battery behavior.
//!
//! Ported from hw/rtc/m48t35.c in jamesbraid/qemu-sgi-origami.

use crate::traits::{BusDevice, BusRead8, Device, Resettable, BUS_ERR, BUS_OK};
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const SIZE: usize = 0x8000;
pub mod reg {
    pub const CONTROL: usize = 0x7ff8;
    pub const SECONDS: usize = 0x7ff9;
    pub const MINUTES: usize = 0x7ffa;
    pub const HOURS: usize = 0x7ffb;
    pub const CLOCK_WRITE: usize = 0x7ffc;
    pub const DAY: usize = 0x7ffd;
    pub const MONTH: usize = 0x7ffe;
    pub const YEAR: usize = 0x7fff;
}
pub const CONTROL: usize = reg::CONTROL;
const CENTURY: i64 = 36525 * 86400;
const MAGIC: &[u8; 8] = b"M48TREF1";
fn bcd(v: i64) -> u8 {
    ((v / 10) * 16 + v % 10) as u8
}
fn unbcd(v: u8) -> i64 {
    i64::from((v >> 4) * 10 + (v & 15))
}
fn month_days(y: i64, m: i64) -> i64 {
    match m {
        2 => 28 + i64::from(y % 4 == 0),
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}
#[derive(Clone, Copy)]
struct Calendar {
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: i64,
}
impl Calendar {
    fn seconds(self) -> i64 {
        let year = (self.year + (self.month - 1).div_euclid(12)).rem_euclid(100);
        let month = (self.month - 1).rem_euclid(12) + 1;
        let mut days = year * 365 + (year + 3) / 4 + self.day - 1;
        for m in 1..month {
            days += month_days(year, m);
        }
        (days * 86400 + self.hour * 3600 + self.minute * 60 + self.second).rem_euclid(CENTURY)
    }
    fn from_seconds(t: i64) -> Self {
        let t = t.rem_euclid(CENTURY);
        let mut days = t / 86400;
        let mut year = 0;
        while days >= 365 + i64::from(year % 4 == 0) {
            days -= 365 + i64::from(year % 4 == 0);
            year += 1;
        }
        let mut month = 1;
        while days >= month_days(year, month) {
            days -= month_days(year, month);
            month += 1;
        }
        Self {
            year,
            month,
            day: days + 1,
            hour: t / 3600 % 24,
            minute: t / 60 % 60,
            second: t % 60,
        }
    }
}
fn saved_calendar(image: &[u8]) -> Option<Calendar> {
    if image.len() != SIZE {
        return None;
    }
    let values = [
        image[0x7ff9] & 0x7f,
        image[0x7ffa],
        image[0x7ffb],
        image[0x7ffd],
        image[0x7ffe],
        image[0x7fff],
    ];
    let limits = [(0, 59), (0, 59), (0, 23), (1, 31), (1, 12), (0, 99)];
    let valid = values
        .iter()
        .zip(limits)
        .all(|(&v, (lo, hi))| v & 15 <= 9 && v >> 4 <= 9 && (lo..=hi).contains(&unbcd(v)));
    if !valid {
        return None;
    }
    Some(Calendar {
        second: unbcd(values[0]),
        minute: unbcd(values[1]),
        hour: unbcd(values[2]),
        day: unbcd(values[3]),
        month: unbcd(values[4]),
        year: unbcd(values[5]),
    })
}
struct State {
    cells: Vec<u8>,
    offset: i64,
    stopped: Option<i64>,
    frozen: Option<i64>,
    staged: Option<(Calendar, bool)>,
    file: Option<PathBuf>,
    #[cfg(test)]
    extra: i64,
}
impl State {
    fn running(&self, host: i64) -> i64 {
        self.stopped
            .unwrap_or_else(|| (host + self.offset).rem_euclid(CENTURY))
    }
    fn calendar(&self, host: i64) -> Calendar {
        Calendar::from_seconds(self.frozen.unwrap_or_else(|| self.running(host)))
    }
}

/// Fuel's 1968-based clock. Ordinary storage starts blank; the guest owns it.
/// The lower candidate clock addresses are independent RAM, with no aliases.
pub struct Timekeeper {
    initial_epoch: i64,
    st: Mutex<State>,
    clock: Arc<dyn Fn() -> i64 + Send + Sync>,
}
impl Timekeeper {
    pub fn fuel() -> Result<Self, String> {
        let host = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs() as i64;
        // Gregorian host date, used only to initialize the part's own calendar.
        let mut days = host / 86400;
        let mut year = 1970;
        loop {
            let n = 365 + i64::from(year % 4 == 0 && (year % 100 != 0 || year % 400 == 0));
            if days < n {
                break;
            }
            days -= n;
            year += 1;
        }
        let mut month = 1;
        while days >= month_days(year, month) {
            days -= month_days(year, month);
            month += 1;
        }
        let t = Self::fuel_at(
            year,
            month,
            days + 1,
            host / 3600 % 24,
            host / 60 % 60,
            host % 60,
        )?;
        Ok(t)
    }
    /// Open authoritative saved clock state before applying new-store year limits.
    pub fn fuel_with_nvram_file(path: &Path) -> Result<Self, String> {
        Self::backed(path, Self::fuel())
    }
    fn backed(path: &Path, initial: Result<Self, String>) -> Result<Self, String> {
        let saved = match std::fs::read(path) {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e.to_string()),
        };
        let clock = if saved_calendar(&saved).is_some() {
            // This seed is replaced by the authoritative image before exposure.
            Self::fuel_at(1968, 1, 1, 0, 0, 0)?
        } else {
            initial?
        };
        clock.set_nvram_file(path)?;
        Ok(clock)
    }
    /// A deterministic initial date; subsequent seconds follow monotonic host time.
    pub fn fuel_at(
        year: i64,
        month: i64,
        day: i64,
        hour: i64,
        minute: i64,
        second: i64,
    ) -> Result<Self, String> {
        if !(1968..=2067).contains(&year) {
            return Err("Timekeeper: initial year is outside 1968-2067".into());
        }
        if !(1..=12).contains(&month)
            || !(1..=month_days(year, month)).contains(&day)
            || !(0..24).contains(&hour)
            || !(0..60).contains(&minute)
            || !(0..60).contains(&second)
        {
            return Err("Timekeeper: invalid initial calendar".into());
        }
        let epoch = Calendar {
            year: year - 1968,
            month,
            day,
            hour,
            minute,
            second,
        }
        .seconds();
        let at = Instant::now();
        Ok(Self {
            initial_epoch: epoch,
            st: Mutex::new(State {
                cells: vec![0; SIZE],
                offset: epoch,
                stopped: None,
                frozen: None,
                staged: None,
                file: None,
                #[cfg(test)]
                extra: 0,
            }),
            clock: Arc::new(move || at.elapsed().as_secs() as i64),
        })
    }
    fn host(&self, st: &State) -> i64 {
        #[cfg(not(test))]
        let _ = st;
        (self.clock)() + {
            #[cfg(test)]
            {
                st.extra
            }
            #[cfg(not(test))]
            {
                0
            }
        }
    }
    #[cfg(test)]
    fn advance_for_test(&self, seconds: i64) {
        self.st.lock().extra += seconds;
    }

    fn image(st: &State, host: i64, saved_at: i64) -> (Vec<u8>, [u8; 16]) {
        let c = Calendar::from_seconds(st.running(host));
        let mut image = st.cells.clone();
        image[CONTROL + 1] = bcd(c.second) | if st.stopped.is_some() { 0x80 } else { 0 };
        image[CONTROL + 2] = bcd(c.minute);
        image[CONTROL + 3] = bcd(c.hour);
        image[CONTROL + 5] = bcd(c.day);
        image[CONTROL + 6] = bcd(c.month);
        image[CONTROL + 7] = bcd(c.year);
        let mut reference = [0; 16];
        reference[..8].copy_from_slice(MAGIC);
        reference[8..].copy_from_slice(&saved_at.to_le_bytes());
        (image, reference)
    }
    /// Export the exact raw image and companion clock reference, without changing RAM.
    pub fn persistence_image_at(&self, saved_at: i64) -> (Vec<u8>, [u8; 16]) {
        let st = self.st.lock();
        Self::image(&st, self.host(&st), saved_at)
    }
    /// Restore validated storage. Invalid/unwritten clock cells keep the initial clock.
    /// A missing/invalid reference adds no elapsed time, matching the model's policy.
    pub fn restore_image_at(
        &self,
        image: &[u8],
        reference: Option<&[u8]>,
        now: i64,
    ) -> Result<(), String> {
        if image.len() != SIZE {
            return Err(format!(
                "Timekeeper: backing image has {} bytes, expected {SIZE}",
                image.len()
            ));
        }
        if reference.is_some_and(|r| r.len() < 16) {
            return Err("Timekeeper: clock reference is shorter than 16 bytes".into());
        }
        let mut st = self.st.lock();
        st.cells.copy_from_slice(image);
        st.frozen = None;
        st.staged = None;
        st.stopped = None;
        if let Some(c) = saved_calendar(image) {
            let stopped = image[0x7ff9] & 0x80 != 0;
            let elapsed = reference.filter(|r| &r[..8] == MAGIC).map_or(0, |r| {
                now.saturating_sub(i64::from_le_bytes(r[8..16].try_into().unwrap()))
                    .max(0)
            });
            let epoch = (c.seconds()
                + if stopped {
                    0
                } else {
                    elapsed.rem_euclid(CENTURY)
                })
            .rem_euclid(CENTURY);
            st.offset = epoch - self.host(&st);
            if stopped {
                st.stopped = Some(epoch);
            }
        } else {
            st.offset = self.initial_epoch;
            crate::dlog!(crate::devlog::LogModule::Rtc, "Timekeeper: unwritten or invalid clock cells; following initial clock");
        }
        Ok(())
    }
    /// Bind a raw store and its `.clock` companion. Validate both before creating either.
    pub fn set_nvram_file(&self, path: &Path) -> Result<(), String> {
        let clock_path = PathBuf::from(format!("{}.clock", path.display()));
        let image = match std::fs::read(path) {
            Ok(v) => Some(v),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.to_string()),
        };
        if image.as_ref().is_some_and(|v| v.len() != SIZE) {
            return Err("Timekeeper: backing image must contain exactly 32768 bytes".into());
        }
        let reference = match std::fs::read(&clock_path) {
            Ok(v) => Some(v),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.to_string()),
        };
        if reference.as_ref().is_some_and(|v| v.len() < 16) {
            return Err("Timekeeper: clock reference is shorter than 16 bytes".into());
        }
        if let Some(image) = image.as_ref() {
            self.restore_image_at(image, reference.as_deref(), unix_now())?;
        }
        use std::io::Write;
        if image.is_none() {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .and_then(|mut f| f.write_all(&vec![0; SIZE]))
                .map_err(|e| e.to_string())?;
        }
        if reference.is_none() {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(clock_path)
                .and_then(|mut f| f.write_all(&[0; 16]))
                .map_err(|e| e.to_string())?;
        }
        self.st.lock().file = Some(path.to_path_buf());
        Ok(())
    }
    pub fn flush_nvram(&self) -> Result<(), String> {
        let st = self.st.lock();
        if let Some(path) = &st.file {
            let (image, reference) = Self::image(&st, self.host(&st), unix_now());
            std::fs::write(path, image).map_err(|e| e.to_string())?;
            use std::io::Write;
            std::fs::OpenOptions::new()
                .write(true)
                .open(format!("{}.clock", path.display()))
                .and_then(|mut f| f.write_all(&reference))
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}
impl BusDevice for Timekeeper {
    fn read8(&self, addr: u64) -> BusRead8 {
        let Ok(a) = usize::try_from(addr) else {
            return BusRead8::err();
        };
        if a >= SIZE {
            return BusRead8::err();
        }
        let st = self.st.lock();
        let c = st.calendar(self.host(&st));
        let data = match a {
            reg::SECONDS => bcd(c.second) | if st.stopped.is_some() { 0x80 } else { 0 },
            reg::MINUTES => bcd(c.minute),
            reg::HOURS => bcd(c.hour),
            reg::DAY => bcd(c.day),
            reg::MONTH => bcd(c.month),
            reg::YEAR => bcd(c.year),
            _ => st.cells[a],
        };
        BusRead8 {
            status: BUS_OK,
            data,
        }
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        let Ok(a) = usize::try_from(addr) else {
            return BUS_ERR;
        };
        if a >= SIZE {
            return BUS_ERR;
        }
        let mut st = self.st.lock();
        let host = self.host(&st);
        let old_control = st.cells[CONTROL];
        let transfer = a == CONTROL && st.cells[CONTROL] & 0x80 != 0 && val & 0x80 == 0;
        if a == CONTROL {
            let old = st.cells[a];
            if val & 0x40 != 0 && old & 0x40 == 0 {
                st.frozen = Some(st.running(host));
            } else if val & 0x40 == 0 {
                st.frozen = None;
            }
            if val & 0x80 != 0 && old & 0x80 == 0 {
                st.staged = Some((st.calendar(host), st.stopped.is_some()));
            } else if val & 0x80 == 0 && old & 0x80 != 0 {
                if let Some((c, stop)) = st.staged.take() {
                    let t = c.seconds();
                    st.offset = t - host;
                    st.stopped = if stop { Some(t) } else { None };
                }
            }
            st.cells[a] = val;
        } else if a > CONTROL {
            if a == reg::CLOCK_WRITE && st.staged.is_some() {
                st.cells[a] = val;
            } else if let Some((c, stop)) = &mut st.staged {
                match a {
                    reg::SECONDS => {
                        c.second = unbcd(val & 0x7f);
                        *stop = val & 0x80 != 0;
                    }
                    reg::MINUTES => c.minute = unbcd(val),
                    reg::HOURS => c.hour = unbcd(val),
                    reg::DAY => c.day = unbcd(val),
                    reg::MONTH => c.month = unbcd(val),
                    reg::YEAR => c.year = unbcd(val),
                    _ => {}
                }
            } else {
                crate::dlog!(crate::devlog::LogModule::Rtc, "Timekeeper: clock write at {a:#x} without write enable");
            }
        } else {
            st.cells[a] = val;
        }
        if !transfer
            && (a < CONTROL
                || a == CONTROL && (old_control ^ val) & !0xc0 != 0
                || a == reg::CLOCK_WRITE && st.staged.is_some())
        {
            // Ordinary storage writes must not save or replace the clock cells.
            if let Some(path) = &st.file {
                use std::io::{Seek, SeekFrom, Write};
                let result =
                    std::fs::OpenOptions::new()
                        .write(true)
                        .open(path)
                        .and_then(|mut f| {
                            f.seek(SeekFrom::Start(a as u64))?;
                            f.write_all(&[st.cells[a]])
                        });
                if let Err(e) = result {
                    log::error!("Timekeeper: backing write failed: {e}");
                }
            }
        }
        drop(st);
        if transfer {
            if let Err(e) = self.flush_nvram() {
                log::error!("Timekeeper: clock save failed: {e}");
            }
        }
        BUS_OK
    }
}
impl Resettable for Timekeeper {
    fn power_on(&self) {
        let mut st = self.st.lock();
        st.frozen = None;
        st.staged = None;
        st.cells[CONTROL] &= !0xc0;
    }
}
impl Device for Timekeeper {
    fn start(&self) {}
    fn step(&self, _cycles: u64) {}
    fn is_running(&self) -> bool {
        false
    }
    fn get_clock(&self) -> u64 {
        0
    }
    fn stop(&self) {
        if let Err(e) = self.flush_nvram() {
            log::error!("Timekeeper: saving failed: {e}");
        }
    }
}
impl Drop for Timekeeper {
    fn drop(&mut self) {
        Device::stop(self);
    }
}
#[cfg(test)]
mod tests {
    // Cases ported from tests/qtest/sn1-fuel-test.c in jamesbraid/qemu-sgi-origami.
    use super::*;
    use crate::traits::{BusDevice, Resettable};

    /// The guest can freeze, stop, stage, and resume the M48T35 calendar.
    #[test]
    fn clock_control_freezes_stops_and_resets_the_calendar() {
        let t = Timekeeper::fuel_at(2067, 12, 31, 23, 59, 59).unwrap();
        t.write8(0x7f8, 0x42);
        t.write8(CONTROL as u64, 0x40);
        t.advance_for_test(2);
        assert_eq!(t.read8(reg::YEAR as u64).data, 0x99);
        t.power_on();
        assert_eq!(t.read8(CONTROL as u64).data, 0);
        assert_eq!(t.read8(reg::YEAR as u64).data, 0);
        assert_eq!(t.read8(reg::MONTH as u64).data, 1);
        assert_eq!(t.read8(reg::DAY as u64).data, 1);
        assert_eq!(t.read8(0x7f8).data, 0x42);
        t.write8(CONTROL as u64, 0x80);
        t.write8(reg::SECONDS as u64, 0x89);
        t.write8(CONTROL as u64, 0);
        t.advance_for_test(60);
        assert_eq!(t.read8(reg::SECONDS as u64).data, 0x89);
    }

    /// A restored guest clock advances by host elapsed time unless its stop bit was saved.
    #[test]
    fn restored_clock_accounts_for_elapsed_time_unless_stopped() {
        for stopped in [false, true] {
            let t = Timekeeper::fuel_at(2024, 2, 28, 23, 59, 59).unwrap();
            t.write8(0x7ff0, 0xa5);
            if stopped {
                t.write8(CONTROL as u64, 0x80);
                t.write8(reg::SECONDS as u64, 0x85);
                t.write8(CONTROL as u64, 0);
            }
            let (image, reference) = t.persistence_image_at(100);
            let restored = Timekeeper::fuel_at(2026, 1, 1, 0, 0, 0).unwrap();
            restored.restore_image_at(&image, Some(&reference), 102).unwrap();
            assert_eq!(restored.read8(reg::DAY as u64).data, if stopped { 0x28 } else { 0x29 });
            assert_eq!(restored.read8(reg::SECONDS as u64).data, if stopped { 0x85 } else { 1 });
            assert_eq!(restored.read8(0x7ff0).data, 0xa5);
            assert!(restored.restore_image_at(&image[..2048], None, 102).is_err());
        }
    }

    /// NVRAM persists across emulator restarts, and malformed backing images are rejected.
    #[test]
    fn backed_nvram_preserves_saved_state_and_rejects_invalid_images() {
        let dir = std::env::current_exe().unwrap().parent().unwrap()
            .join(format!("timekeeper-test-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("nvram");
        let t = Timekeeper::fuel_at(2026, 1, 1, 0, 0, 0).unwrap();
        t.set_nvram_file(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), vec![0; SIZE]);
        t.write8(0x7f8, 0x42);
        assert_eq!(std::fs::read(&path).unwrap()[0x7f8], 0x42);
        t.flush_nvram().unwrap();
        drop(t);
        // Saved clock state is authoritative even if the host date is out of range.
        let restored = Timekeeper::backed(&path, Timekeeper::fuel_at(2068, 1, 1, 0, 0, 0)).unwrap();
        restored.power_on();
        assert_eq!(restored.read8(0x7f8).data, 0x42);
        assert_eq!(restored.read8(reg::YEAR as u64).data, 0x58);
        drop(restored);
        std::fs::remove_file(dir.join("nvram.clock")).unwrap();
        std::fs::write(&path, [0xa5; 2048]).unwrap();
        assert!(Timekeeper::backed(&path, Timekeeper::fuel_at(2026, 1, 1, 0, 0, 0)).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), vec![0xa5; 2048]);
        assert!(!dir.join("nvram.clock").exists());
        std::fs::remove_file(&path).unwrap();
        assert!(Timekeeper::backed(&path, Timekeeper::fuel_at(2068, 1, 1, 0, 0, 0)).is_err());
        assert!(!path.exists());
        std::fs::remove_dir(dir).unwrap();
    }

    const WEEKDAY: u64 = reg::CLOCK_WRITE as u64;

    fn clock_cells(t: &Timekeeper) -> [u8; 7] {
        std::array::from_fn(|i| t.read8(reg::SECONDS as u64 + i as u64).data)
    }

    fn set_clock(t: &Timekeeper, cells: [u8; 7]) {
        t.write8(CONTROL as u64, 0x80);
        for (i, v) in cells.into_iter().enumerate() {
            t.write8(reg::SECONDS as u64 + i as u64, v);
        }
        t.write8(CONTROL as u64, 0);
    }

    // Cells are seconds, minutes, hours, weekday, day, month, year, all BCD.

    /// The PROM and kernel derive file timestamps from this calendar, so
    /// 2024 must have a 29 February and 2026 must not.
    #[test]
    fn the_calendar_rolls_over_february_in_leap_years_only() {
        let t = Timekeeper::fuel_at(2026, 1, 1, 0, 0, 0).unwrap();
        set_clock(&t, [0x59, 0x59, 0x23, 0x05, 0x28, 0x02, 0x56]);
        t.advance_for_test(1);
        assert_eq!(clock_cells(&t), [0, 0, 0, 0x05, 0x29, 0x02, 0x56]);
        set_clock(&t, [0x59, 0x59, 0x23, 0x05, 0x28, 0x02, 0x58]);
        t.advance_for_test(1);
        assert_eq!(clock_cells(&t), [0, 0, 0, 0x05, 0x01, 0x03, 0x58]);
    }

    /// The part counts a two-digit year from 1968, so the calendar wraps after
    /// 2067; the guest must see year 00 again, not garbage.
    #[test]
    fn the_year_wraps_from_2067_to_1968() {
        let t = Timekeeper::fuel_at(2026, 1, 1, 0, 0, 0).unwrap();
        set_clock(&t, [0x59, 0x59, 0x23, 0x05, 0x31, 0x12, 0x99]);
        assert_eq!(t.read8(reg::YEAR as u64).data, 0x99);
        t.advance_for_test(1);
        assert_eq!(clock_cells(&t), [0, 0, 0, 0x05, 0x01, 0x01, 0]);
    }

    /// A guest reading the clock one byte at a time needs the read bit to
    /// freeze every counter at once, or a rollover tears the timestamp.
    #[test]
    fn the_read_bit_freezes_all_counters_until_it_is_cleared() {
        let t = Timekeeper::fuel_at(2026, 1, 1, 0, 0, 0).unwrap();
        let end_of_century = [0x59, 0x59, 0x23, 0x05, 0x31, 0x12, 0x99];
        set_clock(&t, end_of_century);
        t.write8(CONTROL as u64, 0x40);
        assert_eq!(t.read8(CONTROL as u64).data, 0x40);
        t.advance_for_test(2);
        assert_eq!(clock_cells(&t), end_of_century);
        t.write8(CONTROL as u64, 0);
        assert_eq!(clock_cells(&t), [0x01, 0, 0, 0x05, 0x01, 0x01, 0]);
    }

    /// Writes in the write-enable window are staged: the guest sets six
    /// counters and the weekday, and nothing but the weekday is visible until
    /// the window closes.
    #[test]
    fn staged_clock_writes_apply_only_when_the_write_bit_clears() {
        let t = Timekeeper::fuel_at(2026, 1, 2, 3, 4, 5).unwrap();
        let before = clock_cells(&t);
        t.write8(CONTROL as u64, 0x80);
        for (i, v) in [0x59u8, 0x58, 0x17, 0x03, 0x15, 0x06, 0x30].into_iter().enumerate() {
            t.write8(reg::SECONDS as u64 + i as u64, v);
        }
        let during = clock_cells(&t);
        for i in (0..7).filter(|&i| i != 3) {
            assert_eq!(during[i], before[i], "cell {i}");
        }
        assert_eq!(t.read8(WEEKDAY).data, 0x03);
        t.write8(CONTROL as u64, 0);
        assert_eq!(clock_cells(&t), [0x59, 0x58, 0x17, 0x03, 0x15, 0x06, 0x30]);
    }

    /// The stop bit is how a guest parks the oscillator while it sets the time;
    /// the clock must hold until the guest clears it, then continue from there.
    #[test]
    fn the_stop_bit_holds_the_seconds_until_it_is_cleared() {
        let t = Timekeeper::fuel_at(2026, 1, 1, 0, 0, 0).unwrap();
        t.write8(CONTROL as u64, 0x80);
        t.write8(reg::SECONDS as u64, 0x82);
        t.write8(CONTROL as u64, 0);
        t.advance_for_test(3);
        assert_eq!(t.read8(reg::SECONDS as u64).data, 0x82);
        t.write8(CONTROL as u64, 0x80);
        t.write8(reg::SECONDS as u64, 0x02);
        t.write8(CONTROL as u64, 0);
        t.advance_for_test(1);
        assert_eq!(t.read8(reg::SECONDS as u64).data, 0x03);
    }

    /// The 32 KiB window ends at the year register; an access past it must
    /// fault, and ordinary storage must keep its contents across clock writes.
    #[test]
    fn accesses_past_the_window_fault_and_ordinary_storage_is_untouched() {
        let t = Timekeeper::fuel_at(2026, 1, 1, 0, 0, 0).unwrap();
        t.write8(0, 0x5a);
        t.write8(0x7ff7, 0xa5);
        assert_eq!(t.write8(SIZE as u64, 0x69), BUS_ERR);
        assert_ne!(t.read8(SIZE as u64).status, BUS_OK);
        set_clock(&t, [0, 0, 0, 0x05, 0x01, 0x03, 0x58]);
        assert_eq!(t.read8(0).data, 0x5a);
        assert_eq!(t.read8(0x7ff7).data, 0xa5);
    }
}
