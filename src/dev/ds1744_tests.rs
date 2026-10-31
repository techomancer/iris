//! DS1744 register-contract tests, through the byte bus with a clock the
//! test steps.
//!
//! Ported from tests/qtest/ds1742-test.c in jamesbraid/qemu-sgi-origami.

use super::*;
use std::sync::atomic::{AtomicI64, Ordering};

const SECOND: i64 = 1_000_000_000;

fn unix(y: i64, mo: i64, d: i64, h: i64, mi: i64, s: i64) -> i64 {
    days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + s
}

struct Rig {
    d: Ds1744,
    ns: Arc<AtomicI64>,
}

impl Rig {
    fn at(host: i64, base: YearBase) -> Rig {
        let ns = Arc::new(AtomicI64::new(0));
        let n = ns.clone();
        let d = Ds1744::with_clock(host, base, Arc::new(move || n.load(Ordering::SeqCst))).unwrap();
        Rig { d, ns }
    }

    /// The qtest's default: January 1st 2024, on the host's calendar.
    fn new() -> Rig {
        Rig::at(unix(2024, 1, 1, 0, 0, 0), YearBase::Gregorian)
    }

    fn step(&self, ns: i64) {
        self.ns.fetch_add(ns, Ordering::SeqCst);
    }

    fn rd(&self, a: usize) -> u8 {
        let r = self.d.read8(a as u64);
        assert_eq!(r.status, BUS_OK);
        r.data
    }

    fn wr(&self, a: usize, v: u8) {
        assert_eq!(self.d.write8(a as u64, v), BUS_OK);
    }

    /// Set the calendar through the write bit, with a weekday.
    #[allow(clippy::too_many_arguments)]
    fn set_time_weekday(&self, sec: u8, min: u8, hour: u8, date: u8, month: u8, year: u8, century: u8, weekday: u8) {
        self.wr(CONTROL, 0x80);
        for (i, v) in [sec, min, hour, weekday, date, month, year].into_iter().enumerate() {
            self.wr(CONTROL + 1 + i, v);
        }
        self.wr(CONTROL, century);
    }

    #[allow(clippy::too_many_arguments)]
    fn set_time(&self, sec: u8, min: u8, hour: u8, date: u8, month: u8, year: u8, century: u8) {
        self.set_time_weekday(sec, min, hour, date, month, year, century, 0x02);
    }

    /// The guest-visible calendar: control, seconds, minutes, hours, weekday,
    /// date, month, year.
    fn calendar(&self) -> [u8; 8] {
        std::array::from_fn(|i| self.rd(CONTROL + i))
    }
}

/// Ports test_fresh_seed: with the 1968 base the guest's year digits count
/// from 1968 and the weekday follows the host's calendar, across a day.
#[test]
fn a_fresh_seed_counts_years_from_1968() {
    let r = Rig::at(unix(2024, 2, 28, 23, 59, 59), YearBase::From(1968));
    assert_eq!(r.rd(CONTROL), 0x20);
    assert_eq!((r.rd(CONTROL + 7), r.rd(CONTROL + 6), r.rd(CONTROL + 5)), (0x56, 0x02, 0x28));
    assert_eq!(r.rd(CONTROL + 4) & 7, 4);
    r.step(2 * SECOND);
    assert_eq!(r.rd(CONTROL) & 0x3f, 0x20);
    assert_eq!((r.rd(CONTROL + 7), r.rd(CONTROL + 6), r.rd(CONTROL + 5)), (0x56, 0x02, 0x29));
    assert_eq!(r.rd(CONTROL + 1), 0x01);
    assert_eq!(r.rd(CONTROL + 4) & 7, 5);
    r.step(86_400 * SECOND);
    assert_eq!((r.rd(CONTROL + 7), r.rd(CONTROL + 6), r.rd(CONTROL + 5)), (0x56, 0x03, 0x01));
    assert_eq!(r.rd(CONTROL + 4) & 7, 6);
}

/// Ports test_fresh_seed_default: with no base the guest sees the host's year.
#[test]
fn without_a_base_the_guest_sees_the_host_year() {
    let r = Rig::at(unix(2024, 2, 28, 23, 59, 59), YearBase::Gregorian);
    assert_eq!(r.rd(CONTROL), 0x20);
    assert_eq!((r.rd(CONTROL + 7), r.rd(CONTROL + 6), r.rd(CONTROL + 5)), (0x24, 0x02, 0x28));
    assert_eq!(r.rd(CONTROL + 4) & 7, 4);
}

/// Ports test_fresh_seed_year_boundary: 2067 maps to 2099, the century
/// register steps to 21 and the shifted 2100 is not a leap year.
#[test]
fn the_shifted_century_rolls_and_2100_is_not_a_leap_year() {
    let r = Rig::at(unix(2067, 12, 31, 23, 59, 59), YearBase::From(1968));
    assert_eq!(r.rd(CONTROL), 0x20);
    assert_eq!((r.rd(CONTROL + 7), r.rd(CONTROL + 6), r.rd(CONTROL + 5)), (0x99, 0x12, 0x31));
    assert_eq!(r.rd(CONTROL + 4) & 7, 7);
    r.step(SECOND);
    assert_eq!(r.rd(CONTROL), 0x21);
    assert_eq!((r.rd(CONTROL + 7), r.rd(CONTROL + 6), r.rd(CONTROL + 5)), (0x00, 0x01, 0x01));
    assert_eq!(r.rd(CONTROL + 4) & 7, 1);
    r.step((58 * 86_400 + 86_399) * SECOND);
    assert_eq!((r.rd(CONTROL), r.rd(CONTROL + 7), r.rd(CONTROL + 6), r.rd(CONTROL + 5)), (0x21, 0x00, 0x02, 0x28));
    assert_eq!(r.rd(CONTROL + 4) & 7, 3);
    r.step(SECOND);
    assert_eq!((r.rd(CONTROL + 6), r.rd(CONTROL + 5)), (0x03, 0x01));
    assert_eq!(r.rd(CONTROL + 4) & 7, 4);
}

/// Ports test_ds1744_geometry: RAM below the calendar, including its last
/// byte, and an address past the part faults.
#[test]
fn ram_fills_the_part_below_the_calendar() {
    let r = Rig::new();
    for (a, v) in [(0, 0x12), (0x800, 0x34), (0x7ff7, 0x56)] {
        r.wr(a, v);
    }
    assert_eq!((r.rd(0), r.rd(0x800), r.rd(0x7ff7)), (0x12, 0x34, 0x56));
    r.set_time(0x12, 0x23, 0x10, 0x25, 0x09, 0x26, 0x20);
    assert_eq!(r.rd(CONTROL), 0x20);
    assert_eq!((r.rd(CONTROL + 1), r.rd(CONTROL + 2), r.rd(CONTROL + 3)), (0x12, 0x23, 0x10));
    assert_eq!((r.rd(CONTROL + 5), r.rd(CONTROL + 6), r.rd(CONTROL + 7)), (0x25, 0x09, 0x26));
    assert_eq!(r.rd(CONTROL + 4) & 7, 2);
    assert_eq!(r.d.read8(SIZE as u64).status, BUS_ERR);
    assert_eq!(r.d.write8(SIZE as u64, 1), BUS_ERR);
}

/// Ports test_calendar: the control register shows the century beside the
/// write and read bits, the battery flag and weekday advance, and a
/// century carries.
#[test]
fn control_shows_the_century_and_the_calendar_advances() {
    let r = Rig::new();
    assert_eq!(r.rd(CONTROL), 0x20);
    r.wr(CONTROL, 0x40);
    assert_eq!(r.rd(CONTROL), 0x60);
    r.wr(CONTROL, 0x00);
    assert_eq!(r.rd(CONTROL), 0x20);
    r.wr(CONTROL, 0x80);
    assert_eq!(r.rd(CONTROL), 0xa0);
    r.wr(CONTROL, 0x20);
    r.set_time(0x59, 0x59, 0x23, 0x31, 0x12, 0x24, 0x20);
    assert_eq!(r.rd(CONTROL + 1), 0x59);
    assert_eq!(r.rd(CONTROL + 4) & 0x80, 0x80, "battery flag");
    assert_eq!(r.rd(CONTROL + 4) & 7, 2);
    r.step(SECOND);
    assert_eq!(r.rd(CONTROL + 4) & 7, 3, "the weekday advances with the date");
    assert_eq!((r.rd(CONTROL + 1), r.rd(CONTROL + 5), r.rd(CONTROL + 6), r.rd(CONTROL + 7)), (0, 1, 1, 0x25));
    r.set_time(0x59, 0x59, 0x23, 0x31, 0x12, 0x99, 0x20);
    r.step(SECOND);
    assert_eq!(r.rd(CONTROL) & 0x3f, 0x21);
    assert_eq!(r.rd(CONTROL + 7), 0);
    // A commit that is a date is taken, with a different century.
    r.wr(CONTROL, 0x80);
    r.wr(CONTROL + 6, 0);
    r.wr(CONTROL, 0x19);
    assert_eq!(r.rd(CONTROL), 0x21, "month 0 is no date, so the century stays");
    assert_eq!(r.rd(CONTROL + 6), 1);
    // The read bit freezes what the guest sees while the clock runs.
    r.wr(CONTROL, 0x40);
    r.step(2 * SECOND);
    assert_eq!(r.rd(CONTROL + 1), 0);
    r.wr(CONTROL, 0);
    assert_eq!(r.rd(CONTROL + 1), 2);
}

/// Ports test_invalid_dates: a commit that is not a date leaves the calendar
/// alone, a valid leap day is taken, and the stop bit rides in the seconds.
#[test]
fn a_commit_that_is_not_a_date_is_refused() {
    let r = Rig::new();
    r.set_time(0x80 | 0x12, 0x34, 0x05, 0x15, 0x01, 0x24, 0x20);
    for (date, month, year) in [(0x30, 0x02, 0x24), (0x29, 0x02, 0x23)] {
        r.set_time(0x59, 0x59, 0x23, date, month, year, 0x20);
        assert_eq!(r.rd(CONTROL + 1), 0x92, "stopped at the earlier time");
        assert_eq!((r.rd(CONTROL + 5), r.rd(CONTROL + 6), r.rd(CONTROL + 7)), (0x15, 0x01, 0x24));
    }
    r.set_time(0x59, 0x59, 0x23, 0x29, 0x02, 0x24, 0x20);
    r.step(SECOND);
    assert_eq!((r.rd(CONTROL + 1), r.rd(CONTROL + 5), r.rd(CONTROL + 6), r.rd(CONTROL + 7)), (0, 1, 3, 0x24));
}

/// Ports test_stop_and_test_mode: a stopped oscillator holds the seconds,
/// the battery flag is read-only, and the test-mode bit flips the seconds'
/// low bit every 976562 ns unless the oscillator is stopped.
#[test]
fn test_mode_flips_the_low_second_bit() {
    let r = Rig::new();
    r.set_time(0x80 | 0x12, 0x34, 0x05, 0x06, 0x07, 0x26, 0x20);
    r.step(3 * SECOND);
    assert_eq!(r.rd(CONTROL + 1), 0x92);
    r.wr(CONTROL, 0x80);
    r.wr(CONTROL + 1, 0x12);
    r.wr(CONTROL + 4, 0xc2);
    r.wr(CONTROL, 0x20);
    assert_eq!(r.rd(CONTROL + 4), 0xc2);
    assert_eq!(r.rd(CONTROL + 1) & 0xfe, 0x12);
    let bit = r.rd(CONTROL + 1) & 1;
    r.step(976_562);
    assert_eq!(r.rd(CONTROL + 1) & 1, bit ^ 1);
    r.wr(CONTROL, 0x80);
    r.wr(CONTROL + 4, 0x02);
    r.wr(CONTROL, 0x20);
    let bit = r.rd(CONTROL + 1) & 1;
    r.step(976_562);
    assert_eq!(r.rd(CONTROL + 1) & 1, bit, "test mode off");
    r.wr(CONTROL, 0x80);
    r.wr(CONTROL + 4, 0x42);
    r.wr(CONTROL + 1, 0x92);
    r.wr(CONTROL, 0x20);
    let bit = r.rd(CONTROL + 1) & 1;
    r.step(3 * 976_562);
    assert_eq!(r.rd(CONTROL + 1) & 1, bit, "stopped, so it holds");
}

/// Ports test_restart_persistence: RAM and a completed calendar come back
/// from the file, the elapsed host time passes, a stopped clock does not
/// advance, and a file of the wrong size is refused.
#[test]
fn a_backing_file_restores_ram_and_the_calendar() {
    let dir = std::env::temp_dir().join(format!("ds1744-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("clock");
    {
        let r = Rig::new();
        r.d.set_nvram_file(&path).unwrap();
        r.wr(0x100, 0xa5);
        r.set_time(0x30, 0x15, 0x08, 0x15, 0x06, 0x30, 0x20);
        r.d.flush();
    }
    let reopened = Rig::at(unix(2024, 1, 1, 0, 0, 0), YearBase::Gregorian);
    reopened.d.set_nvram_file(&path).unwrap();
    assert_eq!(reopened.rd(0x100), 0xa5);
    let c = reopened.calendar();
    assert_eq!((c[2], c[3], c[5], c[6], c[7]), (0x15, 0x08, 0x15, 0x06, 0x30));
    assert_eq!(c[0], 0x20);

    let short = dir.join("short");
    std::fs::write(&short, [0u8; 16]).unwrap();
    assert!(Rig::new().d.set_nvram_file(&short).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// One step of a register sequence: write, read and expect, or let time pass.
enum Step {
    W(usize, u8),
    R(usize, u8),
    Advance(i64),
}

/// The register sequences QEMU's ds1742-test asserts, as one table so the
/// two models can be held to the same bytes. Offsets are from the 32 KiB
/// part's calendar (`CONTROL`); the seed is 2024-01-01T00:00:00.
///
/// Ports test_calendar, test_invalid_dates and test_stop_and_test_mode
/// from tests/qtest/ds1742-test.c in jamesbraid/qemu-sgi-origami. The
/// model's answers come from the QEMU file's assertions, not from running
/// QEMU here; the end-to-end comparison with a live QEMU is not done.
#[test]
fn the_qemu_register_sequences_read_back_the_same() {
    use Step::{Advance, R, W};
    const C: usize = CONTROL;
    let set = |s, mi, h, wd, d, mo, y, cent| {
        vec![W(C, 0x80), W(C + 1, s), W(C + 2, mi), W(C + 3, h), W(C + 4, wd), W(C + 5, d), W(C + 6, mo), W(C + 7, y), W(C, cent)]
    };
    let mut steps = vec![R(C, 0x20), W(C, 0x40), R(C, 0x60), W(C, 0x00), R(C, 0x20), W(C, 0x80), R(C, 0xa0), W(C, 0x20)];
    // A rollover into 2025.
    steps.extend(set(0x59, 0x59, 0x23, 0x02, 0x31, 0x12, 0x24, 0x20));
    steps.extend([R(C + 1, 0x59), R(C + 4, 0x82), Advance(1_000_000_000)]);
    steps.extend([R(C + 4, 0x83), R(C + 1, 0x00), R(C + 5, 0x01), R(C + 6, 0x01), R(C + 7, 0x25)]);
    // A century carry.
    steps.extend(set(0x59, 0x59, 0x23, 0x02, 0x31, 0x12, 0x99, 0x20));
    steps.extend([Advance(1_000_000_000), R(C, 0x21), R(C + 7, 0x00)]);
    // An impossible month leaves the calendar and its century alone.
    steps.extend([W(C, 0x80), W(C + 6, 0x00), W(C, 0x19), R(C, 0x21), R(C + 6, 0x01)]);
    // The read bit freezes the display; the clock runs on.
    steps.extend([W(C, 0x40), Advance(2_000_000_000), R(C + 1, 0x00), W(C, 0x00), R(C + 1, 0x02)]);
    // An impossible date is refused; a leap day is taken.
    steps.extend(set(0x92, 0x34, 0x05, 0x02, 0x15, 0x01, 0x24, 0x20));
    steps.extend(set(0x59, 0x59, 0x23, 0x02, 0x30, 0x02, 0x24, 0x20));
    steps.extend([R(C + 1, 0x92), R(C + 5, 0x15), R(C + 6, 0x01), R(C + 7, 0x24)]);
    steps.extend(set(0x59, 0x59, 0x23, 0x02, 0x29, 0x02, 0x23, 0x20));
    steps.extend([R(C + 1, 0x92), R(C + 5, 0x15)]);
    steps.extend(set(0x59, 0x59, 0x23, 0x02, 0x29, 0x02, 0x24, 0x20));
    steps.extend([Advance(1_000_000_000), R(C + 1, 0x00), R(C + 5, 0x01), R(C + 6, 0x03), R(C + 7, 0x24)]);
    // The stop bit holds the seconds; test mode makes the low bit run.
    steps.extend(set(0x80 | 0x12, 0x34, 0x05, 0x02, 0x06, 0x07, 0x26, 0x20));
    steps.extend([Advance(3_000_000_000), R(C + 1, 0x92)]);
    steps.extend([W(C, 0x80), W(C + 1, 0x12), W(C + 4, 0xc2), W(C, 0x20), R(C + 4, 0xc2)]);

    let r = Rig::new();
    for (i, step) in steps.iter().enumerate() {
        match *step {
            W(a, v) => r.wr(a, v),
            R(a, want) => assert_eq!(r.rd(a), want, "step {i}: read of {a:#x}"),
            Advance(ns) => r.step(ns),
        }
    }
}
