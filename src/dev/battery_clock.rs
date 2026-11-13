//! The clock of a battery-backed RTC from one iris run to the next.
//!
//! The battery keeps an RTC's oscillator running while the machine is off,
//! so a time the guest set is still there at the next power-up, moved on by
//! however long the machine was off. An RTC model keeps its chip's bytes in
//! the NVRAM file and the clock in a companion file, `<nvram>.clock`: an
//! 8-byte tag naming the part, the host's Unix time at the save, and, for a
//! part whose saved bytes do not give the running time, the guest's Unix
//! time then, both little-endian `i64`. This is the shape QEMU's DS1685 and
//! M48T35 models keep beside their images, so a DS1685 file pair moves
//! between the two emulators unchanged.
//!
//! A run that finds a reference adds the host time that passed since the
//! save. Without one the clock starts where `[rtc_offset]` puts it, host
//! time by default. A model writes a reference only once the guest has set
//! the clock, with a write that moves it (see [`sets`]), so until then an
//! edit to `[rtc_offset]` takes effect at the next run. Such a file pair,
//! with no reference, opens in QEMU at the time of its last save: QEMU then
//! takes the image's calendar without adding the time since.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::config::{format_unix_utc, RtcOffset};

/// A clock reference read back from its file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reference {
    /// Host Unix time of the save.
    pub saved_at: i64,
    /// Guest Unix time at the save, for the parts that store it.
    pub seconds: Option<i64>,
}

impl Reference {
    /// Host seconds since the save, never negative: a host clock that went
    /// backwards adds nothing.
    pub fn elapsed(&self, now: i64) -> i64 {
        now.saturating_sub(self.saved_at).max(0)
    }

    /// The guest's time now, with the time off added.
    pub fn guest_now(&self, now: i64) -> Option<i64> {
        self.seconds.map(|s| s.saturating_add(self.elapsed(now)))
    }
}

/// The companion file for the NVRAM file `nvram`.
pub fn companion(nvram: &Path) -> PathBuf {
    PathBuf::from(format!("{}.clock", nvram.display()))
}

/// A reference for the part `tag`, saved at host time `saved_at`.
pub fn encode(tag: &[u8; 8], saved_at: i64, seconds: Option<i64>) -> Vec<u8> {
    let mut r = tag.to_vec();
    r.extend_from_slice(&saved_at.to_le_bytes());
    if let Some(s) = seconds {
        r.extend_from_slice(&s.to_le_bytes());
    }
    r
}

/// The reference in `bytes` for the part `tag`, with the guest's time when
/// `seconds`; `None` for another part's file, a short one or zeros.
pub fn decode(bytes: &[u8], tag: &[u8; 8], seconds: bool) -> Option<Reference> {
    let word = |at: usize| bytes.get(at..at + 8).map(|b| i64::from_le_bytes(b.try_into().unwrap()));
    if bytes.get(..8) != Some(&tag[..]) {
        return None;
    }
    Some(Reference {
        saved_at: word(8)?,
        seconds: if seconds { Some(word(16)?) } else { None },
    })
}

/// Read a companion file; a missing one is `None`.
pub fn read(path: &Path) -> std::io::Result<Option<Vec<u8>>> {
    match std::fs::read(path) {
        Ok(v) => Ok(Some(v)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// [`read`], for a model that runs without the file: one that cannot be
/// read is reported and treated as missing, so the NVRAM file still loads.
pub fn read_or_warn(path: &Path) -> Option<Vec<u8>> {
    read(path).unwrap_or_else(|e| {
        eprintln!("RTC: clock file {}: {e}; starting the clock without it", path.display());
        None
    })
}

/// Replace a companion file in one step, as the NVRAM files are.
pub fn write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let name = path
        .to_str()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "clock path is not UTF-8"))?;
    crate::nv_storage::save(name, bytes)
}

/// Host Unix time in whole seconds.
pub fn unix_now() -> i64 {
    host_second().0
}

/// Host time in whole seconds since 1970, and the instant that second began.
/// A clock started from it ticks with the host's seconds, so a reference
/// saved from it and read back lands on the same second.
pub fn host_second() -> (i64, Instant) {
    let since = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let now = Instant::now();
    let into = Duration::from_nanos(u64::from(since.subsec_nanos()));
    (since.as_secs() as i64, now.checked_sub(into).unwrap_or(now))
}

/// How far a guest's write may move the clock and still not count as
/// setting it. Firmware raising and dropping SET to change the register
/// mode, or a kernel writing back the time it read, commits the time the
/// clock already had, give or take the second it took.
pub const SET_TOLERANCE: i64 = 2;

/// Whether a guest write that takes the clock from `before` to `after`
/// (guest seconds, both read off the model's own clock at the write) sets
/// it. Judged at the write, so a jump in host time never counts.
pub fn sets(before: i64, after: i64) -> bool {
    (after - before).abs() > SET_TOLERANCE
}

/// Where `[rtc_offset]` puts a clock with no reference, in seconds from
/// host time, clamped to the years the DS1386 holds.
pub fn seed(cfg: &RtcOffset) -> i64 {
    let host = unix_now();
    cfg.apply_clamped(host).0 - host
}

/// Say so when a saved clock overrides a configured `[rtc_offset]`, which
/// otherwise looks ignored.
pub fn note_override(cfg: &RtcOffset, file: &Path, guest_now: i64) {
    if !cfg.is_zero() {
        eprintln!(
            "RTC: {} keeps the clock the guest set ({} UTC); [rtc_offset] {} applies only without it, so delete that file to start from [rtc_offset] again",
            file.display(),
            format_unix_utc(guest_now),
            cfg.describe()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reference_round_trips_and_other_files_are_not_one() {
        let r = encode(b"DS168RF1", 1_794_300_000, Some(1_072_987_200));
        assert_eq!(r.len(), 24);
        let got = decode(&r, b"DS168RF1", true).unwrap();
        assert_eq!(got, Reference { saved_at: 1_794_300_000, seconds: Some(1_072_987_200) });
        assert_eq!(got.guest_now(1_794_300_060), Some(1_072_987_260));
        assert_eq!(got.guest_now(1_794_299_000), Some(1_072_987_200), "a host clock that went back adds nothing");
        assert_eq!(decode(&r, b"M48TREF1", false), None, "another part's tag");
        assert_eq!(decode(&r[..20], b"DS168RF1", true), None, "short");
        assert_eq!(decode(&[0; 24], b"DS168RF1", true), None, "zeros");
        let m = encode(b"M48TREF1", 5, None);
        assert_eq!(decode(&m, b"M48TREF1", false), Some(Reference { saved_at: 5, seconds: None }));
    }

    #[test]
    fn rtc_offset_seeds_the_clock_within_the_ds1386_years() {
        assert_eq!(seed(&RtcOffset::default()), 0);
        assert_eq!(seed(&RtcOffset { hours: -2, ..Default::default() }), -7200);
        let far = unix_now() + seed(&RtcOffset { years: 100, ..Default::default() });
        assert_eq!(far, crate::config::RTC_MAX_UNIX, "clamped");
    }
}
