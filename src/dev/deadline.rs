//! An optional deadline as one atomic word, so a timer can read it without
//! the lock of the device that holds it: nanoseconds after the process epoch,
//! plus one; zero for none.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

pub(crate) fn encode(t: Option<Instant>) -> u64 {
    t.map_or(0, |t| t.checked_duration_since(epoch()).map_or(1, |d| d.as_nanos() as u64 + 1))
}

pub(crate) fn decode(n: u64) -> Option<Instant> {
    (n != 0).then(|| epoch() + Duration::from_nanos(n - 1))
}

// A deadline in a snapshot is the time it had left. Encoded as nanoseconds
// left plus one, zero for none, so a restored guest waits out the same
// interval whatever the host's clock reads at the load.

pub(crate) fn until(due: Option<Instant>, now: Instant) -> u64 {
    due.map_or(0, |d| d.saturating_duration_since(now).as_nanos() as u64 + 1)
}

pub(crate) fn due(left: u64, now: Instant) -> Option<Instant> {
    (left != 0).then(|| now + Duration::from_nanos(left - 1))
}

pub(crate) fn span(d: Option<Duration>) -> u64 {
    d.map_or(0, |d| d.as_nanos() as u64 + 1)
}

pub(crate) fn unspan(n: u64) -> Option<Duration> {
    (n != 0).then(|| Duration::from_nanos(n - 1))
}
