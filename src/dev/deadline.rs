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
