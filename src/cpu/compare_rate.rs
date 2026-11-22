//! The clock-tick rate a guest programs through CP0 Compare, for the status
//! bar's tick light (`status::CpuStatus::tick_rate`).
//!
//! A MIPS guest has no rate register to read: it writes each next deadline
//! into Compare. Two shapes are common, and IRIX uses both:
//!
//! - Compare advances by the interval and Count runs free (the O2).
//! - Count is zeroed at every tick and Compare set to the interval less the
//!   time already spent past the last deadline (the Octane, Origin, Fuel,
//!   Tezro and Origin 3000), so Compare alone is not the interval.
//!
//! Both are one timeline once Count writes are folded back in: every Count
//! write moves the timeline's origin by what it took off Count, measured
//! from the Count the guest last saw (its last read or write), so a deadline
//! is `Compare + epoch` and the interval is the distance between successive
//! deadlines. The distances are smoothed, so a machine that mixes timeouts
//! into its clock (the Octane) reads as its mean interrupt rate, which is
//! what the CPU's tick counter measures.
//!
//! All of this runs on Count reads and writes and Compare writes, never per
//! instruction, and writes the result to one atomic.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Fixed-point fraction bits of the smoothed interval.
const FRAC: u32 = 8;
/// Weight of a new interval in the average, as a shift (1/8).
const WEIGHT: u32 = 3;
/// A distance more than this many times off the average is an outlier.
const OUTLIER_RATIO: u64 = 4;
/// This many outliers in a row are a new rate, not noise: start again from
/// the latest.
const OUTLIERS_TO_RESTART: u8 = 4;

#[derive(Default)]
pub struct CompareTickRate {
    /// Where the rate is published, in millihertz. None: the machine counts
    /// its clock some other way (the Indy's 8254), and this does nothing.
    sink: Option<Arc<AtomicU64>>,
    /// The Count value the guest last read or wrote.
    basis: u32,
    /// What Count writes have taken off the timeline, in Count ticks.
    epoch: u32,
    /// The last deadline, on the timeline.
    last_deadline: Option<u32>,
    /// Smoothed distance between deadlines, Count ticks << FRAC; 0 until
    /// the first distance.
    interval: u64,
    /// Outliers in a row.
    outliers: u8,
}

impl CompareTickRate {
    /// Publish into `sink` from now on.
    pub fn set_sink(&mut self, sink: Arc<AtomicU64>) {
        self.sink = Some(sink);
    }

    /// Start again: a reset or a restored snapshot. `count` is Count now.
    /// Publishes 0 until the guest programs a rate again.
    pub fn reset(&mut self, count: u32) {
        self.basis = count;
        self.epoch = 0;
        self.last_deadline = None;
        self.interval = 0;
        self.outliers = 0;
        if let Some(s) = &self.sink {
            s.store(0, Ordering::Relaxed);
        }
    }

    /// The guest read Count.
    #[inline]
    pub fn count_read(&mut self, count: u32) {
        self.basis = count;
    }

    /// The guest wrote Count.
    pub fn count_write(&mut self, value: u32) {
        self.epoch = self.epoch.wrapping_add(self.basis.wrapping_sub(value));
        self.basis = value;
    }

    /// The guest wrote a Compare value it means as a deadline (not an
    /// acknowledgement or a parking value). `count_hz` is Count's rate.
    pub fn deadline(&mut self, compare: u32, count_hz: u64) {
        let Some(sink) = &self.sink else { return };
        let deadline = compare.wrapping_add(self.epoch);
        let Some(prev) = self.last_deadline.replace(deadline) else { return };
        // A deadline at or before the last one pulls a pending interrupt in
        // (a timeout sooner than the next tick): it replaces, not follows.
        let d = deadline.wrapping_sub(prev) as i32;
        if d <= 0 {
            return;
        }
        let sample = (d as u64) << FRAC;
        if self.interval == 0 {
            self.interval = sample;
        } else if sample > self.interval * OUTLIER_RATIO || sample * OUTLIER_RATIO < self.interval {
            self.outliers += 1;
            if self.outliers < OUTLIERS_TO_RESTART {
                return;
            }
            self.interval = sample;
            self.outliers = 0;
        } else {
            self.outliers = 0;
            self.interval = self.interval - (self.interval >> WEIGHT) + (sample >> WEIGHT);
        }
        sink.store(rate_mhz(count_hz, self.interval), Ordering::Relaxed);
    }
}

/// Millihertz of a deadline every `interval` (Count ticks << FRAC) at
/// `count_hz`.
fn rate_mhz(count_hz: u64, interval: u64) -> u64 {
    if interval == 0 {
        return 0;
    }
    ((count_hz as u128 * 1000 * (1u128 << FRAC) + interval as u128 / 2) / interval as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rate() -> (CompareTickRate, Arc<AtomicU64>) {
        let sink = Arc::new(AtomicU64::new(0));
        let mut r = CompareTickRate::default();
        r.set_sink(sink.clone());
        (r, sink)
    }

    const HZ: u64 = 90_000_000;

    /// Compare advanced by the interval, Count free-running (the O2).
    #[test]
    fn compare_plus_interval_publishes_the_rate() {
        let (mut r, sink) = rate();
        let interval = (HZ / 100) as u32;
        let mut compare = 0x1234_5678u32;
        for _ in 0..10 {
            r.count_read(compare.wrapping_sub(interval / 3));
            compare = compare.wrapping_add(interval);
            r.deadline(compare, HZ);
        }
        assert_eq!(sink.load(Ordering::Relaxed), 100_000);
    }

    /// Count zeroed at each tick, Compare the interval less the overrun (the
    /// SN machines' and the Octane's reset of the counter).
    #[test]
    fn count_reset_with_overrun_publishes_the_interval() {
        let (mut r, sink) = rate();
        let interval = (HZ / 100) as u32;
        let mut compare = interval;
        for late in [500u32, 7_000, 40, 123_456, 9, 2_000, 30_000, 1] {
            // The handler runs `late` ticks after the match, reads Count,
            // zeroes it and sets the next deadline less the overrun.
            r.count_read(compare + late);
            r.count_write(0);
            compare = interval - late;
            r.deadline(compare, HZ);
        }
        assert_eq!(sink.load(Ordering::Relaxed), 100_000);
    }

    /// A Count write the guest restores straight away cancels out.
    #[test]
    fn count_restored_is_no_move() {
        let (mut r, sink) = rate();
        let interval = 1_000_000u32;
        let mut compare = 5u32;
        for _ in 0..5 {
            r.count_read(compare + 10);
            r.count_write(0);
            r.count_write(compare + 10);
            compare = compare.wrapping_add(interval);
            r.deadline(compare, 1_000_000_000);
        }
        assert_eq!(sink.load(Ordering::Relaxed), 1_000_000);
    }

    /// An earlier deadline replaces the pending one without a sample; a new
    /// rate held for a few ticks takes over.
    #[test]
    fn pull_in_is_skipped_and_a_new_rate_takes_over() {
        let (mut r, sink) = rate();
        let mut c = 0u32;
        for _ in 0..5 {
            c += 10_000;
            r.deadline(c, 1_000_000);
        }
        assert_eq!(sink.load(Ordering::Relaxed), 100_000);
        r.deadline(c - 5_000, 1_000_000); // pulled in
        assert_eq!(sink.load(Ordering::Relaxed), 100_000);
        c -= 5_000;
        for _ in 0..OUTLIERS_TO_RESTART {
            c += 1_000;
            r.deadline(c, 1_000_000);
        }
        assert_eq!(sink.load(Ordering::Relaxed), 1_000_000, "1 kHz after a run of 1 ms deadlines");
    }

    /// Without a sink nothing is kept; a reset publishes 0.
    #[test]
    fn no_sink_and_reset() {
        let mut quiet = CompareTickRate::default();
        quiet.deadline(10, HZ);
        quiet.deadline(20, HZ);
        assert!(quiet.last_deadline.is_none());

        let (mut r, sink) = rate();
        r.deadline(1_000, 1_000);
        r.deadline(2_000, 1_000);
        assert_eq!(sink.load(Ordering::Relaxed), 1_000);
        r.reset(0);
        assert_eq!(sink.load(Ordering::Relaxed), 0);
    }
}
