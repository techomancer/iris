//! GPU report accounting, independent of adapter initialization.

use std::sync::atomic::{AtomicU64, Ordering};

const BATCH_HISTOGRAM_BUCKETS: usize = 17;
static NEXT_COUNTER_EPOCH: AtomicU64 = AtomicU64::new(1);

macro_rules! subtract_counters {
    ($after:ident, $before:ident, $($field:ident),+ $(,)?) => {
        $(
            $after.$field = $after.$field.checked_sub($before.$field).ok_or_else(|| {
                format!("{} counter decreased between snapshots", stringify!($field))
            })?;
        )+
    };
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum TimestampSupport {
    #[default]
    Disabled,
    Unsupported(&'static str),
    Unavailable(String),
    Enabled {
        readback_transfer: bool,
    },
}

#[derive(Clone, Debug, Default)]
pub struct ReplayReport {
    /// Identifies one uninterrupted series of cumulative counters.
    pub counter_epoch: u64,
    pub handled_draws: u64,
    pub fallback_draws: u64,
    pub fallback_reasons: std::collections::BTreeMap<String, u64>,
    pub submissions: u64,
    pub fences: u64,
    pub bytes_uploaded: u64,
    pub bytes_read_back: u64,
    pub pending_draws: usize,
    pub error: Option<String>,
    pub timestamp_error: Option<String>,
    pub timestamp_support: TimestampSupport,
    pub telemetry: Option<StageTelemetry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FlushReason {
    BatchFull,
    Idle,
    WaitIdle,
    DisplayFence,
    Exit,
    CpuFallback,
    Explicit,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FlushCounts {
    pub batch_full: u64,
    pub idle: u64,
    pub wait_idle: u64,
    pub display_fence: u64,
    pub exit: u64,
    pub cpu_fallback: u64,
    pub explicit: u64,
}

impl FlushCounts {
    pub(crate) fn record(&mut self, reason: FlushReason) {
        let count = match reason {
            FlushReason::BatchFull => &mut self.batch_full,
            FlushReason::Idle => &mut self.idle,
            FlushReason::WaitIdle => &mut self.wait_idle,
            FlushReason::DisplayFence => &mut self.display_fence,
            FlushReason::Exit => &mut self.exit,
            FlushReason::CpuFallback => &mut self.cpu_fallback,
            FlushReason::Explicit => &mut self.explicit,
        };
        *count += 1;
    }

    fn delta_since(&self, before: &Self) -> Result<Self, String> {
        let mut delta = self.clone();
        subtract_counters!(
            delta,
            before,
            batch_full,
            idle,
            wait_idle,
            display_fence,
            exit,
            cpu_fallback,
            explicit,
        );
        Ok(delta)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TimestampTotals {
    pub count: u64,
    pub nanoseconds: u64,
    /// Samples discarded because the device counter wrapped or went backwards.
    pub discarded: u64,
}

impl TimestampTotals {
    fn delta_since(&self, before: &Self) -> Result<Self, String> {
        let mut delta = self.clone();
        subtract_counters!(delta, before, count, nanoseconds, discarded);
        Ok(delta)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageTelemetry {
    pub match_attempts: u64,
    pub sampled_draws: u64,
    pub sampled_match_ns: u64,
    pub sampled_enqueue_ns: u64,
    pub upload_enqueue_ns: u64,
    pub resource_creation_ns: u64,
    pub command_encoding_ns: u64,
    pub queue_submit_ns: u64,
    pub device_poll_ns: u64,
    pub device_wait_ns: u64,
    pub map_request_ns: u64,
    pub map_callback_wait_ns: u64,
    pub cpu_readback_copy_ns: u64,
    pub timestamp_readback_ns: u64,
    pub timestamp_readback_bytes: u64,
    pub parameter_bytes: u64,
    pub batch_count: u64,
    pub batch_draws: u64,
    /// Bucket i contains batches of 2^i through 2^(i+1)-1 draws.
    pub batch_histogram: [u64; BATCH_HISTOGRAM_BUCKETS],
    pub flush_reasons: FlushCounts,
    pub gpu_compute: Option<TimestampTotals>,
    pub gpu_readback_transfer: Option<TimestampTotals>,
    pub timestamp_submissions_skipped: u64,
}

impl StageTelemetry {
    pub(crate) fn record_batch(&mut self, draws: usize, bytes_per_draw: usize) {
        if draws == 0 {
            return;
        }
        self.batch_count += 1;
        self.batch_draws += draws as u64;
        self.parameter_bytes += (draws * bytes_per_draw) as u64;
        let bucket = (usize::BITS - 1 - draws.leading_zeros()) as usize;
        self.batch_histogram[bucket.min(BATCH_HISTOGRAM_BUCKETS - 1)] += 1;
    }

    fn delta_since(&self, before: &Self) -> Result<Self, String> {
        let mut delta = self.clone();
        subtract_counters!(
            delta,
            before,
            match_attempts,
            sampled_draws,
            sampled_match_ns,
            sampled_enqueue_ns,
            upload_enqueue_ns,
            resource_creation_ns,
            command_encoding_ns,
            queue_submit_ns,
            device_poll_ns,
            device_wait_ns,
            map_request_ns,
            map_callback_wait_ns,
            cpu_readback_copy_ns,
            timestamp_readback_ns,
            timestamp_readback_bytes,
            parameter_bytes,
            batch_count,
            batch_draws,
            timestamp_submissions_skipped,
        );
        for (bucket, previous) in delta.batch_histogram.iter_mut().zip(before.batch_histogram) {
            *bucket = bucket
                .checked_sub(previous)
                .ok_or_else(|| "batch histogram decreased between snapshots".to_owned())?;
        }
        delta.flush_reasons = self.flush_reasons.delta_since(&before.flush_reasons)?;
        delta.gpu_compute = timestamp_delta(&self.gpu_compute, &before.gpu_compute)?;
        delta.gpu_readback_transfer =
            timestamp_delta(&self.gpu_readback_transfer, &before.gpu_readback_transfer)?;
        Ok(delta)
    }
}

fn timestamp_delta(
    after: &Option<TimestampTotals>,
    before: &Option<TimestampTotals>,
) -> Result<Option<TimestampTotals>, String> {
    match (after, before) {
        (Some(after), Some(before)) => Ok(Some(after.delta_since(before)?)),
        (None, None) => Ok(None),
        _ => Err("timestamp availability changed between snapshots".into()),
    }
}

impl ReplayReport {
    pub(crate) fn fresh() -> Self {
        Self {
            counter_epoch: NEXT_COUNTER_EPOCH.fetch_add(1, Ordering::Relaxed),
            ..Self::default()
        }
    }

    pub(crate) fn timestamp_collection_failed(&mut self, error: String) {
        self.timestamp_error = Some(error.clone());
        self.timestamp_support = TimestampSupport::Unavailable(error);
        if let Some(telemetry) = &mut self.telemetry {
            telemetry.gpu_compute = None;
            telemetry.gpu_readback_transfer = None;
        }
    }

    #[cfg(test)]
    pub(crate) fn advance_counter_epoch(&mut self) {
        self.counter_epoch = NEXT_COUNTER_EPOCH.fetch_add(1, Ordering::Relaxed);
    }

    pub fn delta_since(&self, before: &Self) -> Result<Self, String> {
        if self.counter_epoch != before.counter_epoch {
            return Err(
                "report counters were reset, toggled, or belong to different engines".into(),
            );
        }
        if self.timestamp_support != before.timestamp_support
            || self.timestamp_error != before.timestamp_error
        {
            return Err("timestamp support or error changed between snapshots".into());
        }
        let mut delta = self.clone();
        subtract_counters!(
            delta,
            before,
            handled_draws,
            fallback_draws,
            submissions,
            fences,
            bytes_uploaded,
            bytes_read_back,
        );
        for (reason, previous) in &before.fallback_reasons {
            let count = delta.fallback_reasons.get_mut(reason).ok_or_else(|| {
                format!("fallback reason {reason:?} disappeared between snapshots")
            })?;
            *count = count
                .checked_sub(*previous)
                .ok_or_else(|| format!("fallback reason {reason:?} decreased between snapshots"))?;
        }
        delta.fallback_reasons.retain(|_, count| *count != 0);
        delta.telemetry = match (&self.telemetry, &before.telemetry) {
            (Some(after), Some(before)) => Some(after.delta_since(before)?),
            (None, None) => None,
            _ => return Err("telemetry was reset or toggled between snapshots".into()),
        };
        Ok(delta)
    }
}

pub(crate) fn timestamp_duration_ns(
    bytes: &[u8],
    index: u32,
    period: f32,
) -> Result<Option<u64>, String> {
    if !period.is_finite() || period <= 0.0 {
        return Err(format!("invalid GPU timestamp period {period}"));
    }
    let offset = index as usize * std::mem::size_of::<u64>();
    let read = |offset: usize| -> Result<u64, String> {
        let bytes = bytes
            .get(offset..offset + 8)
            .ok_or_else(|| "incomplete timestamp readback".to_owned())?;
        Ok(u64::from_ne_bytes(bytes.try_into().unwrap()))
    };
    let start = read(offset)?;
    let end = read(offset + 8)?;
    let Some(ticks) = end.checked_sub(start) else {
        return Ok(None);
    };
    let nanos = ticks as f64 * f64::from(period);
    if !nanos.is_finite() || nanos >= u64::MAX as f64 {
        return Err(format!("invalid GPU timestamp duration {nanos} ns"));
    }
    Ok(Some(nanos as u64))
}

pub(crate) fn record_timestamp(
    total: &mut TimestampTotals,
    duration: Option<u64>,
) -> Result<(), String> {
    let Some(duration) = duration else {
        total.discarded = total
            .discarded
            .checked_add(1)
            .ok_or("discarded timestamp count overflow")?;
        return Ok(());
    };
    let count = total
        .count
        .checked_add(1)
        .ok_or("timestamp count overflow")?;
    let nanoseconds = total
        .nanoseconds
        .checked_add(duration)
        .ok_or("timestamp duration overflow")?;
    total.count = count;
    total.nanoseconds = nanoseconds;
    Ok(())
}

#[cfg(test)]
pub(crate) fn require_usable_timestamps(
    stage: &str,
    total: &TimestampTotals,
) -> Result<(), String> {
    if total.count != 0 {
        return Ok(());
    }
    if total.discarded != 0 {
        return Err(format!(
            "INCONCLUSIVE: all {} {stage} timestamp samples were discarded",
            total.discarded
        ));
    }
    Err(format!(
        "INCONCLUSIVE: {stage} timestamps produced no usable samples"
    ))
}

#[cfg(test)]
const LIVE_AA_SAMPLE_MS: &str = "REX3_GPU_AA_SAMPLE_MS";
#[cfg(test)]
const LIVE_AA_MAX_ROUNDS: &str = "REX3_GPU_AA_MAX_ROUNDS";
#[cfg(test)]
const LIVE_AA_MIN_ROUNDS: usize = 16;
#[cfg(test)]
const LIVE_AA_ROUND_BATCH: usize = 8;
#[cfg(test)]
const LIVE_AA_SIGN_ALPHA: f64 = 1394.0 / 65536.0;

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LiveAaOptions {
    pub(crate) sample_ms: u64,
    pub(crate) max_rounds: usize,
}

#[cfg(test)]
impl LiveAaOptions {
    pub(crate) fn from_env() -> Result<Self, String> {
        fn read(name: &str) -> Result<Option<String>, String> {
            match std::env::var(name) {
                Ok(value) => Ok(Some(value)),
                Err(std::env::VarError::NotPresent) => Ok(None),
                Err(std::env::VarError::NotUnicode(_)) => {
                    Err(format!("{name}: expected a Unicode integer"))
                }
            }
        }

        let sample_ms = read(LIVE_AA_SAMPLE_MS)?;
        let max_rounds = read(LIVE_AA_MAX_ROUNDS)?;
        Self::parse(sample_ms.as_deref(), max_rounds.as_deref())
    }

    fn parse(sample_ms: Option<&str>, max_rounds: Option<&str>) -> Result<Self, String> {
        fn integer(
            name: &str,
            value: Option<&str>,
            default: u64,
            min: u64,
            max: u64,
        ) -> Result<u64, String> {
            let Some(value) = value else {
                return Ok(default);
            };
            let invalid = || format!("{name}={value:?}: expected an integer in {min}..={max}");
            if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(invalid());
            }
            let parsed = value.parse::<u64>().map_err(|_| invalid())?;
            if !(min..=max).contains(&parsed) {
                return Err(invalid());
            }
            Ok(parsed)
        }

        let sample_ms = integer(LIVE_AA_SAMPLE_MS, sample_ms, 600, 100, 2000)?;
        let max_rounds = integer(LIVE_AA_MAX_ROUNDS, max_rounds, 48, 16, 128)? as usize;
        if max_rounds % LIVE_AA_ROUND_BATCH != 0 {
            return Err(format!(
                "{LIVE_AA_MAX_ROUNDS}={max_rounds}: expected a multiple of \
                 {LIVE_AA_ROUND_BATCH} in 16..=128"
            ));
        }
        Ok(Self {
            sample_ms,
            max_rounds,
        })
    }

    pub(crate) fn is_look(&self, completed: usize) -> bool {
        (LIVE_AA_MIN_ROUNDS..=self.max_rounds).contains(&completed)
            && (completed - LIVE_AA_MIN_ROUNDS) % LIVE_AA_ROUND_BATCH == 0
    }

    pub(crate) fn look_count(&self) -> usize {
        (self.max_rounds - LIVE_AA_MIN_ROUNDS) / LIVE_AA_ROUND_BATCH + 1
    }

    pub(crate) fn look_number(&self, completed: usize) -> usize {
        assert!(self.is_look(completed));
        (completed - LIVE_AA_MIN_ROUNDS) / LIVE_AA_ROUND_BATCH + 1
    }

    pub(crate) fn per_look_alpha(&self) -> f64 {
        LIVE_AA_SIGN_ALPHA / self.look_count() as f64
    }

    pub(crate) fn simultaneous_confidence(&self) -> f64 {
        1.0 - LIVE_AA_SIGN_ALPHA
    }
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct LiveGateStats {
    pub(crate) median: f64,
    pub(crate) coefficient_of_variation: f64,
    pub(crate) range_over_median: f64,
}

#[cfg(test)]
fn sorted_finite(values: &[f64]) -> Vec<f64> {
    assert!(!values.is_empty() && values.iter().all(|value| value.is_finite()));
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted
}

#[cfg(test)]
fn median_sorted(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

#[cfg(test)]
pub(crate) fn live_gate_stats(values: &[f64]) -> LiveGateStats {
    assert!(values.iter().all(|value| *value > 0.0));
    let sorted = sorted_finite(values);
    let median = median_sorted(&sorted);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    LiveGateStats {
        median,
        coefficient_of_variation: variance.sqrt() / mean,
        range_over_median: (sorted[sorted.len() - 1] - sorted[0]) / median,
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct PairedRateStats {
    /// Signed throughput change; negative means the second condition is slower.
    pub(crate) median: f64,
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    pub(crate) median_absolute: f64,
    pub(crate) half_shift: f64,
    pub(crate) confidence: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OverheadDecision {
    Pass,
    Fail,
    Inconclusive,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LiveAaStop {
    Continue,
    Pass,
    FailSlowdown,
    MaxRounds,
}

#[cfg(test)]
pub(crate) fn log_rate_ratio(first: f64, second: f64) -> f64 {
    assert!(first.is_finite() && first > 0.0 && second.is_finite() && second > 0.0);
    (second / first).ln()
}

#[cfg(test)]
pub(crate) fn abba_log_rate_ratio(rates: [f64; 4], times: [f64; 4], on_first: bool) -> f64 {
    assert!(rates.iter().all(|rate| rate.is_finite() && *rate > 0.0));
    assert!(times.iter().all(|time| time.is_finite()));
    assert!(times.windows(2).all(|pair| pair[0] < pair[1]));
    let inner_time = (times[1] + times[2]) / 2.0;
    let weight = (inner_time - times[0]) / (times[3] - times[0]);
    let outer = (1.0 - weight) * rates[0].ln() + weight * rates[3].ln();
    let inner = (rates[1].ln() + rates[2].ln()) / 2.0;
    if on_first {
        outer - inner
    } else {
        inner - outer
    }
}

#[cfg(test)]
pub(crate) fn paired_order_shift(log_ratios: &[f64], on_first: &[bool]) -> f64 {
    assert_eq!(log_ratios.len(), on_first.len());
    let mut first_on = Vec::new();
    let mut first_off = Vec::new();
    for (&ratio, &on) in log_ratios.iter().zip(on_first) {
        if on {
            first_on.push(ratio);
        } else {
            first_off.push(ratio);
        }
    }
    let on = median_sorted(&sorted_finite(&first_on));
    let off = median_sorted(&sorted_finite(&first_off));
    (on - off).abs().exp_m1()
}

#[cfg(test)]
pub(crate) fn paired_rate_stats(log_ratios: &[f64]) -> PairedRateStats {
    paired_rate_stats_with_alpha(log_ratios, 0.05)
}

#[cfg(test)]
pub(crate) fn paired_rate_stats_with_alpha(log_ratios: &[f64], alpha: f64) -> PairedRateStats {
    assert!(alpha.is_finite() && alpha > 0.0 && alpha < 1.0);
    let sorted = sorted_finite(log_ratios);
    let count = sorted.len();
    let mut probability = 2.0_f64.powf(-(count as f64));
    let mut tail = 0.0;
    let mut rank = None;
    for index in 0..count / 2 {
        let next = tail + probability;
        if next > alpha / 2.0 {
            break;
        }
        tail = next;
        rank = Some(index);
        probability *= (count - index) as f64 / (index + 1) as f64;
    }
    let rank = rank.expect("too few pairs for the requested median interval confidence");
    let absolute: Vec<_> = log_ratios.iter().map(|ratio| ratio.abs()).collect();
    let midpoint = count / 2;
    let first = median_sorted(&sorted_finite(&log_ratios[..midpoint]));
    let second = median_sorted(&sorted_finite(&log_ratios[midpoint..]));
    PairedRateStats {
        median: median_sorted(&sorted).exp_m1(),
        lower: sorted[rank].exp_m1(),
        upper: sorted[count - rank - 1].exp_m1(),
        median_absolute: median_sorted(&sorted_finite(&absolute)).exp_m1(),
        half_shift: (first - second).abs().exp_m1(),
        confidence: 1.0 - 2.0 * tail,
    }
}

#[cfg(test)]
pub(crate) fn overhead_decision(
    gpu_aa: PairedRateStats,
    telemetry: PairedRateStats,
    order_shift: f64,
    limit: f64,
) -> OverheadDecision {
    assert!(limit.is_finite() && limit > 0.0);
    assert!(order_shift.is_finite() && order_shift >= 0.0);
    let usable_gpu_control = gpu_aa.lower <= 0.0
        && gpu_aa.upper >= 0.0
        && gpu_aa.lower > -limit
        && gpu_aa.upper < limit
        && gpu_aa.median_absolute < limit
        && gpu_aa.half_shift < limit;
    if !usable_gpu_control || order_shift >= limit {
        return OverheadDecision::Inconclusive;
    }
    if telemetry.lower > -limit && telemetry.upper < limit {
        OverheadDecision::Pass
    } else if telemetry.upper <= -limit {
        OverheadDecision::Fail
    } else {
        OverheadDecision::Inconclusive
    }
}

#[cfg(test)]
pub(crate) fn live_aa_stop(
    options: &LiveAaOptions,
    completed: usize,
    gpu_aa: PairedRateStats,
    telemetry: PairedRateStats,
    order_shift: f64,
    limit: f64,
) -> LiveAaStop {
    assert!(options.is_look(completed));
    match overhead_decision(gpu_aa, telemetry, order_shift, limit) {
        OverheadDecision::Pass => LiveAaStop::Pass,
        OverheadDecision::Fail => LiveAaStop::FailSlowdown,
        OverheadDecision::Inconclusive if completed == options.max_rounds => LiveAaStop::MaxRounds,
        OverheadDecision::Inconclusive => LiveAaStop::Continue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_deltas_account_for_batches_reasons_and_timestamps() {
        let mut before = ReplayReport {
            handled_draws: 8,
            fallback_draws: 1,
            submissions: 1,
            telemetry: Some(StageTelemetry {
                gpu_compute: Some(TimestampTotals::default()),
                ..StageTelemetry::default()
            }),
            ..ReplayReport::default()
        };
        before.fallback_reasons.insert("opcode".into(), 1);
        {
            let telemetry = before.telemetry.as_mut().unwrap();
            telemetry.match_attempts = 9;
            telemetry.record_batch(8, 32);
            telemetry.flush_reasons.record(FlushReason::BatchFull);
        }

        let mut after = before.clone();
        after.handled_draws += 4;
        after.fallback_draws += 1;
        after.submissions += 2;
        after.pending_draws = 2;
        *after.fallback_reasons.get_mut("opcode").unwrap() += 1;
        {
            let telemetry = after.telemetry.as_mut().unwrap();
            telemetry.match_attempts += 5;
            telemetry.sampled_draws += 1;
            telemetry.sampled_match_ns += 13;
            telemetry.timestamp_readback_bytes += 16;
            telemetry.record_batch(1, 32);
            telemetry.record_batch(3, 32);
            telemetry.flush_reasons.record(FlushReason::Idle);
            telemetry.flush_reasons.record(FlushReason::CpuFallback);
            record_timestamp(telemetry.gpu_compute.as_mut().unwrap(), Some(90)).unwrap();
        }
        let delta = after.delta_since(&before).unwrap();
        assert_eq!(
            (delta.handled_draws, delta.fallback_draws, delta.submissions),
            (4, 1, 2)
        );
        assert_eq!(delta.pending_draws, 2);
        assert_eq!(delta.fallback_reasons.get("opcode"), Some(&1));
        let telemetry = delta.telemetry.unwrap();
        assert_eq!(
            (
                telemetry.match_attempts,
                telemetry.sampled_draws,
                telemetry.sampled_match_ns
            ),
            (5, 1, 13)
        );
        assert_eq!((telemetry.batch_count, telemetry.batch_draws), (2, 4));
        assert_eq!(telemetry.parameter_bytes, 4 * 32);
        assert_eq!(telemetry.timestamp_readback_bytes, 16);
        assert_eq!(telemetry.batch_histogram[0], 1);
        assert_eq!(telemetry.batch_histogram[1], 1);
        assert_eq!(telemetry.batch_histogram[3], 0);
        assert_eq!(
            (
                telemetry.flush_reasons.batch_full,
                telemetry.flush_reasons.idle
            ),
            (0, 1)
        );
        assert_eq!(telemetry.flush_reasons.cpu_fallback, 1);
        assert_eq!(
            telemetry.gpu_compute,
            Some(TimestampTotals {
                count: 1,
                nanoseconds: 90,
                discarded: 0,
            })
        );
        assert_eq!(telemetry.gpu_readback_transfer, None);
    }

    #[test]
    fn telemetry_detects_reset_or_changed_availability() {
        let before = ReplayReport {
            handled_draws: 3,
            telemetry: Some(StageTelemetry {
                match_attempts: 3,
                gpu_compute: Some(TimestampTotals::default()),
                ..StageTelemetry::default()
            }),
            ..ReplayReport::default()
        };
        let mut after = before.clone();
        after.telemetry = None;
        assert!(after.delta_since(&before).unwrap_err().contains("toggled"));

        after.telemetry = Some(StageTelemetry::default());
        assert!(after
            .delta_since(&before)
            .unwrap_err()
            .contains("match_attempts"));
        after.telemetry.as_mut().unwrap().match_attempts = 3;
        assert!(after
            .delta_since(&before)
            .unwrap_err()
            .contains("timestamp availability"));

        after = before.clone();
        after.handled_draws = 0;
        assert!(after
            .delta_since(&before)
            .unwrap_err()
            .contains("handled_draws"));
    }

    #[test]
    fn counter_epochs_reject_resets_and_different_engines() {
        let before = ReplayReport::fresh();
        let mut after = before.clone();
        after.advance_counter_epoch();
        after.handled_draws = 100;
        assert!(after.delta_since(&before).unwrap_err().contains("reset"));
        assert!(ReplayReport::fresh()
            .delta_since(&before)
            .unwrap_err()
            .contains("different engines"));
    }

    #[test]
    fn batch_histogram_covers_configured_limits() {
        let mut telemetry = StageTelemetry::default();
        for draws in [0, 1, 2, 3, 1023, 1024, 65536] {
            telemetry.record_batch(draws, 32);
        }
        assert_eq!(telemetry.batch_count, 6);
        assert_eq!(telemetry.batch_draws, 1 + 2 + 3 + 1023 + 1024 + 65536);
        assert_eq!(telemetry.batch_histogram[0], 1);
        assert_eq!(telemetry.batch_histogram[1], 2);
        assert_eq!(telemetry.batch_histogram[9], 1);
        assert_eq!(telemetry.batch_histogram[10], 1);
        assert_eq!(telemetry.batch_histogram[16], 1);
        assert_eq!(
            telemetry.batch_histogram.iter().sum::<u64>(),
            telemetry.batch_count
        );
    }

    #[test]
    fn timestamp_readback_uses_period_and_rejects_bad_data() {
        let bytes: Vec<u8> = [100_u64, 140, 200, 202]
            .into_iter()
            .flat_map(u64::to_ne_bytes)
            .collect();
        assert_eq!(timestamp_duration_ns(&bytes, 0, 2.5), Ok(Some(100)));
        assert_eq!(timestamp_duration_ns(&bytes, 2, 2.5), Ok(Some(5)));
        assert!(timestamp_duration_ns(&bytes, 4, 2.5).is_err());
        assert!(timestamp_duration_ns(&bytes, 0, 0.0).is_err());
        let backwards: Vec<u8> = [2_u64, 1].into_iter().flat_map(u64::to_ne_bytes).collect();
        assert_eq!(timestamp_duration_ns(&backwards, 0, 1.0), Ok(None));
    }

    #[test]
    fn timestamp_failure_invalidates_totals_and_rejects_transition_deltas() {
        let before = ReplayReport {
            timestamp_support: TimestampSupport::Enabled {
                readback_transfer: true,
            },
            telemetry: Some(StageTelemetry {
                gpu_compute: Some(TimestampTotals {
                    count: 1,
                    nanoseconds: 20,
                    discarded: 0,
                }),
                gpu_readback_transfer: Some(TimestampTotals::default()),
                ..StageTelemetry::default()
            }),
            ..ReplayReport::default()
        };
        let mut after = before.clone();
        after.timestamp_collection_failed("readback failed".into());
        let telemetry = after.telemetry.as_ref().unwrap();
        assert_eq!(telemetry.gpu_compute, None);
        assert_eq!(telemetry.gpu_readback_transfer, None);
        assert_eq!(after.timestamp_error.as_deref(), Some("readback failed"));
        assert!(after
            .delta_since(&before)
            .unwrap_err()
            .contains("timestamp support"));

        let mut inconsistent = before.clone();
        inconsistent.timestamp_support = TimestampSupport::Unavailable("failed".into());
        assert!(inconsistent
            .delta_since(&before)
            .unwrap_err()
            .contains("timestamp support"));
        let mut error_only = before.clone();
        error_only.timestamp_error = Some("failed".into());
        assert!(error_only
            .delta_since(&before)
            .unwrap_err()
            .contains("timestamp support"));
        assert!(after.delta_since(&after).is_ok());
    }

    #[test]
    fn timestamp_wrap_discards_only_affected_sample_and_keeps_profiling() {
        let bytes: Vec<u8> = [u64::MAX - 2, 1, 10, 15]
            .into_iter()
            .flat_map(u64::to_ne_bytes)
            .collect();
        let mut total = TimestampTotals::default();
        let before = total.clone();
        record_timestamp(&mut total, timestamp_duration_ns(&bytes, 0, 2.0).unwrap()).unwrap();
        record_timestamp(&mut total, timestamp_duration_ns(&bytes, 2, 2.0).unwrap()).unwrap();
        assert_eq!(
            total.delta_since(&before).unwrap(),
            TimestampTotals {
                count: 1,
                nanoseconds: 10,
                discarded: 1,
            }
        );
    }

    #[test]
    fn timestamp_validation_requires_a_usable_sample() {
        assert!(require_usable_timestamps(
            "compute",
            &TimestampTotals {
                count: 1,
                nanoseconds: 5,
                discarded: 3,
            },
        )
        .is_ok());
        let all_discarded = require_usable_timestamps(
            "compute",
            &TimestampTotals {
                discarded: 3,
                ..TimestampTotals::default()
            },
        )
        .unwrap_err();
        assert!(all_discarded.contains("INCONCLUSIVE"));
        assert!(all_discarded.contains("all 3 compute"));
        assert!(
            require_usable_timestamps("transfer", &TimestampTotals::default())
                .unwrap_err()
                .contains("no usable samples")
        );
    }

    #[test]
    fn live_aa_options_validate_bounds_and_balanced_rounds() {
        assert_eq!(
            LiveAaOptions::parse(None, None).unwrap(),
            LiveAaOptions {
                sample_ms: 600,
                max_rounds: 48,
            }
        );
        assert_eq!(
            LiveAaOptions::parse(Some("100"), Some("16")).unwrap(),
            LiveAaOptions {
                sample_ms: 100,
                max_rounds: 16,
            }
        );
        assert_eq!(
            LiveAaOptions::parse(Some("2000"), Some("128")).unwrap(),
            LiveAaOptions {
                sample_ms: 2000,
                max_rounds: 128,
            }
        );
        for value in [
            "",
            "99",
            "2001",
            "-1",
            "+600",
            " 600",
            "99999999999999999999",
        ] {
            assert!(LiveAaOptions::parse(Some(value), None)
                .unwrap_err()
                .contains(LIVE_AA_SAMPLE_MS));
        }
        for value in ["", "15", "17", "127", "129", "NaN", "99999999999999999999"] {
            assert!(LiveAaOptions::parse(None, Some(value))
                .unwrap_err()
                .contains(LIVE_AA_MAX_ROUNDS));
        }
    }

    #[test]
    fn live_aa_looks_are_balanced_and_spend_confidence_across_looks() {
        for max_rounds in [16, 24, 48, 64, 128] {
            let options = LiveAaOptions::parse(None, Some(&max_rounds.to_string())).unwrap();
            let looks: Vec<_> = (0..=max_rounds)
                .filter(|&round| options.is_look(round))
                .collect();
            assert_eq!(looks.first(), Some(&16));
            assert_eq!(looks.last(), Some(&max_rounds));
            assert_eq!(looks.len(), options.look_count());
            assert!(looks.windows(2).all(|pair| pair[1] - pair[0] == 8));
            let mut spent = 0.0;
            for (index, &rounds) in looks.iter().enumerate() {
                assert_eq!(options.look_number(rounds), index + 1);
                let stats =
                    paired_rate_stats_with_alpha(&vec![0.0; rounds], options.per_look_alpha());
                assert!(1.0 - stats.confidence <= options.per_look_alpha() + 1e-12);
                spent += 1.0 - stats.confidence;
            }
            assert!(spent <= LIVE_AA_SIGN_ALPHA + 1e-12);
        }
        let single_look = LiveAaOptions::parse(None, Some("16")).unwrap();
        let values: Vec<_> = (0..16).map(|value| (value as f64 + 1.0).ln()).collect();
        let old = paired_rate_stats(&values);
        let sequential = paired_rate_stats_with_alpha(&values, single_look.per_look_alpha());
        assert_eq!(sequential.lower, old.lower);
        assert_eq!(sequential.upper, old.upper);
        assert_eq!(sequential.confidence, old.confidence);
    }

    #[test]
    fn live_gate_statistics_handle_odd_even_and_noise() {
        let stats = live_gate_stats(&[5.0, 1.0, 3.0, 2.0, 4.0]);
        assert_eq!(stats.median, 3.0);
        assert!((stats.coefficient_of_variation - 2.0_f64.sqrt() / 3.0).abs() < 1e-12);
        assert!((stats.range_over_median - 4.0 / 3.0).abs() < 1e-12);
        assert_eq!(live_gate_stats(&[1.0, 3.0]).median, 2.0);
    }

    #[test]
    fn abba_rate_ratios_cancel_linear_log_drift_in_both_orders() {
        let times = [0.0, 0.28, 0.72, 1.43];
        for on_first in [false, true] {
            let rates = std::array::from_fn(|index| {
                let on = if index == 0 || index == 3 {
                    on_first
                } else {
                    !on_first
                };
                100.0 * (times[index] * 0.04_f64).exp() * if on { 0.98 } else { 1.0 }
            });
            assert!((abba_log_rate_ratio(rates, times, on_first) - 0.98_f64.ln()).abs() < 1e-12);
        }
        assert!((log_rate_ratio(100.0, 98.0) - 0.98_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn paired_statistics_use_sign_interval_and_resist_outliers() {
        let mut ratios = [0.99_f64.ln(); 16];
        ratios[..3].fill(0.5_f64.ln());
        ratios[13..].fill(2.0_f64.ln());
        let stats = paired_rate_stats(&ratios);
        assert!((stats.median + 0.01).abs() < 1e-12);
        assert!((stats.lower + 0.01).abs() < 1e-12);
        assert!((stats.upper + 0.01).abs() < 1e-12);
        assert!((stats.confidence - (1.0 - 2.0 * 697.0 / 65536.0)).abs() < 1e-12);
    }

    #[test]
    fn overhead_decision_distinguishes_effect_from_noise() {
        let control = paired_rate_stats(&[0.0; 16]);
        let change = |rate: f64| paired_rate_stats(&[rate.ln(); 16]);
        assert_eq!(
            overhead_decision(control, change(0.98), 0.0, 0.03),
            OverheadDecision::Pass
        );
        assert_eq!(
            overhead_decision(control, change(0.96), 0.0, 0.03),
            OverheadDecision::Fail
        );
        assert_eq!(
            overhead_decision(control, change(1.04), 0.0, 0.03),
            OverheadDecision::Inconclusive
        );

        let mut uncertain = [0.98_f64.ln(); 16];
        uncertain[..4].fill(0.95_f64.ln());
        assert_eq!(
            overhead_decision(control, paired_rate_stats(&uncertain), 0.0, 0.03),
            OverheadDecision::Inconclusive
        );
        let mut noisy = [0.96_f64.ln(); 16];
        noisy[..8].fill(1.04_f64.ln());
        assert_eq!(
            overhead_decision(paired_rate_stats(&noisy), change(0.96), 0.0, 0.03),
            OverheadDecision::Inconclusive
        );
        let biased = change(1.01);
        assert_eq!(
            overhead_decision(biased, change(0.99), 0.0, 0.03),
            OverheadDecision::Inconclusive
        );
        let mut order_effects = [0.98_f64.ln(); 16];
        let order: Vec<_> = (0..16).map(|index| index % 2 == 0).collect();
        for (index, effect) in order_effects.iter_mut().enumerate() {
            if order[index] {
                *effect = 1.02_f64.ln();
            }
        }
        let shift = paired_order_shift(&order_effects, &order);
        assert!(shift > 0.03);
        assert_eq!(
            overhead_decision(control, paired_rate_stats(&order_effects), shift, 0.03),
            OverheadDecision::Inconclusive
        );
    }

    #[test]
    fn live_aa_resolved_cpu_interval_allows_noisy_cpu_pairs() {
        fn balanced_log_ratios(low: f64, lower: f64, mid: f64, upper: f64, high: f64) -> Vec<f64> {
            let mut sorted = vec![low.ln(); 19];
            sorted.push(lower.ln());
            sorted.extend([mid.ln(); 24]);
            sorted.push(upper.ln());
            sorted.extend([high.ln(); 19]);
            (0..32)
                .flat_map(|index| [sorted[index], sorted[63 - index]])
                .collect()
        }

        let options = LiveAaOptions::parse(None, Some("64")).unwrap();
        let alpha = options.per_look_alpha();
        let cpu = paired_rate_stats_with_alpha(
            &balanced_log_ratios(1.0 / 1.03225, 0.97524, 1.0, 1.02403, 1.03225),
            alpha,
        );
        let gpu = paired_rate_stats_with_alpha(
            &balanced_log_ratios(1.0 / 1.01, 0.99443, 1.0, 1.00536, 1.01),
            alpha,
        );
        let telemetry = paired_rate_stats_with_alpha(
            &balanced_log_ratios(0.99, 0.99514, 0.99752, 1.00182, 1.01),
            alpha,
        );
        for (actual, expected) in [
            (cpu.lower, -0.02476),
            (cpu.upper, 0.02403),
            (cpu.median_absolute, 0.03225),
            (gpu.lower, -0.00557),
            (gpu.upper, 0.00536),
            (telemetry.median, -0.00248),
            (telemetry.lower, -0.00486),
            (telemetry.upper, 0.00182),
            (cpu.confidence, 0.998437210655172),
        ] {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!(!options.is_look(8));
        assert!(options.is_look(16));
        assert_eq!(
            live_aa_stop(&options, 64, gpu, telemetry, 0.00520, 0.03),
            LiveAaStop::Pass
        );

        let decide = |gpu, telemetry, order_shift| {
            live_aa_stop(&options, 64, gpu, telemetry, order_shift, 0.03)
        };
        assert_eq!(
            decide(
                PairedRateStats {
                    median_absolute: 0.03,
                    ..gpu
                },
                telemetry,
                0.00520
            ),
            LiveAaStop::MaxRounds
        );
        assert_eq!(
            decide(
                PairedRateStats {
                    median: 0.003,
                    lower: 0.001,
                    ..gpu
                },
                telemetry,
                0.00520
            ),
            LiveAaStop::MaxRounds
        );
        assert_eq!(
            decide(PairedRateStats { upper: 0.03, ..gpu }, telemetry, 0.00520),
            LiveAaStop::MaxRounds
        );
        assert_eq!(
            decide(
                PairedRateStats {
                    half_shift: 0.03,
                    ..gpu
                },
                telemetry,
                0.00520
            ),
            LiveAaStop::MaxRounds
        );
        assert_eq!(
            decide(
                gpu,
                PairedRateStats {
                    upper: 0.03,
                    ..telemetry
                },
                0.00520
            ),
            LiveAaStop::MaxRounds
        );
        assert_eq!(
            decide(
                gpu,
                PairedRateStats {
                    half_shift: 0.03,
                    ..telemetry
                },
                0.00520
            ),
            LiveAaStop::Pass
        );
        assert_eq!(
            decide(
                gpu,
                PairedRateStats {
                    median: -0.035,
                    lower: -0.04,
                    upper: -0.03,
                    ..telemetry
                },
                0.00520
            ),
            LiveAaStop::FailSlowdown
        );
        assert_eq!(decide(gpu, telemetry, 0.03), LiveAaStop::MaxRounds);
    }

    #[test]
    fn live_aa_noisy_cpu_does_not_veto_stable_gpu_measurements() {
        fn log_ratios(
            low: f64,
            lower: f64,
            mid: f64,
            upper: f64,
            high: f64,
            balanced: bool,
        ) -> Vec<f64> {
            let mut sorted = vec![low.ln(); 13];
            sorted.push(lower.ln());
            sorted.extend([mid.ln(); 20]);
            sorted.push(upper.ln());
            sorted.extend([high.ln(); 13]);
            if balanced {
                (0..24)
                    .flat_map(|index| [sorted[index], sorted[47 - index]])
                    .collect()
            } else {
                sorted
            }
        }

        let options = LiveAaOptions::parse(None, Some("48")).unwrap();
        let alpha = options.per_look_alpha();
        let cpu = paired_rate_stats_with_alpha(
            &log_ratios(0.94, 0.96116, 1.0, 1.02995, 1.06, false),
            alpha,
        );
        let gpu = paired_rate_stats_with_alpha(
            &log_ratios(1.0 / 1.01, 0.99439, 1.0, 1.00608, 1.01, true),
            alpha,
        );
        let telemetry = paired_rate_stats_with_alpha(
            &log_ratios(0.99, 0.99619, 1.00087, 1.00434, 1.01, true),
            alpha,
        );
        for (actual, expected) in [
            (cpu.lower, -0.03884),
            (cpu.upper, 0.02995),
            (gpu.lower, -0.00561),
            (gpu.upper, 0.00608),
            (telemetry.median, 0.00087),
            (telemetry.lower, -0.00381),
            (telemetry.upper, 0.00434),
            (cpu.confidence, 0.9979118926660036),
        ] {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!(cpu.lower < -0.03 && cpu.half_shift > 0.03 && cpu.median_absolute > 0.03);
        assert_eq!(
            live_aa_stop(&options, 48, gpu, telemetry, 0.00068, 0.03),
            LiveAaStop::Pass
        );
        let noisy_gpu = PairedRateStats {
            median_absolute: 0.04,
            ..gpu
        };
        assert_eq!(
            overhead_decision(noisy_gpu, telemetry, 0.00068, 0.03),
            OverheadDecision::Inconclusive
        );
        assert_eq!(
            live_aa_stop(&options, 48, noisy_gpu, telemetry, 0.00068, 0.03),
            LiveAaStop::MaxRounds
        );
    }

    #[test]
    fn live_aa_sequential_stop_requires_stable_controls_and_a_decisive_interval() {
        let options = LiveAaOptions::parse(None, Some("48")).unwrap();
        let stats = |count: usize, rate: f64| {
            paired_rate_stats_with_alpha(&vec![rate.ln(); count], options.per_look_alpha())
        };
        let control = stats(16, 1.0);
        assert_eq!(
            live_aa_stop(&options, 16, control, stats(16, 0.98), 0.0, 0.03),
            LiveAaStop::Pass
        );
        assert_eq!(
            live_aa_stop(&options, 16, control, stats(16, 0.96), 0.0, 0.03),
            LiveAaStop::FailSlowdown
        );
        assert_eq!(
            live_aa_stop(&options, 16, control, stats(16, 1.04), 0.0, 0.03),
            LiveAaStop::Continue
        );
        assert_eq!(
            live_aa_stop(&options, 48, control, stats(48, 1.04), 0.0, 0.03),
            LiveAaStop::MaxRounds
        );
        let mut noisy_gpu = control;
        noisy_gpu.median_absolute = 0.04;
        assert_eq!(
            live_aa_stop(&options, 24, noisy_gpu, stats(24, 0.98), 0.0, 0.03),
            LiveAaStop::Continue
        );
        assert_eq!(
            live_aa_stop(&options, 48, noisy_gpu, stats(48, 0.96), 0.0, 0.03),
            LiveAaStop::MaxRounds
        );
        let mut crosses_limit = stats(16, 0.99);
        crosses_limit.lower = -0.03;
        assert_eq!(
            live_aa_stop(&options, 16, control, crosses_limit, 0.0, 0.03),
            LiveAaStop::Continue
        );
        assert_eq!(
            live_aa_stop(&options, 16, control, stats(16, 0.98), 0.03, 0.03),
            LiveAaStop::Continue
        );
    }
}
