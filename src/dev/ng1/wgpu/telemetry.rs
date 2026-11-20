//! Bounded, aggregate-only accounting for the test-gated Newport raster.

use std::fmt;

use crate::dev::ng1::rex3::{
    Rex3Context, DRAWMODE0_ADRMODE_A_LINE, DRAWMODE0_ADRMODE_BLOCK,
    DRAWMODE0_ADRMODE_F_LINE, DRAWMODE0_ADRMODE_I_LINE, DRAWMODE0_ADRMODE_SPAN,
    DRAWMODE0_OPCODE_DRAW, DRAWMODE0_OPCODE_NOOP, DRAWMODE0_OPCODE_READ,
    DRAWMODE0_OPCODE_SCR2SCR,
};
use crate::wgpu::tiles::{TILE_H, TILE_W};

const BUCKETS: usize = 32;

macro_rules! add {
    ($field:expr, $value:expr) => {
        $field = $field.checked_add($value).ok_or("resident telemetry counter overflow")?
    };
}

macro_rules! subtract {
    ($after:ident, $before:ident, $($field:ident),+ $(,)?) => {
        $(
            $after.$field = $after.$field.checked_sub($before.$field)
                .ok_or(concat!(stringify!($field), " counter decreased"))?;
        )+
    };
}

/// Bucket 0 is zero; bucket i is 2^(i-1)..2^i-1, with the final bucket
/// unbounded. Quantiles are conservative upper bounds, not raw samples.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Histogram {
    pub(crate) buckets: [u64; BUCKETS],
    pub(crate) samples: u64,
    pub(crate) sum: u64,
    /// Lifetime high-water mark, including in a delta report and after reset.
    pub(crate) high_water: u64,
}

impl Histogram {
    pub(crate) fn record(&mut self, value: u64) -> Result<(), &'static str> {
        let bucket = if value == 0 { 0 } else {
            (u64::BITS - value.leading_zeros()) as usize
        }.min(BUCKETS - 1);
        let count = self.buckets[bucket].checked_add(1).ok_or("histogram bucket overflow")?;
        let samples = self.samples.checked_add(1).ok_or("histogram sample overflow")?;
        let sum = self.sum.checked_add(value).ok_or("histogram sum overflow")?;
        self.buckets[bucket] = count;
        self.samples = samples;
        self.sum = sum;
        self.high_water = self.high_water.max(value);
        Ok(())
    }

    pub(crate) fn delta_since(&self, before: &Self) -> Result<Self, &'static str> {
        if self.high_water < before.high_water {
            return Err("histogram maximum decreased");
        }
        let mut delta = self.clone();
        subtract!(delta, before, samples, sum);
        for (bucket, previous) in delta.buckets.iter_mut().zip(before.buckets) {
            *bucket = bucket.checked_sub(previous).ok_or("histogram bucket decreased")?;
        }
        Ok(delta)
    }

    pub(crate) fn percentile_upper(&self, percent: u32) -> Option<u64> {
        if self.samples == 0 || !(1..=100).contains(&percent) {
            return None;
        }
        let rank = ((self.samples as u128 * percent as u128 + 99) / 100) as u64;
        let mut seen = 0u64;
        for (index, &count) in self.buckets.iter().enumerate() {
            seen = seen.checked_add(count)?;
            if seen >= rank {
                return match index {
                    0 => Some(0),
                    _ if index == BUCKETS - 1 => Some(self.high_water),
                    _ => Some((1u64 << index) - 1),
                };
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum PrimitiveKind {
    Draw,
    Copy,
    HostWrite,
    HostRead,
    Noop,
}

const PRIMITIVE_KINDS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum AddressMode {
    Span,
    Block,
    IntegerLine,
    FractionalLine,
    AntialiasedLine,
    Other,
}

const ADDRESS_MODES: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum ModeFeature {
    Blend,
    Shade,
    ZPattern,
    LinePattern,
    Dither,
    FastClear,
    Scissor,
    CidTest,
    Auxiliary,
    ColorHost,
    AlphaHost,
    PackedHost,
    DoubleHost,
    XyOffset,
    YFlip,
    AlphaCompare,
}

const MODE_FEATURES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum FlushReason {
    Capacity,
    HostTransition,
    CopyHazard,
    Idle,
    WaitIdle,
    DisplayFence,
    FrameCut,
    Observation,
    HostRead,
    Explicit,
    Exit,
    Latency,
    Restore,
    Unsupported,
}

const FLUSH_REASONS: usize = 14;

/// Screen-clipped conservative destination bounds. This is not a count of
/// pixels surviving a per-pixel CID, pattern, alpha, or logic test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Coverage {
    pub(crate) x0: u32,
    pub(crate) y0: u32,
    pub(crate) x1: u32,
    pub(crate) y1: u32,
}

impl Coverage {
    pub(crate) fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Option<Self> {
        let x0 = x0.max(0);
        let y0 = y0.max(0);
        let x1 = x1.min(crate::dev::ng1::rex3::REX3_SCREEN_WIDTH - 1);
        let y1 = y1.min(crate::dev::ng1::rex3::REX3_SCREEN_HEIGHT - 1);
        (x0 <= x1 && y0 <= y1).then_some(Self {
            x0: x0 as u32, y0: y0 as u32, x1: x1 as u32, y1: y1 as u32,
        })
    }

    pub(crate) fn include_address(&mut self, address: u32) {
        let stride = crate::disp::FB_STRIDE as u32;
        let x = address % stride;
        let y = address / stride;
        self.x0 = self.x0.min(x);
        self.y0 = self.y0.min(y);
        self.x1 = self.x1.max(x);
        self.y1 = self.y1.max(y);
    }

    pub(crate) fn from_address(address: u32) -> Self {
        let stride = crate::disp::FB_STRIDE as u32;
        let x = address % stride;
        let y = address / stride;
        Self { x0: x, y0: y, x1: x, y1: y }
    }

    fn area(self) -> u64 {
        u64::from(self.x1 - self.x0 + 1) * u64::from(self.y1 - self.y0 + 1)
    }

    fn tiles(self) -> u64 {
        u64::from(self.x1 / TILE_W - self.x0 / TILE_W + 1)
            * u64::from(self.y1 / TILE_H - self.y0 / TILE_H + 1)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DrawSample {
    kind: PrimitiveKind,
    address: AddressMode,
    features: u32,
}

impl DrawSample {
    pub(crate) fn from_context(context: &Rex3Context) -> Self {
        let dm0 = context.drawmode0;
        let dm1 = context.drawmode1;
        let kind = match dm0.opcode() {
            DRAWMODE0_OPCODE_NOOP => PrimitiveKind::Noop,
            DRAWMODE0_OPCODE_READ => PrimitiveKind::HostRead,
            DRAWMODE0_OPCODE_SCR2SCR => PrimitiveKind::Copy,
            DRAWMODE0_OPCODE_DRAW if dm0.colorhost() || dm0.alphahost() => PrimitiveKind::HostWrite,
            _ => PrimitiveKind::Draw,
        };
        let address = match dm0.adrmode() {
            DRAWMODE0_ADRMODE_SPAN => AddressMode::Span,
            DRAWMODE0_ADRMODE_BLOCK => AddressMode::Block,
            DRAWMODE0_ADRMODE_I_LINE => AddressMode::IntegerLine,
            DRAWMODE0_ADRMODE_F_LINE => AddressMode::FractionalLine,
            DRAWMODE0_ADRMODE_A_LINE => AddressMode::AntialiasedLine,
            _ => AddressMode::Other,
        };
        let mut features = 0u32;
        let mut set = |feature: ModeFeature, enabled: bool| {
            if enabled { features |= 1 << feature as usize; }
        };
        set(ModeFeature::Blend, dm1.blend());
        set(ModeFeature::Shade, dm0.shade());
        set(ModeFeature::ZPattern, dm0.enzpattern());
        set(ModeFeature::LinePattern, dm0.enlspattern());
        set(ModeFeature::Dither, dm1.dither());
        set(ModeFeature::FastClear, dm1.fastclear());
        set(ModeFeature::Scissor, context.clipmode & 0x1f != 0);
        set(ModeFeature::CidTest, super::command::cid_test(context.clipmode));
        set(ModeFeature::Auxiliary, matches!(dm1.planes(), 4 | 5 | 6));
        set(ModeFeature::ColorHost, dm0.colorhost());
        set(ModeFeature::AlphaHost, dm0.alphahost());
        set(ModeFeature::PackedHost, dm1.rwpacked());
        set(ModeFeature::DoubleHost, dm1.rwdouble());
        set(ModeFeature::XyOffset, dm0.xyoffset());
        set(ModeFeature::YFlip, dm1.yflip());
        set(ModeFeature::AlphaCompare, dm1.compare() != 7);
        Self { kind, address, features }
    }

    pub(crate) fn is_host_read(self) -> bool {
        self.kind == PrimitiveKind::HostRead
    }

    pub(crate) fn is_host_write(self) -> bool {
        self.kind == PrimitiveKind::HostWrite
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Workload {
    pub(crate) profiling: bool,
    pub(crate) accepted: u64,
    pub(crate) noops: u64,
    pub(crate) unsupported: u64,
    pub(crate) primitive: [u64; PRIMITIVE_KINDS],
    pub(crate) address: [u64; ADDRESS_MODES],
    pub(crate) features: [u64; MODE_FEATURES],
    pub(crate) clipped_area: Histogram,
    pub(crate) clipped_tiles: Histogram,
    pub(crate) host_write_qwords: Histogram,
    pub(crate) host_read_requested_qwords: Histogram,
    pub(crate) host_read_produced_qwords: Histogram,
    pub(crate) host_write_words: u64,
    pub(crate) host_read_words: u64,
    pub(crate) batches: Histogram,
    pub(crate) batch_tile_pairs: Histogram,
    pub(crate) flush: [u64; FLUSH_REASONS],
    pub(crate) set_waits: u64,
    pub(crate) set_wait_ns: u64,
    pub(crate) command_generation_ns: u64,
    pub(crate) submission_ns: u64,
    /// Profiling maps are never raw-plane or composed-frame observations.
    pub(crate) profile_maps: u64,
    pub(crate) profile_bytes: u64,
    pub(crate) profile_samples: u64,
    pub(crate) profile_busy_tiles: u64,
    pub(crate) profile_tile_pairs: u64,
    pub(crate) profile_bin_ns: u64,
    pub(crate) profile_raster_ns: u64,
}

impl Workload {
    pub(crate) fn record_draw(
        &mut self,
        sample: DrawSample,
        coverage: Option<Coverage>,
        noop: bool,
        host_qwords: u64,
        generation_ns: u64,
    ) -> Result<(), &'static str> {
        add!(self.accepted, 1);
        if noop { add!(self.noops, 1); }
        add!(self.primitive[sample.kind as usize], 1);
        add!(self.address[sample.address as usize], 1);
        for (index, count) in self.features.iter_mut().enumerate() {
            if sample.features & (1 << index) != 0 { add!(*count, 1); }
        }
        if let Some(coverage) = coverage {
            self.clipped_area.record(coverage.area())?;
            self.clipped_tiles.record(coverage.tiles())?;
        }
        if sample.is_host_write() {
            self.host_write_qwords.record(host_qwords)?;
            add!(self.host_write_words, host_qwords.checked_mul(2).ok_or("HostW word count overflow")?);
        } else if sample.is_host_read() {
            self.host_read_requested_qwords.record(host_qwords)?;
        }
        add!(self.command_generation_ns, generation_ns);
        Ok(())
    }

    pub(crate) fn record_host_read(&mut self, produced: u64, generation_ns: u64) -> Result<(), &'static str> {
        self.host_read_produced_qwords.record(produced)?;
        add!(self.host_read_words, produced.checked_mul(2).ok_or("HostR word count overflow")?);
        add!(self.command_generation_ns, generation_ns);
        Ok(())
    }

    pub(crate) fn record_flush(
        &mut self, reason: FlushReason, commands: usize, tile_pairs: u64, ns: u64,
        waits: u64, wait_ns: u64,
    ) -> Result<(), &'static str> {
        if commands == 0 {
            return Ok(());
        }
        self.batches.record(commands as u64)?;
        self.batch_tile_pairs.record(tile_pairs)?;
        add!(self.flush[reason as usize], 1);
        add!(self.submission_ns, ns);
        add!(self.set_waits, waits);
        add!(self.set_wait_ns, wait_ns);
        Ok(())
    }

    pub(crate) fn delta_since(&self, before: &Self) -> Result<Self, &'static str> {
        if self.profiling != before.profiling {
            return Err("resident profiling was toggled");
        }
        let mut delta = self.clone();
        subtract!(delta, before, accepted, noops, unsupported, host_write_words, host_read_words,
            set_waits, set_wait_ns, command_generation_ns, submission_ns, profile_maps,
            profile_bytes, profile_samples, profile_busy_tiles, profile_tile_pairs, profile_bin_ns,
            profile_raster_ns);
        for (a, b) in delta.primitive.iter_mut().zip(before.primitive) {
            *a = a.checked_sub(b).ok_or("primitive count decreased")?;
        }
        for (a, b) in delta.address.iter_mut().zip(before.address) {
            *a = a.checked_sub(b).ok_or("address-mode count decreased")?;
        }
        for (a, b) in delta.features.iter_mut().zip(before.features) {
            *a = a.checked_sub(b).ok_or("mode-feature count decreased")?;
        }
        for (a, b) in delta.flush.iter_mut().zip(before.flush) {
            *a = a.checked_sub(b).ok_or("flush count decreased")?;
        }
        delta.clipped_area = self.clipped_area.delta_since(&before.clipped_area)?;
        delta.clipped_tiles = self.clipped_tiles.delta_since(&before.clipped_tiles)?;
        delta.host_write_qwords = self.host_write_qwords.delta_since(&before.host_write_qwords)?;
        delta.host_read_requested_qwords = self.host_read_requested_qwords.delta_since(&before.host_read_requested_qwords)?;
        delta.host_read_produced_qwords = self.host_read_produced_qwords.delta_since(&before.host_read_produced_qwords)?;
        delta.batches = self.batches.delta_since(&before.batches)?;
        delta.batch_tile_pairs = self.batch_tile_pairs.delta_since(&before.batch_tile_pairs)?;
        Ok(delta)
    }
}

/// Occupancy samples can race the consumer; reservation failures themselves
/// distinguish a full published ring from producer lock contention exactly.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Board {
    pub(crate) occupancy: Histogram,
    pub(crate) reservations: u64,
    pub(crate) backpressure: u64,
    pub(crate) full: u64,
    pub(crate) bus_busy: u64,
    pub(crate) cut_latency_ns: Histogram,
    /// Render entry to successful presenter submission; not display scanout.
    pub(crate) presentation_latency_ns: Histogram,
    pub(crate) errors: bool,
}

impl Board {
    pub(crate) fn reservation(&mut self, occupancy: usize, result: Result<(), crate::gfifo::PushFailure>) {
        let result = (|| {
            self.occupancy.record(occupancy as u64)?;
            add!(self.reservations, 1);
            if let Err(failure) = result {
                add!(self.backpressure, 1);
                if failure == crate::gfifo::PushFailure::Full { add!(self.full, 1); }
            }
            Ok::<(), &'static str>(())
        })();
        self.errors |= result.is_err();
    }

    pub(crate) fn bus_busy(&mut self) {
        match self.bus_busy.checked_add(1) {
            Some(count) => self.bus_busy = count,
            None => self.errors = true,
        }
    }

    pub(crate) fn cut(&mut self, duration_ns: u64) {
        self.errors |= self.cut_latency_ns.record(duration_ns).is_err();
    }

    pub(crate) fn presentation(&mut self, duration_ns: u64) {
        self.errors |= self.presentation_latency_ns.record(duration_ns).is_err();
    }

    pub(crate) fn delta_since(&self, before: &Self) -> Result<Self, &'static str> {
        if self.errors || before.errors { return Err("board telemetry overflow"); }
        let mut delta = self.clone();
        subtract!(delta, before, reservations, backpressure, full, bus_busy);
        delta.occupancy = self.occupancy.delta_since(&before.occupancy)?;
        delta.cut_latency_ns = self.cut_latency_ns.delta_since(&before.cut_latency_ns)?;
        delta.presentation_latency_ns = self.presentation_latency_ns.delta_since(&before.presentation_latency_ns)?;
        Ok(delta)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Traffic {
    pub(crate) raw_attempts: [u64; 6],
    pub(crate) raw_bytes: [u64; 6],
    pub(crate) raw_maps: u64,
    /// Direct and readback-compatibility composed captures, respectively.
    pub(crate) composed_attempts: [u64; 2],
    pub(crate) composed_bytes: [u64; 2],
}

impl Traffic {
    fn delta_since(&self, before: &Self) -> Result<Self, &'static str> {
        let mut delta = self.clone();
        subtract!(delta, before, raw_maps);
        for (a, b) in delta.raw_attempts.iter_mut().zip(before.raw_attempts) {
            *a = a.checked_sub(b).ok_or("raw observation attempts decreased")?;
        }
        for (a, b) in delta.raw_bytes.iter_mut().zip(before.raw_bytes) {
            *a = a.checked_sub(b).ok_or("raw observation bytes decreased")?;
        }
        for (a, b) in delta.composed_attempts.iter_mut().zip(before.composed_attempts) {
            *a = a.checked_sub(b).ok_or("composed capture attempts decreased")?;
        }
        for (a, b) in delta.composed_bytes.iter_mut().zip(before.composed_bytes) {
            *a = a.checked_sub(b).ok_or("composed capture bytes decreased")?;
        }
        Ok(delta)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Snapshot {
    pub(crate) epoch: u64,
    pub(crate) workload: Workload,
    pub(crate) board: Board,
    pub(crate) traffic: Traffic,
}

impl Snapshot {
    pub(crate) fn delta_since(&self, before: &Self) -> Result<Self, &'static str> {
        if self.epoch != before.epoch {
            return Err("resident telemetry epoch changed");
        }
        Ok(Self {
            epoch: self.epoch,
            workload: self.workload.delta_since(&before.workload)?,
            board: self.board.delta_since(&before.board)?,
            traffic: self.traffic.delta_since(&before.traffic)?,
        })
    }
}

impl fmt::Display for Snapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let w = &self.workload;
        let b = &self.board;
        write!(f,
            "resident epoch={} accepted={} noop={} unsupported={} \
             kind(draw/copy/hostw/hostr/noop)={:?} address(span/block/i/f/a/other)={:?} \
             features(blend/shade/z/line/dither/clear/scissor/cid/aux/colorhost/alphahost/packed/double/xy/yflip/compare)={:?} \
             batches={} commands={} batch_p50/p95={:?}/{:?} \
             flush(capacity/host-transition/copy/idle/wait/fence/cut/observe/hostr/explicit/exit/latency/restore/unsupported)={:?} \
             area_p50/p95={:?}/{:?} tiles_p50/p95={:?}/{:?} \
             hostw(qwords/words)={}/{} hostw_qwords_p50/p95={:?}/{:?} \
             hostr(requested/produced_qwords/words)={}/{}/{} hostr_qwords_p50/p95={:?}/{:?} \
             gfifo_max={} gfifo_p50/p95={:?}/{:?} backpressure={} full={} busy={} \
             cut_p50/p95_ns={:?}/{:?} present_p50/p95_ns={:?}/{:?} generation_ns={} submit_ns={} waits={} wait_ns={} \
             raw_maps={} raw_bytes(hostr/snapshot/monitor/recording/validation/compat)={:?} \
             composed_attempts(direct/compat)={:?} composed_bytes(direct/compat)={:?} \
             profiling={} profile_maps={} profile_bytes={} profile_samples={} \
             profile_bin_ns={} profile_raster_ns={} busy_tiles={} tile_pairs={}",
            self.epoch, w.accepted, w.noops, w.unsupported, w.primitive, w.address, w.features,
            w.batches.samples, w.batches.sum, w.batches.percentile_upper(50), w.batches.percentile_upper(95), w.flush,
            w.clipped_area.percentile_upper(50), w.clipped_area.percentile_upper(95),
            w.clipped_tiles.percentile_upper(50), w.clipped_tiles.percentile_upper(95),
            w.host_write_qwords.sum, w.host_write_words,
            w.host_write_qwords.percentile_upper(50), w.host_write_qwords.percentile_upper(95),
            w.host_read_requested_qwords.sum, w.host_read_produced_qwords.sum, w.host_read_words,
            w.host_read_produced_qwords.percentile_upper(50), w.host_read_produced_qwords.percentile_upper(95),
            b.occupancy.high_water, b.occupancy.percentile_upper(50), b.occupancy.percentile_upper(95),
            b.backpressure, b.full, b.bus_busy,
            b.cut_latency_ns.percentile_upper(50), b.cut_latency_ns.percentile_upper(95),
            b.presentation_latency_ns.percentile_upper(50), b.presentation_latency_ns.percentile_upper(95),
            w.command_generation_ns, w.submission_ns, w.set_waits, w.set_wait_ns, self.traffic.raw_maps,
            self.traffic.raw_bytes, self.traffic.composed_attempts, self.traffic.composed_bytes,
            w.profiling, w.profile_maps, w.profile_bytes, w.profile_samples,
            w.profile_bin_ns, w.profile_raster_ns, w.profile_busy_tiles, w.profile_tile_pairs)
    }
}

pub(crate) fn duration_ns(duration: std::time::Duration) -> Result<u64, &'static str> {
    u64::try_from(duration.as_nanos()).map_err(|_| "resident telemetry duration overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::ng1::rex3::{DrawMode0, DrawMode1, DRAWMODE0_ADRMODE_BLOCK_SH};
    use crate::gfifo::PushFailure;

    #[test]
    fn resident_histogram_percentiles_and_overflow_are_checked() {
        let mut h = Histogram::default();
        for value in [0, 1, 2, 3, 10, 4_000_000_000] { h.record(value).unwrap(); }
        assert_eq!(h.samples, 6);
        assert_eq!(h.percentile_upper(50), Some(3));
        assert_eq!(h.percentile_upper(95), Some(4_000_000_000));
        assert_eq!(h.percentile_upper(0), None);
        let before = h.clone();
        h.sum = u64::MAX;
        let exhausted = h.clone();
        assert!(h.record(1).is_err());
        assert_eq!(h, exhausted, "a failed histogram update must be atomic");
        assert!(before.delta_since(&h).is_err());
    }

    #[test]
    fn resident_typed_workload_records_transfers_and_batch_reasons() {
        std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(|| {
            let mut context = Rex3Context::power_on_default();
            context.drawmode0 = DrawMode0(DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK_SH | (1 << 6) | (1 << 18));
            context.drawmode1 = DrawMode1(4 | (1 << 7) | (1 << 10) | (1 << 18));
            let sample = DrawSample::from_context(&context);
            assert_eq!(sample.kind, PrimitiveKind::HostWrite);
            assert_eq!(sample.address, AddressMode::Block);
            let mut workload = Workload::default();
            let bounds = Coverage::new(7, 7, 8, 8).unwrap();
            workload.record_draw(sample, Some(bounds), false, 3, 17).unwrap();
            assert_eq!(workload.accepted, 1);
            assert_eq!(workload.primitive[PrimitiveKind::HostWrite as usize], 1);
            assert_eq!(workload.address[AddressMode::Block as usize], 1);
            for feature in [ModeFeature::Blend, ModeFeature::Shade, ModeFeature::Auxiliary,
                ModeFeature::ColorHost, ModeFeature::PackedHost, ModeFeature::DoubleHost] {
                assert_eq!(workload.features[feature as usize], 1, "{feature:?}");
            }
            assert_eq!((workload.clipped_area.sum, workload.clipped_tiles.sum), (4, 4));
            assert_eq!((workload.host_write_qwords.sum, workload.host_write_words), (3, 6));

            context.drawmode0.0 = DRAWMODE0_OPCODE_READ | DRAWMODE0_ADRMODE_BLOCK_SH;
            let read = DrawSample::from_context(&context);
            assert_eq!(read.kind, PrimitiveKind::HostRead);
            workload.record_draw(read, None, true, 4, 11).unwrap();
            workload.record_host_read(2, 13).unwrap();
            workload.record_flush(FlushReason::Idle, 0, 0, 5, 0, 0).unwrap();
            workload.record_flush(FlushReason::Latency, 1, 4, 19, 1, 7).unwrap();
            assert_eq!((workload.host_read_requested_qwords.sum, workload.host_read_produced_qwords.sum), (4, 2));
            assert_eq!((workload.host_read_words, workload.noops, workload.command_generation_ns), (4, 1, 41));
            assert_eq!((workload.batches.sum, workload.batch_tile_pairs.sum, workload.submission_ns), (1, 4, 19));
            assert_eq!((workload.set_waits, workload.set_wait_ns), (1, 7));
            assert_eq!(workload.flush[FlushReason::Idle as usize], 0);
            assert_eq!(workload.flush[FlushReason::Latency as usize], 1);
        }).unwrap().join().unwrap();
    }

    #[test]
    fn resident_coverage_is_screen_clipped_and_conservative() {
        let bounds = Coverage::new(-8, -5, 15, 8).unwrap();
        assert_eq!(bounds.area(), 16 * 9);
        assert_eq!(bounds.tiles(), 4);
        assert!(Coverage::new(-20, -20, -1, -1).is_none());
        let mut pixels = Coverage::from_address(2048 + 5);
        pixels.include_address(3 * 2048 + 9);
        assert_eq!((pixels.x0, pixels.y0, pixels.x1, pixels.y1), (5, 1, 9, 3));
    }

    #[test]
    fn resident_snapshot_deltas_detect_epoch_decrease_and_profile_changes() {
        let mut before = Snapshot { epoch: 7, workload: Workload::default(), board: Board::default(), traffic: Traffic::default() };
        before.workload.accepted = 3;
        before.board.occupancy.record(8).unwrap();
        before.traffic.raw_maps = 2;
        let mut after = before.clone();
        after.workload.accepted = 5;
        after.board.occupancy.record(2).unwrap();
        after.traffic.raw_maps = 3;
        let delta = after.delta_since(&before).unwrap();
        assert_eq!((delta.workload.accepted, delta.board.occupancy.samples, delta.traffic.raw_maps), (2, 1, 1));
        assert_eq!(delta.board.occupancy.high_water, 8, "delta maxima remain lifetime gauges");
        let report = delta.to_string();
        assert!(report.contains("gfifo_p50/p95="));
        assert!(report.contains("hostw(qwords/words)="));
        assert!(report.contains("profile_bin_ns="));
        after.workload.accepted = 2;
        assert!(after.delta_since(&before).is_err());
        after.workload.accepted = 5;
        after.epoch += 1;
        assert!(after.delta_since(&before).is_err());
        after.epoch = before.epoch;
        after.workload.profiling = true;
        assert!(after.delta_since(&before).is_err());
        after.workload.profiling = false;
        after.traffic.raw_maps = 1;
        assert!(after.delta_since(&before).is_err());
    }

    #[test]
    fn resident_board_counts_full_contention_busy_and_latency() {
        let mut board = Board::default();
        board.reservation(4, Ok(()));
        board.reservation(7, Err(PushFailure::Full));
        board.reservation(1, Err(PushFailure::Contended));
        board.bus_busy();
        board.cut(40);
        board.presentation(200);
        assert_eq!((board.reservations, board.backpressure, board.full, board.bus_busy), (3, 2, 1, 1));
        assert_eq!(board.occupancy.high_water, 7);
        assert_eq!(board.cut_latency_ns.percentile_upper(95), Some(63));
        assert_eq!(board.presentation_latency_ns.percentile_upper(50), Some(255));
        board.bus_busy = u64::MAX;
        board.bus_busy();
        assert!(board.errors);
        assert!(board.delta_since(&Board::default()).is_err());
    }
}
