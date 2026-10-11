//! The writer thread: drains the per-thread rings, orders each batch by the
//! session's timeline, encodes TracePackets and appends them to the file.
//!
//! Everything goes out on one packet sequence, so a single interning table
//! covers every thread: an event name is sent the first time it appears and
//! referred to by number afterwards. Tracks are explicit (`track_uuid` on
//! every event), so one sequence carrying many threads' events is fine.
//!
//! The file is flushed after every batch: a process that ends with
//! `std::process::exit` (monitor/CI `quit`) loses at most the last batch.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::proto::{self, ProtoWriter};
use super::{rings, Clock, EventDef, Kind, Record, Ring, Track};

/// How often the writer wakes to drain. At 64Ki slots a ring overflows only
/// above ~30M events/s from one thread.
const DRAIN_PERIOD: Duration = Duration::from_millis(2);

/// How often the writer samples the guest instruction counter for the
/// `guest MIPS` counter track.
const MIPS_PERIOD: Duration = Duration::from_millis(10);

const SEQ_ID: u64 = 1;
const PROCESS_UUID: u64 = 1;
const THREAD_UUID_BASE: u64 = 0x1_0000;
const MIPS_UUID: u64 = 2;
const DROPPED_UUID: u64 = 3;

#[derive(Default)]
pub(crate) struct Stats {
    pub events: u64,
    pub dropped: u64,
    pub bytes: u64,
}

#[derive(Default)]
struct SharedStats {
    events: AtomicU64,
    dropped: AtomicU64,
    bytes: AtomicU64,
}

pub(crate) struct Session {
    pub path: String,
    pub clock: Clock,
    pub cats: AtomicU32,
    pub started: Instant,
    stop: Arc<AtomicBool>,
    stats: Arc<SharedStats>,
    handle: JoinHandle<std::io::Result<()>>,
}

impl Session {
    pub fn spawn(file: File, path: String, cats: u32, clock: Clock) -> Session {
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(SharedStats::default());
        let handle = {
            let stop = stop.clone();
            let stats = stats.clone();
            std::thread::Builder::new()
                .name("perfetto-writer".into())
                .spawn(move || Writer::new(file, clock, stats).run(&stop))
                .expect("spawn perfetto writer")
        };
        Session { path, clock, cats: AtomicU32::new(cats), started: Instant::now(), stop, stats, handle }
    }

    pub fn stats(&self) -> Stats {
        Stats {
            events: self.stats.events.load(Ordering::Relaxed),
            dropped: self.stats.dropped.load(Ordering::Relaxed),
            bytes: self.stats.bytes.load(Ordering::Relaxed),
        }
    }

    /// Final drain, flush, join.
    pub fn finish(self) -> Result<Stats, String> {
        self.stop.store(true, Ordering::Release);
        self.handle.thread().unpark();
        let res = self.handle.join().map_err(|_| "perfetto writer panicked".to_string())?;
        res.map_err(|e| format!("perfetto: write failed: {e}"))?;
        Ok(Stats {
            events: self.stats.events.load(Ordering::Relaxed),
            dropped: self.stats.dropped.load(Ordering::Relaxed),
            bytes: self.stats.bytes.load(Ordering::Relaxed),
        })
    }
}

/// Maps `host_ticks` to `host_ns`. Anchors are (ticks, ns) pairs taken at
/// each drain; a record is placed by linear interpolation between the
/// previous anchor and the current one (extrapolating for the odd record
/// stamped just outside them). Short segments mean the clock-read jitter
/// never accumulates into drift, and the mapping is continuous, so ordering
/// by ticks and by ns agree.
struct HostClock {
    prev: (u64, u64),
    cur: (u64, u64),
}

impl HostClock {
    fn anchor() -> (u64, u64) {
        // Bracket the ns read with tick reads and take the midpoint.
        let t0 = super::host_ticks();
        let ns = super::host_ns();
        let t1 = super::host_ticks();
        (t0 + (t1.wrapping_sub(t0)) / 2, ns)
    }

    fn new() -> Self {
        let a = Self::anchor();
        // A first segment of zero length has no slope: wait out a little.
        let start = Instant::now();
        let mut b = Self::anchor();
        while b.0 == a.0 || start.elapsed() < Duration::from_micros(200) {
            std::hint::spin_loop();
            b = Self::anchor();
        }
        HostClock { prev: a, cur: b }
    }

    /// Start a new segment ending now.
    fn advance(&mut self) {
        let next = Self::anchor();
        if next.0 > self.cur.0 && next.1 > self.cur.1 {
            self.prev = self.cur;
            self.cur = next;
        }
    }

    fn to_ns(&self, ticks: u64) -> u64 {
        let (t0, n0) = self.prev;
        let (t1, n1) = self.cur;
        let dt = ticks as i128 - t0 as i128;
        let ns = n0 as i128 + dt * (n1 - n0) as i128 / (t1 - t0) as i128;
        ns.max(0) as u64
    }
}

fn fnv1a(tag: u8, s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ tag as u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    // Keep clear of the small fixed uuids and the thread range.
    h | (1 << 63)
}

struct Writer {
    out: BufWriter<File>,
    clock: Clock,
    host: HostClock,
    stats: Arc<SharedStats>,
    w: ProtoWriter,
    pid: i64,
    /// Interned ids. Names are keyed by `EventDef` address, which is unique
    /// per static; two statics with the same name get two ids, which is
    /// harmless.
    name_iids: HashMap<usize, u64>,
    cat_iids: HashMap<u32, u64>,
    arg_iids: HashMap<&'static str, u64>,
    next_iid: u64,
    /// Tracks whose descriptor has been written.
    described: HashSet<u64>,
    batch: Vec<(usize, Record)>,
    ring_snapshot: Vec<Arc<Ring>>,
    dropped_total: u64,
    last_mips: Option<(Instant, u64, u64)>,
}

impl Writer {
    fn new(file: File, clock: Clock, stats: Arc<SharedStats>) -> Self {
        Writer {
            out: BufWriter::with_capacity(1 << 20, file),
            clock,
            host: HostClock::new(),
            stats,
            w: ProtoWriter::new(),
            pid: std::process::id() as i64,
            name_iids: HashMap::new(),
            cat_iids: HashMap::new(),
            arg_iids: HashMap::new(),
            next_iid: 1,
            described: HashSet::new(),
            batch: Vec::with_capacity(1 << 16),
            ring_snapshot: Vec::new(),
            dropped_total: 0,
            last_mips: None,
        }
    }

    fn run(mut self, stop: &AtomicBool) -> std::io::Result<()> {
        self.preamble();
        loop {
            // Read the flag before draining: whatever was published before
            // `stop` was set is then guaranteed to be in this last batch.
            let stopping = stop.load(Ordering::Acquire);
            self.drain();
            self.sample_mips();
            self.flush()?;
            if stopping {
                return Ok(());
            }
            std::thread::park_timeout(DRAIN_PERIOD);
        }
    }

    fn ts(&self, r: &Record) -> u64 {
        match self.clock {
            Clock::Host => self.host.to_ns(r.host_ticks),
            Clock::Guest => r.guest(),
        }
    }

    fn now(&self) -> u64 {
        match self.clock {
            Clock::Host => super::host_ns(),
            Clock::Guest => super::guest_now(),
        }
    }

    /// First packet: clears incremental state and describes the process.
    fn preamble(&mut self) {
        use proto::{packet, process_descriptor as pd, track_descriptor as td};
        let pid = self.pid;
        let p = self.w.begin(proto::TRACE_PACKET);
        self.w.varint(packet::TRUSTED_PACKET_SEQUENCE_ID, SEQ_ID);
        self.w.varint(packet::SEQUENCE_FLAGS, packet::SEQ_INCREMENTAL_STATE_CLEARED);
        let d = self.w.begin(packet::TRACK_DESCRIPTOR);
        self.w.varint(td::UUID, PROCESS_UUID);
        let pr = self.w.begin(td::PROCESS);
        self.w.varint(pd::PID, pid as u64);
        self.w.string(pd::PROCESS_NAME, &format!("iris ({} clock)", self.clock.name()));
        self.w.end(pr);
        self.w.end(d);
        self.w.end(p);
        self.described.insert(PROCESS_UUID);
    }

    fn drain(&mut self) {
        // Snapshot the ring list so emitting threads registering new rings
        // never wait on encoding.
        {
            let all = rings().lock();
            if all.len() != self.ring_snapshot.len() {
                self.ring_snapshot = all.clone();
            }
        }
        // Anchor before draining, so nearly every record drained is older
        // than it and lands inside the segment rather than past its end.
        self.host.advance();
        let mut batch = std::mem::take(&mut self.batch);
        batch.clear();
        let mut dropped = 0;
        for ring in &self.ring_snapshot {
            ring.drain_into(&mut batch);
            dropped += ring.take_dropped();
        }
        // Stable: equal timestamps keep each thread's emission order, which
        // matters for a begin/end pair stamped in the same tick.
        batch.sort_by_key(|(_, r)| self.ts(r));

        for i in 0..batch.len() {
            let (ring_idx, r) = batch[i];
            self.event(ring_idx, &r);
        }
        self.stats.events.fetch_add(batch.len() as u64, Ordering::Relaxed);
        self.batch = batch;

        if dropped > 0 {
            self.dropped_total += dropped;
            self.stats.dropped.fetch_add(dropped, Ordering::Relaxed);
            let ts = self.now();
            self.writer_counter(DROPPED_UUID, "perfetto dropped events", ts, self.dropped_total as i64);
        }
    }

    fn sample_mips(&mut self) {
        let now = Instant::now();
        let g = super::guest_now();
        let ts = self.now();
        match self.last_mips {
            Some((t, g0, _)) if now.duration_since(t) >= MIPS_PERIOD => {
                let ns = now.duration_since(t).as_nanos().max(1) as u64;
                let mips = g.saturating_sub(g0).saturating_mul(1000) / ns;
                self.writer_counter(MIPS_UUID, "guest MIPS", ts, mips as i64);
                self.last_mips = Some((now, g, ts));
            }
            Some(_) => {}
            None => self.last_mips = Some((now, g, ts)),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if !self.w.buf.is_empty() {
            self.out.write_all(&self.w.buf)?;
            self.stats.bytes.fetch_add(self.w.buf.len() as u64, Ordering::Relaxed);
            self.w.clear();
        }
        self.out.flush()
    }

    fn thread_uuid(ring_idx: usize) -> u64 {
        THREAD_UUID_BASE + ring_idx as u64
    }

    fn track_for(&mut self, ring_idx: usize, r: &Record) -> u64 {
        let def = r.def();
        let parent = match def.group {
            None => PROCESS_UUID,
            Some(g) => {
                let uuid = fnv1a(b'g', g);
                if !self.described.contains(&uuid) {
                    self.describe_named(uuid, PROCESS_UUID, g, false);
                }
                uuid
            }
        };
        if r.kind() == Kind::Counter {
            let uuid = fnv1a(b'c', def.name);
            if !self.described.contains(&uuid) {
                self.describe_named(uuid, parent, def.name, true);
            }
            return uuid;
        }
        match def.track {
            Track::Thread => {
                let uuid = Self::thread_uuid(ring_idx);
                if !self.described.contains(&uuid) {
                    self.describe_thread(ring_idx);
                }
                uuid
            }
            Track::Global(name) => {
                let uuid = fnv1a(b't', name);
                if !self.described.contains(&uuid) {
                    self.describe_named(uuid, parent, name, false);
                }
                uuid
            }
        }
    }

    fn describe_thread(&mut self, ring_idx: usize) {
        use proto::{packet, thread_descriptor as thd, track_descriptor as td};
        let ring = self.ring_snapshot[ring_idx].clone();
        let uuid = Self::thread_uuid(ring_idx);
        let pid = self.pid;
        let p = self.w.begin(proto::TRACE_PACKET);
        self.w.varint(packet::TRUSTED_PACKET_SEQUENCE_ID, SEQ_ID);
        let d = self.w.begin(packet::TRACK_DESCRIPTOR);
        self.w.varint(td::UUID, uuid);
        let t = self.w.begin(td::THREAD);
        self.w.varint(thd::PID, pid as u64);
        self.w.int64(thd::TID, ring.tid);
        self.w.string(thd::THREAD_NAME, &ring.name);
        self.w.end(t);
        self.w.end(d);
        self.w.end(p);
        self.described.insert(uuid);
    }

    /// A named slice track, counter track or group under `parent`.
    fn describe_named(&mut self, uuid: u64, parent: u64, name: &str, counter: bool) {
        use proto::{packet, track_descriptor as td};
        let p = self.w.begin(proto::TRACE_PACKET);
        self.w.varint(packet::TRUSTED_PACKET_SEQUENCE_ID, SEQ_ID);
        let d = self.w.begin(packet::TRACK_DESCRIPTOR);
        self.w.varint(td::UUID, uuid);
        self.w.varint(td::PARENT_UUID, parent);
        self.w.string(td::NAME, name);
        if counter {
            self.w.empty(td::COUNTER);
        }
        self.w.end(d);
        self.w.end(p);
        self.described.insert(uuid);
    }

    /// Writer-generated counter (no EventDef, no interning).
    fn writer_counter(&mut self, uuid: u64, name: &str, ts: u64, value: i64) {
        use proto::{packet, track_event as te};
        if !self.described.contains(&uuid) {
            self.describe_named(uuid, PROCESS_UUID, name, true);
        }
        let p = self.w.begin(proto::TRACE_PACKET);
        self.w.varint(packet::TIMESTAMP, ts);
        self.w.varint(packet::TRUSTED_PACKET_SEQUENCE_ID, SEQ_ID);
        let e = self.w.begin(packet::TRACK_EVENT);
        self.w.varint(te::TYPE, te::TYPE_COUNTER);
        self.w.varint(te::TRACK_UUID, uuid);
        self.w.int64(te::COUNTER_VALUE, value);
        self.w.end(e);
        self.w.end(p);
    }

    fn event(&mut self, ring_idx: usize, r: &Record) {
        use proto::{debug_annotation as da, interned_data as idata, packet, track_event as te};
        let def: &'static EventDef = r.def();
        let kind = r.kind();
        let track = self.track_for(ring_idx, r);
        let ts = self.ts(r);

        // Interning for this packet: (name, category, arg name), each
        // `Some(new iid)` only the first time it is seen.
        let named = matches!(kind, Kind::Begin | Kind::Instant);
        let with_arg = named && def.arg.is_some();
        let (mut new_name, mut new_cat, mut new_arg) = (None, None, None);
        let mut name_iid = 0;
        let mut cat_iid = 0;
        let mut arg_iid = 0;
        if named {
            name_iid = match self.name_iids.get(&(def as *const _ as usize)) {
                Some(&i) => i,
                None => {
                    let i = self.next_iid;
                    self.next_iid += 1;
                    self.name_iids.insert(def as *const _ as usize, i);
                    new_name = Some(i);
                    i
                }
            };
            cat_iid = match self.cat_iids.get(&def.cat) {
                Some(&i) => i,
                None => {
                    let i = self.next_iid;
                    self.next_iid += 1;
                    self.cat_iids.insert(def.cat, i);
                    new_cat = Some(i);
                    i
                }
            };
        }
        if with_arg {
            let a = def.arg.unwrap();
            arg_iid = match self.arg_iids.get(a) {
                Some(&i) => i,
                None => {
                    let i = self.next_iid;
                    self.next_iid += 1;
                    self.arg_iids.insert(a, i);
                    new_arg = Some(i);
                    i
                }
            };
        }

        let p = self.w.begin(proto::TRACE_PACKET);
        self.w.varint(packet::TIMESTAMP, ts);
        self.w.varint(packet::TRUSTED_PACKET_SEQUENCE_ID, SEQ_ID);
        self.w.varint(packet::SEQUENCE_FLAGS, packet::SEQ_NEEDS_INCREMENTAL_STATE);
        if new_name.is_some() || new_cat.is_some() || new_arg.is_some() {
            let id = self.w.begin(packet::INTERNED_DATA);
            if let Some(i) = new_cat {
                let m = self.w.begin(idata::EVENT_CATEGORIES);
                self.w.varint(idata::IID, i);
                self.w.string(idata::NAME, &super::cat::format(def.cat));
                self.w.end(m);
            }
            if let Some(i) = new_name {
                let m = self.w.begin(idata::EVENT_NAMES);
                self.w.varint(idata::IID, i);
                self.w.string(idata::NAME, def.name);
                self.w.end(m);
            }
            if let Some(i) = new_arg {
                let m = self.w.begin(idata::DEBUG_ANNOTATION_NAMES);
                self.w.varint(idata::IID, i);
                self.w.string(idata::NAME, def.arg.unwrap());
                self.w.end(m);
            }
            self.w.end(id);
        }
        let e = self.w.begin(packet::TRACK_EVENT);
        self.w.varint(te::TYPE, kind as u64);
        self.w.varint(te::TRACK_UUID, track);
        if named {
            self.w.varint(te::NAME_IID, name_iid);
            self.w.varint(te::CATEGORY_IIDS, cat_iid);
        }
        if kind == Kind::Counter {
            self.w.int64(te::COUNTER_VALUE, r.arg as i64);
        }
        if with_arg {
            let a = self.w.begin(te::DEBUG_ANNOTATIONS);
            self.w.varint(da::NAME_IID, arg_iid);
            self.w.varint(da::UINT_VALUE, r.arg);
            self.w.end(a);
        }
        self.w.end(e);
        self.w.end(p);
    }
}
