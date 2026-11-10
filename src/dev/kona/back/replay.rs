//! Replay of a recorded command stream into a board.
//!
//! A recording is what a guest sent a pipe and what the pipe answered, made by
//! another model of the same board (the file is `KONAREC`, a NUL, a `u32`
//! version, then tagged little-endian records; QEMU's KONA manual page,
//! "Recording the stream", defines it). Replaying it into a board with
//! A null back end checks the front end and the file format before anything
//! draws: the board is built from the `C` record, each `W` is a bus store and
//! each `R` a bus load, in order, retrying while the bus is busy; the board's
//! reads of guest memory are served from the `D` records and its writes are
//! compared with the `O` records.
//!
//! The recorder is another model, not a hardware measurement, so a match
//! shows the front end behaves as that model does and nothing more. What a
//! back end adds (pixel reads, query replies, image DMA, timing) a front end
//! on a null back end cannot reproduce; the report lists those apart.
//!
//! `KONA_SHOT_DIR`, when set, keeps the frame the desktop test composes there as
//! `irix-desktop-ours.png`, to look at beside the capture.
//!
//! The recordings are not in the repository. `KONA_RECORDINGS` names the
//! directory that holds them; without it, or when it is not readable (as in a
//! build container without the storage), the tests that need them skip, with a
//! line on stderr that says so.

use crate::dev::kona::tests::{ScratchDir, TestHost};
use super::super::defs::{ACTIVITY, CAUSE_VIDEO, CREDIT, INT_CAUSE, PROBE};
use super::super::ident::{KonaConfig, Product};
use super::super::Kona;
use super::census::Census;
use crate::traits::{Resettable, BUS_BUSY, BUS_OK};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;

/// One record of a recording, after the board's `C` record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Rec {
    Store { size: u8, off: u32, val: u64 },
    Load { size: u8, off: u32, val: u64 },
    /// Guest memory the pipe read.
    DmaRead { src: u8, addr: u64, bytes: Vec<u8> },
    /// Guest memory the pipe wrote.
    DmaWrite { src: u8, addr: u64, bytes: Vec<u8> },
    Interrupt { kind: u8, vector: u8, delivered: u8 },
    Marker { kind: u32, arg: u32 },
}

/// The board a recording was made on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Board {
    pub part: u32,
    pub revision: u32,
    pub rm_count: u32,
    pub texture_mb: u32,
    pub product: String,
}

#[derive(Debug)]
pub(super) struct Recording {
    pub board: Board,
    pub records: Vec<Rec>,
    /// The record count the end record gave: None for a file cut off.
    pub end: Option<u64>,
}

/// One thing in a recording: the board, a record, or the end with its count.
pub(super) enum Item {
    Board(Board),
    Rec(Rec),
    End(u64),
}

/// A recording read record by record, so a file too big for memory can be
/// replayed.
pub(super) struct Stream<R: Read> {
    r: R,
    /// Bytes read so far, for messages.
    at: u64,
    seen_board: bool,
    ended: bool,
}

impl<R: Read> Stream<R> {
    pub fn new(mut r: R) -> Result<Stream<R>, String> {
        let mut head = [0u8; 12];
        r.read_exact(&mut head).map_err(|_| "not a KONAREC file".to_string())?;
        if &head[..8] != b"KONAREC\0" {
            return Err("not a KONAREC file".into());
        }
        let version = u32::from_le_bytes(head[8..12].try_into().unwrap());
        if version != 1 {
            return Err(format!("format version {version}, not 1"));
        }
        Ok(Stream { r, at: 12, seen_board: false, ended: false })
    }

    fn fill(&mut self, buf: &mut [u8]) -> Result<(), String> {
        self.r.read_exact(buf).map_err(|_| format!("record cut off at byte {}", self.at))?;
        self.at += buf.len() as u64;
        Ok(())
    }

    fn u8(&mut self) -> Result<u8, String> {
        let mut b = [0u8; 1];
        self.fill(&mut b)?;
        Ok(b[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.fill(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let mut b = [0u8; 8];
        self.fill(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, String> {
        let mut v = vec![0u8; n];
        self.fill(&mut v)?;
        Ok(v)
    }

    /// The next item; None at the end of the file between records.
    pub fn next(&mut self) -> Result<Option<Item>, String> {
        let tag_at = self.at;
        let mut t = [0u8; 1];
        let n = self.r.read(&mut t).map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(None);
        }
        self.at += 1;
        if self.ended {
            return Err(format!("data after the end record, at byte {tag_at}"));
        }
        let tag = t[0];
        if tag != b'C' && !self.seen_board {
            return Err("the board record is not first".into());
        }
        Ok(Some(match tag {
            b'C' => {
                let (part, revision, rm_count, texture_mb) = (self.u32()?, self.u32()?, self.u32()?, self.u32()?);
                let n = self.u8()? as usize;
                let product = String::from_utf8_lossy(&self.bytes(n)?).into_owned();
                if std::mem::replace(&mut self.seen_board, true) {
                    return Err(format!("second board record at byte {tag_at}"));
                }
                Item::Board(Board { part, revision, rm_count, texture_mb, product })
            }
            b'W' | b'R' => {
                let (size, off, val) = (self.u8()?, self.u32()?, self.u64()?);
                Item::Rec(if tag == b'W' { Rec::Store { size, off, val } } else { Rec::Load { size, off, val } })
            }
            b'D' | b'O' => {
                let (src, addr, n) = (self.u8()?, self.u64()?, self.u32()? as usize);
                let bytes = self.bytes(n)?;
                Item::Rec(if tag == b'D' { Rec::DmaRead { src, addr, bytes } } else { Rec::DmaWrite { src, addr, bytes } })
            }
            b'I' => Item::Rec(Rec::Interrupt { kind: self.u8()?, vector: self.u8()?, delivered: self.u8()? }),
            b'M' => Item::Rec(Rec::Marker { kind: self.u32()?, arg: self.u32()? }),
            b'E' => {
                self.ended = true;
                Item::End(self.u64()?)
            }
            _ => return Err(format!("unknown record tag {tag:#04x} at byte {tag_at}")),
        }))
    }
}

impl Recording {
    /// Parse a file. A file with no end record parses, with `end` None and
    /// whole records only: the recorder flushes at record boundaries, but the
    /// last record of a killed run may be partial and is an error.
    pub fn parse(bytes: &[u8]) -> Result<Recording, String> {
        let mut stream = Stream::new(bytes)?;
        let mut board = None;
        let mut records = Vec::new();
        let mut end = None;
        while let Some(item) = stream.next()? {
            match item {
                Item::Board(b) => board = Some(b),
                Item::Rec(r) => records.push(r),
                Item::End(n) => end = Some(n),
            }
        }
        let board = board.ok_or("no board record")?;
        if let Some(n) = end {
            // The count is of the records before the end, the board's included.
            if n != records.len() as u64 + 1 {
                return Err(format!("end record counts {n} records, the file has {}", records.len() + 1));
            }
        }
        Ok(Recording { board, records, end })
    }

    /// The `KonaConfig` of the recorded board: the product its record names,
    /// with the texture size that product has.
    pub fn config(&self) -> Result<KonaConfig, String> {
        let product = Product::from_name(&self.board.product).map_err(|e| format!("recorded {e}"))?;
        let want = product.spec().texture_mb();
        if self.board.texture_mb != want {
            return Err(format!("product {:?} has {want} MB of texture memory, the recording says {}", self.board.product, self.board.texture_mb));
        }
        KonaConfig::new(product, self.board.rm_count)
    }
}

/// The directory the recordings are in, if the environment names one that
/// exists. Prints why a test is skipping when not.
pub(super) fn recordings_dir() -> Option<std::path::PathBuf> {
    match std::env::var_os("KONA_RECORDINGS") {
        Some(d) if std::path::Path::new(&d).is_dir() => Some(d.into()),
        Some(d) => {
            skip_note(&format!("KONA_RECORDINGS={d:?} is not a readable directory here"));
            None
        }
        None => {
            skip_note("KONA_RECORDINGS is not set");
            None
        }
    }
}

/// One line saying a replay test is skipping, written to stderr directly so the
/// test harness's capture does not hide it. A build container that cannot see
/// the recordings' storage skips this way: run the built test binary on the
/// host with `KONA_RECORDINGS` set to run them.
fn skip_note(why: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "kona replay test skipped: {why}; the recordings are not readable from here (run the test binary on a host that has them)");
}

/// Builds the machine side of a replay: a test host that serves the board's reads
/// from the recording's `D` records and keeps its writes with the record index.
fn replay_host(rec: &Recording, at: Arc<AtomicUsize>) -> Arc<TestHost> {
    let reads = rec.records.iter().filter_map(|r| match r {
        Rec::DmaRead { src, addr, bytes } => Some((*src, *addr, bytes.clone())),
        _ => None,
    });
    TestHost::replaying(reads.collect(), at)
}

/// A load whose answer depends on timing the replay does not reproduce.
/// Returns the mask of the bits that are compared: zero skips the load.
fn load_mask(off: u32) -> u64 {
    match off {
        // The FIFO's fill level and the watchdog latch follow the thread's
        // pace, which a replay without the thread does not have.
        CREDIT | ACTIVITY => 0,
        // The video bit rises at a retrace, which the recorder times against
        // its own clock.
        INT_CAUSE => !(CAUSE_VIDEO as u64),
        _ => !0,
    }
}

/// What a replay found.
#[derive(Default, Debug)]
pub(super) struct Report {
    pub stores: u64,
    pub loads: u64,
    pub loads_compared: u64,
    /// Loads that differed: (offset, expected, got) to (count, record index
    /// of the first).
    pub load_mismatches: BTreeMap<(u32, u64, u64), (u64, usize)>,
    pub resets: u64,
    /// The board's failure, if it stopped taking commands.
    pub failure: Option<String>,
    /// Recorded interrupts of the geometry-engine kind, and those the board sent.
    pub ge_expected: Vec<u8>,
    pub ge_sent: Vec<u8>,
    pub general_expected: u64,
    pub general_sent: u64,
    /// Retrace completions the recording marked, and the tags the board's
    /// back end was asked for.
    pub retrace_expected: Vec<u32>,
    pub retrace_asked: Vec<u32>,
    /// Swap markers: only a back end that draws can produce them.
    pub swap_markers: u64,
    pub reads_matched: u64,
    pub reads_skipped_over: u64,
    pub reads_unmatched: u64,
    pub reads_unmatched_at: Vec<(u64, usize)>,
    /// Recorded reads no read of the board used, by source.
    pub reads_unused: BTreeMap<u8, (u64, u64)>,
    /// Bytes the board wrote that the recording has the same (run, bytes) for,
    /// bytes it wrote that the recording does not have, and recorded bytes it
    /// never wrote, each by source of the recorded write where known.
    pub writes_same: u64,
    pub writes_extra: Vec<(u64, usize)>,
    pub writes_extra_bytes: u64,
    pub writes_missing: Vec<(u8, u64, usize)>,
    pub writes_missing_bytes: u64,
    pub writes_missing_by_src: BTreeMap<u8, (u64, u64)>,
    /// Writes in order: board runs found in the recorded runs in order.
    pub runs_in_order: u64,
    pub runs_board: u64,
    /// The video Gets' replies the recording shows (a reply id, the data its size takes and the
    /// sum), and how many of them the board wrote, in order; the recorded ones it did not.
    pub video_replies: (u64, u64),
    pub video_replies_lost: Vec<(u64, Vec<u8>)>,
    /// The first record index a load differed at.
    pub first_mismatch: Option<usize>,
    /// The record index of the 64th dropped word in a row (the run of data the
    /// front end could not size began a little before), where a replay into a null back end stops being exact (the back
    /// end answers `Reply::Data` for those; a null one does not).
    pub exact_until: usize,
    /// Records in the recording, after the board's.
    pub records: usize,
    /// Before that index: runs of bytes the board wrote that the recording's
    /// writes also hold, in order, out of all the board wrote; how many runs
    /// the recording has (it has more: what only a back end writes); and the
    /// first board runs it lacks.
    pub exact_runs: (u64, u64),
    pub exact_runs_recorded: u64,
    pub exact_runs_lost: Vec<(u64, usize)>,
}

/// Merge consecutive writes that touch end to start into runs.
fn runs(writes: impl IntoIterator<Item = (u64, Vec<u8>)>) -> Vec<(u64, Vec<u8>)> {
    let mut out: Vec<(u64, Vec<u8>)> = Vec::new();
    for (a, d) in writes {
        match out.last_mut() {
            Some((la, ld)) if *la + ld.len() as u64 == a => ld.extend_from_slice(&d),
            _ => out.push((a, d)),
        }
    }
    out
}

/// Whether a run of written bytes is a video Get's reply: a reply id below 32, the data its size
/// in the server's table takes, and the id plus the data as the last word.
fn is_video_reply(d: &[u8]) -> bool {
    let word = |i: usize| u32::from_be_bytes(d[4 * i..4 * i + 4].try_into().unwrap());
    if d.len() < 12 || d.len() % 4 != 0 {
        return false;
    }
    let id = word(0) as usize;
    let size = crate::dev::kona::tables::VIDEO_REPLY.iter().flatten().find(|(i, _)| *i as usize == id).map(|(_, b)| *b as usize);
    let last = d.len() / 4 - 1;
    size == Some(d.len() - 8) && word(last) == (0..last).fold(0u32, |s, i| s.wrapping_add(word(i)))
}

/// What a replay's board has behind its census tap: a back end that answers every
/// job as nothing (retrace requests complete at once; DMA rows are unclaimed), or the real one.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Behind {
    Null,
    Real,
}

impl Behind {
    /// Put the census tap on `kona`'s back end, and the null double in place of its work if asked.
    fn install(self, kona: &Kona, census: &Arc<Mutex<Census>>, at: &Arc<AtomicUsize>) {
        kona.with_backend(|b| {
            b.tap = Some((census.clone(), at.clone()));
            if self == Behind::Null {
                b.hook = Some(super::null_hook());
            }
        });
    }
}

/// Replay `rec` into a board with a null back end, through the census tap.
pub(super) fn replay(rec: &Recording) -> Result<(Report, Census), String> {
    replay_into(rec, Behind::Null).map(|(rep, census, _)| (rep, census))
}

/// As `replay`, with `behind` it, under the census tap; the board is returned so
/// a test can look at what the back end made.
pub(super) fn replay_into(rec: &Recording, behind: Behind) -> Result<(Report, Census, Arc<Kona>), String> {
    let cfg = rec.config()?;
    let at = Arc::new(AtomicUsize::new(0));
    let host = replay_host(rec, at.clone());
    let census = Arc::new(Mutex::new(Census::default()));
    // The probe register's value is the board's configuration, which the
    // recording leaves out; its first answer is the value the recorder's
    // board had.
    let probe = rec.records.iter().find_map(|r| match r {
        Rec::Load { off, val, .. } if *off == PROBE => Some(*val as u32),
        _ => None,
    });
    let kona: Arc<Kona> = Kona::new(KonaConfig { probe: probe.unwrap_or(0), ..cfg }, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None)?;
    behind.install(&kona, &census, &at);
    let mut rep = Report::default();
    let busy_limit = 50_000_000u32;
    for (index, r) in rec.records.iter().enumerate() {
        at.store(index, Relaxed);
        match *r {
            Rec::Store { size, off, val } => {
                rep.stores += 1;
                let mut n = 0;
                while kona.bus_write(off as u64, size as u32, val) == BUS_BUSY {
                    n += 1;
                    if n > busy_limit {
                        return Err(format!("store at {off:#x} stayed busy"));
                    }
                    kona.run_stores();
                    std::thread::yield_now();
                }
                kona.run_stores();
            }
            Rec::Load { size, off, val } => {
                rep.loads += 1;
                let mut n = 0;
                let got = loop {
                    let (status, data) = kona.bus_read(off as u64, size as u32);
                    if status == BUS_OK {
                        break data;
                    }
                    n += 1;
                    if status != BUS_BUSY || n > busy_limit {
                        return Err(format!("load at {off:#x} answered {status:#x}"));
                    }
                    std::thread::yield_now();
                };
                let mask = load_mask(off);
                if mask != 0 {
                    rep.loads_compared += 1;
                    if got & mask != val & mask {
                        rep.load_mismatches.entry((off, val & mask, got & mask)).or_insert((0, index)).0 += 1;
                        rep.first_mismatch.get_or_insert(index);
                    }
                }
            }
            Rec::Marker { kind: 1, .. } => {
                rep.resets += 1;
                kona.power_on();
            }
            Rec::Marker { kind: 2, .. } => rep.swap_markers += 1,
            Rec::Marker { kind: 3, arg } => rep.retrace_expected.push(arg),
            Rec::Marker { .. } | Rec::DmaRead { .. } | Rec::DmaWrite { .. } => {}
            Rec::Interrupt { kind: 1, vector, .. } => rep.ge_expected.push(vector),
            Rec::Interrupt { .. } => rep.general_expected += 1,
        }
    }
    rep.failure = kona.failure();

    // Interrupts. The recording names a kind and a vector; a packet carries
    // only the vector, so the geometry-engine vector is told from the
    // general one by the vectors the recording used for each.
    let general: Vec<u8> = rec
        .records
        .iter()
        .filter_map(|r| if let Rec::Interrupt { kind: 0, vector, .. } = r { Some(*vector) } else { None })
        .collect();
    for vector in host.sent().iter().filter(|p| p.level).map(|p| p.vector) {
        if rep.ge_expected.contains(&vector) && !general.contains(&vector) {
            rep.ge_sent.push(vector);
        } else {
            rep.general_sent += 1;
        }
    }
    rep.retrace_asked = census.lock().retraces.clone();

    // Reads.
    {
        let r = host.recorded().reads.lock();
        rep.reads_matched = r.matched;
        rep.reads_skipped_over = r.skipped_over;
        rep.reads_unmatched = r.unmatched;
        rep.reads_unmatched_at = r.unmatched_at.clone();
        for rec in r.recs.iter().filter(|x| x.served == 0) {
            let e = rep.reads_unused.entry(rec.src).or_default();
            e.0 += 1;
            e.1 += rec.bytes.len() as u64;
        }
    }

    // Writes: as byte maps, so chunking does not matter, and as runs, so
    // order does.
    let recorded: Vec<(u8, u64, Vec<u8>)> = rec
        .records
        .iter()
        .filter_map(|r| if let Rec::DmaWrite { src, addr, bytes } = r { Some((*src, *addr, bytes.clone())) } else { None })
        .collect();
    let board_all = host.recorded().writes.lock().clone();
    let board: Vec<(u64, Vec<u8>)> = board_all.iter().map(|(_, a, d)| (*a, d.clone())).collect();
    let mut rec_final: BTreeMap<u64, (u8, u8)> = BTreeMap::new();
    for (src, a, d) in &recorded {
        for (i, b) in d.iter().enumerate() {
            rec_final.insert(a + i as u64, (*b, *src));
        }
    }
    let mut board_final: BTreeMap<u64, u8> = BTreeMap::new();
    for (a, d) in &board {
        for (i, b) in d.iter().enumerate() {
            board_final.insert(a + i as u64, *b);
        }
    }
    let mut extra: Vec<u64> = Vec::new();
    for (a, b) in &board_final {
        match rec_final.get(a) {
            Some((rb, _)) if rb == b => rep.writes_same += 1,
            _ => extra.push(*a),
        }
    }
    rep.writes_extra_bytes = extra.len() as u64;
    for run in extra.chunk_by(|a, b| *b == *a + 1) {
        if rep.writes_extra.len() < 8 {
            rep.writes_extra.push((run[0], run.len()));
        }
    }
    let mut missing: Vec<(u64, u8)> = Vec::new();
    for (a, (b, src)) in &rec_final {
        if board_final.get(a) != Some(b) {
            missing.push((*a, *src));
        }
    }
    rep.writes_missing_bytes = missing.len() as u64;
    for run in missing.chunk_by(|a, b| b.0 == a.0 + 1 && b.1 == a.1) {
        let e = rep.writes_missing_by_src.entry(run[0].1).or_default();
        e.0 += 1;
        e.1 += run.len() as u64;
        if rep.writes_missing.len() < 8 {
            rep.writes_missing.push((run[0].1, run[0].0, run.len()));
        }
    }
    let rec_runs = runs(recorded.iter().map(|(_, a, d)| (*a, d.clone())));
    let board_runs = runs(board);
    let mut at = 0;
    for run in &board_runs {
        rep.runs_board += 1;
        if let Some(i) = rec_runs[at..].iter().position(|x| x == run) {
            at += i + 1;
            rep.runs_in_order += 1;
        }
    }

    let mut at = 0;
    for run in rec_runs.iter().filter(|(_, d)| is_video_reply(d)) {
        rep.video_replies.0 += 1;
        match board_runs[at..].iter().position(|x| x == run) {
            Some(i) => {
                at += i + 1;
                rep.video_replies.1 += 1;
            }
            None => rep.video_replies_lost.push(run.clone()),
        }
    }

    let census = std::mem::take(&mut *census.lock());
    rep.records = rec.records.len();
    rep.exact_until = census.first_long_drop.unwrap_or(rep.records);
    let before = |at: usize| at < rep.exact_until;
    let exact_board = runs(board_all.iter().filter(|(at, ..)| before(*at)).map(|(_, a, d)| (*a, d.clone())));
    let exact_rec = runs(rec.records.iter().enumerate().filter(|(at, _)| before(*at)).filter_map(|(_, r)| match r {
        Rec::DmaWrite { addr, bytes, .. } => Some((*addr, bytes.clone())),
        _ => None,
    }));
    let mut at = 0;
    for run in &exact_board {
        rep.exact_runs.1 += 1;
        match exact_rec[at..].iter().position(|x| x == run) {
            Some(i) => {
                at += i + 1;
                rep.exact_runs.0 += 1;
            }
            None if rep.exact_runs_lost.len() < 8 => rep.exact_runs_lost.push((run.0, run.1.len())),
            None => {}
        }
    }
    rep.exact_runs_recorded = exact_rec.len() as u64;
    Ok((rep, census, kona))
}

impl Report {
    pub fn render(&self) -> String {
        let mut o = String::new();
        let _ = writeln!(o, "stores {}  loads {} ({} compared)  resets {}", self.stores, self.loads, self.loads_compared, self.resets);
        let _ = writeln!(o, "board failure: {:?}", self.failure);
        if self.exact_until >= self.records {
            let _ = writeln!(o, "exact over the whole stream; first load mismatch at {:?}", self.first_mismatch);
        } else {
            let _ = writeln!(
                o,
                "exact until record {}, where the {}th dropped word in a row came (a data run the front end could not size, begun a little before); first load mismatch at {:?}",
                self.exact_until,
                super::census::LONG_DROP,
                self.first_mismatch
            );
        }
        let _ = writeln!(
            o,
            "writes before that: {} of the board's {} runs are in the recording's {} in order; not found: {:x?}",
            self.exact_runs.0, self.exact_runs.1, self.exact_runs_recorded, self.exact_runs_lost
        );
        let _ = writeln!(o, "load mismatches: {}", self.load_mismatches.len());
        for ((off, want, got), (n, first)) in &self.load_mismatches {
            let _ = writeln!(o, "  {off:#08x} wanted {want:#x}, got {got:#x}: {n} times, first at record {first}");
        }
        let _ = writeln!(
            o,
            "geometry-engine interrupts: recorded {:?}, sent {:?}; general: recorded {}, sent {}",
            self.ge_expected, self.ge_sent, self.general_expected, self.general_sent
        );
        let _ = writeln!(o, "retraces: recorded {:?}, asked of the back end {:?}; swap markers {}", self.retrace_expected, self.retrace_asked, self.swap_markers);
        let _ = writeln!(
            o,
            "board reads: {} served from records ({} skipped over earlier ones), {} bytes unmatched {:?}",
            self.reads_matched, self.reads_skipped_over, self.reads_unmatched, self.reads_unmatched_at
        );
        let _ = writeln!(o, "recorded reads never used, by source: {:?}", self.reads_unused);
        let _ = writeln!(
            o,
            "board writes: {} bytes equal to the recording, {} bytes not in it {:?}, {} recorded bytes the board did not write {:?}",
            self.writes_same, self.writes_extra_bytes, self.writes_extra, self.writes_missing_bytes, self.writes_missing
        );
        let _ = writeln!(o, "recorded bytes not written, by source (runs, bytes): {:?}", self.writes_missing_by_src);
        let _ = writeln!(o, "board write runs found in the recorded runs, in order: {} of {}", self.runs_in_order, self.runs_board);
        let _ = writeln!(o, "video Get replies the recording shows, written by the board in order: {} of {}; lost: {:x?}", self.video_replies.1, self.video_replies.0, self.video_replies_lost);
        o
    }
}

/// Replay a recording into a board whose back end is a `KonaBackend`, loads
/// and stores only, and return the monitor's `status` and `hash` at the end,
/// after `limit` records or all of them. The recording's resets are performed
/// as in `replay`.
pub(super) fn replay_drawing(rec: &Recording, limit: Option<usize>) -> Result<(String, String), String> {
    let kona = replay_board(rec, limit)?;
    Ok((monitor(&kona, &["status"])?, monitor(&kona, &["hash"])?.trim().to_string()))
}

/// A monitor command of a board's back end, as text.
pub(super) fn monitor(kona: &Kona, args: &[&str]) -> Result<String, String> {
    let mut v = Vec::new();
    kona.with_backend(|b| b.monitor(args, &mut v)).map_err(|e| e.to_string())?;
    String::from_utf8(v).map_err(|e| e.to_string())
}

/// A board with a `KonaBackend` after `limit` records (all of them if None) of
/// `rec`, loads and stores only.
pub(super) fn replay_board(rec: &Recording, limit: Option<usize>) -> Result<Arc<Kona>, String> {
    replay_board_with_display(rec, limit).map(|(kona, _)| kona)
}

/// As `replay_board`, with the window the board shows the guest.
pub(super) fn replay_board_with_display(rec: &Recording, limit: Option<usize>) -> Result<(Arc<Kona>, Arc<super::display::screen::Display>), String> {
    let kona = new_board(rec)?;
    let display = kona.with_backend(|b| b.display());
    play(&kona, rec.records.iter().take(limit.unwrap_or(usize::MAX)));
    Ok((kona, display))
}

/// A board with a `KonaBackend` as the recording's recorder had it, with the recording's
/// reads to serve and no record played.
pub(super) fn new_board(rec: &Recording) -> Result<Arc<Kona>, String> {
    let cfg = rec.config()?;
    let at = Arc::new(AtomicUsize::new(0));
    let host = replay_host(rec, at);
    let probe = rec.records.iter().find_map(|r| match r {
        Rec::Load { off, val, .. } if *off == PROBE => Some(*val as u32),
        _ => None,
    });
    Kona::new(KonaConfig { probe: probe.unwrap_or(0), ..cfg }, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None)
}

/// Play loads, stores and resets of a recording into `kona`.
pub(super) fn play<'a>(kona: &Kona, records: impl Iterator<Item = &'a Rec>) {
    for r in records {
        match *r {
            Rec::Store { size, off, val } => {
                while kona.bus_write(off as u64, size as u32, val) == BUS_BUSY {
                    std::thread::yield_now();
                }
                kona.run_stores();
            }
            Rec::Load { size, off, .. } => {
                while kona.bus_read(off as u64, size as u32).0 == BUS_BUSY {
                    std::thread::yield_now();
                }
            }
            Rec::Marker { kind: 1, .. } => kona.power_on(),
            _ => {}
        }
    }
}

/// The GL demo recording played into a second board loaded from a snapshot of the first at each of `splits`
/// (record numbers; the snapshot is taken at the first record from there on where the stream is between
/// commands) and driven from then on with it. At each of `checks` (record numbers; the first record from
/// there on, once every snapshot is taken, where the stream is between commands) every second board is compared with the first: the chunks
/// it saves (the store, the texture arena and the small state) and its hash. Returns the record each snapshot
/// was taken at, and for each check the record it was made at and the boards that differed.
pub(super) fn gl_resume(path: &std::path::Path, splits: &[usize], checks: &[usize]) -> Result<(Vec<usize>, Vec<(usize, Vec<usize>)>), String> {
    let open = || -> Result<Stream<std::io::BufReader<std::fs::File>>, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Stream::new(std::io::BufReader::with_capacity(1 << 20, f))
    };
    let (mut board, mut reads, mut probe) = (None, Vec::new(), None);
    let mut stream = open()?;
    while let Some(item) = stream.next()? {
        match item {
            Item::Board(b) => board = Some(b),
            Item::Rec(Rec::DmaRead { src, addr, bytes }) => reads.push((src, addr, bytes)),
            Item::Rec(Rec::Load { off, val, .. }) if off == PROBE && probe.is_none() => probe = Some(val as u32),
            _ => {}
        }
    }
    let cfg = Recording { board: board.ok_or("no board record")?, records: Vec::new(), end: None }.config()?;
    let make = || -> Result<(Arc<Kona>, Arc<TestHost>), String> {
        let host = TestHost::replaying(reads.clone(), Arc::new(AtomicUsize::new(0)));
        Ok((Kona::new(KonaConfig { probe: probe.unwrap_or(0), ..cfg.clone() }, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None)?, host))
    };
    let (a, a_host) = make()?;
    let saves = ScratchDir::new("gl-resume");
    let cs = crate::chunk_store::ChunkStore::new(saves.path());
    let mut seconds: Vec<Arc<Kona>> = Vec::new();
    let (mut taken, mut done) = (Vec::new(), Vec::new());
    let mut stream = open()?;
    let mut index = 0usize;
    while let Some(item) = stream.next()? {
        let Item::Rec(r) = item else { continue };
        index += 1;
        play(&a, std::iter::once(&r));
        for b in &seconds {
            play(b, std::iter::once(&r));
        }
        for (k, &split) in splits.iter().enumerate() {
            // A snapshot is refused between the words of a command; the next record that is not.
            if index >= split && taken.len() == k {
                if let Ok(saved) = a.save_chunks(&cs) {
                    let (b, b_host) = make()?;
                    b.load_chunks(&cs, &saved)?;
                    // The second board's memory reads continue where the first's have got to.
                    b_host.recorded().continue_from(a_host.recorded());
                    taken.push(index);
                    seconds.push(b);
                }
            }
        }
        if let Some(&at) = checks.get(done.len()).filter(|&&at| index >= at && taken.len() == splits.len()) {
            if let Ok(ours) = a.save_chunks(&cs) {
                let hash = monitor(&a, &["hash"])?;
                let mut differ = Vec::new();
                for (k, b) in seconds.iter().enumerate() {
                    if b.save_chunks(&cs).map_err(|e| e.to_string())? != ours || monitor(b, &["hash"])? != hash {
                        differ.push(k);
                    }
                }
                done.push((index.max(at), differ));
            }
        }
    }
    Ok((taken, done))
}

/// What a stretch of a stream between two user marks sent.
pub(super) struct Span {
    /// The mark the span starts at (0 for the start of the file).
    pub mark: u32,
    /// Records in it, swap markers and retrace-completion markers it holds.
    pub records: u64,
    pub swaps: u64,
    pub retraces: u64,
    pub census: Census,
    /// The GL path's counts for the span (`kona gl`), when a drawing back end ran.
    pub gl: String,
    /// The store's hash at the swaps `GL_GOLDEN_SWAPS` names in a GL demo's span,
    /// and the pixels in the windmill tower's box that are near red: swap, hash, count.
    pub frames: Vec<(u64, String, u32)>,
}

/// The swaps of each GL demo's span at which the replay takes a hash and a shot.
const GL_GOLDEN_SWAPS: [u64; 3] = [51, 301, 901];

/// The windmill tower's box in screen pixels (x0, y0, x1, y1), where the capture
/// shows a red tower.
const TOWER_BOX: (u32, u32, u32, u32) = (299, 279, 407, 510);

/// How many pixels of the box are near red in the frame the display would show.
fn tower_pixels(kona: &Kona) -> u32 {
    let (w, frame) = kona.with_backend(|b| (b.store.width() as usize, b.pixels().frame()));
    let (x0, y0, x1, y1) = TOWER_BOX;
    let mut n = 0;
    for y in y0 as usize..y1 as usize {
        for x in x0 as usize..x1 as usize {
            let p = frame[y * w + x];
            let (r, g, b) = (p & 0xff, p >> 8 & 0xff, p >> 16 & 0xff);
            n += (r > 150 && g < 80 && b < 80) as u32;
        }
    }
    n
}

/// Replay a recording too big to hold in memory into a board with a census
/// tap, cutting a census at every user mark. Loads are not compared (the
/// stream is past the point where a replay into a null back end stays exact);
/// the point is which tokens the stream sends. Reads the file twice: the
/// board's DMA reads first, since they are recorded after the store that
/// causes them.
pub(super) fn census_spans(path: &std::path::Path) -> Result<Vec<Span>, String> {
    census_spans_into(path, Behind::Null)
}

/// As `census_spans`, with `behind` it, under the census tap; its `gl` monitor
/// text is taken at each mark and cleared.
pub(super) fn census_spans_into(path: &std::path::Path, behind: Behind) -> Result<Vec<Span>, String> {
    census_spans_replies(path, behind, None)
}

/// The query replies the recording shows the board wrote (the data at byte 8 of the return page,
/// at least 52 bytes) and those the board under replay wrote, in order.
#[derive(Default)]
pub(super) struct Replies {
    pub recorded: Vec<Vec<u8>>,
    pub board: Vec<Vec<u8>>,
}

/// As `census_spans_into`, also collecting the query replies if asked.
pub(super) fn census_spans_replies(path: &std::path::Path, behind: Behind, mut replies: Option<&mut Replies>) -> Result<Vec<Span>, String> {
    // KONA_SHOT_DIR: keep the composed frame at some swaps of each span.
    let shots = std::env::var_os("KONA_SHOT_DIR").map(std::path::PathBuf::from);
    let open = || -> Result<Stream<std::io::BufReader<std::fs::File>>, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Stream::new(std::io::BufReader::with_capacity(1 << 20, f))
    };
    let mut board = None;
    let mut reads = Vec::new();
    let mut probe = None;
    let mut stream = open()?;
    while let Some(item) = stream.next()? {
        match item {
            Item::Board(b) => board = Some(b),
            Item::Rec(Rec::DmaRead { src, addr, bytes }) => reads.push((src, addr, bytes)),
            Item::Rec(Rec::DmaWrite { bytes, .. }) if bytes.len() >= 52 => {
                if let Some(r) = replies.as_deref_mut() {
                    r.recorded.push(bytes);
                }
            }
            Item::Rec(Rec::Load { off, val, .. }) if off == PROBE && probe.is_none() => probe = Some(val as u32),
            _ => {}
        }
    }
    let board = board.ok_or("no board record")?;
    let cfg = Recording { board, records: Vec::new(), end: None }.config()?;
    let at = Arc::new(AtomicUsize::new(0));
    let host = TestHost::replaying(reads, at.clone());
    let census = Arc::new(Mutex::new(Census::default()));
    let kona = Kona::new(KonaConfig { probe: probe.unwrap_or(0), ..cfg }, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None)?;
    behind.install(&kona, &census, &at);
    let _ = monitor(&kona, &["gl", "census"]);
    let mut spans = Vec::new();
    let mut frames = Vec::new();
    let (mut mark, mut records, mut swaps, mut retraces) = (0u32, 0u64, 0u64, 0u64);
    let mut stream = open()?;
    let mut index = 0usize;
    let progress = std::env::var("KONA_PROGRESS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0);
    // KONA_STOP_AT=N ends the replay after N records, to compare two builds at the same point.
    let stop_at = std::env::var("KONA_STOP_AT").ok().and_then(|v| v.parse::<usize>().ok());
    while let Some(item) = stream.next()? {
        let Item::Rec(r) = item else { continue };
        at.store(index, Relaxed);
        index += 1;
        records += 1;
        // KONA_PROGRESS=N prints the texture counts every N records, to see how far a long replay got.
        if progress.is_some_and(|n| index % n == 0) {
            let line = monitor(&kona, &["gl"]).unwrap_or_default().lines().find(|l| l.starts_with("textures")).map(str::to_string).unwrap_or_default();
            eprintln!("progress: record {index}, mark {mark}, {swaps} swaps: {line}");
        }
        if stop_at.is_some_and(|n| index >= n) {
            break;
        }
        match r {
            Rec::Store { size, off, val } => {
                let mut n = 0u32;
                while kona.bus_write(off as u64, size as u32, val) == BUS_BUSY {
                    n += 1;
                    if n > 50_000_000 {
                        return Err(format!("store at {off:#x} stayed busy"));
                    }
                    kona.run_stores();
                    std::thread::yield_now();
                }
                kona.run_stores();
            }
            Rec::Load { size, off, .. } => {
                let mut n = 0u32;
                while kona.bus_read(off as u64, size as u32).0 == BUS_BUSY {
                    n += 1;
                    if n > 50_000_000 {
                        return Err(format!("load at {off:#x} stayed busy"));
                    }
                    std::thread::yield_now();
                }
            }
            Rec::Marker { kind: 1, .. } => kona.power_on(),
            Rec::Marker { kind: 2, .. } => {
                swaps += 1;
                // The tex_cube span has two swaps in all: both are kept.
                if (GL_GOLDEN_SWAPS.contains(&swaps) && [210, 220].contains(&mark)) || (mark == 230 && swaps <= 2) {
                    let hash = monitor(&kona, &["hash"]).unwrap_or_default().trim().to_string();
                    frames.push((swaps, hash, tower_pixels(&kona)));
                }
                if let (Some(dir), true) = (&shots, ([51, 301, 901].contains(&swaps) && [210, 220].contains(&mark)) || (mark == 230 && swaps <= 2)) {
                    let file = dir.join(format!("gl-mark{mark}-swap{swaps}.png"));
                    let _ = monitor(&kona, &["shot", file.to_str().unwrap_or("x.png")]);
                }
            }
            Rec::Marker { kind: 3, .. } => retraces += 1,
            Rec::Marker { kind: 0x100, arg } => {
                let done = std::mem::take(&mut *census.lock());
                let gl = monitor(&kona, &["gl"]).unwrap_or_default();
                let _ = monitor(&kona, &["gl", "clear"]);
                spans.push(Span { mark, records, swaps, retraces, census: done, gl, frames: std::mem::take(&mut frames) });
                (mark, records, swaps, retraces) = (arg, 0, 0, 0);
            }
            _ => {}
        }
    }
    let gl = monitor(&kona, &["gl"]).unwrap_or_default();
    spans.push(Span { mark, records, swaps, retraces, census: std::mem::take(&mut *census.lock()), gl, frames });
    if let Some(r) = replies {
        // The board writes the count at byte 4 and the data at byte 8 in one write; the recorder's
        // model wrote the data alone, so the count's four bytes are cut.
        // The host keeps each bus write as it came; adjacent ones in order make a run.
        let mut runs: Vec<(u64, Vec<u8>)> = Vec::new();
        for (_, addr, d) in host.recorded().writes.lock().iter() {
            match runs.last_mut() {
                Some((a, run)) if *a + run.len() as u64 == *addr => run.extend_from_slice(d),
                _ => runs.push((*addr, d.clone())),
            }
        }
        // A reply is the count word then that many words; the previous reply's completion word, when it is
        // adjacent, leads the run.
        let word = |d: &[u8], i: usize| u32::from_be_bytes(d[4 * i..4 * i + 4].try_into().unwrap()) as usize;
        r.board = runs
            .into_iter()
            .filter_map(|(_, d)| match d.len() {
                n if n >= 12 && n % 4 == 0 && word(&d, 0) * 4 + 4 == n => Some(d[4..].to_vec()),
                n if n >= 16 && n % 4 == 0 && word(&d, 1) * 4 + 8 == n => Some(d[8..].to_vec()),
                _ => None,
            })
            .filter(|d| d.len() >= 52)
            .collect();
    }
    Ok(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A recording's bytes, built record by record.
    #[derive(Default)]
    struct Builder(Vec<u8>);

    impl Builder {
        fn new(rm: u32, product: &str) -> Builder {
            let mut b = Builder(b"KONAREC\0".to_vec());
            b.0.extend(1u32.to_le_bytes());
            b.0.push(b'C');
            for v in [0xc102u32, 0, rm, 64] {
                b.0.extend(v.to_le_bytes());
            }
            b.0.push(product.len() as u8);
            b.0.extend(product.as_bytes());
            b
        }

        fn rw(mut self, tag: u8, size: u8, off: u32, val: u64) -> Self {
            self.0.push(tag);
            self.0.push(size);
            self.0.extend(off.to_le_bytes());
            self.0.extend(val.to_le_bytes());
            self
        }

        fn mem(mut self, tag: u8, src: u8, addr: u64, bytes: &[u8]) -> Self {
            self.0.push(tag);
            self.0.push(src);
            self.0.extend(addr.to_le_bytes());
            self.0.extend((bytes.len() as u32).to_le_bytes());
            self.0.extend(bytes);
            self
        }

        fn marker(mut self, kind: u32, arg: u32) -> Self {
            self.0.push(b'M');
            self.0.extend(kind.to_le_bytes());
            self.0.extend(arg.to_le_bytes());
            self
        }

        fn end(mut self, count: u64) -> Vec<u8> {
            self.0.push(b'E');
            self.0.extend(count.to_le_bytes());
            self.0
        }
    }

    /// The store after the PROM recording (see `the_prom_recording_draws_into_the_store`).
    const PROM_STORE_HASH: &str = "48d707b87081b952";

    /// The records of the IRIX recording the front end frames exactly (see
    /// `the_irix_recording_replays`).
    const IRIX_EXACT_RECORDS: usize = 1_415_538;
    /// The GL demo recording's store hash at the swaps its shots are taken at, with
    /// the count of near-red pixels in the windmill tower's box (mark, swap, hash,
    /// tower pixels). Pinned from this model's own output on 2026-11-08, after the
    /// hand-off stopped cutting in window space (which is what made the ivview tower
    /// draw: the capture shows about 11,700 red pixels, the model drew 54 before). Re-pinned
    /// when depth became a 32.16 integer plane: the stored depth words round differently,
    /// so the hashes moved, and the frames moved by three windmill pixels where exact depth
    /// ties now resolve differently (one at swap 51, two at swap 901); Atlantis is identical.
    /// Re-pinned again when lines, points and the polygon stipple began to draw: the
    /// windmill's sails are stippled, so its hashes and tower counts moved (the capture shows
    /// about 11,700); the Atlantis hashes did not. Re-pinned a third time when textures began to draw (and
    /// the matrix double forms made tex_cube's cube draw): the Atlantis and windmill hashes
    /// moved and the windmill's tower counts did not; the tex_cube span, two swaps, is pinned
    /// from this run, and its second frame is within 24 levels of the hardware capture
    /// everywhere but 16 pixels. Re-pinned a fourth time when bitmaps began to draw: the tex_cube
    /// span's 19 bitmaps (its "Loading texture" label, 460 fragments) changed both its swaps,
    /// where the second frame still matches the capture to the same 16 pixels; the Atlantis and
    /// windmill hashes did not move. Re-pinned a fifth time when multisampling began to draw: the
    /// Atlantis window is multisampled (four samples, 16-bit depth), so its span's hashes moved, and
    /// the later spans' with them, since the store keeps the sample words of the earlier window. The
    /// windmill tower counts and the tex_cube frame's distance from the capture (the same sixteen
    /// pixels) did not move. Re-pinned a sixth time when a small texture level took the origin and size the
    /// client's copies give it, instead of the box of every copy: the tex_cube frame's second swap moved in 216
    /// pixels, none by more than 8 levels, and its distance from the capture went from 16 to 17 pixels past 24
    /// levels; the other swaps did not move. Re-pinned a seventh time when window positions began to round to
    /// the nearest 1/256 with ties to even, as the engine's subtract of 49152.0 does, instead of truncating
    /// toward minus infinity: every hash moved (every vertex can land a step away), and the frames moved by 14
    /// pixels at Atlantis's last pin, 94 at the windmill's second and none at the cube's; the windmill tower
    /// counts moved by four pixels at two pins; no frame's distance from its capture changed by more than
    /// three pixels. Re-pinned an eighth time when texture-coordinate generation began to run: the demos
    /// enable generation for s and t (eye linear, the start-up mode) around their textured draws, so
    /// generated coordinates now replace the current ones and every hash moved; the windmill tower
    /// counts did not. They
    /// guard the GL path; they are not hardware truth. A span's hash covers the store as the earlier
    /// spans left it (the sample words of a multisampled window stay), so a change to one demo moves the
    /// pins of the demos after it. Only the windmill span has a tower, so the other counts are zero.
    const GL_GOLDEN: [(u32, u64, &str, u32); 8] = [
        (210, 51, "314e04178c634396", 0),
        (210, 301, "060ac500cc3c9b2c", 0),
        (210, 901, "26e030b433aaa6e8", 0),
        (220, 51, "d968e2f0c8716891", 13711),
        (220, 301, "c238b10a810661ff", 11622),
        (220, 901, "1fc80fbf204ae224", 11629),
        (230, 1, "15c23026e7755afa", 0),
        (230, 2, "44e73640bb50bcf6", 0),
    ];
    const IRIX_STORE_HASH: &str = "3a39733d18360884";
    const IRIX_FULL_STORE_HASH: &str = "ba6c19868b1d31f9";

    /// The share of pixels the PROM's composed frame may differ from its capture by:
    /// none. It measures zero, so any difference is a regression to look at.
    const PROM_FRAME_TOLERANCE: f64 = 0.0;

    const PORT: u32 = 0xc10000;

    #[test]
    fn the_parser_reads_every_record_kind() {
        let mut b = Builder::new(2, "ir2")
            .rw(b'W', 4, PORT, 7)
            .rw(b'R', 8, 0x800240, 0x1122_3344_5566_7788)
            .mem(b'D', 3, 0x1000, &[1, 2, 3])
            .mem(b'O', 1, 0x2000, &[9]);
        b.0.extend([b'I', 1, 0x12, 1]);
        let b = b.marker(1, 0).end(7);
        let r = Recording::parse(&b).unwrap();
        assert_eq!(r.board, Board { part: 0xc102, revision: 0, rm_count: 2, texture_mb: 64, product: "ir2".into() });
        assert_eq!(r.end, Some(7));
        assert_eq!(
            r.records,
            [
                Rec::Store { size: 4, off: PORT, val: 7 },
                Rec::Load { size: 8, off: 0x800240, val: 0x1122_3344_5566_7788 },
                Rec::DmaRead { src: 3, addr: 0x1000, bytes: vec![1, 2, 3] },
                Rec::DmaWrite { src: 1, addr: 0x2000, bytes: vec![9] },
                Rec::Interrupt { kind: 1, vector: 0x12, delivered: 1 },
                Rec::Marker { kind: 1, arg: 0 },
            ]
        );
        assert!(r.config().is_ok());
    }

    #[test]
    fn the_parser_refuses_what_it_cannot_trust() {
        assert!(Recording::parse(b"ODSYREC\0\x01\0\0\0").is_err());
        let ok = Builder::new(2, "ir2").rw(b'W', 4, PORT, 7).end(2);
        assert!(Recording::parse(&ok).is_ok());
        // A wrong count, a cut-off record, an unknown tag, a second version.
        assert!(Recording::parse(&Builder::new(2, "ir2").rw(b'W', 4, PORT, 7).end(3)).unwrap_err().contains("counts 3"));
        // The end record cut in half is a cut-off record, not a short file.
        assert!(Recording::parse(&ok[..ok.len() - 4]).unwrap_err().contains("cut off"));
        let cut = Builder::new(2, "ir2").rw(b'W', 4, PORT, 7).0;
        assert!(Recording::parse(&cut[..cut.len() - 3]).unwrap_err().contains("cut off"));
        let mut bad = Builder::new(2, "ir2").0;
        bad.push(b'Z');
        assert!(Recording::parse(&bad).unwrap_err().contains("unknown record tag"));
        let mut v2 = Builder::new(2, "ir2").0;
        v2[8] = 2;
        assert!(Recording::parse(&v2).unwrap_err().contains("version 2"));
        // A file with no end record parses, and says so.
        assert_eq!(Recording::parse(&cut).unwrap().end, None);
        // Data after the end record, a record before the board, a second board.
        let mut after = ok.clone();
        after.push(b'W');
        assert!(Recording::parse(&after).unwrap_err().contains("after the end record"));
        let mut early = b"KONAREC\0\x01\0\0\0".to_vec();
        early.extend([b'W', 4]);
        assert!(Recording::parse(&early).unwrap_err().contains("not first"));
        let board = Builder::new(2, "ir2").0;
        let mut twice = board.clone();
        twice.extend(&board[12..]);
        assert!(Recording::parse(&twice).unwrap_err().contains("second board"));
        assert!(Recording::parse(&board[..12]).unwrap_err().contains("no board record"));
        // The InfiniteReality4 is not modelled.
        let e = Recording::parse(&Builder::new(2, "ir4").0).unwrap().config().unwrap_err();
        assert!(e.contains("ir4") && e.contains("ir2e"), "{e}");
    }

    /// A recording's product picks the pipe, and its texture size has to be
    /// the product's.
    #[test]
    fn a_recording_names_any_modelled_product() {
        let board = |product: &str, texture_mb| Recording {
            board: Board { part: 0xc102, revision: 0, rm_count: 1, texture_mb, product: product.into() },
            records: Vec::new(),
            end: None,
        };
        for (product, mb) in [("ir2", 64), ("ir2e", 64), ("ir3", 256)] {
            let cfg = board(product, mb).config().unwrap();
            assert_eq!(cfg.spec().name, product);
            assert_eq!(cfg.texture_bytes_per_rm(), (mb as u64) << 20);
        }
        let e = board("ir3", 64).config().unwrap_err();
        assert!(e.contains("256"), "{e}");
    }

    /// The textport probe and a colour select, a mailbox read the reply
    /// answers, a retrace request and a reset marker, through the board.
    #[test]
    fn a_small_stream_replays_and_is_compared() {
        let b = Builder::new(2, "ir2")
            .rw(b'W', 4, PORT, 1)
            .rw(b'W', 4, PORT, 9)
            .rw(b'W', 4, PORT, 0)
            .rw(b'R', 4, 0x800240, 0x1234abcd)
            .rw(b'W', 4, PORT, 0x1e7)
            .rw(b'W', 4, PORT, 0x42)
            .rw(b'R', 4, 0x800248, 0x42)
            .marker(3, 0x42)
            .marker(1, 0)
            .rw(b'R', 4, 0x800248, 0)
            .end(11);
        let rec = Recording::parse(&b).unwrap();
        let (rep, census) = replay(&rec).unwrap();
        assert_eq!(rep.loads_compared, 3, "{}", rep.render());
        assert!(rep.load_mismatches.is_empty(), "{}", rep.render());
        assert_eq!(rep.resets, 1);
        assert_eq!(rep.retrace_asked, [0x42]);
        assert_eq!(rep.retrace_asked, rep.retrace_expected);
        assert_eq!(census.retraces, [0x42]);
        assert!(rep.failure.is_none());
    }

    /// A load that differs from the recorded answer is reported, one that
    /// depends on timing is not compared.
    #[test]
    fn a_wrong_answer_is_reported_and_a_timing_load_is_not() {
        let b = Builder::new(2, "ir2")
            .rw(b'R', 4, 0x04, 0xdead_beef)
            .rw(b'R', 4, CREDIT, 0x7fe)
            .rw(b'R', 4, ACTIVITY, 4)
            .end(4);
        let (rep, ..) = replay(&Recording::parse(&b).unwrap()).unwrap();
        assert_eq!(rep.loads, 3);
        assert_eq!(rep.loads_compared, 1);
        assert_eq!(rep.load_mismatches.keys().map(|k| k.0).collect::<Vec<_>>(), [0x04]);
    }

    /// The board's reads are served from the recorded ones in order, a read
    /// no record covers is counted, and what the board writes is compared
    /// with the recorded writes.
    #[test]
    fn dma_is_served_and_compared() {
        let host = replay_host(&Recording::parse(&Builder::new(2, "ir2").mem(b'D', 3, 0x1000, &[1, 2, 3, 4]).mem(b'D', 3, 0x2000, &[5, 6]).end(3)).unwrap(), Arc::default());
        let mut a = [0u8; 2];
        host.recorded().read_into(0x1000, &mut a);
        assert_eq!(a, [1, 2]);
        let mut a = [0u8; 2];
        host.recorded().read_into(0x1002, &mut a);
        assert_eq!(a, [3, 4]);
        let mut a = [0xffu8; 2];
        host.recorded().read_into(0x9000, &mut a);
        assert_eq!(a, [0, 0]);
        let r = host.recorded().reads.lock();
        assert_eq!((r.matched, r.unmatched, r.cursor), (2, 2, 1));
    }

    /// The recordings, when KONA_RECORDINGS names them. The PROM stream uses
    /// the register aperture only; the IRIX stream adds DMA, contexts and
    /// the GL grammar.
    fn replay_file(name: &str) -> Option<(Recording, Report, Census)> {
        let dir = recordings_dir()?;
        let path = dir.join(name);
        if !path.exists() {
            eprintln!("skipping: {} is not there", path.display());
            return None;
        }
        let rec = Recording::parse(&std::fs::read(&path).unwrap()).unwrap();
        let (rep, census) = replay(&rec).unwrap();
        eprintln!("{name}\n{}", rep.render());
        eprintln!("{}", census.render());
        if let Some(before) = &census.before_long_drop {
            eprintln!("\n=== the same census up to record {}, where the front end could no longer size the stream ===\n{}", rep.exact_until, before.render());
        }
        Some((rec, rep, census))
    }

    /// The PROM stream drawn into the store, to its end. The hash is the
    /// store after the last record: a regression net for the drawing, not
    /// evidence of the pixels, which the display step compares with
    /// the capture. Reset markers clear the store, so this is the last
    /// screen the console drew.
    #[test]
    fn the_prom_recording_draws_into_the_store() {
        let Some(dir) = recordings_dir() else { return };
        let path = dir.join("onyx2-deskside-prom.kona.rec");
        if !path.exists() {
            eprintln!("skipping: {} is not there", path.display());
            return;
        }
        let rec = Recording::parse(&std::fs::read(&path).unwrap()).unwrap();
        let (status, hash) = replay_drawing(&rec, None).unwrap();
        eprintln!("{status}\nstore hash {hash}");
        assert!(status.contains("1 word arrays"), "{status}");
        assert_eq!(hash, PROM_STORE_HASH, "{status}");
        // The PROM draws no GL, so no drain ever finds work queued.
        assert!(status.contains("0 with work queued"), "{status}");
    }

    /// The IRIX stream drawn into the store as far as the front end frames it
    /// exactly with no back end behind it: the X server's state, boxes,
    /// rectangles, spans, points, stipples and the pixel writes and copies of
    /// the greeter's start. The hash is a regression net, not evidence of the
    /// pixels. The whole stream, which only a back end can frame, is
    /// `the_irix_recording_replays_into_the_back_end`.
    #[test]
    fn the_irix_recording_draws_into_the_store_over_its_exact_window() {
        let Some(dir) = recordings_dir() else { return };
        let path = dir.join("onyx2-deskside-irix-4dwm.kona.rec");
        if !path.exists() {
            eprintln!("skipping: {} is not there", path.display());
            return;
        }
        let rec = Recording::parse(&std::fs::read(&path).unwrap()).unwrap();
        let (status, hash) = replay_drawing(&rec, Some(IRIX_EXACT_RECORDS)).unwrap();
        eprintln!("{status}\nstore hash {hash}");
        assert_eq!(hash, IRIX_STORE_HASH, "{status}");
    }

    /// A snapshot taken part way through a stream, loaded into a fresh board, which then plays the
    /// rest, ends with the store and the chunks the uninterrupted board ends with. Chunks of a
    /// saved board that is loaded and saved again are identical.
    fn resumed_from_a_snapshot(name: &str, splits: &[usize]) {
        let Some(dir) = recordings_dir() else { return };
        let path = dir.join(name);
        if !path.exists() {
            eprintln!("skipping: {} is not there", path.display());
            return;
        }
        let rec = Recording::parse(&std::fs::read(&path).unwrap()).unwrap();
        let whole = replay_board(&rec, None).unwrap();
        let want = monitor(&whole, &["hash"]).unwrap();
        for &split in splits {
            let mut split = split.min(rec.records.len());
            let saves = ScratchDir::new(&format!("replay-snap-{split}"));
            let cs = crate::chunk_store::ChunkStore::new(saves.path());
            // A snapshot is refused in the middle of a command; take the next record that is not.
            let (first, saved) = loop {
                let first = replay_board(&rec, Some(split)).unwrap();
                match first.save_chunks(&cs) {
                    Ok(saved) => break (first, saved),
                    Err(e) if e.to_string().contains("middle of a command") && split < rec.records.len() => split += 1,
                    Err(e) => panic!("{name} at {split}: {e}"),
                }
            };
            let second = new_board(&rec).unwrap();
            second.load_chunks(&cs, &saved).unwrap();
            let again = second.save_chunks(&cs).unwrap();
            assert_eq!(saved, again, "{name} at {split}: save, load, save changed the chunks");
            assert_eq!(monitor(&first, &["hash"]).unwrap(), monitor(&second, &["hash"]).unwrap(), "{name} at {split}: the store differs after the load");
            play(&second, rec.records.iter().skip(split));
            assert_eq!(monitor(&second, &["hash"]).unwrap(), want, "{name}: resumed at {split} ends with another store");
            eprintln!("{name}: resumed at {split} of {} records, {} chunk lists", rec.records.len(), saved.len());
        }
    }

    #[test]
    fn the_prom_replay_resumed_from_a_snapshot_reaches_the_same_frame() {
        resumed_from_a_snapshot("onyx2-deskside-prom.kona.rec", &[1000, 20_000, 40_000]);
    }

    #[test]
    fn the_irix_replay_resumed_from_a_snapshot_reaches_the_same_store() {
        resumed_from_a_snapshot("onyx2-deskside-irix-4dwm.kona.rec", &[IRIX_EXACT_RECORDS / 2, IRIX_EXACT_RECORDS]);
    }

    /// A snapshot taken part way through each of the three GL demos, loaded into a fresh board that is then driven with
    /// the first, ends each demo with the same chunks (the store, the texture arena, the small state:
    /// contexts, texture level maps, pixel, feedback and multisample state) and the same hash. The splits are
    /// inside atlantis (a multisampled window with a texture), ivview and tex_cube. Slow: the file is replayed once
    /// with up to four boards.
    #[test]
    fn the_gl_replay_resumed_from_snapshots_ends_the_same() {
        let Some(path) = std::env::var_os("KONA_GL_RECORDING").map(std::path::PathBuf::from) else {
            eprintln!("skipping: set KONA_GL_RECORDING to the GL demo recording");
            return;
        };
        // The boards are compared once all three exist (tex_cube's), at its start, at its end and at the recording's.
        let (taken, checks) = gl_resume(&path, &[33_586_058 + 1_000_000, 65_508_173 + 500_000, 71_263_133 + 10_000], &[71_273_200, 71_283_374, 71_292_000]).unwrap();
        eprintln!("snapshots at {taken:?}; checks {checks:?}");
        assert_eq!(taken.len(), 3);
        assert_eq!(checks.len(), 3, "every check was made");
        assert!(checks.iter().all(|(_, differ)| differ.is_empty()), "boards that ended differently: {checks:?}");
    }


    /// The GL demo recording's query replies: what the recorder's board wrote for the current-value, generic-state (the
    /// matrices the demos read back every frame) and line-limit queries, against what this board writes. The recorder
    /// is a model of the same board, so agreement is a behaviour match and not a claim about the hardware.
    #[test]
    fn the_gl_demo_query_replies_match_the_recording() {
        let Some(path) = std::env::var_os("KONA_GL_RECORDING").map(std::path::PathBuf::from) else {
            eprintln!("skipping: set KONA_GL_RECORDING to the GL demo recording");
            return;
        };
        let mut replies = Replies::default();
        census_spans_replies(&path, Behind::Real, Some(&mut replies)).unwrap();
        // The replies of the three query kinds, by size: current values (52 bytes), a matrix (64) and the line limits (128).
        let by_len = |v: &Vec<Vec<u8>>, n: usize| v.iter().filter(|r| r.len() == n).cloned().collect::<Vec<_>>();
        let mut report = Vec::new();
        for (n, kind) in [(52, "current values"), (64, "matrices"), (128, "line limits")] {
            let (a, b) = (by_len(&replies.recorded, n), by_len(&replies.board, n));
            let same = a.iter().zip(&b).filter(|(x, y)| x == y).count();
            let first = a.iter().zip(&b).position(|(x, y)| x != y);
            eprintln!("{kind}: {} recorded, {} written by this board, {same} identical, first difference at {first:?}", a.len(), b.len());
            if let Some(i) = first {
                eprintln!("  recorded {:08x?}\n  board    {:08x?}", a[i].chunks(4).map(|c| u32::from_be_bytes(c.try_into().unwrap())).collect::<Vec<_>>(), b[i].chunks(4).map(|c| u32::from_be_bytes(c.try_into().unwrap())).collect::<Vec<_>>());
            }
            report.push((n, a.len(), b.len(), same));
        }
        // The line limits are the recorder's own values and match exactly. The current-value replies, now in the
        // table's established layout (colour, index, normal, texture coordinates, edge flag), match word for word
        // except the edge flag: the recorder writes 0 where the default is set and this board writes the float 1.0, which
        // is what the client needs for a set flag (it truncates the word to an integer). The table grades the flag's
        // exact bit pattern inferred, so that one word is not held, and the other twelve are. The
        // matrices are held to within float rounding.
        let (a, b) = (by_len(&replies.recorded, 52), by_len(&replies.board, 52));
        let body = |r: &Vec<u8>| r[..48].to_vec();
        let differ = a.iter().zip(&b).filter(|(x, y)| body(x) != body(y)).count();
        eprintln!("current values, edge flag aside: {differ} differ");
        assert_eq!(differ, 0, "current-value replies that differ outside the edge flag");
        assert_eq!(report[1].1, report[1].2, "a matrix reply for every one recorded");
        // The matrices are built by each side's own float arithmetic from the same commands, so they agree to the
        // last bit or two of the largest element, not always exactly.
        let (a, b) = (by_len(&replies.recorded, 64), by_len(&replies.board, 64));
        let fl = |v: &Vec<u8>| v.chunks(4).map(|c| f32::from_bits(u32::from_be_bytes(c.try_into().unwrap()))).collect::<Vec<_>>();
        let worst = a.iter().zip(&b).map(|(x, y)| {
            let (x, y) = (fl(x), fl(y));
            let scale = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
            x.iter().zip(&y).map(|(p, q)| (p - q).abs() / scale).fold(0f32, f32::max)
        }).fold(0f32, f32::max);
        eprintln!("worst matrix difference, relative to the matrix's largest element: {worst:e}");
        assert!(worst < 2e-7, "{worst:e}");
        assert_eq!(report[2].3, report[2].1, "line-limit replies that differ");
        assert_eq!(report[0].1, report[0].2, "a current-value reply for every one recorded");
    }

    /// The frame the PROM stream composes, against the recorded screen
    /// capture. Both renderers are approximations of one board, so the check is
    /// a tolerance, set from the first measurement: the share of pixels whose
    /// colour differs by more than a few levels in any channel stays under
    /// `PROM_FRAME_TOLERANCE`, zero (the first measurement found no pixel differing;
    /// the recorder is the same model, so that agreement is not hardware
    /// evidence). It catches a layout, placement or colour-map
    /// disagreement to look into; it is not a claim about the hardware's pixels.
    #[test]
    fn the_prom_frame_matches_its_capture_within_a_tolerance() {
        let Some(dir) = recordings_dir() else { return };
        let (rec_path, png_path) = (dir.join("onyx2-deskside-prom.kona.rec"), dir.join("onyx2-deskside-prom-graphics-console.png"));
        if !rec_path.exists() || !png_path.exists() {
            eprintln!("skipping: the PROM recording or its capture is not in {}", dir.display());
            return;
        }
        let rec = Recording::parse(&std::fs::read(&rec_path).unwrap()).unwrap();
        let kona = replay_board(&rec, None).unwrap();
        let shots = ScratchDir::new("prom-shot");
        let shot = shots.join("shot.png");
        let said = monitor(&kona, &["shot", shot.to_str().unwrap()]).unwrap();
        assert!(said.contains("1280 x 1024"), "{said}");
        let (ours, theirs) = (decode_png(&shot), decode_png(&png_path));
        assert_eq!((ours.0, ours.1), (theirs.0, theirs.1));
        let bad = ours.2.chunks_exact(3).zip(theirs.2.chunks_exact(3)).filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.abs_diff(*y) > 8)).count();
        let share = bad as f64 / (ours.0 * ours.1) as f64;
        eprintln!("{bad} of {} pixels differ by more than 8 levels ({:.4})", ours.0 * ours.1, share);
        assert!(share <= PROM_FRAME_TOLERANCE, "{share}");
    }

    /// The window the board shows the guest returns the composed PROM frame:
    /// once the board runs, a frame request is answered at a retrace with the
    /// frame in the screen's buffer, which matches the capture as the shot does.
    #[test]
    fn a_frame_request_on_the_running_board_returns_the_prom_frame() {
        use crate::gfx_display::GfxDisplay;
        use crate::traits::Device;
        let Some(dir) = recordings_dir() else { return };
        let (rec_path, png_path) = (dir.join("onyx2-deskside-prom.kona.rec"), dir.join("onyx2-deskside-prom-graphics-console.png"));
        if !rec_path.exists() || !png_path.exists() {
            eprintln!("skipping: the PROM recording or its capture is not in {}", dir.display());
            return;
        }
        let rec = Recording::parse(&std::fs::read(&rec_path).unwrap()).unwrap();
        let (kona, display) = replay_board_with_display(&rec, None).unwrap();
        kona.start();
        display.request_frame();
        let end = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while display.frame_pending() && std::time::Instant::now() < end {
            std::thread::yield_now();
        }
        assert!(!display.frame_pending(), "the request was answered at a retrace");
        let theirs = decode_png(&png_path);
        let screen = display.screen().lock();
        assert_eq!((screen.width, screen.height), (theirs.0, theirs.1));
        let mut bad = 0;
        for y in 0..theirs.1 {
            for x in 0..theirs.0 {
                let p = screen.rgba[y * crate::disp::FB_STRIDE + x];
                let ours = [p as u8, (p >> 8) as u8, (p >> 16) as u8];
                let at = (y * theirs.0 + x) * 3;
                bad += ours.iter().zip(&theirs.2[at..at + 3]).any(|(a, b)| a.abs_diff(*b) > 8) as usize;
            }
        }
        drop(screen);
        kona.stop();
        let share = bad as f64 / (theirs.0 * theirs.1) as f64;
        eprintln!("{bad} pixels differ ({share:.4})");
        assert!(share <= PROM_FRAME_TOLERANCE, "{share}");
    }

    /// The IRIX recording replayed whole into the drawing back end, with the
    /// board's board built from its `C` record.
    fn replay_irix_into_the_back_end() -> Option<(Report, Census, Arc<Kona>)> {
        let dir = recordings_dir()?;
        let path = dir.join("onyx2-deskside-irix-4dwm.kona.rec");
        if !path.exists() {
            eprintln!("skipping: {} is not there", path.display());
            return None;
        }
        let rec = Recording::parse(&std::fs::read(&path).unwrap()).unwrap();
        Some(replay_into(&rec, Behind::Real).unwrap())
    }

    /// The video Get replies the IRIX recording holds.
    const VIDEO_REPLIES_RECORDED: u64 = 63;

    /// The whole IRIX stream, which the front end can now frame to its end: the
    /// back end sizes the pixel writes' rows, so no word is dropped. What the
    /// recording shows the board wrote to guest memory is what the back end
    /// writes, byte for byte: the pixel reads' rows (about 250 KB of them, by
    /// DMA into host buffers) come out of this store the same as out of the
    /// recorder's, and the two completions that follow them match. Both models
    /// are approximations of one board, so that agreement is a behaviour match
    /// and not a statement about the hardware's pixels.
    #[test]
    fn the_irix_recording_replays_into_the_back_end() {
        let Some((rep, census, kona)) = replay_irix_into_the_back_end() else { return };
        eprintln!("{}", rep.render());
        let status = monitor(&kona, &["status"]).unwrap();
        eprintln!("{status}");
        assert!(rep.failure.is_none(), "{}", rep.render());
        assert!(rep.load_mismatches.is_empty(), "{}", rep.render());
        assert_eq!(rep.exact_until, rep.records, "no run of dropped words: the stream frames whole");
        assert_eq!(census.dropped, 0);
        assert_eq!(rep.ge_sent, rep.ge_expected, "{}", rep.render());
        assert_eq!((rep.writes_extra_bytes, rep.writes_missing_bytes), (0, 0), "{}", rep.render());
        assert_eq!(rep.writes_same, 251_920, "{}", rep.render());
        // The display generator's Gets, which the stream's start-up sends and the server waits on: every
        // reply the recording shows, written the same, in order.
        assert_eq!(rep.video_replies, (VIDEO_REPLIES_RECORDED, VIDEO_REPLIES_RECORDED), "{}", rep.render());
        assert_eq!(rep.runs_in_order, rep.runs_board, "{}", rep.render());
        // Every pixel command the stream sends is drawn, none refused.
        assert!(status.contains("448 setups, 692 writes (6 by DMA), 113 copies, 63 reads, 272 finishes"), "{status}");
        assert!(status.contains("refused         0 context, 0 target, 0 destination, 0 window copies, 0 reads answered with zeros (0 frame-buffer word)"), "{status}");
        assert!(status.contains("0 short or left owing, 0 with no write waiting"), "{status}");
        // No GL is queued while the desktop draws, so no drain ever found work waiting.
        assert!(status.contains("0 with work queued"), "{status}");
        // The desktop's pointer is the XMAP glyph; the stream never loads or enables a DAC cursor.
        assert!(status.contains("DAC cursor        0 loads, 0 refused; held for channels []; channel 0's is not drawn"), "{status}");
        let hash = monitor(&kona, &["hash"]).unwrap().trim().to_string();
        eprintln!("store hash {hash}");
        assert_eq!(hash, IRIX_FULL_STORE_HASH, "{status}");
    }

    /// The frame the board composes at the end of the IRIX stream, against the
    /// recorded capture of the desktop. The desktop is index windows through the
    /// X colour map and the gamma RAM, composed from what the stream loaded and
    /// drew. Both renderers are approximations of one board and the capture is the
    /// recorder's own, so this agreement shows the composition reads the tables as
    /// the recorder's model does and says nothing about the hardware's pixels.
    ///
    /// Every pixel is within one level of the capture, and no pixel is further. The only
    /// remaining cause is the recorder's widening of the 10-bit colour-map entry to the
    /// 12-bit gamma index: it rounds the entry to 8 bits first and widens that, where the
    /// pipe's own rule replicates the top two bits (graded inferred: the only rule the
    /// board's gamma addressing is known to fit), which with the identity gamma RAM is the
    /// entry's top eight bits. The 12-bit gamma output reaches the 8-bit DAC inputs by dropping
    /// the low four bits, as the recorder does (established on the link from the colour-map
    /// chips to the video-output chips, by the board's wiring). With the widening the recorder's
    /// way too, the frame matched the capture exactly (measured on 2026-11-10: 0 of 1310720
    /// pixels differ). Measured with the truncation in: 644505 of 1310720 pixels differ, all by
    /// one level (rounding the entry and dropping its low two bits part at about every other
    /// entry). The glyph cursor is enabled at the end of the stream, but neither this shot nor
    /// the capture shows one, so none is compared.
    #[test]
    fn the_irix_desktop_frame_matches_its_capture_within_a_tolerance() {
        let Some(dir) = recordings_dir() else { return };
        let png_path = dir.join("onyx2-deskside-irix-4dwm-desktop.png");
        if !png_path.exists() {
            eprintln!("skipping: {} is not there", png_path.display());
            return;
        }
        let Some((_, _, kona)) = replay_irix_into_the_back_end() else { return };
        let shots = ScratchDir::new("irix-shot");
        let shot = shots.join("shot.png");
        monitor(&kona, &["shot", shot.to_str().unwrap()]).unwrap();
        let (ours, theirs) = (decode_png(&shot), decode_png(&png_path));
        // KONA_SHOT_DIR keeps the frame, to look at beside the capture.
        if let Some(dir) = std::env::var_os("KONA_SHOT_DIR") {
            let _ = std::fs::copy(&shot, std::path::Path::new(&dir).join("irix-desktop-ours.png"));
        }
        assert_eq!((ours.0, ours.1), (theirs.0, theirs.1));
        let widest = ours.2.iter().zip(&theirs.2).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        let off_by_one = ours.2.chunks_exact(3).zip(theirs.2.chunks_exact(3)).filter(|(a, b)| a != b).count();
        let bad = ours.2.chunks_exact(3).zip(theirs.2.chunks_exact(3)).filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.abs_diff(*y) > 1)).count();
        eprintln!("{off_by_one} of {} pixels differ, the widest by {widest} level, {bad} by more than one", ours.0 * ours.1);
        assert_eq!(bad, 0, "pixels further than one level from the capture");
    }

    /// A PNG's width, height and RGB bytes.
    fn decode_png(path: &std::path::Path) -> (usize, usize, Vec<u8>) {
        let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
        let mut r = dec.read_info().unwrap();
        let mut buf = vec![0; r.output_buffer_size().expect("a PNG that fits memory")];
        let info = r.next_frame(&mut buf).unwrap();
        let bytes = &buf[..info.buffer_size()];
        let (w, h) = (info.width as usize, info.height as usize);
        let rgb = match info.color_type {
            png::ColorType::Rgb => bytes.to_vec(),
            png::ColorType::Rgba => bytes.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect(),
            other => panic!("unexpected PNG colour type {other:?}"),
        };
        (w, h, rgb)
    }

    #[test]
    fn the_prom_recording_replays() {
        let Some((rec, rep, ..)) = replay_file("onyx2-deskside-prom.kona.rec") else { return };
        assert_eq!(rec.end, Some(rec.records.len() as u64 + 1));
        assert!(rep.failure.is_none(), "{}", rep.render());
        assert!(rep.load_mismatches.is_empty(), "{}", rep.render());
        assert_eq!(rep.ge_expected, rep.ge_sent);
        assert_eq!(rep.exact_runs, (0, 0));
    }

    #[test]
    fn the_irix_recording_replays() {
        let Some((rec, rep, census)) = replay_file("onyx2-deskside-irix-4dwm.kona.rec") else { return };
        assert_eq!(rec.end, Some(rec.records.len() as u64 + 1));
        assert!(rep.failure.is_none(), "{}", rep.render());
        // Exact as far as the front end can frame the stream alone: every
        // deterministic load, and every byte the board wrote to guest memory,
        // matches the recording before the first run of data only a back end
        // can size (measured as the 64th dropped word in a row). Past it the stream misframes, as it must on a null back end.
        // The window is pinned: a front-end change that makes the stream
        // misframe sooner shrinks what is checked, and must fail here.
        assert_eq!(rep.exact_until, 1_415_538, "{}", rep.render());
        assert_eq!(rep.exact_runs, (4, 4), "{}", rep.render());
        assert_eq!(rep.ge_sent, rep.ge_expected, "{}", rep.render());
        assert!(rep.first_mismatch.is_none_or(|at| at >= rep.exact_until), "{}", rep.render());
        assert!(rep.exact_runs.1 > 0 && rep.exact_runs.0 == rep.exact_runs.1, "{}", rep.render());
        // The recording has more: the data a pixel read's reply carries, which
        // only a back end posts.
        assert!(rep.exact_runs.1 < rep.exact_runs_recorded, "{}", rep.render());
        // Video interrupts depend on timing, so only that some came is
        // checked: the back end completes no retrace here, so none is sent.
        assert!(rep.general_expected > 0);
        assert!(census.jobs > 0);
    }

    /// The GL demo recording, when KONA_GL_RECORDING names it (a gigabyte: a
    /// minute or two in a debug build). Prints each demo's span; asserts that
    /// the marks are there and that GL commands reached the back end.
    #[test]
    fn the_gl_demo_recording_is_censused() {
        let Some(path) = std::env::var_os("KONA_GL_RECORDING").map(std::path::PathBuf::from) else {
            eprintln!("skipping: set KONA_GL_RECORDING to the GL demo recording");
            return;
        };
        let spans = census_spans(&path).unwrap();
        let marks: Vec<u32> = spans.iter().map(|s| s.mark).collect();
        for m in [210, 220, 230] {
            assert!(marks.contains(&m), "mark {m} not in {marks:?}");
        }
        for sp in spans.iter().filter(|s| [210, 220, 230].contains(&s.mark)) {
            eprintln!("=== span from mark {}: {} records, {} swap markers, {} retrace markers ===", sp.mark, sp.records, sp.swaps, sp.retraces);
            eprintln!("{}", sp.census.render());
            eprintln!("{}", sp.census.render_families());
            assert!(sp.census.jobs > 0);
        }
    }

    /// The GL demo recording through the GL path with the census on: where its
    /// vertices fall against the window, which texture-coordinate forms it
    /// sends, what the path does not take. Prints each demo's span. The
    /// recording is a gigabyte (a few minutes in a debug build).
    #[test]
    fn the_gl_demo_recording_is_censused_through_the_gl_path() {
        let Some(path) = std::env::var_os("KONA_GL_RECORDING").map(std::path::PathBuf::from) else {
            eprintln!("skipping: set KONA_GL_RECORDING to the GL demo recording");
            return;
        };
        let cfg = KonaConfig::default();
        let spans = census_spans_into(&path, Behind::Real).unwrap();
        for sp in spans.iter().filter(|s| [210, 220, 230].contains(&s.mark)) {
            eprintln!("=== GL path, span from mark {}: {} records ===\n{}", sp.mark, sp.records, sp.gl);
            assert!(sp.gl.contains("vertex census"), "{}", sp.gl);
            for (swap, hash, tower) in &sp.frames {
                eprintln!("golden mark {} swap {swap}: store hash {hash}, tower pixels {tower}", sp.mark);
            }
        }
        let got: Vec<(u32, u64, &str, u32)> = spans.iter().filter(|s| [210, 220, 230].contains(&s.mark)).flat_map(|s| s.frames.iter().map(move |(w, h, t)| (s.mark, *w, h.as_str(), *t))).collect();
        assert_eq!(got, GL_GOLDEN.to_vec(), "the GL demos' store hashes at the shot swaps moved: a change to the GL path or the pixel module, or a fix; look at the shots");
    }

    /// The Quake recording (SGI Quake GL, then TyrQuake) through the GL path: how many texture loads
    /// and binds each span's replay refuses. Texture memory is one pool the games load into and reload
    /// within; a level reloaded at the same place overwrites it and a retired level's space is free again,
    /// so a long run must not fill the product's memory (`back::store::texture`). The recording is four
    /// gigabytes (several minutes in a release build).
    #[test]
    fn the_quake_recording_keeps_texture_memory_in_bounds() {
        let Some(path) = std::env::var_os("KONA_QUAKE_RECORDING").map(std::path::PathBuf::from) else {
            eprintln!("skipping: set KONA_QUAKE_RECORDING to the Quake recording");
            return;
        };
        let spans = census_spans_into(&path, Behind::Real).unwrap();
        // The "textures" line reads "... {E} past memory), {F} finishes, {G} binds ({H} of volumes, {I} refused) ...".
        let past_of = |text: &str| -> u64 { text.lines().find(|l| l.starts_with("textures")).and_then(|l| l.split(" past memory").next()).and_then(|l| l.split_whitespace().last()).and_then(|t| t.parse().ok()).unwrap_or(0) };
        let binds_refused_of = |text: &str| -> u64 { text.lines().find(|l| l.starts_with("textures")).and_then(|l| l.split(" of volumes, ").nth(1)).and_then(|l| l.split_whitespace().next()).and_then(|t| t.parse().ok()).unwrap_or(0) };
        let (mut past, mut refused_binds) = (0u64, 0u64);
        for sp in &spans {
            let (p, b) = (past_of(&sp.gl), binds_refused_of(&sp.gl));
            eprintln!("quake span from mark {}: {} records, {p} loads past texture memory, {b} binds refused", sp.mark, sp.records);
            for l in sp.gl.lines().filter(|l| l.starts_with("textures") || l.starts_with("texture memory")) {
                eprintln!("  {l}");
            }
            past += p;
            refused_binds += b;
        }
        eprintln!("quake total: {past} loads past texture memory, {refused_binds} refused binds");
        assert_eq!(past, 0, "a load was refused as past texture memory");
        assert_eq!(refused_binds, 0, "a bind was refused");
    }

    /// A list pull is recorded by its offset in the pool, but the board reads it at
    /// the host addresses its map RAM gives: the replay serves it in order, and a
    /// read nothing covers is still answered with zeros and counted.
    #[test]
    fn a_list_pull_is_served_from_the_record_keyed_by_pool_offset() {
        let at = Arc::new(AtomicUsize::new(0));
        let reads = vec![(2, 0x1000, vec![1, 2, 3, 4]), (4, 0x80c, vec![9, 8, 7, 6, 5, 4, 3, 2]), (4, 0xc0c, vec![1; 8])];
        let host = TestHost::replaying(reads, at);
        let mut word = [0u8; 4];
        host.recorded().read_into(0x1000, &mut word);
        assert_eq!(word, [1, 2, 3, 4]);
        let mut pull = [0u8; 8];
        host.recorded().read_into(0x1_03f2_480c, &mut pull);
        assert_eq!(pull, [9, 8, 7, 6, 5, 4, 3, 2]);
        host.recorded().read_into(0x055f_c00c, &mut pull);
        assert_eq!(pull, [1; 8]);
        let mut none = [7u8; 8];
        host.recorded().read_into(0x7777, &mut none);
        assert_eq!(none, [0; 8]);
        assert_eq!(host.recorded().reads.lock().unmatched, 8);
    }
}
