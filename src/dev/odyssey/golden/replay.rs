//! Replay of recorded guest sessions (`record.rs`), headless.
//!
//! The recording's stores go through a fresh board's register interface in
//! order, its DMA reads are answered from the recording, and the board's
//! planes are hashed at checkpoints and compared with the hashes kept
//! beside the recording in `testdata/replay/<name>.hashes`. The recordings
//! themselves stay out of git (see `testdata/README.md`); they are found in
//! `VPRO_REPLAY_DIR`, default `ignore/vpro-golden/` under the checkout.
//!
//! `VPRO_REPLAY=<file.rec>` replays one file, `VPRO_REPLAY_OUT=<file.png>`
//! saves the screen it composes, `VPRO_REPLAY_DUMP=<dir>` board memory as
//! `board.bin`, as `vpro fbdump` writes it (to diff against a live board's
//! dump), `VPRO_REPLAY_BLESS=1` with
//! `cargo test ... replay_bless -- --ignored` re-takes every recording's
//! checkpoints. `VPRO_REPLAY_LIMIT=<n>` stops after n records, so a dump
//! shows the board there; `VPRO_REPLAY_THREADS=1` draws on the raster and
//! back-end threads; `VPRO_REPLAY_GPU=1` draws on the GPU raster unit
//! (with the `wgpu` feature).

use crate::dev::odyssey::board::{Board, ODSY_HEIGHT, ODSY_WIDTH};
use crate::dev::odyssey::consts::{cmd, gl_cmd, image, reg, ODSY_BUZZ_1};
use crate::dev::odyssey::clip::ClipRect;
use crate::dev::odyssey::pixel_dma::{read_guest, GuestMemory};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use super::super::record::{self, Rec};
use crate::traits::{BusDevice, BusRead8, BUS_ERR, BUS_OK};

/// Records between checkpoints.
pub const EVERY: usize = 500_000;

use super::super::debug::{dump_memory, memory_hash, memory_hash_bytes};

/// Guest memory for replay: answers the board's DMA reads with the recorded
/// values in order, and notes any that do not match the recorded address.
pub struct ReplayMem {
    reads: Vec<Rec>,
    next: Mutex<usize>,
    pub mismatches: Mutex<u64>,
    /// What the board wrote to guest memory (pixel reads, fences, query
    /// results), hashed in order with each write's address: two replays of
    /// one recording on different raster units read back the same pixels
    /// exactly when this matches.
    pub writes: Mutex<(blake3::Hasher, u64)>,
    /// `VPRO_REPLAY_WRITES=<file>`: each write as a line, address and words.
    log: Option<Mutex<std::io::BufWriter<std::fs::File>>>,
}

impl ReplayMem {
    pub fn new(recs: &[Rec]) -> Self {
        let reads = recs.iter().filter(|r| matches!(r, Rec::DmaBlock { .. } | Rec::DmaByte { .. })).cloned().collect();
        let log = std::env::var("VPRO_REPLAY_WRITES").ok().and_then(|f| std::fs::File::create(f).ok()).map(|f| Mutex::new(std::io::BufWriter::new(f)));
        ReplayMem { reads, next: Mutex::new(0), mismatches: Mutex::new(0), writes: Mutex::new((blake3::Hasher::new(), 0)), log }
    }
    pub fn consumed(&self) -> (usize, usize) {
        (*self.next.lock(), self.reads.len())
    }
    fn take(&self) -> Option<&Rec> {
        let mut n = self.next.lock();
        let r = self.reads.get(*n);
        *n += r.is_some() as usize;
        r
    }
}

impl BusDevice for ReplayMem {
    fn read8(&self, addr: u64) -> BusRead8 {
        match self.take() {
            Some(Rec::DmaByte { addr: a, val }) if *a == addr => BusRead8::ok(*val),
            _ => {
                *self.mismatches.lock() += 1;
                BusRead8::ok(0)
            }
        }
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        let mut w = self.writes.lock();
        w.0.update(&addr.to_be_bytes());
        w.0.update(&[val]);
        w.1 += 1;
        if let Some(l) = &self.log {
            use std::io::Write;
            let _ = writeln!(l.lock(), "{addr:#x} b {val:02x}");
        }
        BUS_OK
    }
    fn read_block(&self, addr: u64, buf: &mut [u64]) -> u32 {
        match self.take() {
            Some(Rec::DmaBlock { addr: a, words }) if *a == addr && words.len() == buf.len() => {
                buf.copy_from_slice(words);
                BUS_OK
            }
            _ => {
                *self.mismatches.lock() += 1;
                buf.fill(0);
                BUS_ERR
            }
        }
    }
    fn write_block(&self, addr: u64, buf: &[u64]) -> u32 {
        let mut w = self.writes.lock();
        w.0.update(&addr.to_be_bytes());
        for v in buf {
            w.0.update(&v.to_be_bytes());
        }
        w.1 += 8 * buf.len() as u64;
        if let Some(l) = &self.log {
            use std::io::Write;
            let mut l = l.lock();
            let _ = write!(l, "{addr:#x} {}", buf.len());
            for v in buf {
                let _ = write!(l, " {v:016x}");
            }
            let _ = writeln!(l);
        }
        BUS_OK
    }
}

/// What a replay saw.
#[derive(Debug, Default)]
pub struct Report {
    pub records: usize,
    /// The checkpoints reached: record index and plane hash.
    pub hashes: Vec<(usize, String)>,
    pub dma_used: usize,
    pub dma_total: usize,
    pub dma_mismatched: u64,
    pub reads_differ: u64,
    /// The board's writes to guest memory: their hash and byte count.
    pub host_writes: (String, u64),
    /// The recording's marks (`vpro rec`), and those whose hash the replayed
    /// planes did not reach.
    pub marks: usize,
    pub marks_failed: Vec<usize>,
    /// The recording's opening mark is one a board at power-on does not
    /// reach: it was started mid-session, so its marks cannot match; its
    /// replay is still deterministic and its checkpoints still a regression
    /// net.
    pub mid_session: bool,
    /// The time the stores took to replay and draw, without the
    /// checkpoints' hashing and screenshots.
    pub draw_secs: f64,
}

/// Whether `recs` store to the PB&J scan-out entries (0x2800-0x280f) through
/// the back end's word port.
fn programs_scan_out(recs: &[Rec]) -> bool {
    let (mut index, mut count) = (0u32, 0u32);
    for r in recs {
        let Rec::Write { size, off, val } = r else { continue };
        if *off as u64 != super::super::consts::backend::SELECT {
            continue;
        }
        let words: &[u32] = if *size == 8 { &[(*val >> 32) as u32, *val as u32] } else { &[*val as u32] };
        for &w in words {
            if count == 0 {
                if (w >> 28) & 3 == 3 {
                    (index, count) = ((w >> 14) & 0x3fff, w & 0x3fff);
                }
            } else {
                if (0x2800..0x2810).contains(&index) {
                    return true;
                }
                index += 1;
                count -= 1;
            }
        }
    }
    false
}

/// Feed `recs` through a fresh board's registers, `setup` run on it first.
/// `board_out` receives the board when it is done.
/// How a replay draws: on the GPU raster unit or not, with its readbacks
/// skipped (`lazy`, for timing), ending a board slice every `tick` stores.
#[derive(Clone, Copy, Debug, Default)]
pub struct Opts {
    pub gpu: bool,
    pub lazy: bool,
    pub tick: Option<usize>,
}

impl Opts {
    /// `VPRO_REPLAY_GPU`, `VPRO_REPLAY_LAZY`, `VPRO_REPLAY_TICK`.
    pub fn from_env() -> Opts {
        Opts {
            gpu: std::env::var("VPRO_REPLAY_GPU").is_ok(),
            lazy: std::env::var("VPRO_REPLAY_LAZY").is_ok(),
            tick: std::env::var("VPRO_REPLAY_TICK").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0),
        }
    }
}

pub fn run(recs: &[Rec], setup: impl FnOnce(&mut Board), every: usize) -> (Report, Box<Board>) {
    run_with(recs, setup, every, Opts::from_env())
}

pub fn run_with(recs: &[Rec], setup: impl FnOnce(&mut Board), every: usize, opts: Opts) -> (Report, Box<Board>) {
    let mut b = Board::new("v12", 11).expect("board");
    // A recording with marks (version 2) was taken from a board the machine
    // reset at power-on (the depth plane at all ones): start there, as the
    // machine did. Older recordings replay on the bare board their
    // checkpoints were taken on.
    if recs.iter().any(|r| matches!(r, Rec::Mark(_))) {
        b.reset();
    }
    // A recording that never programs the scan-out (one started after the
    // X server set it up) starts from the board as X leaves it
    // (Board::test_screen()).
    if !programs_scan_out(recs) {
        b.test_screen();
    }
    let mem = Arc::new(ReplayMem::new(recs));
    b.guest = Some(mem.clone());
    setup(&mut b);
    // VPRO_REPLAY_GPU=1: draw on the GPU raster unit.
    if opts.gpu {
        #[cfg(feature = "wgpu")]
        {
            b.enable_gpu(&crate::wgpu::GpuOptions::default()).expect("the GPU raster unit");
            // VPRO_REPLAY_LAZY=1: no readbacks between checkpoints, to time
            // the GPU path alone (the planes the front end reads go stale,
            // so the checkpoints are not expected to hold).
            if opts.lazy {
                b.raster.run_now(|ctx| ctx.gpu.as_mut().expect("the unit").lazy = true);
            }
        }
        #[cfg(not(feature = "wgpu"))]
        panic!("VPRO_REPLAY_GPU needs a build with the wgpu feature");
    }
    // VPRO_REPLAY_THREADS=1: draw on the raster and back-end threads, as a
    // running machine does; the checkpoints must not move.
    let threads = std::env::var("VPRO_REPLAY_THREADS").is_ok();
    if threads {
        // VPRO_REPLAY_POOL=<threads>,<tile>: the pipeline's workers and tile
        // size (default 2, 32, as `[graphics]`).
        if let Ok(v) = std::env::var("VPRO_REPLAY_POOL") {
            let mut it = v.split(',').map(|n| n.trim().parse::<u32>().unwrap_or(0));
            let (n, t) = (it.next().unwrap_or(2), it.next().unwrap_or(32));
            b.raster.set_pool(n.max(1) as usize, t);
        }
        b.raster.start();
    }
    let mut rep = Report { records: recs.len(), ..Default::default() };
    // VPRO_REPLAY_LIMIT=<n>: stop after n records (to dump the board there).
    let limit = std::env::var("VPRO_REPLAY_LIMIT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(usize::MAX);
    // VPRO_REPLAY_SHOTS=<dir>: the screen at each mark, as mark-NNN.png.
    let shots = std::env::var("VPRO_REPLAY_SHOTS").ok();
    // VPRO_REPLAY_EVERY=<n> with VPRO_REPLAY_SHOTS: the screen every n
    // records too, as rec-NNNNNNNNN.png.
    let shot_every = std::env::var("VPRO_REPLAY_EVERY").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0);
    // VPRO_REPLAY_TICK=<n>: end a slice every n stores, as the board's thread
    // does between runs of stores (`gpu_upload_tick`, the board going idle):
    // the GPU unit then submits mid-stream as it does live.
    let tick = opts.tick;
    let mut stores = 0usize;
    let t0 = std::time::Instant::now();
    let mut aside = std::time::Duration::ZERO;
    for (i, r) in recs.iter().enumerate().take(limit) {
        if let (Some(n), Some(dir)) = (shot_every, &shots) {
            if i % n == 0 && i != 0 {
                b.gpu_sync();
                let _ = std::fs::create_dir_all(dir);
                save_screen(&b, &format!("{dir}/rec-{i:09}.png"));
            }
        }
        match r {
            Rec::Write { size, off, val } => {
                b.mmio_write(*off as u64, *val, *size as u32);
                stores += 1;
                if tick.is_some_and(|n| stores % n == 0) {
                    b.gpu_upload_tick(true);
                }
            }
            Rec::Read { size, off, val } => {
                let mask = if *size >= 8 { u64::MAX } else { (1u64 << (8 * *size as u32)) - 1 };
                if b.mmio_read(*off as u64, *size as u32) & mask != val & mask {
                    rep.reads_differ += 1;
                }
            }
            Rec::DmaBlock { .. } | Rec::DmaByte { .. } => {}
            Rec::Mark(h) => {
                b.gpu_sync();
                let t = std::time::Instant::now();
                let got = memory_hash_bytes(&b);
                rep.marks += 1;
                if got != *h {
                    // The opening mark not reached: the recording did not
                    // start from power-on.
                    rep.mid_session |= rep.marks == 1;
                    rep.marks_failed.push(i + 1);
                }
                rep.hashes.push((i + 1, blake3::Hash::from(got).to_hex().to_string()));
                if let Some(dir) = &shots {
                    let _ = std::fs::create_dir_all(dir);
                    save_screen(&b, &format!("{dir}/mark-{:03}.png", rep.marks));
                }
                aside += t.elapsed();
            }
        }
        if every != 0 && (i + 1) % every == 0 && rep.hashes.last().is_none_or(|(j, _)| *j != i + 1) {
            b.gpu_sync();
            let t = std::time::Instant::now();
            rep.hashes.push((i + 1, memory_hash(&b)));
            if let Some(dir) = &shots {
                let _ = std::fs::create_dir_all(dir);
                save_screen(&b, &format!("{dir}/check-{:03}.png", rep.hashes.len()));
            }
            aside += t.elapsed();
        }
    }
    b.gpu_sync();
    rep.draw_secs = (t0.elapsed() - aside).as_secs_f64();
    if threads {
        b.raster.stop();
    }
    #[cfg(feature = "wgpu")]
    if b.gpu_unit() {
        let (batches, records, dropped, up, blocks, tiles, down, readbacks) = b.raster.run_now(|ctx| {
            let g = ctx.gpu.as_ref().expect("the unit");
            let (batches, records) = g.counts();
            (batches, records, g.dropped, g.uploaded, g.uploaded_blocks, g.uploaded_tiles, g.read_back_bytes, g.readbacks)
        });
        eprintln!("gpu unit: {records} records in {batches} batches ({dropped} dropped), {} MiB uploaded ({} MiB dirty blocks, {} MiB regions), {} MiB read back in {readbacks} readbacks", up >> 20, blocks >> 20, tiles >> 20, down >> 20);
        b.raster.run_now(|ctx| {
            if let Some(by) = &ctx.gpu.as_ref().expect("the unit").waits_by {
                let mut v: Vec<_> = by.iter().collect();
                v.sort_by(|a, b| b.1.cmp(a.1));
                for (who, n) in v {
                    eprintln!("  {n:8} readbacks from {who}");
                }
            }
        });
    }
    if every != 0 && recs.len() % every != 0 && rep.hashes.last().is_none_or(|(j, _)| *j != recs.len()) {
        rep.hashes.push((recs.len(), memory_hash(&b)));
    }
    (rep.dma_used, rep.dma_total) = mem.consumed();
    rep.dma_mismatched = *mem.mismatches.lock();
    {
        let w = mem.writes.lock();
        rep.host_writes = (w.0.clone().finalize().to_hex()[..16].to_string(), w.1);
    }
    (rep, b)
}

pub fn recordings_dir() -> PathBuf {
    match std::env::var("VPRO_REPLAY_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR")).join("ignore/vpro-golden"),
    }
}

fn hashes_path(rec: &Path) -> PathBuf {
    let stem = rec.file_stem().unwrap().to_string_lossy().to_string();
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/dev/odyssey/testdata/replay").join(format!("{stem}.hashes"))
}

fn load_hashes(rec: &Path) -> Result<Vec<(usize, String)>, String> {
    let p = hashes_path(rec);
    let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e} (run scripts/vpro-goldens.sh replay-bless)", p.display()))?;
    Ok(text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let (i, h) = l.split_once(' ').expect("index hash");
            (i.parse().expect("record index"), h.to_string())
        })
        .collect())
}

fn save_hashes(rec: &Path, rep: &Report) {
    let p = hashes_path(rec);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    let mut s = format!(
        "# Checkpoints of {} (record index, blake3 of the planes), every {EVERY} records and at each mark.\n# {} records, {} DMA reads, {} marks{}.\n",
        rec.file_name().unwrap().to_string_lossy(),
        rep.records,
        rep.dma_total,
        rep.marks,
        if rep.mid_session { "; started mid-session, so the marks are the replay's, not the live board's" } else { "" }
    );
    for (i, h) in &rep.hashes {
        s += &format!("{i} {h}\n");
    }
    std::fs::write(p, s).unwrap();
}

/// Replay one recording and check its checkpoints; `png` saves the final
/// colour buffer.
pub fn replay_file(path: &Path, png: Option<&str>) -> Result<Report, String> {
    let recs = record::load(&path.to_string_lossy()).map_err(|e| format!("{}: {e}", path.display()))?;
    let t0 = std::time::Instant::now();
    let (rep, board) = run(&recs, |_| {}, EVERY);
    // The memory as drawn, on either raster unit.
    board.gpu_sync();
    eprintln!(
        "{}: {} records in {:.1} s ({:.3} s drawing), DMA reads {}/{} ({} mismatched), {} bus loads differ, host writes {} bytes {}, memory {}, {} of {} marks reached{}",
        path.display(),
        rep.records,
        t0.elapsed().as_secs_f64(),
        rep.draw_secs,
        rep.dma_used,
        rep.dma_total,
        rep.dma_mismatched,
        rep.reads_differ,
        rep.host_writes.1,
        rep.host_writes.0,
        &memory_hash(&board)[..16],
        rep.marks - rep.marks_failed.len(),
        rep.marks,
        if rep.mid_session { " (started mid-session: the marks are informational)" } else { "" }
    );
    if let Some(p) = png {
        save_screen(&board, p);
    }
    if let Ok(dir) = std::env::var("VPRO_REPLAY_DUMP") {
        let dir = Path::new(&dir);
        dump_memory(board.vram.ptr_shared(), dir).map_err(|e| format!("VPRO_REPLAY_DUMP {}: {e}", dir.display()))?;
        eprintln!("{}: board memory dumped to {}", path.display(), dir.display());
    }
    if rep.dma_mismatched != 0 || rep.dma_used != rep.dma_total {
        return Err(format!("{}: the replay's DMA reads do not follow the recording ({}/{} used, {} mismatched)", path.display(), rep.dma_used, rep.dma_total, rep.dma_mismatched));
    }
    if !rep.mid_session && !rep.marks_failed.is_empty() {
        return Err(format!("{}: {} of {} marks not reached, at records {:?}", path.display(), rep.marks_failed.len(), rep.marks, &rep.marks_failed[..rep.marks_failed.len().min(8)]));
    }
    // Read last, so a recording without checkpoints still gives its
    // picture and memory.
    let mut want = load_hashes(path)?;
    if let Some(limit) = std::env::var("VPRO_REPLAY_LIMIT").ok().and_then(|v| v.parse::<usize>().ok()) {
        want.retain(|(i, _)| *i <= limit);
    }
    let failed: Vec<usize> = want.iter().filter(|(i, h)| rep.hashes.iter().find(|(j, _)| j == i).is_none_or(|(_, g)| g != h)).map(|(i, _)| *i).collect();
    if want.is_empty() || !failed.is_empty() || want.len() != rep.hashes.len() {
        return Err(format!("{}: {} of {} checkpoints failed, at records {:?}", path.display(), failed.len(), want.len(), &failed[..failed.len().min(8)]));
    }
    Ok(rep)
}

/// The screen as the scan-out composes it from board memory.
fn save_screen(b: &Board, path: &str) {
    let regs = b.regs().read();
    let Some(frame) = super::super::frame::Frame::snapshot(&regs, b.front.lock().displayed_buffer) else {
        eprintln!("{path}: the scan-out is not set up");
        return;
    };
    let (w, h) = (frame.width, frame.height);
    let mut px = vec![0u32; w as usize * h as usize];
    frame.compose(b.vram.ptr_shared(), &regs, &mut px, w as usize, h as usize);
    let mut rgb = Vec::with_capacity(px.len() * 3);
    for p in px {
        rgb.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, p as u8]);
    }
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&rgb).unwrap();
}

fn recordings() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(recordings_dir())
        .map(|d| d.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "rec")).collect())
        .unwrap_or_default();
    v.sort();
    v
}

/// Every recording in the recordings directory replays to its checkpoints.
/// The two raster units agree on recorded guest sessions: each recording in
/// `VPRO_PARITY_DIR` (default `ignore/recordings/gpu-cases`) replays on the
/// CPU unit, on the GPU unit, and on the GPU unit ending a board slice every
/// 64 stores as a running machine does; all three must write the same bytes
/// to guest memory (the pixels the guest reads back) and leave the same
/// board memory. The recordings are short guest test programs (vissweep
/// cases, pixtest groups, cicopy); like the others they are not in git, and
/// the test skips without them.
#[cfg(feature = "wgpu")]
#[test]
fn the_raster_units_agree_on_recorded_sessions() {
    let dir = std::env::var("VPRO_PARITY_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("ignore/recordings/gpu-cases"));
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "rec")).collect())
        .unwrap_or_default();
    files.sort();
    if files.is_empty() {
        eprintln!("SKIPPED: no recordings in {}", dir.display());
        return;
    }
    let mut bad = vec![];
    for f in &files {
        let recs = record::load(&f.to_string_lossy()).expect("a recording");
        let outcome = |opts: Opts| {
            let (rep, b) = run_with(&recs, |_| {}, 0, opts);
            b.gpu_sync();
            (rep.host_writes, memory_hash(&b))
        };
        let cpu = outcome(Opts::default());
        let gpu = outcome(Opts { gpu: true, ..Default::default() });
        let ticked = outcome(Opts { gpu: true, tick: Some(64), ..Default::default() });
        eprintln!("{}: host writes {} bytes", f.display(), cpu.0 .1);
        if gpu != cpu {
            bad.push(format!("{}: the GPU unit differs ({:?} vs {:?})", f.display(), gpu, cpu));
        }
        if ticked != cpu {
            bad.push(format!("{}: the GPU unit with slice ends differs ({:?} vs {:?})", f.display(), ticked, cpu));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// With no recordings the test says so and passes, since they are not in the
/// tree; `VPRO_REPLAY_REQUIRE=1` (the script's `replay`) makes that a failure.
#[test]
fn recorded_sessions_replay_to_their_checkpoints() {
    let files = match std::env::var("VPRO_REPLAY") {
        Ok(f) => vec![PathBuf::from(f)],
        Err(_) => recordings(),
    };
    if files.is_empty() {
        let msg = format!("no recordings in {}: see testdata/README.md", recordings_dir().display());
        assert!(std::env::var("VPRO_REPLAY_REQUIRE").is_err(), "{msg}");
        eprintln!("SKIPPED: {msg}");
        return;
    }
    let png = std::env::var("VPRO_REPLAY_OUT").ok();
    let mut errs = vec![];
    for f in &files {
        if let Err(e) = replay_file(f, png.as_deref()) {
            errs.push(e);
        }
    }
    assert!(errs.is_empty(), "{}", errs.join("\n"));
}

/// Re-take every recording's checkpoints after a deliberate change.
#[test]
#[ignore]
fn replay_bless() {
    assert!(std::env::var("VPRO_REPLAY_BLESS").is_ok(), "set VPRO_REPLAY_BLESS=1: blessing rewrites the checkpoints");
    let files = recordings();
    assert!(!files.is_empty(), "no recordings in {}", recordings_dir().display());
    for f in files {
        let recs = record::load(&f.to_string_lossy()).unwrap();
        let (rep, _) = run(&recs, |_| {}, EVERY);
        assert!(rep.dma_mismatched == 0 && rep.dma_used == rep.dma_total, "{}: the DMA reads do not follow the recording; the session cannot be replayed", f.display());
        save_hashes(&f, &rep);
        eprintln!("{}: {} checkpoints", f.display(), rep.hashes.len());
    }
}

// The replay machinery on a short session built in memory.

fn window(b: &mut Board) {
    // The screen a replay without its own scan-out set-up runs on.
    b.test_screen();
    b.win.clip_kept = 1;
    b.win.clip_rect[0] = ClipRect { x0: 0, y0: 0, x1: ODSY_WIDTH as i32 - 1, y1: ODSY_HEIGHT as i32 - 1 };
}

/// A glDrawPixels of two pixels by pixel burst, as a store sequence: the same
/// words the library sends for a small image.
fn image_session(pixels: &[u32]) -> Vec<Rec> {
    let w = |off: u64, val: u32| Rec::Write { size: 4, off: off as u32, val: val as u64 };
    let mut v = vec![];
    let mut cmd = |word: u32| v.push(w(cmd::FIFO, word));
    cmd(cmd::REGISTER_TAG | reg::GL_BEGIN);
    cmd(gl_cmd::PRIMITIVE_IMAGE);
    cmd(cmd::IMAGE_BOX);
    for i in 0..10 {
        let val = match i {
            image::BOX_X1_INDEX => 40,
            image::BOX_Y1_INDEX => 10,
            image::BOX_X2_INDEX => 40 + pixels.len() as u32,
            image::BOX_Y2_INDEX => 11,
            _ => 0,
        };
        cmd((val as f32).to_bits());
    }
    cmd(cmd::IMAGE_HEADER);
    cmd(0x0021_104b);
    assert_eq!(pixels.len(), 2, "the header is a 2 x 1 image's");
    cmd(cmd::REGISTER_TAG | (image::BURST_BASE | (pixels.len() as u32) << cmd::ARITY_SHIFT | image::FMT_RGB));
    for &p in pixels {
        cmd(p);
    }
    cmd(cmd::REGISTER_TAG | reg::GL_END);
    // A load between the stores, as a polling driver makes.
    v.push(Rec::Read { size: 4, off: ODSY_BUZZ_1 as u32, val: 0 });
    v
}

/// A session replayed gives the planes the same stores gave a board
/// directly, and a changed store gives other planes.
#[test]
fn a_replayed_session_reaches_the_planes_the_stores_do() {
    let session = image_session(&[0x116c_c722, 0x2233_4400]);
    let (rep, b) = run(&session, window, 10);
    assert!(rep.dma_total == 0 && rep.dma_mismatched == 0);
    assert_eq!(b.buffer_load(0, 40, 10), 0x116c_c722);
    assert_eq!(b.buffer_load(0, 41, 10), 0x2233_4400);

    // The same stores, applied to a board directly.
    let mut direct = Board::new("v12", 11).unwrap();
    window(&mut direct);
    for r in &session {
        if let Rec::Write { size, off, val } = r {
            direct.mmio_write(*off as u64, *val, *size as u32);
        }
    }
    direct.gpu_sync();
    assert_eq!(memory_hash(&direct), rep.hashes.last().unwrap().1);

    let mut changed = session.clone();
    for r in changed.iter_mut() {
        if let Rec::Write { val, .. } = r {
            if *val == 0x2233_4400 {
                *val = 0x2233_4401;
            }
        }
    }
    let (rep2, _) = run(&changed, window, 10);
    assert_ne!(rep2.hashes.last().unwrap().1, rep.hashes.last().unwrap().1);
}

/// A recording written and read back is the same records, and the DMA the
/// replay serves comes back in order.
#[test]
fn a_recording_reads_back_as_written() {
    let dir = std::env::var("CARGO_TARGET_DIR").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir());
    let path = dir.join(format!("vpro_rec_test_{}.rec", std::process::id()));
    let p = path.to_string_lossy().to_string();
    let want = vec![
        Rec::Write { size: 4, off: 0x1000, val: 0xdead_beef },
        Rec::Write { size: 8, off: 0x2000, val: 0x0123_4567_89ab_cdef },
        Rec::Read { size: 4, off: 0x106c, val: 0xa100_0000 },
        Rec::DmaBlock { addr: 0x2000_0000, words: vec![1, 2, 3] },
        Rec::DmaByte { addr: 0x2000_0003, val: 0x7f },
        Rec::Write { size: 4, off: 0x1004, val: 9 },
        Rec::Mark([7u8; 32]),
    ];
    {
        let mut r = record::Recorder::create(&p).unwrap();
        for x in &want {
            match x {
                Rec::Write { size, off, val } => r.write(*size, *off, *val),
                Rec::Read { size, off, val } => {
                    r.read(*size, *off, *val);
                    // An identical load straight after is not kept.
                    r.read(*size, *off, *val);
                }
                Rec::Mark(h) => r.mark(*h),
                _ => {}
            }
        }
        assert_eq!((r.records, r.marks), (5, 1));
    }
    let got = record::load(&p).unwrap();
    let _ = std::fs::remove_file(&path);
    let kept: Vec<Rec> = want.iter().filter(|r| matches!(r, Rec::Write { .. } | Rec::Read { .. } | Rec::Mark(_))).cloned().collect();
    assert_eq!(got, kept);
}

/// A recording's marks are checked against the replayed planes: a
/// power-on recording's are reached, one started mid-session is said so and
/// its marks only reported.
#[test]
fn marks_are_checked_against_the_replayed_planes() {
    let session = image_session(&[0x116c_c722, 0x2233_4400]);
    let (first, _) = run(&[], window, 0);
    assert_eq!(first.marks, 0);
    // A marked recording starts from the power-on state, so its reference
    // board is reset too.
    let mut fresh = Board::new("v12", 11).unwrap();
    fresh.reset();
    window(&mut fresh);
    fresh.gpu_sync();
    let h0 = memory_hash_bytes(&fresh);
    let (rep, b) = run(&session, window, 0);
    assert_eq!(rep.hashes.len(), 0);
    assert_ne!(memory_hash_bytes(&b), h0, "a bare board is not the power-on one");
    for r in &session {
        if let Rec::Write { size, off, val } = r {
            fresh.mmio_write(*off as u64, *val, *size as u32);
        }
    }
    fresh.gpu_sync();
    let h1 = memory_hash_bytes(&fresh);

    let mut marked = vec![Rec::Mark(h0)];
    marked.extend(session.iter().cloned());
    marked.push(Rec::Mark(h1));
    let (rep, _) = run(&marked, window, 0);
    assert_eq!((rep.marks, rep.marks_failed.len(), rep.mid_session), (2, 0, false));
    assert_eq!(rep.hashes.len(), 2, "each mark is a checkpoint");
    assert_eq!(rep.hashes[1].1, blake3::Hash::from(h1).to_hex().to_string());

    marked[0] = Rec::Mark([0u8; 32]);
    let (rep, _) = run(&marked, window, 0);
    assert_eq!((rep.marks, rep.marks_failed.clone(), rep.mid_session), (2, vec![1], true));

    // A mark that is not the first and is not reached is a divergence.
    marked[0] = Rec::Mark(h0);
    let n = marked.len();
    marked[n - 1] = Rec::Mark([0u8; 32]);
    let (rep, _) = run(&marked, window, 0);
    assert_eq!((rep.marks_failed.clone(), rep.mid_session), (vec![n], false));
}

/// The board's DMA reads, recorded through `RecordingMem` and served back by
/// `ReplayMem`, give the same bytes in the same order; a read at another
/// address is counted as a mismatch.
#[test]
fn recorded_dma_reads_are_served_back_in_order() {
    use super::super::tests::Ram;

    let ram = Ram::new(1 << 16);
    let bytes: Vec<u8> = (0..64u8).collect();
    ram.put(0x100, &bytes);
    let dir = std::env::var("CARGO_TARGET_DIR").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir());
    let path = dir.join(format!("vpro_dma_test_{}.rec", std::process::id()));
    let p = path.to_string_lossy().to_string();
    let mut live = [vec![0u8; 21], vec![0u8; 8], vec![0u8; 16]];
    {
        let h = record::RecSlot::new(Some(record::Recorder::create(&p).unwrap()));
        let mem: GuestMemory = Some(Arc::new(record::RecordingMem { inner: ram.clone(), rec: h.clone() }));
        // Ragged at both ends, then a whole qword, then two.
        assert!(read_guest(&mem, 0x103, &mut live[0]));
        assert!(read_guest(&mem, 0x120, &mut live[1]));
        assert!(read_guest(&mem, 0x128, &mut live[2]));
        // Taking the recorder out finishes the file.
        assert!(h.set(None).is_some());
        assert!(!h.active());
    }
    let recs = record::load(&p).unwrap();
    let _ = std::fs::remove_file(&path);
    let served: GuestMemory = Some(Arc::new(ReplayMem::new(&recs)));
    let mut again = [vec![0u8; 21], vec![0u8; 8], vec![0u8; 16]];
    assert!(read_guest(&served, 0x103, &mut again[0]));
    assert!(read_guest(&served, 0x120, &mut again[1]));
    assert!(read_guest(&served, 0x128, &mut again[2]));
    assert_eq!(live, again);
    assert_eq!(&live[0][..4], &[3, 4, 5, 6]);

    let m = Arc::new(ReplayMem::new(&recs));
    let wrong: GuestMemory = Some(m.clone());
    let mut buf = [0u8; 21];
    read_guest(&wrong, 0x104, &mut buf);
    assert_ne!(*m.mismatches.lock(), 0);
}
