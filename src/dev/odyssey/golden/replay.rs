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
//! saves its final colour buffer, `VPRO_REPLAY_BLESS=1` with
//! `cargo test ... replay_bless -- --ignored` re-takes every recording's
//! checkpoints.

use crate::dev::odyssey::board::{Board, ODSY_HEIGHT, ODSY_WIDTH};
use crate::dev::odyssey::consts::{cmd, gl_cmd, image, reg, ODSY_BUZZ_1};
use crate::dev::odyssey::memory::ClipRect;
use crate::dev::odyssey::pixel_dma::{read_guest, GuestMemory};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use super::super::record::{self, Rec};
use crate::traits::{BusDevice, BusRead8, BUS_ERR, BUS_OK};

/// Records between checkpoints.
pub const EVERY: usize = 500_000;

/// The planes a checkpoint hashes: both colour buffers, depth and stencil,
/// and the object memory in use.
pub fn plane_hash(b: &Board) -> String {
    let mut h = blake3::Hasher::new();
    let (w, ht) = (ODSY_WIDTH as i32, ODSY_HEIGHT as i32);
    let depth = b.planes.depth.plane(b.planes.depth_plane).expect("the depth plane");
    let stencil = b.planes.stencil.plane(b.planes.stencil_plane).expect("the stencil plane");
    let mut row = Vec::with_capacity(w as usize * 16);
    for y in 0..ht {
        row.clear();
        for x in 0..w {
            row.extend_from_slice(&b.buffer_load(0, x as i64, y as i64).to_be_bytes());
            row.extend_from_slice(&b.buffer_load(1, x as i64, y as i64).to_be_bytes());
            row.extend_from_slice(&b.planes.depth.plane_load(&depth, x, y).to_be_bytes());
            row.extend_from_slice(&b.planes.stencil.plane_load(&stencil, x, y).to_be_bytes());
        }
        h.update(&row);
    }
    let used = (b.objects.used as usize * 4).min(b.objects.mem.bytes.len());
    h.update(&b.objects.mem.bytes[..used]);
    h.finalize().to_hex().to_string()
}

/// Guest memory for replay: answers the board's DMA reads with the recorded
/// values in order, and notes any that do not match the recorded address.
pub struct ReplayMem {
    reads: Vec<Rec>,
    next: Mutex<usize>,
    pub mismatches: Mutex<u64>,
}

impl ReplayMem {
    pub fn new(recs: &[Rec]) -> Self {
        let reads = recs.iter().filter(|r| matches!(r, Rec::DmaBlock { .. } | Rec::DmaByte { .. })).cloned().collect();
        ReplayMem { reads, next: Mutex::new(0), mismatches: Mutex::new(0) }
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
    fn write8(&self, _addr: u64, _val: u8) -> u32 {
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
    fn write_block(&self, _addr: u64, _buf: &[u64]) -> u32 {
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
}

/// Feed `recs` through a fresh board's registers, `setup` run on it first.
/// `board_out` receives the board when it is done.
pub fn run(recs: &[Rec], setup: impl FnOnce(&mut Board), every: usize) -> (Report, Box<Board>) {
    let mut b = Board::new("v12", 11).expect("board");
    let mem = Arc::new(ReplayMem::new(recs));
    b.guest = Some(mem.clone());
    setup(&mut b);
    let mut rep = Report { records: recs.len(), ..Default::default() };
    for (i, r) in recs.iter().enumerate() {
        match r {
            Rec::Write { size, off, val } => b.mmio_write(*off as u64, *val, *size as u32),
            Rec::Read { size, off, val } => {
                let mask = if *size >= 8 { u64::MAX } else { (1u64 << (8 * *size as u32)) - 1 };
                if b.mmio_read(*off as u64, *size as u32) & mask != val & mask {
                    rep.reads_differ += 1;
                }
            }
            Rec::DmaBlock { .. } | Rec::DmaByte { .. } => {}
        }
        if every != 0 && (i + 1) % every == 0 {
            rep.hashes.push((i + 1, plane_hash(&b)));
        }
    }
    b.raster_wait();
    if every != 0 && recs.len() % every != 0 {
        rep.hashes.push((recs.len(), plane_hash(&b)));
    }
    (rep.dma_used, rep.dma_total) = mem.consumed();
    rep.dma_mismatched = *mem.mismatches.lock();
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
        "# Checkpoints of {} (record index, blake3 of the planes), every {EVERY} records.\n# {} records, {} DMA reads.\n",
        rec.file_name().unwrap().to_string_lossy(),
        rep.records,
        rep.dma_total
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
    let want = load_hashes(path)?;
    let t0 = std::time::Instant::now();
    let (rep, board) = run(&recs, |_| {}, EVERY);
    eprintln!("{}: {} records in {:.1} s, DMA reads {}/{} ({} mismatched), {} bus loads differ", path.display(), rep.records, t0.elapsed().as_secs_f64(), rep.dma_used, rep.dma_total, rep.dma_mismatched, rep.reads_differ);
    if let Some(p) = png {
        save_screen(&board, p);
    }
    if rep.dma_mismatched != 0 || rep.dma_used != rep.dma_total {
        return Err(format!("{}: the replay's DMA reads do not follow the recording ({}/{} used, {} mismatched)", path.display(), rep.dma_used, rep.dma_total, rep.dma_mismatched));
    }
    let failed: Vec<usize> = want.iter().filter(|(i, h)| rep.hashes.iter().find(|(j, _)| j == i).is_none_or(|(_, g)| g != h)).map(|(i, _)| *i).collect();
    if want.is_empty() || !failed.is_empty() || want.len() != rep.hashes.len() {
        return Err(format!("{}: {} of {} checkpoints failed, at records {:?}", path.display(), failed.len(), want.len(), &failed[..failed.len().min(8)]));
    }
    Ok(rep)
}

fn save_screen(b: &Board, path: &str) {
    let (w, h) = (ODSY_WIDTH as u32, ODSY_HEIGHT as u32);
    let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
    for y in 0..h {
        for x in 0..w {
            let p = b.buffer_load(0, x as i64, y as i64);
            rgb.extend_from_slice(&[(p >> 24) as u8, (p >> 16) as u8, (p >> 8) as u8]);
        }
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
    assert_eq!(plane_hash(&direct), rep.hashes.last().unwrap().1);

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
    ];
    {
        let h = record::Recorder::create(&p).unwrap();
        let mut r = h.lock();
        for x in &want {
            match x {
                Rec::Write { size, off, val } => r.write(*size, *off, *val),
                Rec::Read { size, off, val } => {
                    r.read(*size, *off, *val);
                    // An identical load straight after is not kept.
                    r.read(*size, *off, *val);
                }
                _ => {}
            }
        }
        r.flush();
    }
    let got = record::load(&p).unwrap();
    let _ = std::fs::remove_file(&path);
    let kept: Vec<Rec> = want.iter().filter(|r| matches!(r, Rec::Write { .. } | Rec::Read { .. })).cloned().collect();
    assert_eq!(got, kept);
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
        let h = record::Recorder::create(&p).unwrap();
        let mem: GuestMemory = Some(Arc::new(record::RecordingMem { inner: ram.clone(), rec: h.clone() }));
        // Ragged at both ends, then a whole qword, then two.
        assert!(read_guest(&mem, 0x103, &mut live[0]));
        assert!(read_guest(&mem, 0x120, &mut live[1]));
        assert!(read_guest(&mem, 0x128, &mut live[2]));
        h.lock().flush();
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
