//! Persistent code cache: compiled pages kept on disk across runs, keyed by
//! what they were compiled from. Design, measurements and verification plan:
//! `docs/jitv2-persistent-cache.md`.
//!
//! Opt-in with `[jitv2] cache = true` in `iris.toml` (or the GUI's jitv2
//! section), which `Jitv2Config::apply_env` turns into the `IRIS_JIT_CACHE`
//! env var this module actually reads — that var (and `IRIS_JIT_CACHE_DIR`,
//! `[jitv2] cache_dir`'s counterpart, which moves the cache; default: the
//! platform's user cache directory, `iris/jitv2` inside it; see
//! `default_base`) still work directly too, same override rule as the rest
//! of `[debug]`/`[jitv2]`.
//!
//! Layout, under the base directory:
//!
//! ```text
//! <build-id>/<fingerprint>/<page-hash>-<fr>/<entries-hash>.jc
//! ```
//!
//! - `build-id` is a BLAKE3 hash of the running executable, so a rebuild never
//!   sees another build's code.
//! - `fingerprint` hashes every runtime switch that shapes emitted code
//!   (`Codegen::cache_fingerprint`); a run with `j2 alu off` has its own space.
//! - Each page directory holds that page's variants, one per entry set. A blob
//!   serves any request whose entries are a subset of its own, and a new
//!   variant that covers an older one replaces it.
//!
//! There is no index: a lookup lists one small directory, which is the only
//! filesystem work on a page that has never been cached. Several emulators
//! can share a cache, since every write is a temp file plus a rename.
//!
//! A hit is always verified against the full 4 KB of page words stored in the
//! blob, so no hash collision can serve code for other bytes.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;

use crate::jitv2::{BITMAP_WORDS, ENTRIES_PER_PAGE};

pub type Fingerprint = [u8; 16];
pub type PageHash = [u8; 16];
pub type Entries = [u64; BITMAP_WORDS];

const MAGIC: [u8; 8] = *b"IRISJC\0\0";
/// Bump when the file layout changes. (A codegen change needs no bump: it
/// changes the executable, and with it the build id.)
const FORMAT: u32 = 1;
/// How many build directories survive startup pruning.
const KEEP_BUILDS: usize = 3;
/// A lookup reads at most this many variants of one page.
const MAX_VARIANTS: usize = 16;
const REPORT_EVERY: u64 = 500;

const HEADER_LEN: usize = 8 + 4 + 4 + 16 + 16 + BITMAP_WORDS * 8 * 2 + 4 + 4 + 4;
const WORDS_LEN: usize = ENTRIES_PER_PAGE * 4;
const SUM_LEN: usize = 16;

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = matches!(std::env::var("IRIS_JIT_CACHE").as_deref(), Ok("1") | Ok("on"));
        if on {
            match root() {
                Some(dir) => eprintln!("jitcache: on, {}", dir.display()),
                None => eprintln!("jitcache: requested but no cache directory could be set up; off"),
            }
        }
        on && root().is_some()
    })
}

/// The user cache directory's `iris/jitv2`: `~/Library/Caches` on macOS,
/// `%LOCALAPPDATA%` on Windows, `$XDG_CACHE_HOME` or `~/.cache` elsewhere.
fn default_base() -> Option<PathBuf> {
    let cache = if cfg!(target_os = "macos") {
        PathBuf::from(std::env::var_os("HOME")?).join("Library/Caches")
    } else if cfg!(windows) {
        PathBuf::from(std::env::var_os("LOCALAPPDATA")?)
    } else {
        match std::env::var_os("XDG_CACHE_HOME") {
            Some(d) if !d.is_empty() => PathBuf::from(d),
            _ => PathBuf::from(std::env::var_os("HOME")?).join(".cache"),
        }
    };
    Some(cache.join("iris").join("jitv2"))
}

/// `<base>/<build-id>`, created on first use; `None` if it can't be.
fn root() -> Option<&'static PathBuf> {
    static ROOT: OnceLock<Option<PathBuf>> = OnceLock::new();
    ROOT.get_or_init(|| {
        let base = match std::env::var_os("IRIS_JIT_CACHE_DIR") {
            Some(d) => PathBuf::from(d),
            None => default_base()?,
        };
        let exe = std::fs::read(std::env::current_exe().ok()?).ok()?;
        let id = hex(&blake3::hash(&exe).as_bytes()[..16]);
        let dir = base.join(id);
        std::fs::create_dir_all(&dir).ok()?;
        // Mark this build as the most recently used one, for pruning.
        let _ = std::fs::File::create(dir.join(".used"));
        prune_builds(&base);
        Some(dir)
    }).as_ref()
}

/// Keep the `KEEP_BUILDS` most recently used build directories.
fn prune_builds(base: &Path) {
    let Ok(rd) = std::fs::read_dir(base) else { return };
    let mut builds: Vec<(std::time::SystemTime, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| {
            let used = std::fs::metadata(e.path().join(".used")).and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            (used, e.path())
        })
        .collect();
    builds.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, old) in builds.into_iter().skip(KEEP_BUILDS) {
        let _ = std::fs::remove_dir_all(old);
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn page_hash(words: &[u32; ENTRIES_PER_PAGE]) -> PageHash {
    let mut h = blake3::Hasher::new();
    for w in words {
        h.update(&w.to_le_bytes());
    }
    first16(h.finalize())
}

fn first16(h: blake3::Hash) -> [u8; 16] {
    h.as_bytes()[..16].try_into().unwrap()
}

fn entries_hash(e: &Entries) -> [u8; 8] {
    let mut h = blake3::Hasher::new();
    for w in e {
        h.update(&w.to_le_bytes());
    }
    h.finalize().as_bytes()[..8].try_into().unwrap()
}

fn page_dir(fp: &Fingerprint, ph: &PageHash, fr1: bool) -> Option<PathBuf> {
    Some(root()?.join(hex(fp)).join(format!("{}-{}", hex(ph), fr1 as u8)))
}

fn covers(have: &Entries, want: &Entries) -> bool {
    have.iter().zip(want).all(|(h, w)| w & !h == 0)
}

fn count(e: &Entries) -> u32 {
    e.iter().map(|w| w.count_ones()).sum()
}

// ---- statistics ---------------------------------------------------------

static LOOKUPS: AtomicU64 = AtomicU64::new(0);
static HITS: AtomicU64 = AtomicU64::new(0);
/// Same page hash, but the stored words differed: a hash collision, or the
/// fault-injected skip turned off. Should stay 0.
static COMPARE_FAILS: AtomicU64 = AtomicU64::new(0);
static BAD_FILES: AtomicU64 = AtomicU64::new(0);
static STORES: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);
static UNION_ADDED: AtomicU64 = AtomicU64::new(0);
static LOAD_NS: AtomicU64 = AtomicU64::new(0);

/// A compile whose output can't be stored (relocations, or a configuration
/// that bakes host addresses).
pub fn note_refused() {
    REFUSED.fetch_add(1, Relaxed);
}

/// Entries union-on-miss added to a compile.
pub fn note_union_added(n: u32) {
    UNION_ADDED.fetch_add(n as u64, Relaxed);
}

pub fn note_load_time(d: std::time::Duration) {
    LOAD_NS.fetch_add(d.as_nanos() as u64, Relaxed);
}

pub fn summary() -> String {
    let (l, h) = (LOOKUPS.load(Relaxed), HITS.load(Relaxed));
    format!(
        "jitcache: lookups={l} hits={h} ({:.1}%) stored={} refused={} union_added={} compare_fail={} bad_files={} load_us_avg={:.0}",
        if l == 0 { 0.0 } else { 100.0 * h as f64 / l as f64 },
        STORES.load(Relaxed), REFUSED.load(Relaxed), UNION_ADDED.load(Relaxed),
        COMPARE_FAILS.load(Relaxed), BAD_FILES.load(Relaxed),
        if h == 0 { 0.0 } else { LOAD_NS.load(Relaxed) as f64 / h as f64 / 1000.0 },
    )
}

// ---- blobs ---------------------------------------------------------------

/// One compiled page, as stored.
pub struct Blob {
    /// Entry points the code has a dispatch case for.
    pub entries: Entries,
    /// Words the compile decoded (the churn-avoidance snapshot's mask).
    pub used: Entries,
    pub instr_count: u32,
    pub align: u32,
    pub words: Box<[u32; ENTRIES_PER_PAGE]>,
    pub code: Vec<u8>,
}

struct Header {
    fr1: bool,
    fp: Fingerprint,
    ph: PageHash,
    entries: Entries,
    used: Entries,
    instr_count: u32,
    align: u32,
    code_len: usize,
}

fn put_entries(out: &mut Vec<u8>, e: &Entries) {
    for w in e {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

fn encode(fp: &Fingerprint, ph: &PageHash, fr1: bool, b: &Blob) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + WORDS_LEN + b.code.len() + SUM_LEN);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT.to_le_bytes());
    out.extend_from_slice(&(fr1 as u32).to_le_bytes());
    out.extend_from_slice(fp);
    out.extend_from_slice(ph);
    put_entries(&mut out, &b.entries);
    put_entries(&mut out, &b.used);
    out.extend_from_slice(&b.instr_count.to_le_bytes());
    out.extend_from_slice(&b.align.to_le_bytes());
    out.extend_from_slice(&(b.code.len() as u32).to_le_bytes());
    debug_assert_eq!(out.len(), HEADER_LEN);
    for w in b.words.iter() {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out.extend_from_slice(&b.code);
    let sum = first16(blake3::hash(&out));
    out.extend_from_slice(&sum);
    out
}

struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        a
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn arr16(&mut self) -> [u8; 16] {
        self.take(16).try_into().unwrap()
    }
    fn entries(&mut self) -> Entries {
        let mut e = [0u64; BITMAP_WORDS];
        for w in &mut e {
            *w = u64::from_le_bytes(self.take(8).try_into().unwrap());
        }
        e
    }
}

fn decode_header(buf: &[u8]) -> Option<Header> {
    if buf.len() < HEADER_LEN {
        return None;
    }
    let mut r = Reader(&buf[..HEADER_LEN]);
    if r.take(8) != MAGIC || r.u32() != FORMAT {
        return None;
    }
    let fr = r.u32();
    if fr > 1 {
        return None;
    }
    Some(Header {
        fr1: fr == 1,
        fp: r.arr16(),
        ph: r.arr16(),
        entries: r.entries(),
        used: r.entries(),
        instr_count: r.u32(),
        align: r.u32(),
        code_len: r.u32() as usize,
    })
}

/// Parse and fully check one file against the key it was found under.
fn decode(buf: &[u8], fp: &Fingerprint, ph: &PageHash, fr1: bool) -> Option<Blob> {
    let h = decode_header(buf)?;
    if h.fr1 != fr1 || &h.fp != fp || &h.ph != ph || !h.align.is_power_of_two() {
        return None;
    }
    if buf.len() != HEADER_LEN + WORDS_LEN + h.code_len + SUM_LEN {
        return None;
    }
    let body = &buf[..buf.len() - SUM_LEN];
    if first16(blake3::hash(body)) != buf[body.len()..] {
        return None;
    }
    let mut words = Box::new([0u32; ENTRIES_PER_PAGE]);
    for (w, c) in words.iter_mut().zip(buf[HEADER_LEN..HEADER_LEN + WORDS_LEN].chunks_exact(4)) {
        *w = u32::from_le_bytes(c.try_into().unwrap());
    }
    Some(Blob {
        entries: h.entries,
        used: h.used,
        instr_count: h.instr_count,
        align: h.align,
        words,
        code: buf[HEADER_LEN + WORDS_LEN..body.len()].to_vec(),
    })
}

fn variant_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    rd.filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "jc"))
        .take(MAX_VARIANTS)
        .collect()
}

fn discard_bad(path: &Path) {
    BAD_FILES.fetch_add(1, Relaxed);
    let _ = std::fs::remove_file(path);
}

/// Find a stored compile of exactly `words` whose entries cover `want`. When
/// several do, the one with the most entries wins.
pub fn lookup(
    fp: &Fingerprint,
    ph: &PageHash,
    fr1: bool,
    words: &[u32; ENTRIES_PER_PAGE],
    want: &Entries,
) -> Option<Blob> {
    let n = LOOKUPS.fetch_add(1, Relaxed) + 1;
    if n % REPORT_EVERY == 0 {
        eprintln!("{}", summary());
    }
    let dir = page_dir(fp, ph, fr1)?;
    let mut best: Option<Blob> = None;
    for path in variant_files(&dir) {
        let Ok(buf) = std::fs::read(&path) else { continue };
        // Only a covering variant is worth the full check.
        match decode_header(&buf) {
            Some(h) if covers(&h.entries, want) => {}
            Some(_) => continue,
            None => { discard_bad(&path); continue; }
        }
        let Some(blob) = decode(&buf, fp, ph, fr1) else { discard_bad(&path); continue };
        if *blob.words != *words {
            COMPARE_FAILS.fetch_add(1, Relaxed);
            continue;
        }
        if best.as_ref().is_none_or(|b| count(&blob.entries) > count(&b.entries)) {
            best = Some(blob);
        }
    }
    if best.is_some() {
        HITS.fetch_add(1, Relaxed);
    }
    best
}

/// Every entry any stored variant of this page has, for union-on-miss. Only
/// headers are read; the walk that follows decides what is really covered.
pub fn known_entries(fp: &Fingerprint, ph: &PageHash, fr1: bool) -> Entries {
    let mut all = [0u64; BITMAP_WORDS];
    let Some(dir) = page_dir(fp, ph, fr1) else { return all };
    for path in variant_files(&dir) {
        let Ok(mut f) = std::fs::File::open(&path) else { continue };
        let mut buf = vec![0u8; HEADER_LEN];
        if std::io::Read::read_exact(&mut f, &mut buf).is_err() {
            continue;
        }
        if let Some(h) = decode_header(&buf) {
            if &h.fp == fp && &h.ph == ph && h.fr1 == fr1 {
                for (a, e) in all.iter_mut().zip(&h.entries) {
                    *a |= e;
                }
            }
        }
    }
    all
}

// ---- writer --------------------------------------------------------------

struct Job {
    fp: Fingerprint,
    ph: PageHash,
    fr1: bool,
    blob: Blob,
}

/// Queue a successful compile for writing. Returns at once; a background
/// thread does the filesystem work.
pub fn store(fp: Fingerprint, ph: PageHash, fr1: bool, blob: Blob) {
    static TX: OnceLock<std::sync::Mutex<std::sync::mpsc::Sender<Job>>> = OnceLock::new();
    let tx = TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("jitcache-writer".into())
            .spawn(move || {
                for job in rx {
                    write_job(job);
                }
            })
            .expect("spawn jitcache writer");
        std::sync::Mutex::new(tx)
    });
    let _ = tx.lock().unwrap().send(Job { fp, ph, fr1, blob });
}

fn write_job(job: Job) {
    let Some(dir) = page_dir(&job.fp, &job.ph, job.fr1) else { return };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let bytes = encode(&job.fp, &job.ph, job.fr1, &job.blob);
    let name = format!("{}.jc", hex(&entries_hash(&job.blob.entries)));
    let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    if std::fs::write(&tmp, &bytes).is_err() || std::fs::rename(&tmp, dir.join(&name)).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    STORES.fetch_add(1, Relaxed);
    // A variant the new one covers can never be chosen over it again.
    for path in variant_files(&dir) {
        if path.file_name().is_some_and(|n| n == name.as_str()) {
            continue;
        }
        let Ok(mut f) = std::fs::File::open(&path) else { continue };
        let mut buf = vec![0u8; HEADER_LEN];
        if std::io::Read::read_exact(&mut f, &mut buf).is_ok() {
            if let Some(h) = decode_header(&buf) {
                if covers(&job.blob.entries, &h.entries) {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(entries: Entries, seed: u32) -> Blob {
        let mut words = Box::new([0u32; ENTRIES_PER_PAGE]);
        for (i, w) in words.iter_mut().enumerate() {
            *w = seed.wrapping_mul(2654435761).wrapping_add(i as u32);
        }
        Blob { entries, used: [!0; BITMAP_WORDS], instr_count: 7, align: 16, words, code: vec![0xd5, 0x03, 0x20, 0x1f, 1, 2, 3] }
    }

    #[test]
    fn roundtrip_and_tamper() {
        let fp = [1u8; 16];
        let mut e = [0u64; BITMAP_WORDS];
        e[0] = 0b1011;
        let b = blob(e, 5);
        let ph = page_hash(&b.words);
        let bytes = encode(&fp, &ph, true, &b);
        let back = decode(&bytes, &fp, &ph, true).expect("decodes");
        assert_eq!(back.entries, b.entries);
        assert_eq!(back.code, b.code);
        assert_eq!(*back.words, *b.words);
        assert_eq!(back.align, 16);
        // Wrong key, wrong FR, one flipped bit, truncation: all rejected.
        assert!(decode(&bytes, &[2u8; 16], &ph, true).is_none());
        assert!(decode(&bytes, &fp, &ph, false).is_none());
        for i in [HEADER_LEN + 3, bytes.len() - SUM_LEN - 1] {
            let mut bad = bytes.clone();
            bad[i] ^= 0x40;
            assert!(decode(&bad, &fp, &ph, true).is_none(), "flip at {i}");
        }
        assert!(decode(&bytes[..bytes.len() - 1], &fp, &ph, true).is_none());
    }

    #[test]
    fn subset_rule() {
        let mut have = [0u64; BITMAP_WORDS];
        have[1] = 0b110;
        let mut want = [0u64; BITMAP_WORDS];
        want[1] = 0b100;
        assert!(covers(&have, &want));
        want[2] = 1;
        assert!(!covers(&have, &want));
    }
}
