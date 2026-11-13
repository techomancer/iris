//! SGI PROM image loading: version-six single-segment containers, the IRIX
//! installer's 256-byte header (the O2's `ip32prom.image`) and raw payloads.
//! A recognized but malformed container is rejected rather than retried as a
//! raw image.
//!
//! Ported from hw/mips/sgi/prom-image.c in jamesbraid/qemu-sgi-origami.

use std::io::{self, Read};

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn word(data: &[u8], off: usize) -> u64 {
    u64::from_be_bytes(data[off..off + 8].try_into().unwrap())
}
fn byte_sum(data: &[u8]) -> u64 {
    data.iter().map(|b| *b as u64).sum()
}

/// Read a PROM file and decode its payload, retaining the path in errors.
pub fn load(path: &std::path::Path, payload_limit: usize) -> io::Result<Vec<u8>> {
    let with_path = |error: io::Error| {
        io::Error::new(error.kind(), format!("PROM image {}: {error}", path.display()))
    };
    let file = std::fs::File::open(path).map_err(with_path)?;
    // Bound the encoded file by zlib's compression bound, the container
    // header, gzip framing and the accepted zero tail.
    let size = payload_limit as u64;
    let file_limit = size
        .saturating_add(size >> 12)
        .saturating_add(size >> 14)
        .saturating_add(size >> 25)
        .saturating_add(13 + 4096 + 12 + 7);
    if file.metadata().map_err(with_path)?.len() > file_limit {
        return Err(with_path(invalid(
            "encoded PROM image exceeds flash capacity",
        )));
    }
    let mut bytes = Vec::new();
    file.take(file_limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(with_path)?;
    if bytes.len() as u64 > file_limit {
        return Err(with_path(invalid(
            "encoded PROM image exceeds flash capacity",
        )));
    }
    decode(&bytes, payload_limit).map_err(with_path)
}

/// A PROM image and where it came from.
pub struct Loaded {
    pub bytes: Vec<u8>,
    /// The file it was read from; None for the PROM built into iris.
    pub source: Option<std::path::PathBuf>,
}

impl Loaded {
    /// The file's path, or "the built-in PROM", for messages.
    pub fn name(&self) -> String {
        self.source.as_ref().map_or_else(|| "the built-in PROM".into(), |p| p.display().to_string())
    }
}

/// `prom = "(embedded)"`: use the PROM built into iris, whatever files exist.
pub fn is_embedded_keyword(prom: &str) -> bool {
    prom == "(embedded)"
}

/// The PROM `profile` runs: the file `prom` names, which must load; or, when `prom` is
/// left at its default, `prom.bin` or the file name the machine's releases shipped
/// under if one is in the working directory; else the PROM built into iris.
pub fn resolve(profile: crate::config::MachineProfile, prom: &str, payload_limit: usize) -> Result<Loaded, String> {
    let implicit: Vec<&str> = std::iter::once(crate::config::DEFAULT_PROM).chain(profile.release_prom_file()).collect();
    resolve_with(profile.embedded_prom(), prom, &implicit, payload_limit)
}

fn resolve_with(
    built_in: crate::dev::embedded_prom::EmbeddedProm, prom: &str, implicit: &[&str], payload_limit: usize,
) -> Result<Loaded, String> {
    let from_file = |path: &str| {
        let loaded = load(std::path::Path::new(path), payload_limit).map(|bytes| Loaded { bytes, source: Some(path.into()) });
        if loaded.is_ok() {
            println!("Loaded PROM from {path}");
        }
        loaded
    };
    if prom != crate::config::DEFAULT_PROM && !is_embedded_keyword(prom) {
        return from_file(prom).map_err(|e| e.to_string());
    }
    if prom == crate::config::DEFAULT_PROM {
        for path in implicit {
            match from_file(path) {
                Ok(loaded) => return Ok(loaded),
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.to_string()),
            }
        }
    }
    let bytes = crate::dev::embedded_prom::get(built_in)?;
    println!("Using the built-in {built_in:?} PROM");
    Ok(Loaded { bytes, source: None })
}

/// The magic at 0x40 of a version-six container's header.
const CONTAINER_MAGIC: u64 = 0x4a46_4b53_5743_534d;

/// Whether `file` starts a version-six container (its header may still be cut short).
fn is_container(file: &[u8]) -> bool {
    file.len() >= 0x48 && word(file, 0x40) == CONTAINER_MAGIC
}

/// What a PROM file is, from its first bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Flash contents, with no header.
    Raw,
    /// The IRIX installer's 256-byte header, then the flash contents.
    Installer,
    /// A version-six container: a 4096-byte header, then its segment.
    Container,
}

/// One segment record of a version-six container: the fields at 0x80 in the container's
/// header, or a copy of such a record inside a payload (the IP35 PROM carries the BaseIO
/// board's PROM that way).
#[derive(Debug, Clone)]
pub struct Segment {
    pub name: String,
    /// Where the record starts in the bytes it was read from.
    pub record: usize,
    /// Where its container starts: the record sits 0x80 in, the stored bytes `offset` on.
    pub start: usize,
    pub flags: u64,
    pub offset: u64,
    /// The length once unpacked.
    pub length: u64,
    /// The stored length, when gzipped.
    pub stored: u64,
    pub sum_decoded: u64,
    pub sum_stored: u64,
    /// The length of an uncompressed segment, stated again.
    pub plain: u64,
}

impl Segment {
    /// Read the record at `record`; None if `buf` is too short for one.
    pub fn read(buf: &[u8], record: usize) -> Option<Segment> {
        let b = buf.get(record..record.checked_add(0x58)?)?;
        let name = b[..16].iter().take_while(|c| **c != 0).map(|c| *c as char).collect();
        Some(Segment {
            name,
            record,
            start: record.saturating_sub(0x80),
            flags: word(b, 0x10),
            offset: word(b, 0x18),
            length: word(b, 0x30),
            stored: word(b, 0x38),
            sum_decoded: word(b, 0x40),
            sum_stored: word(b, 0x48),
            plain: word(b, 0x50),
        })
    }

    /// Whether the segment is gzipped (flags 0x13) rather than stored (0x10).
    pub fn gzip(&self) -> bool {
        self.flags == 0x13
    }

    /// Inflate `stored`, the gzipped bytes, checking their sum and the unpacked length.
    fn gunzip(&self, stored: &[u8]) -> io::Result<Vec<u8>> {
        if byte_sum(stored) != self.sum_stored {
            return Err(invalid("SGI PROM stored checksum mismatch"));
        }
        let mut decoder = flate2::bufread::GzDecoder::new(stored);
        let mut decoded = Vec::new();
        (&mut decoder).take(self.length.saturating_add(1)).read_to_end(&mut decoded)?;
        if decoded.len() as u64 != self.length || !decoder.get_ref().is_empty() {
            return Err(invalid("invalid SGI PROM gzip segment or decoded length"));
        }
        Ok(decoded)
    }

    /// The segment's bytes unpacked from `buf`, which holds its container, verified
    /// against the record's lengths and checksums. An unpacked length of zero or over
    /// `limit` is refused before anything is inflated.
    pub fn unpack(&self, buf: &[u8], limit: usize) -> io::Result<Vec<u8>> {
        if self.length == 0 || self.length > limit as u64 {
            return Err(invalid("SGI PROM decoded length is zero or exceeds capacity"));
        }
        let at = self.start.saturating_add(usize::try_from(self.offset).unwrap_or(usize::MAX));
        let len = usize::try_from(if self.gzip() { self.stored } else { self.length }).unwrap_or(usize::MAX);
        let data = buf.get(at..at.saturating_add(len)).ok_or_else(|| invalid("SGI PROM segment lies outside its container"))?;
        let payload = if self.gzip() { self.gunzip(data)? } else { data.to_vec() };
        if byte_sum(&payload) != self.sum_decoded {
            return Err(invalid("SGI PROM decoded checksum mismatch"));
        }
        Ok(payload)
    }

    /// Whether this looks like a segment record and not data that happens to fit: a
    /// name, stored or gzip flags, the 4096-byte offset, and the stored bytes in range.
    /// A gzip segment must also start with the gzip magic; a stored one has nothing
    /// like that to check, so it is the weaker test (`unpack` still checks its sum).
    fn plausible(&self, buf: &[u8]) -> bool {
        if self.offset != 4096 || self.length == 0 || !(self.flags == 0x10 || self.flags == 0x13) {
            return false;
        }
        if self.name.len() < 3 || !self.name.bytes().all(|c| c.is_ascii_graphic()) {
            return false;
        }
        let at = self.start.saturating_add(4096);
        let len = if self.gzip() { self.stored } else { self.length };
        (!self.gzip() || self.stored > 0 && buf.get(at..at.saturating_add(2)) == Some(&[0x1f, 0x8b]))
            && (at as u64).saturating_add(len) <= buf.len() as u64
    }
}

/// What a PROM file says about itself.
#[derive(Debug)]
pub struct Info {
    pub kind: Kind,
    /// The version its header states: the installer's "4.18", a container's "6.156".
    pub version: Option<String>,
    /// A container's own segment: the first, which is the one iris reads.
    pub segment: Option<Segment>,
}

/// Read a PROM file's header. See [`decode`] for what iris accepts.
pub fn inspect(file: &[u8]) -> Info {
    if installer_payload(file).is_some() {
        let half = |at: usize| u32::from_be_bytes(file[at..at + 4].try_into().unwrap());
        return Info { kind: Kind::Installer, version: Some(format!("{}.{}", half(0x10), half(0x14))), segment: None };
    }
    if is_container(file) {
        if file.len() < 4096 {
            return Info { kind: Kind::Container, version: None, segment: None };
        }
        let count = u32::from_be_bytes(file[0x7c..0x80].try_into().unwrap());
        return Info {
            kind: Kind::Container,
            version: Some(format!("{}.{}", word(file, 0x48), word(file, 0x38))),
            segment: if count > 0 { Segment::read(file, 0x80) } else { None },
        };
    }
    Info { kind: Kind::Raw, version: None, segment: None }
}

/// The segment records inside `payload`, 16-aligned, that read as real ones.
pub fn nested(payload: &[u8]) -> Vec<Segment> {
    (0x80..payload.len().saturating_sub(0x58))
        .step_by(16)
        // The cheap words first: a name is only built for a likely record.
        .filter(|&at| matches!(word(payload, at + 0x10), 0x10 | 0x13) && word(payload, at + 0x18) == 4096)
        .filter_map(|at| Segment::read(payload, at))
        .filter(|s| s.plausible(payload))
        .collect()
}

/// Where a banner sits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Place {
    /// An offset in the file itself.
    File(usize),
    /// An offset inside a segment once unpacked (a gzipped container's segment, or a
    /// segment carried in the payload).
    Segment { name: String, offset: usize },
}

impl std::fmt::Display for Place {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Place::File(at) => write!(f, "{at:#x}"),
            Place::Segment { name, offset } => write!(f, "{name}+{offset:#x}"),
        }
    }
}

/// A version banner a PROM carries, as a whole printable line (or, for the O2, its
/// four-line version record).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Banner {
    pub text: String,
    pub at: Place,
}

fn printable_run(data: &[u8], from: usize, max: usize) -> usize {
    data[from.min(data.len())..].iter().take(max).take_while(|c| (0x20..0x7f).contains(*c)).count()
}

/// The end of the lazily shortest `[ -~]{min,max}? SGI` starting at `from`, if any.
fn up_to_sgi(data: &[u8], from: usize, min: usize, max: usize) -> Option<usize> {
    let run = printable_run(data, from, max + 4);
    (min..=max.min(run)).find(|k| data[from + k..].starts_with(b" SGI")).map(|k| from + k + 4)
}

/// The O2 PROM's record: build date, `VERSION n.n`, target, and a line naming IP32prom.
fn o2_record(data: &[u8], at: usize) -> Option<(usize, usize)> {
    if !data[at..].starts_with(b"VERSION ") || at == 0 || data[at - 1] != b'\n' {
        return None;
    }
    let digits = data[at + 8..].iter().take_while(|c| c.is_ascii_digit() || **c == b'.').count();
    if digits == 0 || data.get(at + 8 + digits) != Some(&b'\n') {
        return None;
    }
    let start = data[..at - 1].iter().rposition(|c| !(0x20..0x7f).contains(c)).map_or(0, |p| p + 1);
    let date = &data[start..at - 1];
    if date.len() < 20 || date.len() > 32 || !date.iter().all(|c| (0x20..0x7f).contains(c)) || !date.ends_with(&[b' ']) {
        return None;
    }
    let mut end = at + 8 + digits + 1;
    for line in 0..2 {
        let len = printable_run(data, end, 120);
        if len == 0 || data.get(end + len) != Some(&b'\n') {
            return None;
        }
        if line == 1 && !data[end..end + len].windows(8).any(|w| w == b"IP32prom") {
            return None;
        }
        end += len + 1;
    }
    Some((start, end - 1))
}

/// The banner lines in one buffer: "SGI Version ..." (every monitor since 4.0), the
/// 4D-era "Version 4D1-... SGI" and "Version n.n IPn ... SGI" forms, and the O2 record.
fn scan(data: &[u8]) -> Vec<(usize, String)> {
    let text = |a: usize, b: usize| data[a..b].iter().map(|c| *c as char).collect::<String>();
    let mut found = Vec::new();
    for at in 0..data.len() {
        let rest = &data[at..];
        if rest.starts_with(b"SGI Version ") {
            let run = printable_run(data, at + 12, 100);
            if run >= 4 {
                found.push((at, text(at, at + 12 + run)));
            }
        } else if rest.starts_with(b"Version 4D1-") {
            if let Some(end) = up_to_sgi(data, at + 12, 4, 100) {
                found.push((at, text(at, end)));
            }
        } else if rest.starts_with(b"Version ") {
            let digits = rest[8..].iter().take_while(|c| c.is_ascii_digit() || **c == b'.').count();
            let ip = &rest[8 + digits..];
            if digits > 0 && ip.starts_with(b" IP") {
                let n = ip[3..].iter().take_while(|c| c.is_ascii_digit()).count();
                if n > 0 && ip.get(3 + n) == Some(&b' ') {
                    if let Some(end) = up_to_sgi(data, at + 8 + digits + 3 + n + 1, 4, 60) {
                        found.push((at, text(at, end)));
                    }
                }
            }
        } else if let Some((start, end)) = o2_record(data, at) {
            found.push((start, text(start, end)));
        }
    }
    found
}

/// The version banners a PROM file carries, read from its own bytes without running
/// it: in the file or its payload, and inside the segments it carries gzipped (the
/// IO6 PROM is gzipped whole; the Origin and Fuel CPU PROMs carry the BaseIO monitor).
/// Several are real: the Octane holds its first-stage banner and the main monitor's.
pub fn banners(file: &[u8]) -> Vec<Banner> {
    let Ok(payload) = decode(file, 64 << 20) else { return Vec::new() };
    let info = inspect(file);
    let mut out = Vec::new();
    let base = file.len() - payload.len().min(file.len());
    match info.segment.as_ref().filter(|s| s.gzip()) {
        Some(seg) => out.extend(scan(&payload).into_iter().map(|(offset, text)| Banner {
            text,
            at: Place::Segment { name: seg.name.clone(), offset },
        })),
        None => out.extend(scan(&payload).into_iter().map(|(at, text)| Banner { text, at: Place::File(base + at) })),
    }
    for seg in nested(&payload) {
        if let Ok(inner) = seg.unpack(&payload, 64 << 20) {
            out.extend(scan(&inner).into_iter().map(|(offset, text)| Banner {
                text,
                at: Place::Segment { name: seg.name.clone(), offset },
            }));
        }
    }
    out
}

/// The payload behind the installer's header: "PROM", then the payload's
/// length at +12 (big-endian), in 256 bytes. Recognized only when that length
/// matches the file, so a raw image that happens to start "PROM" stays raw.
fn installer_payload(file: &[u8]) -> Option<&[u8]> {
    let payload = file.get(0x100..)?;
    let said = u32::from_be_bytes(file[12..16].try_into().unwrap()) as usize;
    (file.starts_with(b"PROM") && said == payload.len()).then_some(payload)
}

/// Accept raw payload bytes, the installer's header form or the supported
/// stored/gzip container subset. Gzip output is bounded before decoding, and
/// exactly one member is required.
pub fn decode(file: &[u8], payload_limit: usize) -> io::Result<Vec<u8>> {
    if let Some(payload) = installer_payload(file) {
        if payload.len() > payload_limit {
            return Err(invalid("PROM image exceeds flash capacity"));
        }
        return Ok(payload.to_vec());
    }
    if !is_container(file) {
        if file.len() > payload_limit {
            return Err(invalid("PROM image exceeds flash capacity"));
        }
        return Ok(file.to_vec());
    }
    if file.len() < 4096 {
        return Err(invalid("truncated SGI PROM container header"));
    }
    if word(file, 0x48) != 6 {
        return Err(invalid("unsupported SGI PROM container version"));
    }
    if word(file, 0x50) != file.len() as u64 {
        return Err(invalid("SGI PROM declared end does not match file size"));
    }
    if u32::from_be_bytes(file[0x7c..0x80].try_into().unwrap()) != 1 {
        return Err(invalid("unsupported SGI PROM segment count"));
    }
    let seg = Segment::read(file, 0x80).unwrap();
    if seg.offset != 4096 {
        return Err(invalid("unsupported SGI PROM payload offset"));
    }
    let flags = seg.flags;
    if flags != 0x10 && flags != 0x13 {
        return Err(invalid("unsupported SGI PROM flags"));
    }
    let length = seg.length;
    if length == 0 || length > payload_limit as u64 {
        return Err(invalid(
            "SGI PROM decoded length is zero or exceeds capacity",
        ));
    }
    if flags == 0x10 {
        if seg.stored != 0 || seg.sum_stored != 0 || seg.plain != length {
            return Err(invalid("unsupported SGI PROM uncompressed metadata"));
        }
        if length != (file.len() - 4096) as u64 {
            return Err(invalid("invalid SGI PROM stored length"));
        }
    } else {
        let stored = seg.stored;
        if stored == 0 || stored > (file.len() - 4096) as u64 {
            return Err(invalid("invalid SGI PROM stored length"));
        }
        let end = 4096 + stored as usize;
        if file.len() - end > 7 || file[end..].iter().any(|b| *b != 0) {
            return Err(invalid("unsupported SGI PROM trailing bytes"));
        }
    }
    seg.unpack(file, payload_limit)
}

#[cfg(test)]
mod tests {
    // Cases follow tests/unit/test-sgi-prom-image.c in
    // jamesbraid/qemu-sgi-origami: wrapped, gzip, raw, malformed and files.
    use super::*;
    use std::io::Write;
    fn put(v: &mut [u8], off: usize, n: u64) {
        v[off..off + 8].copy_from_slice(&n.to_be_bytes());
    }
    fn container(payload: &[u8], gzip: bool) -> Vec<u8> {
        let stored = if gzip {
            let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            enc.write_all(payload).unwrap();
            enc.finish().unwrap()
        } else {
            payload.to_vec()
        };
        let mut v = vec![0; 4096];
        put(&mut v, 0x40, CONTAINER_MAGIC);
        put(&mut v, 0x48, 6);
        put(&mut v, 0x50, (4096 + stored.len()) as u64);
        v[0x7c..0x80].copy_from_slice(&1u32.to_be_bytes());
        put(&mut v, 0x90, if gzip { 0x13 } else { 0x10 });
        put(&mut v, 0x98, 4096);
        put(&mut v, 0xb0, payload.len() as u64);
        put(&mut v, 0xc0, byte_sum(payload));
        if gzip {
            put(&mut v, 0xb8, stored.len() as u64);
            put(&mut v, 0xc8, byte_sum(&stored));
        } else {
            put(&mut v, 0xd0, payload.len() as u64);
        }
        v.extend(stored);
        v
    }
    /// Equivalent PROM payloads must boot from each supported image format.
    #[test]
    fn raw_stored_and_gzip_images_load_with_the_same_capacity_limit() {
        let path = crate::test_temp::TempPath::new("prom-load");
        for file in [vec![9, 8, 7, 6], container(&[9, 8, 7, 6], false), container(&[9, 8, 7, 6], true)] {
            std::fs::write(&path, file).unwrap();
            assert_eq!(load(&path, 4).unwrap(), [9, 8, 7, 6]);
            assert!(load(&path, 3).is_err(), "decoded payload exceeds capacity");
        }
        std::fs::write(&path, []).unwrap();
        assert!(load(&path, 4).unwrap().is_empty());
    }

    /// An invalid PROM must identify its input file instead of executing a
    /// malformed container.
    #[test]
    fn invalid_or_unavailable_images_report_their_path() {
        let path = crate::test_temp::TempPath::new("prom-invalid");
        let error = load(&path, 4).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(error.to_string().contains(&path.display().to_string()));
        let reject = |file: &[u8]| {
            std::fs::write(&path, file).unwrap();
            let error = load(&path, 4).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert!(error.to_string().contains(&path.display().to_string()));
        };
        reject(&vec![0; 5000]); // encoded file bound
        let original = container(&[4, 3, 2, 1], false);
        reject(&original[..100]); // recognized but truncated header
        for (off, n) in [(0x48, 7), (0x50, 4096), (0x90, 0x14), (0x98, 4095), (0xb0, 0), (0xd0, 3)] {
            let mut file = original.clone();
            put(&mut file, off, n);
            reject(&file);
        }
        for gzip in [false, true] {
            let mut file = container(&[4, 3, 2, 1], gzip);
            file[0xc7] ^= 1;
            reject(&file);
        }
        let gzip = container(&[4, 3, 2, 1], true);
        for tail in [7, 8] {
            let mut file = gzip.clone();
            file.extend(vec![0; tail]);
            let size = file.len();
            put(&mut file, 0x50, size as u64);
            if tail == 7 {
                std::fs::write(&path, &file).unwrap();
                assert_eq!(load(&path, 4).unwrap(), [4, 3, 2, 1]);
            } else {
                reject(&file);
            }
        }
    }

    /// The installer's form: "PROM", a word iris does not read (0x4400 in
    /// the O2's `ip32prom.image`), then the payload length, in 256 bytes.
    fn installer(payload: &[u8], said: u32) -> Vec<u8> {
        let mut v = vec![0; 0x100];
        v[..4].copy_from_slice(b"PROM");
        v[8..12].copy_from_slice(&0x4400u32.to_be_bytes());
        v[12..16].copy_from_slice(&said.to_be_bytes());
        v.extend(payload);
        v
    }

    /// The header is stripped only when its length matches the file; a raw
    /// image that starts "PROM" is still raw. The header is checked before
    /// the size, so an image whose header and payload fill the whole part
    /// still loads from the right place.
    #[test]
    fn installer_header_is_stripped_only_when_its_length_matches() {
        assert_eq!(decode(&installer(&[9, 8, 7, 6], 4), 4).unwrap(), [9, 8, 7, 6]);
        assert!(decode(&installer(&[9, 8, 7, 6], 4), 3).is_err(), "payload exceeds capacity");
        let full = installer(&[0x5a; 0x300], 0x300);
        assert_eq!(decode(&full, full.len()).unwrap(), [0x5a; 0x300]);
        let wrong = installer(&[9, 8, 7, 6], 5);
        assert_eq!(decode(&wrong, wrong.len()).unwrap(), wrong, "a mismatched length is raw");
    }

    /// Every file under `dir`, for the real-image test below.
    fn files_under(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for path in entries.filter_map(|e| e.ok().map(|e| e.path())) {
            if path.is_dir() {
                files_under(&path, out);
            } else {
                out.push(path);
            }
        }
        out.sort();
    }

    /// Real PROM files load to the flash contents each machine runs from:
    /// the Origin, Fuel and Tezro `.img` containers to the `.payload.bin`
    /// beside them, the O2's `ip32prom.image` to the bytes behind its
    /// header, and the Octane's raw `IP30prom.bin` to itself. The images
    /// are not in git; `IRIS_PROM_DIR` (default `ignore/proms`) is searched
    /// for them, whatever the layout, and the test skips without any.
    #[test]
    fn real_prom_files_load_to_their_payload() {
        let dir = std::env::var("IRIS_PROM_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| "ignore/proms".into());
        let mut files = Vec::new();
        files_under(&dir, &mut files);
        let mut checked = 0;
        for path in &files {
            let name = path.file_name().unwrap().to_string_lossy();
            let raw = || std::fs::read(path).unwrap();
            if let Some(stem) = name.strip_suffix(".img").filter(|s| s.ends_with("prom")) {
                // The Origin 200/2000 PROM is a 1 MiB part, the Fuel's and Tezro's 8 MiB.
                let limit = if stem.starts_with("ip27") { 1 << 20 } else { 8 << 20 };
                let Ok(payload) = std::fs::read(path.with_file_name(format!("{stem}.payload.bin"))) else { continue };
                assert_eq!(load(path, limit).unwrap(), payload, "{}", path.display());
            } else if name == "ip32prom.image" {
                let file = raw();
                let payload = load(path, 1 << 19).unwrap();
                assert_eq!(payload, &file[0x100..], "{}: the header is stripped", path.display());
                assert!(payload.len() < 1 << 19, "{}: shorter than the part", path.display());
            } else if name == "IP30prom.bin" {
                assert_eq!(load(path, 1 << 20).unwrap(), raw(), "{}", path.display());
            } else {
                continue;
            }
            checked += 1;
        }
        if checked == 0 {
            eprintln!("SKIPPED: no PROM images under {}", dir.display());
        } else {
            eprintln!("checked {checked} PROM images under {}", dir.display());
        }
    }

    /// A PROM file the user names must load, and wins over the built-in PROM; left at the
    /// default, the files in the working directory are tried and a missing one is no error.
    #[cfg(feature = "embedded-proms")]
    #[test]
    fn a_named_file_wins_and_must_load() {
        use crate::dev::embedded_prom::{self, EmbeddedProm::Ip32};
        let dir = crate::test_temp::TempPath::dir("prom-resolve");
        let (good, bad, missing) = (dir.join("good.bin"), dir.join("bad.bin"), dir.join("missing.bin"));
        std::fs::write(&good, [1, 2, 3, 4]).unwrap();
        std::fs::write(&bad, vec![0; 5000]).unwrap(); // over the limit
        let run = |prom: &std::path::Path, implicit: &[&std::path::Path]| {
            let implicit: Vec<&str> = implicit.iter().map(|p| p.to_str().unwrap()).collect();
            resolve_with(Ip32, prom.to_str().unwrap(), &implicit, 4096)
        };
        let default = std::path::Path::new(crate::config::DEFAULT_PROM);
        let built_in = embedded_prom::get(Ip32).unwrap();
        // A named file wins, and a named file that will not load is an error.
        assert_eq!(run(&good, &[&missing]).unwrap().bytes, [1, 2, 3, 4]);
        assert!(run(&missing, &[&good]).err().unwrap().contains("missing.bin"));
        assert!(run(&bad, &[&good]).is_err());
        // At the default, an implicit file is used when there, skipped when missing, and an
        // error when it is there but will not load.
        assert_eq!(run(default, &[&missing, &good]).unwrap().bytes, [1, 2, 3, 4]);
        let none = run(default, &[&missing]).unwrap();
        assert!(none.source.is_none() && none.bytes == built_in);
        assert!(run(default, &[&bad, &good]).is_err());
        // The keyword is the built-in PROM whatever files exist.
        let keyword = resolve_with(Ip32, "(embedded)", &[good.to_str().unwrap()], 4096).unwrap();
        assert!(keyword.source.is_none() && keyword.bytes == built_in);
    }

    /// Every profile starts from its built-in PROM when no file is named.
    #[cfg(feature = "embedded-proms")]
    #[test]
    fn every_machine_starts_without_a_file() {
        for p in crate::config::MachineProfile::EVERY {
            let loaded = resolve(p, "(embedded)", 8 << 20).unwrap();
            assert_eq!(loaded.bytes, crate::dev::embedded_prom::get(p.embedded_prom()).unwrap(), "{p:?}");
        }
    }

    /// Without the `embedded-proms` feature the built-in PROM is an error naming the feature.
    #[cfg(not(feature = "embedded-proms"))]
    #[test]
    fn without_the_feature_a_machine_needs_a_prom_file() {
        let e = resolve(crate::config::MachineProfile::OctaneIp30, "(embedded)", 8 << 20).err().unwrap();
        assert!(e.contains("embedded-proms") && e.contains("Ip30"), "{e}");
    }

    /// `inspect` tells the three forms apart and reads the version each states.
    #[test]
    fn inspect_reads_each_form() {
        let raw = inspect(&[1, 2, 3, 4]);
        assert_eq!((raw.kind, raw.version, raw.segment.is_none()), (Kind::Raw, None, true));
        let mut installer = vec![0u8; 0x100];
        installer[..4].copy_from_slice(b"PROM");
        installer[12..16].copy_from_slice(&4u32.to_be_bytes());
        installer[0x10..0x14].copy_from_slice(&4u32.to_be_bytes());
        installer[0x14..0x18].copy_from_slice(&18u32.to_be_bytes());
        installer.extend([0x0b, 0xf0, 0x00, 0xf0]);
        let info = inspect(&installer);
        assert_eq!((info.kind, info.version.as_deref()), (Kind::Installer, Some("4.18")));
        let mut file = container(&[9, 8, 7, 6], true);
        put(&mut file, 0x38, 156);
        let info = inspect(&file);
        assert_eq!((info.kind, info.version.as_deref()), (Kind::Container, Some("6.156")));
        let seg = info.segment.as_ref().unwrap();
        assert!(seg.gzip() && seg.length == 4 && seg.offset == 4096);
        assert_eq!(seg.unpack(&file, 1 << 20).unwrap(), [9, 8, 7, 6]);
    }

    /// A segment record carried inside a payload is found by `nested`, unpacks through
    /// its own container, and a bad checksum is refused.
    #[test]
    fn a_nested_segment_is_found_and_unpacked() {
        let inner = b"the BaseIO board's PROM".repeat(40);
        let mut copy = container(&inner, true);
        copy[0x80..0x87].copy_from_slice(b"io7prom");
        let mut payload = vec![0x5a; 0x300];
        payload.extend(&copy);
        payload.extend([0u8; 16]);
        let found = nested(&payload);
        assert_eq!(found.len(), 1, "{found:?}");
        let seg = &found[0];
        assert_eq!((seg.name.as_str(), seg.record, seg.start), ("io7prom", 0x380, 0x300));
        assert_eq!(seg.unpack(&payload, 1 << 20).unwrap(), inner);
        let mut damaged = payload.clone();
        let last = seg.start + 4096 + seg.stored as usize - 1;
        damaged[last] ^= 1;
        assert!(seg.unpack(&damaged, 1 << 20).is_err(), "the stored checksum catches it");
        assert!(seg.unpack(&payload, 100).is_err(), "over the limit");
        for fill in [0x00, 0x5a, 0xff] {
            assert!(nested(&vec![fill; 0x400]).is_empty(), "{fill:#x} filler is not a segment");
        }
    }

    /// A record with absurd lengths neither panics nor unpacks, and `inspect`, `nested`
    /// and `decode` survive truncated and filled buffers of every size around the
    /// header boundaries, with and without the container magic.
    #[test]
    fn hostile_input_is_refused_not_a_panic() {
        let mut record = vec![0u8; 0x200];
        record[0x80..0x87].copy_from_slice(b"io7prom");
        for (off, n) in [(0x90, 0x13), (0x98, 4096), (0xb0, u64::MAX), (0xb8, u64::MAX)] {
            put(&mut record, off, n);
        }
        let found = Segment::read(&record, 0x80).unwrap();
        assert!(found.unpack(&record, 1 << 20).is_err());
        assert!(!found.plausible(&record));
        let erased = vec![0xff; 0x200];
        assert!(!Segment::read(&erased, 0x80).unwrap().plausible(&erased), "0xff words are no offset");
        for len in [0, 1, 0x47, 0x48, 0x57, 0x58, 0x7f, 0x80, 0xd8, 0x100, 4095, 4096, 4097, 9000] {
            for fill in [0x00u8, 0x5a, 0xff] {
                let mut buf = vec![fill; len];
                for magic in [false, true] {
                    if magic && len >= 0x48 {
                        put(&mut buf, 0x40, CONTAINER_MAGIC);
                    }
                    let _ = inspect(&buf);
                    let _ = nested(&buf);
                    let _ = decode(&buf, 1 << 20);
                }
            }
        }
    }

    /// A 4 KiB erased-flash file is a raw image, and a version-seven container is refused.
    #[test]
    fn erased_flash_and_an_unknown_version_are_handled() {
        let erased = vec![0xff; 4096];
        assert_eq!(inspect(&erased).kind, Kind::Raw);
        assert!(nested(&erased).is_empty());
        let mut seven = container(&[1, 2, 3, 4], false);
        put(&mut seven, 0x48, 7);
        assert!(decode(&seven, 1 << 20).is_err());
        assert_eq!(inspect(&seven).version.as_deref(), Some("7.0"));
        assert_eq!(inspect(&seven[..100]).version, None);
        assert!(decode(&seven[..100], 1 << 20).is_err(), "a truncated container header");
    }

    /// Each banner shape is found whole, in a raw file, in a gzipped container and in a
    /// segment a payload carries; filler and near misses are not banners.
    #[test]
    fn banners_are_found_in_every_place_they_sit() {
        let mut raw = vec![0u8; 64];
        raw.extend(b"SGI Version 6.5 Rev 4.17 IP30 Jan 20, 2004\0junk");
        raw.extend(b"\0\0Version 4D1-3.1 Feb 03, 1992 SGI\0");
        raw.extend(b"\0Version 4.0 IP7 Mar 02, 1990 SGI\0");
        raw.extend(b"\0Tue Oct 22 10:58:00 PDT 2002 \nVERSION 4.18\nO2 R5K\nIRIX 6.5.x IP32prom v4\n");
        let found = banners(&raw);
        let texts: Vec<&str> = found.iter().map(|b| b.text.as_str()).collect();
        assert_eq!(texts, [
            "SGI Version 6.5 Rev 4.17 IP30 Jan 20, 2004",
            "Version 4D1-3.1 Feb 03, 1992 SGI",
            "Version 4.0 IP7 Mar 02, 1990 SGI",
            "Tue Oct 22 10:58:00 PDT 2002 \nVERSION 4.18\nO2 R5K\nIRIX 6.5.x IP32prom v4",
        ]);
        assert_eq!(found[0].at, Place::File(64));
        // The same text inside a gzipped container names the segment.
        let mut zipped = container(&raw, true);
        zipped[0x80..0x86].copy_from_slice(b"io6pro");
        let found = banners(&zipped);
        assert_eq!(found[0].at, Place::Segment { name: "io6pro".into(), offset: 64 });
        // And a record the payload carries holds its own.
        let mut inner = b"padding SGI Version 6.156  built 11:26:28 AM Nov 18, 2003\0".to_vec();
        inner.extend(vec![0u8; 600]); // compresses, so the text is not stored in the clear
        let mut copy = container(&inner, true);
        copy[0x80..0x87].copy_from_slice(b"io6prom");
        let mut payload = vec![0x5a; 0x300];
        payload.extend(&copy);
        payload.extend([0u8; 16]);
        let found = banners(&payload);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].at, Place::Segment { name: "io6prom".into(), offset: 8 });
        assert!(banners(&vec![0xff; 4096]).is_empty() && banners(b"SGI Version ").is_empty());
    }
}
